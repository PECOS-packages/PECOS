// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_decoder_core::dem::SparseDem;
use std::{env, fmt::Display, fs, path::PathBuf, str::FromStr};

#[cfg(feature = "replay")]
#[path = "src/replay.rs"]
#[allow(dead_code)]
mod replay;
const MAX_OBSERVABLES: u32 = 128;

fn main() {
    println!("cargo:rerun-if-env-changed=FRONTIER_DEM_PATH");
    println!("cargo:rerun-if-env-changed=FRONTIER_K");
    println!("cargo:rerun-if-env-changed=FRONTIER_DELTA");
    println!("cargo:rerun-if-env-changed=FRONTIER_REPLAY_PATH");
    println!("cargo:rerun-if-changed=model.dem");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = env::var_os("FRONTIER_DEM_PATH")
        .map_or_else(|| manifest_dir.join("model.dem"), PathBuf::from);
    println!("cargo:rerun-if-changed={}", source.display());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("model.dem");

    let dem = fs::read_to_string(&source)
        .unwrap_or_else(|error| panic!("failed to read DEM {}: {error}", source.display()));
    let parsed = SparseDem::from_dem_str(&dem)
        .unwrap_or_else(|error| panic!("invalid flattened DEM {}: {error}", source.display()));
    assert!(
        parsed.num_observables <= MAX_OBSERVABLES as usize,
        "DEM {} exceeds the 128-observable WebAssembly result ABI",
        source.display()
    );
    fs::write(output, dem).expect("failed to stage embedded DEM");
    stage_search_parameters();

    #[cfg(feature = "replay")]
    stage_replay(&parsed);
}

/// Validate optional `FRONTIER_K` / `FRONTIER_DELTA` so a bad value fails the build
/// instead of the deployed module's `init`. The rules mirror `TrellisConfig`.
fn stage_search_parameters() {
    let k = parsed_env::<u64>("FRONTIER_K");
    if let Some(k) = k {
        let pointer_width: u32 = env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            k > 0 && (pointer_width >= u64::BITS || k < 1 << pointer_width),
            "FRONTIER_K={k} must be positive and fit the target usize"
        );
    }
    let delta = parsed_env::<f64>("FRONTIER_DELTA");
    if let Some(delta) = delta {
        assert!(
            delta >= 0.0,
            "FRONTIER_DELTA={delta} must be nonnegative and not NaN"
        );
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("search_parameters.rs");
    fs::write(
        output,
        format!(
            "const FRONTIER_K: Option<usize> = {k:?};\n\
             const FRONTIER_DELTA_BITS: Option<u64> = {:?};\n",
            delta.map(f64::to_bits)
        ),
    )
    .expect("failed to stage search parameters");
}

fn parsed_env<T: FromStr>(name: &str) -> Option<T>
where
    T::Err: Display,
{
    let value = env::var_os(name)?;
    let value = value
        .to_str()
        .unwrap_or_else(|| panic!("{name} must be valid UTF-8"));
    Some(
        value
            .parse()
            .unwrap_or_else(|error| panic!("invalid {name}={value:?}: {error}")),
    )
}

#[cfg(feature = "replay")]
fn stage_replay(parsed: &SparseDem) {
    let replay = if let Some(path) = env::var_os("FRONTIER_REPLAY_PATH") {
        let path = PathBuf::from(path);
        println!("cargo:rerun-if-changed={}", path.display());
        fs::read(&path).unwrap_or_else(|error| {
            panic!("failed to read replay fixture {}: {error}", path.display())
        })
    } else {
        let mut empty = b"FWR1".to_vec();
        empty.extend_from_slice(&2_u32.to_le_bytes());
        empty.extend_from_slice(&[0_u8; 12]);
        empty
    };
    let fixture = replay::Replay::parse(&replay).expect("invalid FRONTIER_REPLAY_PATH");
    if fixture.shots > 0 {
        assert_eq!(
            fixture.detectors, parsed.num_detectors,
            "replay detector width does not match DEM"
        );
        assert_eq!(
            fixture.observables, parsed.num_observables,
            "replay observable width does not match DEM"
        );
    }
    let replay_output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("replay.fwr");
    fs::write(replay_output, replay).expect("failed to stage embedded replay fixture");
}
