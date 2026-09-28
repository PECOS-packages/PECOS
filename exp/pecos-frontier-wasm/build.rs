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
use std::{env, fs, path::PathBuf};

const REPLAY_HEADER_BYTES: usize = 20;
const MAX_OBSERVABLES: u32 = 128;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn validate_replay(bytes: &[u8], source: &str) {
    assert!(
        bytes.len() >= REPLAY_HEADER_BYTES && &bytes[..4] == b"FWR1",
        "invalid Frontier replay fixture header in {source}"
    );
    let version = read_u32(bytes, 4);
    let shots = read_u32(bytes, 8) as usize;
    let detectors = read_u32(bytes, 12);
    let observables = read_u32(bytes, 16);
    assert!(
        version == 1 || version == 2,
        "unsupported Frontier replay fixture version {version} in {source}"
    );
    assert!(
        observables <= MAX_OBSERVABLES,
        "Frontier replay fixture in {source} exceeds the 128-observable result ABI"
    );
    let record_bytes = if version == 1 {
        assert!(
            detectors <= 128,
            "version-1 replay fixture in {source} exceeds its 128-detector limit"
        );
        32
    } else {
        let detector_words = usize::try_from(detectors.div_ceil(32)).unwrap();
        let observable_words = usize::try_from(observables.div_ceil(32)).unwrap();
        (detector_words + observable_words) * 4
    };
    let expected = REPLAY_HEADER_BYTES
        .checked_add(
            shots
                .checked_mul(record_bytes)
                .expect("replay fixture is too large"),
        )
        .expect("replay fixture is too large");
    assert_eq!(
        bytes.len(),
        expected,
        "invalid Frontier replay fixture length in {source}"
    );
}

fn main() {
    println!("cargo:rerun-if-env-changed=FRONTIER_DEM_PATH");
    println!("cargo:rerun-if-env-changed=FRONTIER_REPLAY_PATH");
    println!("cargo:rerun-if-changed=model.dem");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = env::var_os("FRONTIER_DEM_PATH")
        .map_or_else(|| manifest_dir.join("model.dem"), PathBuf::from);
    println!("cargo:rerun-if-changed={}", source.display());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("model.dem");

    let dem = fs::read_to_string(&source)
        .unwrap_or_else(|error| panic!("failed to read DEM {}: {error}", source.display()));
    if dem.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("repeat") || line.starts_with("shift_detectors")
    }) {
        panic!(
            "{} is not flattened; flatten it first (for example with stim.DetectorErrorModel.flattened())",
            source.display()
        );
    }
    let parsed = SparseDem::from_dem_str(&dem)
        .unwrap_or_else(|error| panic!("invalid flattened DEM {}: {error}", source.display()));
    assert!(
        parsed.num_observables <= MAX_OBSERVABLES as usize,
        "DEM {} exceeds the 128-observable WebAssembly result ABI",
        source.display()
    );
    fs::write(output, dem).expect("failed to stage embedded DEM");

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
    validate_replay(&replay, "FRONTIER_REPLAY_PATH");
    if read_u32(&replay, 8) > 0 {
        assert_eq!(
            read_u32(&replay, 12) as usize,
            parsed.num_detectors,
            "replay detector width does not match DEM {}",
            source.display()
        );
        assert_eq!(
            read_u32(&replay, 16) as usize,
            parsed.num_observables,
            "replay observable width does not match DEM {}",
            source.display()
        );
    }
    let replay_output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("replay.fwr");
    fs::write(replay_output, replay).expect("failed to stage embedded replay fixture");
}
