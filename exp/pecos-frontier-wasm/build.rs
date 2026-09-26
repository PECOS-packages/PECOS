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

const MAX_BITS: usize = 128;

fn main() {
    println!("cargo:rerun-if-env-changed=FRONTIER_DEM_PATH");
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
        parsed.num_detectors <= MAX_BITS && parsed.num_observables <= MAX_BITS,
        "DEM {} exceeds the 128-detector or 128-observable WebAssembly ABI limit",
        source.display()
    );
    fs::write(output, dem).expect("failed to stage embedded DEM");
}
