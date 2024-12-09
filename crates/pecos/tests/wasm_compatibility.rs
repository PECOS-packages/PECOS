// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

// wasm_bindgen_test_configure!(run_in_browser);

use pecos::sets::{Set, VecSet};
use pecos_qsims::SparseStab;
// use pecos-qsim;

#[wasm_bindgen_test]
fn wasm_compatibility_test() {
    // Small example program using your library
    let mut state = SparseStab::<VecSet<u32>, u32>::new(3);

    // Perform some operations
    state.h(0);
    state.cx(0, 1);
    state.cz(1, 2);

    // Perform measurements
    let (m0, _) = state.mz(0);
    let (m1, _) = state.mx(1);
    let (m2, _) = state.my(2);

    // Check that the operations completed without error
    // We can't assert specific measurement outcomes as they're probabilistic
    println!("Measurements: {} {} {}", m0, m1, m2);

    // Test VecSet operations
    let mut set = VecSet::<u32>::new();
    set.insert(1);
    set.insert(2);
    set.insert(3);

    assert_eq!(set.len(), 3);
    assert!(set.contains(&2));

    set.remove(&2);
    assert_eq!(set.len(), 2);
    assert!(!set.contains(&2));

    println!("WASM compatibility test passed!");
}
