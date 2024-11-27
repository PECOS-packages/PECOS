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

use super::sim_rng::SimRng;
pub use rand_xoshiro::{
    Xoshiro128PlusPlus, Xoshiro128StarStar, Xoshiro256PlusPlus, Xoshiro256StarStar,
    Xoshiro512PlusPlus, Xoshiro512StarStar,
};

impl SimRng for Xoshiro128PlusPlus {} // Recommended for 32-bit systems
impl SimRng for Xoshiro256PlusPlus {} // Recommended for 64-bit systems
impl SimRng for Xoshiro512PlusPlus {} // Recommended if paranoid about parallelization (bigger state)

// StarStar can be faster for some systems
impl SimRng for Xoshiro128StarStar {}
impl SimRng for Xoshiro256StarStar {}
impl SimRng for Xoshiro512StarStar {}
