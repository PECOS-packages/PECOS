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
pub use rand_chacha::{ChaCha12Rng, ChaCha20Rng, ChaCha8Rng};

impl SimRng for ChaCha8Rng {} // fastest but less cryptographically secure
impl SimRng for ChaCha12Rng {}
impl SimRng for ChaCha20Rng {} // slowest but most cryptographically secure
