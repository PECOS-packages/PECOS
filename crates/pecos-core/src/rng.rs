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

pub mod choices;
pub mod recording_rng;
pub mod replaying_rng;
pub mod rng_manageable;
pub mod rng_utils;

pub use recording_rng::RecordingRng;
pub use replaying_rng::ReplayingRng;
pub use rng_manageable::{RngManageable, derive_seed};

// Re-export RngProbabilityExt from pecos-random for convenience
pub use pecos_random::rng_ext::RngProbabilityExt;

// Export the utility functions from rng_utils
pub use rng_utils::{choose_weighted, coin_flip, gen_bools};

// Export the new RandomUtils struct
pub use rng_utils::RandomUtils;
