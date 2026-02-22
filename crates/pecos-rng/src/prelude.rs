// Copyright 2025 The PECOS Developers
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

//! A prelude for users of the `pecos-rng` crate.
//!
//! This prelude re-exports the PCG random number generator module.

// Re-export RNG modules
pub use crate::quality_rng;
pub use crate::rng;
pub use crate::rng_ext;
pub use crate::rng_manageable;
pub use crate::rng_pcg;
pub use crate::scalar_rng;

// Re-export rand traits for convenience
pub use rand::RngExt;
pub use rand_core::{Rng, SeedableRng, TryRng};

// Re-export types
pub use crate::quality_rng::{PecosQualityRng, SimdXoshiro256PlusPlus};
pub use crate::rng::{ParallelRapidRng, PecosRng};
pub use crate::rng_ext::{RngBulkExt, RngProbabilityExt};
pub use crate::rng_manageable::{RngManageable, derive_seed, resolve_seed, time_seed};
pub use crate::rng_pcg::{PCG64Fast, PCGRandom};
pub use crate::scalar_rng::PecosScalarRng;
