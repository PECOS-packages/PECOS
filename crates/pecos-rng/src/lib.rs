pub mod prelude;
pub mod quality_rng;
pub mod rng;
pub mod rng_ext;
pub mod rng_manageable;
pub mod rng_pcg;
pub mod scalar_rng;

// Re-export key types at crate root for convenience
pub use quality_rng::{PecosQualityRng, SimdXoshiro256PlusPlus};
pub use rand_core::{RngCore, SeedableRng};
pub use rng::{ParallelRapidRng, PecosRng};
pub use rng_ext::{RngBulkExt, RngProbabilityExt};
pub use rng_manageable::{RngManageable, derive_seed};
pub use rng_pcg::{PCG64Fast, PCGRandom};
pub use scalar_rng::PecosScalarRng;

// Re-export Rng trait from rand for convenience (provides .gen(), .gen_range(), etc.)
pub use rand::Rng;
