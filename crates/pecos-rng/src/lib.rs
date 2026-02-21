pub mod prelude;
pub mod quality_rng;
pub mod rng;
pub mod rng_ext;
pub mod rng_manageable;
pub mod rng_pcg;
pub mod scalar_rng;

// Re-export key types at crate root for convenience
pub use quality_rng::{PecosQualityRng, SimdXoshiro256PlusPlus};
pub use rng::{ParallelRapidRng, PecosRng};
pub use rng_ext::{RngBulkExt, RngProbabilityExt};
pub use rng_manageable::{RngManageable, derive_seed};
pub use rng_pcg::{PCG64Fast, PCGRandom};
pub use scalar_rng::PecosScalarRng;

// Re-export rand_core traits
// Note: In rand 0.10, RngCore was renamed to Rng, and TryRngCore to TryRng
pub use rand_core::{Rng, SeedableRng, TryRng};

// Backwards compatibility alias (RngCore was renamed to Rng in rand_core 0.10)
#[allow(deprecated)]
pub use rand_core::RngCore;

// Re-export RngExt trait from rand for convenience (provides .random(), .random_range(), etc.)
// Note: In rand 0.10, the old Rng trait was renamed to RngExt
pub use rand::RngExt;
