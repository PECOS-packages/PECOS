//! Gate support declaration for foreign simulators.
//!
//! Foreign simulators implicitly declare their native gate support through
//! the vtable: `sz`, `h`, `cx`, `mz` are always supported, and `rx`, `rz`, `rzz`
//! are supported if the corresponding function pointers are non-null.
//!
//! For integration with pecos-neo's `CircuitRunner`, this module provides helpers
//! that configure the runner's decomposition system based on what the foreign
//! simulator actually supports. Gates the simulator doesn't support are automatically
//! decomposed into supported primitives.
//!
//! # Example
//!
//! ```rust,ignore
//! let sim = ForeignSimulator::new(handle, vtable);
//! let runner = configure_runner_for_foreign(&sim);
//! // runner will decompose unsupported gates into {SZ, H, CX, MZ, RX?, RZ?, RZZ?}
//! ```

use crate::simulator::ForeignSimulator;
use pecos_neo::prelude::CircuitRunner;

/// Create a `CircuitRunner<ForeignSimulator>` configured for the foreign simulator's
/// native gate set.
///
/// The runner will:
/// - Execute `SZ`, `H`, `CX`, `MZ` natively (always supported)
/// - Execute `RX`, `RZ`, `RZZ` natively if the simulator supports rotations
/// - Decompose everything else (X, Y, Z, SWAP, T, RXX, RYY, etc.) into the above
///
/// The decomposition uses pecos-neo's `GateDefinitions` which provides standard
/// decomposition rules (e.g., SWAP -> 3 CX gates, T -> RZ(pi/4), X -> H SZ SZ H).
#[must_use]
pub fn configure_runner_for_foreign(sim: &ForeignSimulator) -> CircuitRunner<ForeignSimulator> {
    if sim.supports_rotations() {
        // Use the rotations constructor which enables RX, RZ, RZZ, T, Tdg, etc.
        CircuitRunner::<ForeignSimulator>::rotations()
    } else {
        // Clifford-only: SZ, H, CX, MZ + decompositions for X, Y, Z, SWAP, etc.
        CircuitRunner::<ForeignSimulator>::new()
    }
}
