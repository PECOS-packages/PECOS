//! Foundations for loss-conditioned detector error models.
//!
//! Locations use tick, stored batch, operand, and side. Records always use emission
//! indices, never measurement IDs. Branch enumeration and decoding are separate work.
//!
//! The built-settings bridge is an allow-list:
//!
//! | Engine setting | `NoiseConfig` target | Conversion contract |
//! | --- | --- | --- |
//! | `p1`, `p1_pauli_model` | `p1`, `p1_weights` | Same ordinary DEM gate channel and signature fit; residual retained |
//! | `p2`, `p2_pauli_model` | `p2`, `p2_weights` | Same ordinary DEM ordered-pair channel and signature fit; residual retained |
//! | `p_prep` | `p_prep` | X after PZ; both engines and the ordinary DEM omit MPZ's Pauli prep half |
//! | symmetric `p_meas_0/1` | `p_meas` | No surviving measurement-X effect on later records |
//! | linear idle rate/model | `p_idle_linear` | Rates at or below `f64::EPSILON` are zero, matching the engine; same `TimeUnits` and axis weights |
//! | noiseless unitary types | `p1_gate_rates`, `p2_gate_rates` = 0 | Shared engine exemption predicate, including phase-shaped U |
//!
//! Unsupported nondefault settings are errors even when inert. In particular, leakage,
//! seepage, emission, crosstalk, non-loss hooks, coherent/nonlinear idle, angle dependence,
//! and record-consuming readout rows are rejected. Detect rows are allowed because flags
//! are trusted. Record-free Z-herald rows are allowed because they never enter detectors.
//! `idle_after_2q` is allowed only with zero Pauli idle noise; it still matters for future
//! loss branches. Noiseless PZ/MZ/Idle require zero matching rates; MPZ checks only measurement noise.
//!
//! The neo builder allow-list remains unchanged. This bridge works on scaled built values,
//! accepts custom Pauli/linear-idle models with ordinary DEM fit diagnostics and per-type exemptions, and handles loss
//! through location channels. Unlike neo it rejects asymmetric readout and inert seepage
//! or leakage-scale settings. Configuring a sampler equal to the compiled default is
//! indistinguishable from leaving it unset and has identical sampling semantics.
//!
//! Measurement-noise caveat: PZ; MZ(record 0); MZ(record 1), with `p_meas=1`,
//! returns [1,1] from the engine but the ordinary DEM predicts flips [1,0]. Its X before
//! the first measurement survives into the second; the engine flips only the reported
//! bit. The bridge rejects this lifecycle (and cross-qubit downstream effects), without
//! changing the existing `DemBuilder` model.
//!
//! Recovery false alarms use the shared inline categorical conversion: each of X/Y/Z
//! has independent probability `(1 - sqrt(1 - FP))/2`, per present operand. Before-site
//! Paulis are conjugated forward as a whole through the location's resolved Clifford.
//! E2 must replace these terms at sites with lost operands. Event labels containing L
//! contribute identity here, a first-order background approximation.
//!
//! General categorical channels need not admit exact independent nonnegative mechanisms.
//! Binary and representable uniform channels retain the exact inline conversion. Other
//! distributions use the ordinary `DemBuilder` signature fit, including its approximate
//! nonnegative boundary fit. `DetectorErrorModel::idle_noise_residuals()` reports the
//! location, channel kind, discrepancy signature, total-variation magnitude and requested
//! nonidentity channel weight, just as for ordinary gate noise. No-loss Pauli mechanisms
//! and their residuals are identical to the ordinary builder, including uniform defaults.
//! No branch enumeration, shot-dependent DEM mixing, or decoder construction is included.
mod bridge;
mod channels;
mod runner;
pub use bridge::pauli_noise_config;
pub use channels::{LocationChannel, background_dem};
mod symbolic;
pub use crate::fault_tolerance::fault_sampler::{AffineParity, AffineParityError, affine_parity};
pub use runner::{OutputSlot, OutputTable, Probe, ProbeId, Shot, TickRunner};
pub use symbolic::{Span, located_pauli_flips};

/// Side of a stored gate batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Immediately before the operation.
    Before,
    /// Immediately after the operation.
    After,
}

/// Position in a tick circuit's stored batch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    /// Tick index.
    pub tick: usize,
    /// Index in `iter_gate_batches()` within that tick.
    pub batch: usize,
    /// Index in the batch's qubit list.
    pub operand: usize,
    /// Injection side.
    pub side: Side,
}

/// Failures retain their category and the original error source where available.
#[derive(Debug, thiserror::Error)]
pub enum LossDemError {
    /// A compiled setting outside the supported noise model.
    #[error("{name} = {value} is not supported by the loss-conditioned DEM; {fix}")]
    UnsupportedSetting {
        /// Setting name.
        name: String,
        /// Rejected value.
        value: String,
        /// Suggested correction.
        fix: String,
    },
    /// Invalid circuit, gate, location, channel, or algebra input.
    #[error("{message}")]
    Circuit {
        /// Context, offending value/location, and suggested correction.
        message: String,
        /// Original circuit or propagation error, if one was forwarded.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// Invalid physical record or runtime parity evaluation.
    #[error("{message}")]
    Record {
        /// Record value/index or count, and suggested correction.
        message: String,
        /// Original runtime evaluator error, when applicable.
        #[source]
        source: Option<Box<crate::DemBuildError>>,
    },
    /// Execution or output decoding failure.
    #[error("{context}: {source}; check the named operation and its circuit inputs")]
    Engine {
        /// Operation and gate coordinates, or whole-shot context.
        context: String,
        /// Original engine error.
        #[source]
        source: pecos_core::errors::PecosError,
    },
    /// Invalid detector or observable definitions.
    #[error("invalid detector/observable definitions: {0}; correct the named record or definition")]
    Definition(#[source] Box<crate::fault_tolerance::circuit_definitions::DefinitionError>),
}

fn invalid(message: impl Into<String>) -> LossDemError {
    LossDemError::Circuit {
        message: message.into(),
        source: None,
    }
}

fn circuit_error(
    context: impl std::fmt::Display,
    source: impl std::error::Error + Send + Sync + 'static,
) -> LossDemError {
    LossDemError::Circuit {
        message: format!(
            "{context}: {source}; correct the named circuit gate, location, or channel"
        ),
        source: Some(Box::new(source)),
    }
}

impl From<crate::fault_tolerance::circuit_definitions::DefinitionError> for LossDemError {
    fn from(error: crate::fault_tolerance::circuit_definitions::DefinitionError) -> Self {
        Self::Definition(Box::new(error))
    }
}

impl From<crate::DemBuilderError> for LossDemError {
    fn from(error: crate::DemBuilderError) -> Self {
        match error {
            crate::DemBuilderError::Definition(source) => Self::Definition(source),
            other => circuit_error("DEM construction", other),
        }
    }
}

#[cfg(test)]
mod tests;
