//! Precision-auto wrapper that picks f64 when available and falls back to f32.
//!
//! Provides [`GpuStateVecAuto`] for cross-platform code that wants *some* GPU
//! state vector without caring whether the adapter supports `SHADER_F64`.
//! Explicit users should keep reaching for [`GpuStateVec32`] or [`GpuStateVec64`]
//! directly -- this wrapper is intentionally opt-in.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};

use crate::{GpuError, GpuStateVec32, GpuStateVec64, RequiredFeature};

/// GPU state vector simulator that selects precision at runtime.
///
/// Construct with [`GpuStateVecAuto::new`]. It first tries f64
/// ([`GpuStateVec64`]); if the adapter lacks `SHADER_F64` (e.g. Metal on Apple
/// Silicon) it falls back to f32 ([`GpuStateVec32`]).
///
/// Implements the standard gate traits so it can be used interchangeably with
/// either concrete backend in code that does not depend on precision.
pub enum GpuStateVecAuto {
    /// f64 backend (preferred; selected when `SHADER_F64` is available).
    F64(GpuStateVec64),
    /// f32 backend (fallback for adapters without `SHADER_F64`).
    F32(GpuStateVec32),
}

impl GpuStateVecAuto {
    /// Create a GPU state vector simulator, preferring f64 precision.
    ///
    /// Falls back to f32 only when the f64 path reports
    /// `UnsupportedFeature(ShaderF64)`. Any other error (no adapter, too many
    /// qubits, etc.) is propagated as-is so callers don't silently get a less
    /// precise simulator for an unrelated reason.
    ///
    /// # Errors
    /// Returns a [`GpuError`] from whichever constructor was used. The f64
    /// error is *not* preserved if the fallback succeeds; if the fallback also
    /// fails, only its error is surfaced.
    pub fn new(num_qubits: u32) -> Result<Self, GpuError> {
        match GpuStateVec64::new(num_qubits) {
            Ok(sim) => Ok(GpuStateVecAuto::F64(sim)),
            Err(GpuError::UnsupportedFeature(RequiredFeature::ShaderF64)) => {
                GpuStateVec32::new(num_qubits).map(GpuStateVecAuto::F32)
            }
            Err(e) => Err(e),
        }
    }

    /// True if the selected backend is f64.
    #[must_use]
    pub fn is_f64(&self) -> bool {
        matches!(self, Self::F64(_))
    }
}

/// Dispatch a `&mut self -> &mut Self` method to the inner backend.
macro_rules! dispatch_mut {
    ($self:ident, $method:ident ( $($arg:expr),* $(,)? )) => {{
        match $self {
            Self::F64(s) => { s.$method($($arg),*); }
            Self::F32(s) => { s.$method($($arg),*); }
        }
        $self
    }};
}

impl QuantumSimulator for GpuStateVecAuto {
    fn reset(&mut self) -> &mut Self {
        dispatch_mut!(self, reset())
    }
}

impl CliffordGateable for GpuStateVecAuto {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        dispatch_mut!(self, h(qubits))
    }
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        dispatch_mut!(self, sz(qubits))
    }
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        dispatch_mut!(self, cx(pairs))
    }
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        match self {
            Self::F64(s) => s.mz(qubits),
            Self::F32(s) => s.mz(qubits),
        }
    }
}

impl ArbitraryRotationGateable for GpuStateVecAuto {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        dispatch_mut!(self, rx(theta, qubits))
    }
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        dispatch_mut!(self, rz(theta, qubits))
    }
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        dispatch_mut!(self, rzz(theta, pairs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::qid;

    #[test]
    fn auto_falls_back_or_uses_f64() {
        // Either succeeds with whatever backend the adapter supports, or skips
        // cleanly (no GPU available at all).
        let Ok(mut sim) = GpuStateVecAuto::new(3) else {
            return;
        };
        sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].outcome, results[1].outcome);
    }
}
