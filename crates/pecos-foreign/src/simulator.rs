//! Foreign simulator plugin interface.
//!
//! A foreign language implements a simulator by providing:
//! - An opaque handle (`*mut ()`) to its simulator instance
//! - A vtable of C-ABI function pointers ([`ForeignSimulatorVTable`])
//!
//! The Rust [`ForeignSimulator`] wraps these into types that implement
//! [`QuantumSimulator`], [`CliffordGateable`], and optionally
//! [`ArbitraryRotationGateable`].
//!
//! # Narrow gate interface
//!
//! Foreign simulators implement a small gate set. The full Clifford group
//! (56 methods) is decomposed into these primitives by the default trait
//! implementations -- the foreign author only needs:
//!
//! **Clifford (required):**
//! - `sz` -- phase gate S
//! - `h` -- Hadamard
//! - `cx` -- CNOT
//! - `mz` -- Z-basis measurement
//!
//! **Rotation (optional, for non-Clifford simulators):**
//! - `rx` -- X rotation
//! - `rz` -- Z rotation
//! - `rzz` -- ZZ rotation
//!
//! **Lifecycle:**
//! - `reset` -- reset to initial state
//! - `destroy` -- free the simulator

use pecos_core::{Angle64, QubitId};
use pecos_random::PecosRng;
use pecos_random::rng_manageable::RngManageable;
use pecos_simulators::clifford_gateable::MeasurementResult;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};

/// Measurement result returned over the C ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ForeignMeasurementResult {
    /// 0 = |0>, 1 = |1>
    pub outcome: u8,
    /// 0 = random, 1 = deterministic
    pub is_deterministic: u8,
}

/// Vtable for a foreign Clifford simulator.
///
/// All functions use C calling convention. Qubit indices are passed as `usize`.
/// Two-qubit gates receive interleaved pairs: [control0, target0, control1, target1, ...].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ForeignSimulatorVTable {
    /// ABI version. Must equal [`crate::version::SIMULATOR_VTABLE_VERSION`].
    /// Checked on construction; mismatches are rejected with a clear error.
    pub version: u32,

    // -- Clifford gates (required) --
    /// Apply S (sqrt-Z) gate to each qubit.
    pub sz: unsafe extern "C" fn(handle: *mut (), qubits: *const usize, num_qubits: usize),

    /// Apply Hadamard gate to each qubit.
    pub h: unsafe extern "C" fn(handle: *mut (), qubits: *const usize, num_qubits: usize),

    /// Apply CNOT gate to each (control, target) pair.
    /// `pairs` is [c0, t0, c1, t1, ...], `num_pairs` is the number of pairs.
    pub cx: unsafe extern "C" fn(handle: *mut (), pairs: *const usize, num_pairs: usize),

    /// Measure qubits in the Z basis.
    /// Writes results into `results_out` (caller-allocated, length = `num_qubits`).
    pub mz: unsafe extern "C" fn(
        handle: *mut (),
        qubits: *const usize,
        num_qubits: usize,
        results_out: *mut ForeignMeasurementResult,
    ),

    // -- Rotation gates (optional, null if Clifford-only) --
    /// Apply RX(theta) to each qubit. `theta` is in radians.
    /// May be null if the simulator only supports Clifford gates.
    pub rx: Option<
        unsafe extern "C" fn(handle: *mut (), theta: f64, qubits: *const usize, num_qubits: usize),
    >,

    /// Apply RZ(theta) to each qubit. `theta` is in radians.
    pub rz: Option<
        unsafe extern "C" fn(handle: *mut (), theta: f64, qubits: *const usize, num_qubits: usize),
    >,

    /// Apply RZZ(theta) to each (q0, q1) pair. `theta` is in radians.
    /// Pairs layout: `[q0_0, q1_0, q0_1, q1_1, ...]`, `num_pairs` is the number of pairs.
    pub rzz: Option<
        unsafe extern "C" fn(handle: *mut (), theta: f64, pairs: *const usize, num_pairs: usize),
    >,

    // -- Lifecycle --
    /// Reset the simulator to initial state (all qubits to |0>).
    pub reset: unsafe extern "C" fn(handle: *mut ()),

    /// Set the RNG seed on the foreign simulator for reproducibility.
    /// May be null if the foreign simulator does not support seeding.
    pub set_seed: Option<unsafe extern "C" fn(handle: *mut (), seed: u64)>,

    /// Destroy the simulator and free all resources.
    pub destroy: unsafe extern "C" fn(handle: *mut ()),
}

// SAFETY: The foreign simulator handle is opaque and accessed only through the vtable
// function pointers. We require that the foreign implementation is safe to transfer
// between threads (single-owner semantics). The GIL / mutex in the foreign code
// provides mutual exclusion.
unsafe impl Send for ForeignSimulator {}

/// A quantum simulator implemented in a foreign language via C ABI.
pub struct ForeignSimulator {
    handle: *mut (),
    vtable: ForeignSimulatorVTable,
    /// RNG used by PECOS's noise system. The foreign simulator has its own
    /// internal RNG; this one is for the Rust framework (noise injection, etc.).
    rng: PecosRng,
}

impl ForeignSimulator {
    /// Create a new `ForeignSimulator` from an opaque handle and vtable.
    ///
    /// # Safety
    ///
    /// The caller must guarantee:
    /// - `handle` is a valid pointer to a foreign simulator instance
    /// - All non-Option function pointers in `vtable` are valid
    /// - The foreign simulator lives until `destroy` is called
    /// - The foreign simulator is thread-safe (Send)
    ///
    /// Returns `None` if the vtable version does not match the expected ABI version.
    pub unsafe fn new(handle: *mut (), vtable: ForeignSimulatorVTable) -> Option<Self> {
        if vtable.version != crate::version::SIMULATOR_VTABLE_VERSION {
            log::error!(
                "Foreign simulator ABI version mismatch: plugin has v{}, PECOS expects v{}",
                vtable.version,
                crate::version::SIMULATOR_VTABLE_VERSION,
            );
            return None;
        }
        Some(Self {
            handle,
            vtable,
            rng: PecosRng::seed_from_u64(0),
        })
    }

    /// Whether this simulator supports arbitrary rotation gates.
    #[must_use]
    pub fn supports_rotations(&self) -> bool {
        self.vtable.rx.is_some() && self.vtable.rz.is_some() && self.vtable.rzz.is_some()
    }

    /// Helper: convert `&[QubitId]` to a temporary Vec of raw usize indices.
    fn qubit_indices(qubits: &[QubitId]) -> Vec<usize> {
        qubits.iter().map(QubitId::index).collect()
    }

    /// Helper: convert `&[(QubitId, QubitId)]` to interleaved [c0, t0, c1, t1, ...].
    fn pair_indices(pairs: &[(QubitId, QubitId)]) -> Vec<usize> {
        let mut out = Vec::with_capacity(pairs.len() * 2);
        for &(c, t) in pairs {
            out.push(c.index());
            out.push(t.index());
        }
        out
    }
}

impl Drop for ForeignSimulator {
    fn drop(&mut self) {
        unsafe {
            (self.vtable.destroy)(self.handle);
        }
    }
}

impl QuantumSimulator for ForeignSimulator {
    fn reset(&mut self) -> &mut Self {
        unsafe {
            (self.vtable.reset)(self.handle);
        }
        self
    }
}

impl CliffordGateable for ForeignSimulator {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        let indices = Self::qubit_indices(qubits);
        unsafe {
            (self.vtable.sz)(self.handle, indices.as_ptr(), indices.len());
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let indices = Self::qubit_indices(qubits);
        unsafe {
            (self.vtable.h)(self.handle, indices.as_ptr(), indices.len());
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let flat = Self::pair_indices(pairs);
        unsafe {
            (self.vtable.cx)(self.handle, flat.as_ptr(), pairs.len());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let indices = Self::qubit_indices(qubits);
        let mut raw_results = vec![
            ForeignMeasurementResult {
                outcome: 0,
                is_deterministic: 0,
            };
            qubits.len()
        ];

        unsafe {
            (self.vtable.mz)(
                self.handle,
                indices.as_ptr(),
                indices.len(),
                raw_results.as_mut_ptr(),
            );
        }

        raw_results
            .iter()
            .map(|r| MeasurementResult {
                outcome: r.outcome != 0,
                is_deterministic: r.is_deterministic != 0,
            })
            .collect()
    }
}

impl ArbitraryRotationGateable for ForeignSimulator {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        if let Some(rx_fn) = self.vtable.rx {
            let indices = Self::qubit_indices(qubits);
            let radians = theta.to_radians();
            unsafe {
                rx_fn(self.handle, radians, indices.as_ptr(), indices.len());
            }
            self
        } else {
            panic!("foreign simulator does not support rotation gates (rx is null)")
        }
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        if let Some(rz_fn) = self.vtable.rz {
            let indices = Self::qubit_indices(qubits);
            let radians = theta.to_radians();
            unsafe {
                rz_fn(self.handle, radians, indices.as_ptr(), indices.len());
            }
            self
        } else {
            panic!("foreign simulator does not support rotation gates (rz is null)")
        }
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        if let Some(rzz_fn) = self.vtable.rzz {
            let flat = Self::pair_indices(pairs);
            let radians = theta.to_radians();
            unsafe {
                rzz_fn(self.handle, radians, flat.as_ptr(), pairs.len());
            }
            self
        } else {
            panic!("foreign simulator does not support rotation gates (rzz is null)")
        }
    }
}

impl RngManageable for ForeignSimulator {
    type Rng = PecosRng;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn set_seed(&mut self, seed: u64) {
        // Seed the Rust-side RNG (used by PECOS noise system).
        self.rng = PecosRng::seed_from_u64(seed);
        // Forward the seed to the foreign simulator's own RNG.
        if let Some(set_seed_fn) = self.vtable.set_seed {
            unsafe {
                set_seed_fn(self.handle, seed);
            }
        }
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}
