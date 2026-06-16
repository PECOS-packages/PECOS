//! Core traits for Zlup FFI integration.
//!
//! Implement these traits to create decoders, noise models, and simulators
//! that can be called from Zlup.

use crate::types::{CorrectionData, GateType, QubitId, SyndromeData};

/// A decoder that maps syndromes to corrections.
///
/// This is the primary trait for implementing QEC decoders.
///
/// # Example
///
/// ```rust,ignore
/// use zlup_ffi::prelude::*;
///
/// pub struct LookupDecoder {
///     table: Vec<u64>,
/// }
///
/// impl Decoder for LookupDecoder {
///     type Syndrome = u64;
///     type Correction = u64;
///
///     fn decode(&self, syndrome: u64) -> u64 {
///         self.table.get(syndrome as usize).copied().unwrap_or(0)
///     }
/// }
/// ```
pub trait Decoder: Send + Sync {
    /// The syndrome type (typically u64 or PackedBits).
    type Syndrome: SyndromeData;

    /// The correction type (typically u64 or PackedBits).
    type Correction: CorrectionData;

    /// Decode a syndrome into a correction.
    ///
    /// This is the core decoding operation. Given a syndrome (pattern of
    /// stabilizer measurement outcomes), return the correction to apply.
    fn decode(&self, syndrome: Self::Syndrome) -> Self::Correction;

    /// Decode with soft information (for ML decoders).
    ///
    /// Some decoders (e.g., neural network decoders) can use soft information
    /// like measurement probabilities. The default implementation ignores
    /// soft info and calls the standard `decode`.
    fn decode_soft(&self, syndrome: Self::Syndrome, _soft_info: &[f32]) -> Self::Correction {
        self.decode(syndrome)
    }

    /// Reset decoder state between shots.
    ///
    /// Some decoders maintain state (e.g., for temporal decoding).
    /// Call this between independent decoding problems.
    fn reset(&mut self) {}

    /// Get the code distance this decoder is configured for.
    fn distance(&self) -> Option<usize> {
        None
    }

    /// Get the number of syndrome bits expected.
    fn syndrome_bits(&self) -> Option<usize> {
        None
    }
}

/// A noise model for quantum simulation.
///
/// Implement this trait to define custom noise channels.
pub trait NoiseModel: Send + Sync {
    /// Apply noise after a gate operation.
    ///
    /// Called after each gate in the circuit. The noise model can
    /// inject errors based on the gate type and affected qubits.
    fn apply_gate_noise(
        &self,
        gate: GateType,
        qubits: &[QubitId],
        rng: &mut dyn RngCore,
    );

    /// Apply measurement noise.
    ///
    /// Returns true if the measurement outcome should be flipped.
    fn apply_measurement_noise(&self, qubit: QubitId, rng: &mut dyn RngCore) -> bool;

    /// Apply idle noise for a time step.
    ///
    /// Called for qubits that are idle during a tick.
    fn apply_idle_noise(&self, qubits: &[QubitId], rng: &mut dyn RngCore);

    /// Get the error rate for a specific gate type.
    fn gate_error_rate(&self, gate: GateType) -> f64 {
        let _ = gate;
        0.0
    }

    /// Get the measurement error rate.
    fn measurement_error_rate(&self) -> f64 {
        0.0
    }
}

/// A quantum state simulator.
///
/// Implement this trait to create custom simulation backends.
pub trait Simulator: Send + Sync {
    /// Apply a single-qubit gate.
    fn apply_single_qubit_gate(&mut self, gate: GateType, qubit: QubitId);

    /// Apply a two-qubit gate.
    fn apply_two_qubit_gate(&mut self, gate: GateType, control: QubitId, target: QubitId);

    /// Apply a three-qubit gate.
    fn apply_three_qubit_gate(
        &mut self,
        gate: GateType,
        q0: QubitId,
        q1: QubitId,
        q2: QubitId,
    );

    /// Measure a qubit in the Z basis.
    ///
    /// Returns the measurement outcome (0 or 1).
    fn measure_z(&mut self, qubit: QubitId) -> bool;

    /// Reset a qubit to |0⟩.
    fn reset(&mut self, qubit: QubitId);

    /// Initialize the simulator with a given number of qubits.
    fn initialize(&mut self, num_qubits: usize);

    /// Get the current number of qubits.
    fn num_qubits(&self) -> usize;
}

/// Minimal RNG trait for noise models.
///
/// This is a simplified version of `rand::RngCore` to avoid
/// requiring the full rand crate as a dependency.
pub trait RngCore {
    /// Generate a random u64.
    fn next_u64(&mut self) -> u64;

    /// Generate a random f64 in [0, 1).
    fn gen_f64(&mut self) -> f64 {
        // Standard conversion from u64 to [0, 1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate a random bool with given probability of true.
    fn gen_bool(&mut self, probability: f64) -> bool {
        self.gen_f64() < probability
    }
}

/// A simple XorShift64 RNG for when you don't need cryptographic randomness.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
}

impl RngCore for XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Extension trait for decoders that support batch decoding.
pub trait BatchDecoder: Decoder {
    /// Decode multiple syndromes at once.
    ///
    /// This can be more efficient than calling `decode` repeatedly
    /// due to better cache utilization or parallelization.
    fn decode_batch(
        &self,
        syndromes: &[Self::Syndrome],
        corrections: &mut [Self::Correction],
    ) {
        assert_eq!(syndromes.len(), corrections.len());
        for (syn, cor) in syndromes.iter().zip(corrections.iter_mut()) {
            *cor = self.decode(*syn);
        }
    }
}

/// Extension trait for decoders that support streaming/temporal decoding.
pub trait StreamingDecoder: Decoder {
    /// Feed a syndrome from a new round.
    ///
    /// For temporal decoders that look at syndrome history.
    fn feed_round(&mut self, syndrome: Self::Syndrome);

    /// Get the current correction estimate.
    fn current_correction(&self) -> Self::Correction;

    /// Commit the current state (e.g., after a logical measurement).
    fn commit(&mut self) -> Self::Correction;
}
