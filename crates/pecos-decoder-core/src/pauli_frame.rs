// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Pauli frame accumulator for real-time QEC.
//!
//! In a real QEC system, the decoder runs once per QEC cycle, producing
//! an observable mask. These masks accumulate into a Pauli frame that
//! tracks the net logical correction. The frame is consumed only when
//! a logical operation requires it (T-gate injection, logical measurement).
//!
//! # Example
//!
//! ```
//! use pecos_decoder_core::{DecoderError, ObservableDecoder};
//! use pecos_decoder_core::pauli_frame::PauliFrameAccumulator;
//!
//! struct FixedDecoder(u64);
//!
//! impl ObservableDecoder for FixedDecoder {
//!     fn decode_to_observables(&mut self, _syndrome: &[u8]) -> Result<u64, DecoderError> {
//!         Ok(self.0)
//!     }
//! }
//!
//! let mut frame = PauliFrameAccumulator::new(Box::new(FixedDecoder(0b01)));
//!
//! // QEC cycles
//! for syndrome in [&[1, 0][..], &[0, 1][..]] {
//!     frame.decode_cycle(syndrome).unwrap();
//! }
//! assert_eq!(frame.current_frame(), 0b00);
//!
//! // At logical measurement: consume frame
//! let correction = frame.consume_frame();
//! let raw_measurement = 1;
//! let logical_result = raw_measurement ^ (correction & 1);
//! assert_eq!(logical_result, 1);
//! ```

use crate::ObservableDecoder;
use crate::errors::DecoderError;

/// Accumulates Pauli frame corrections across QEC cycles.
///
/// Wraps any `ObservableDecoder` and XORs each cycle's observable mask
/// into a running frame. The frame represents the net logical correction
/// needed at the current point in the computation.
pub struct PauliFrameAccumulator {
    decoder: Box<dyn ObservableDecoder>,
    frame: u64,
    cycle_count: usize,
}

impl PauliFrameAccumulator {
    /// Create from any observable decoder.
    #[must_use]
    pub fn new(decoder: Box<dyn ObservableDecoder>) -> Self {
        Self {
            decoder,
            frame: 0,
            cycle_count: 0,
        }
    }

    /// Decode one QEC cycle's syndrome and accumulate into the frame.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the inner decoder fails.
    pub fn decode_cycle(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let obs = self.decoder.decode_to_observables(syndrome)?;
        self.frame ^= obs;
        self.cycle_count += 1;
        Ok(obs)
    }

    /// Current accumulated Pauli frame (does not reset).
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.frame
    }

    /// Consume the frame: returns the accumulated mask and resets to zero.
    ///
    /// Call this at logical operations (T-gate, measurement) to get the
    /// correction and start fresh for the next logical cycle.
    pub fn consume_frame(&mut self) -> u64 {
        let f = self.frame;
        self.frame = 0;
        self.cycle_count = 0;
        f
    }

    /// Number of QEC cycles since last reset/consume.
    #[must_use]
    pub fn cycle_count(&self) -> usize {
        self.cycle_count
    }

    /// Manually flip a frame bit (e.g., for deterministic corrections).
    pub fn flip_bit(&mut self, bit: u32) {
        self.frame ^= 1u64 << bit;
    }

    /// Direct access to the frame bits.
    #[must_use]
    pub fn frame_mut(&mut self) -> &mut u64 {
        &mut self.frame
    }

    /// Access the inner decoder.
    pub fn decoder_mut(&mut self) -> &mut dyn ObservableDecoder {
        &mut *self.decoder
    }
}

/// Propagate Pauli frames through a transversal CNOT.
///
/// When a logical CNOT is applied from control to target:
/// - X errors propagate: control → target (`X_c` → `X_c` ⊗ `X_t`)
/// - Z errors propagate: target → control (`Z_t` → `Z_c` ⊗ `Z_t`)
///
/// For observable masks (bit k = observable k):
/// - X-type observables on control propagate to target
/// - Z-type observables on target propagate to control
///
/// `x_obs_mask`: which observable bits are X-type (propagate forward)
/// `z_obs_mask`: which observable bits are Z-type (propagate backward)
pub fn propagate_cnot_frames(
    control: &mut PauliFrameAccumulator,
    target: &mut PauliFrameAccumulator,
    x_obs_mask: u64,
    z_obs_mask: u64,
) {
    let ctrl_frame = control.current_frame();
    let tgt_frame = target.current_frame();

    // X-type bits on control propagate to target: target ^= control & x_mask
    *target.frame_mut() ^= ctrl_frame & x_obs_mask;

    // Z-type bits on target propagate to control: control ^= target & z_mask
    *control.frame_mut() ^= tgt_frame & z_obs_mask;
}

/// Propagate Pauli frame through a logical S gate (phase gate).
///
/// S gate: X → Y = iXZ, Z → Z. For the frame:
/// - Z-type bits are unchanged
/// - X-type bits that are set also flip the corresponding Z-type bit
pub fn propagate_s_gate_frame(frame: &mut PauliFrameAccumulator, x_obs_mask: u64, z_obs_mask: u64) {
    let f = frame.current_frame();
    // X-type corrections also induce Z-type corrections after S gate
    *frame.frame_mut() ^= (f & x_obs_mask) & z_obs_mask;
}

/// Propagate Pauli frame through a logical Hadamard.
///
/// H gate: X ↔ Z. Swaps X-type and Z-type frame bits.
pub fn propagate_h_gate_frame(frame: &mut PauliFrameAccumulator, x_obs_mask: u64, z_obs_mask: u64) {
    let f = frame.current_frame();
    let x_bits = f & x_obs_mask;
    let z_bits = f & z_obs_mask;
    // Clear both, then swap
    *frame.frame_mut() &= !(x_obs_mask | z_obs_mask);
    // X bits go to Z positions, Z bits go to X positions
    // (This assumes x_obs_mask and z_obs_mask don't overlap and have
    // matching bit positions. For a single logical qubit with obs 0 = X
    // and obs 1 = Z: x_mask=0b01, z_mask=0b10, swap bits 0 and 1.)
    *frame.frame_mut() |= if x_bits != 0 { z_obs_mask } else { 0 };
    *frame.frame_mut() |= if z_bits != 0 { x_obs_mask } else { 0 };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDecoder(u64);
    impl ObservableDecoder for FixedDecoder {
        fn decode_to_observables(&mut self, _: &[u8]) -> Result<u64, DecoderError> {
            Ok(self.0)
        }
    }

    #[test]
    fn test_accumulate_xor() {
        let mut frame = PauliFrameAccumulator::new(Box::new(FixedDecoder(0b01)));
        frame.decode_cycle(&[]).unwrap();
        assert_eq!(frame.current_frame(), 0b01);
        frame.decode_cycle(&[]).unwrap();
        assert_eq!(frame.current_frame(), 0b00); // XOR cancels
        frame.decode_cycle(&[]).unwrap();
        assert_eq!(frame.current_frame(), 0b01);
        assert_eq!(frame.cycle_count(), 3);
    }

    #[test]
    fn test_consume_resets() {
        let mut frame = PauliFrameAccumulator::new(Box::new(FixedDecoder(0b11)));
        frame.decode_cycle(&[]).unwrap();
        assert_eq!(frame.consume_frame(), 0b11);
        assert_eq!(frame.current_frame(), 0);
        assert_eq!(frame.cycle_count(), 0);
    }

    #[test]
    fn test_flip_bit() {
        let mut frame = PauliFrameAccumulator::new(Box::new(FixedDecoder(0)));
        frame.flip_bit(2);
        assert_eq!(frame.current_frame(), 0b100);
        frame.flip_bit(2);
        assert_eq!(frame.current_frame(), 0);
    }

    #[test]
    fn test_cnot_frame_propagation() {
        // Two logical qubits with obs bit 0 = X-type, bit 1 = Z-type.
        let mut ctrl = PauliFrameAccumulator::new(Box::new(FixedDecoder(0)));
        let mut tgt = PauliFrameAccumulator::new(Box::new(FixedDecoder(0)));

        // Control has X correction (bit 0).
        ctrl.flip_bit(0);
        assert_eq!(ctrl.current_frame(), 0b01);
        assert_eq!(tgt.current_frame(), 0b00);

        // CNOT: X on control propagates to target.
        propagate_cnot_frames(&mut ctrl, &mut tgt, 0b01, 0b10);
        assert_eq!(ctrl.current_frame(), 0b01); // unchanged
        assert_eq!(tgt.current_frame(), 0b01); // X propagated

        // Now target has Z correction (bit 1).
        tgt.flip_bit(1);
        assert_eq!(tgt.current_frame(), 0b11);

        // CNOT: Z on target propagates to control.
        propagate_cnot_frames(&mut ctrl, &mut tgt, 0b01, 0b10);
        assert_eq!(ctrl.current_frame(), 0b11); // Z propagated back
    }

    #[test]
    fn test_hadamard_frame() {
        let mut frame = PauliFrameAccumulator::new(Box::new(FixedDecoder(0)));
        // Set X correction (bit 0).
        frame.flip_bit(0);
        assert_eq!(frame.current_frame(), 0b01);

        // Hadamard: X ↔ Z (swap bits 0 and 1).
        propagate_h_gate_frame(&mut frame, 0b01, 0b10);
        assert_eq!(frame.current_frame(), 0b10); // X became Z
    }
}
