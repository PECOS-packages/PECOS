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

//! Multi-logical-qubit decoder manager.
//!
//! Manages decoder instances for K logical qubits, each with its own
//! Pauli frame. Routes syndromes to the right decoder and maintains
//! per-qubit frames.

use crate::ObservableDecoder;
use crate::errors::DecoderError;

/// Manages multiple logical qubit decoders with per-qubit Pauli frames.
pub struct MultiDecoderManager {
    /// (label, decoder) per logical qubit.
    decoders: Vec<(String, Box<dyn ObservableDecoder>)>,
    /// Per-qubit accumulated Pauli frame.
    frames: Vec<u64>,
    /// Per-qubit cycle count.
    cycle_counts: Vec<usize>,
}

impl MultiDecoderManager {
    /// Create with no decoders.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decoders: Vec::new(),
            frames: Vec::new(),
            cycle_counts: Vec::new(),
        }
    }

    /// Add a logical qubit decoder with a label.
    pub fn add_qubit(&mut self, label: impl Into<String>, decoder: Box<dyn ObservableDecoder>) {
        self.decoders.push((label.into(), decoder));
        self.frames.push(0);
        self.cycle_counts.push(0);
    }

    /// Number of logical qubits managed.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.decoders.len()
    }

    /// Decode one QEC cycle for a specific logical qubit.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the decoder fails or the qubit index is out of bounds.
    pub fn decode_cycle(&mut self, qubit_idx: usize, syndrome: &[u8]) -> Result<u64, DecoderError> {
        if qubit_idx >= self.decoders.len() {
            return Err(DecoderError::InvalidNodeIndex {
                index: qubit_idx,
                max: self.decoders.len(),
            });
        }
        let obs = self.decoders[qubit_idx].1.decode_to_observables(syndrome)?;
        self.frames[qubit_idx] ^= obs;
        self.cycle_counts[qubit_idx] += 1;
        Ok(obs)
    }

    /// Get the current Pauli frame for a logical qubit.
    #[must_use]
    pub fn frame(&self, qubit_idx: usize) -> u64 {
        self.frames.get(qubit_idx).copied().unwrap_or(0)
    }

    /// Consume and reset the frame for a logical qubit.
    pub fn consume_frame(&mut self, qubit_idx: usize) -> u64 {
        if qubit_idx >= self.frames.len() {
            return 0;
        }
        let f = self.frames[qubit_idx];
        self.frames[qubit_idx] = 0;
        self.cycle_counts[qubit_idx] = 0;
        f
    }

    /// Label of a logical qubit.
    #[must_use]
    pub fn label(&self, qubit_idx: usize) -> Option<&str> {
        self.decoders.get(qubit_idx).map(|(l, _)| l.as_str())
    }

    /// Apply a transversal CNOT between two logical qubits' Pauli frames.
    ///
    /// `x_obs_mask`: observable bits that are X-type (propagate control→target).
    /// `z_obs_mask`: observable bits that are Z-type (propagate target→control).
    ///
    /// For a standard surface code with observable 0 = logical observable:
    /// use `x_obs_mask = 1, z_obs_mask = 1` (single observable, both X and Z
    /// corrections matter depending on the basis).
    pub fn apply_transversal_cnot(
        &mut self,
        control_idx: usize,
        target_idx: usize,
        x_obs_mask: u64,
        z_obs_mask: u64,
    ) {
        if control_idx >= self.frames.len() || target_idx >= self.frames.len() {
            return;
        }
        let ctrl_frame = self.frames[control_idx];
        let tgt_frame = self.frames[target_idx];

        // X-type: control → target
        self.frames[target_idx] ^= ctrl_frame & x_obs_mask;
        // Z-type: target → control
        self.frames[control_idx] ^= tgt_frame & z_obs_mask;
    }

    /// Apply a logical Hadamard to a qubit's Pauli frame.
    ///
    /// Swaps X-type and Z-type frame bits.
    pub fn apply_hadamard(&mut self, qubit_idx: usize, x_obs_mask: u64, z_obs_mask: u64) {
        if qubit_idx >= self.frames.len() {
            return;
        }
        let f = self.frames[qubit_idx];
        let x_bits = f & x_obs_mask;
        let z_bits = f & z_obs_mask;
        self.frames[qubit_idx] &= !(x_obs_mask | z_obs_mask);
        self.frames[qubit_idx] |= if x_bits != 0 { z_obs_mask } else { 0 };
        self.frames[qubit_idx] |= if z_bits != 0 { x_obs_mask } else { 0 };
    }

    /// Mutable access to the frame for a qubit (for custom gate propagation).
    pub fn frame_mut(&mut self, qubit_idx: usize) -> Option<&mut u64> {
        self.frames.get_mut(qubit_idx)
    }

    /// Replace the decoder for a qubit (e.g., after lattice surgery changes the DEM).
    ///
    /// # Errors
    ///
    /// Returns error if `qubit_idx` is out of bounds.
    pub fn replace_decoder(
        &mut self,
        qubit_idx: usize,
        decoder: Box<dyn ObservableDecoder>,
    ) -> Result<(), crate::errors::DecoderError> {
        if qubit_idx >= self.decoders.len() {
            return Err(crate::errors::DecoderError::InvalidNodeIndex {
                index: qubit_idx,
                max: self.decoders.len(),
            });
        }
        self.decoders[qubit_idx].1 = decoder;
        Ok(())
    }
}

impl Default for MultiDecoderManager {
    fn default() -> Self {
        Self::new()
    }
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
    fn test_multi_qubit() {
        let mut mgr = MultiDecoderManager::new();
        mgr.add_qubit("q0", Box::new(FixedDecoder(0b01)));
        mgr.add_qubit("q1", Box::new(FixedDecoder(0b10)));

        assert_eq!(mgr.num_qubits(), 2);
        assert_eq!(mgr.label(0), Some("q0"));

        mgr.decode_cycle(0, &[]).unwrap();
        mgr.decode_cycle(1, &[]).unwrap();

        assert_eq!(mgr.frame(0), 0b01);
        assert_eq!(mgr.frame(1), 0b10);
    }

    #[test]
    fn test_consume_frame() {
        let mut mgr = MultiDecoderManager::new();
        mgr.add_qubit("q0", Box::new(FixedDecoder(1)));
        mgr.decode_cycle(0, &[]).unwrap();
        assert_eq!(mgr.consume_frame(0), 1);
        assert_eq!(mgr.frame(0), 0);
    }

    #[test]
    fn test_transversal_cnot() {
        let mut mgr = MultiDecoderManager::new();
        mgr.add_qubit("ctrl", Box::new(FixedDecoder(0)));
        mgr.add_qubit("tgt", Box::new(FixedDecoder(0)));

        // Set X correction on control (bit 0).
        *mgr.frame_mut(0).unwrap() = 0b01;

        // Transversal CNOT: X propagates ctrl→tgt.
        mgr.apply_transversal_cnot(0, 1, 0b01, 0b10);
        assert_eq!(mgr.frame(0), 0b01);
        assert_eq!(mgr.frame(1), 0b01); // X propagated
    }

    #[test]
    fn test_hadamard() {
        let mut mgr = MultiDecoderManager::new();
        mgr.add_qubit("q0", Box::new(FixedDecoder(0)));
        *mgr.frame_mut(0).unwrap() = 0b01; // X correction

        mgr.apply_hadamard(0, 0b01, 0b10);
        assert_eq!(mgr.frame(0), 0b10); // became Z correction
    }

    #[test]
    fn test_replace_decoder() {
        let mut mgr = MultiDecoderManager::new();
        mgr.add_qubit("q0", Box::new(FixedDecoder(0b01)));
        mgr.decode_cycle(0, &[]).unwrap();
        assert_eq!(mgr.frame(0), 0b01);

        // Replace decoder (e.g., after lattice surgery changes the DEM).
        mgr.replace_decoder(0, Box::new(FixedDecoder(0b10)))
            .unwrap();
        mgr.decode_cycle(0, &[]).unwrap();
        assert_eq!(mgr.frame(0), 0b11); // 01 ^ 10 = 11
    }
}
