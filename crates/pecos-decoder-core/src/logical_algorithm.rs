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

//! Logical algorithm decoder for real-time QEC.
//!
//! Decodes logical algorithms (sequences of memory segments separated by
//! transversal gates) using the full-circuit DEM for accuracy, with
//! segment structure metadata for streaming and frame propagation.
//!
//! # Decoding Modes
//!
//! - **Full-circuit**: Uses the full DEM's OSD for maximum accuracy.
//!   Equivalent to `ObservableSubgraphDecoder` on the full circuit.
//! - **Per-segment** (future streaming): Each segment decoded independently
//!   with buffer overlap at gate boundaries.

use crate::ObservableDecoder;
use crate::errors::DecoderError;

/// One segment of a logical algorithm.
pub struct SegmentDescriptor {
    /// Number of detectors in this segment's DEM.
    pub num_detectors: usize,
    /// Number of observables in this segment's DEM.
    pub num_observables: usize,
}

/// Gate at a segment boundary.
#[derive(Debug, Clone)]
pub enum BoundaryGate {
    /// Transversal Hadamard: swaps X↔Z frame bits for a qubit.
    Hadamard { x_obs_bit: u32, z_obs_bit: u32 },
    /// Transversal CNOT: propagates X forward, Z backward.
    Cnot {
        ctrl_x_bit: u32,
        ctrl_z_bit: u32,
        tgt_x_bit: u32,
        tgt_z_bit: u32,
    },
    /// Transversal S gate: X corrections induce Z corrections.
    SGate { x_obs_bit: u32, z_obs_bit: u32 },
    /// T-gate via magic state injection (decision point).
    ///
    /// At this boundary, the decoder MUST produce a correction before
    /// the hardware can proceed. The corrected measurement outcome
    /// determines whether an S correction is applied:
    ///   corrected = `raw_measurement` XOR frame[`z_obs_bit`]
    ///   if corrected == 1: apply S gate on the data qubit
    ///
    /// This is a feed-forward decision point with a reaction time
    /// deadline. The decoder's frame must be ready.
    TGateInjection {
        /// Observable bit for the data qubit's Z correction.
        z_obs_bit: u32,
        /// Observable bit for the ancilla's Z measurement.
        ancilla_z_bit: u32,
    },
}

/// Marks whether a segment boundary is a decision point.
///
/// At decision points, the decoder must provide the Pauli frame
/// within the reaction time budget. At non-decision boundaries
/// (Clifford gates), the frame is metadata — no deadline.
impl BoundaryGate {
    /// Whether this gate is a feed-forward decision point.
    #[must_use]
    pub fn is_decision_point(&self) -> bool {
        matches!(self, Self::TGateInjection { .. })
    }
}

/// Full description of a logical algorithm for decoding.
pub struct AlgorithmDescriptor {
    /// Per-segment descriptors.
    pub segments: Vec<SegmentDescriptor>,
    /// Gates at segment boundaries. `boundary_gates[i]` between segment i and i+1.
    pub boundary_gates: Vec<Vec<BoundaryGate>>,
    /// Total number of observables.
    pub num_observables: usize,
}

/// Decoder for logical quantum algorithms.
///
/// Wraps a full-circuit decoder (OSD) with segment metadata. The
/// segment structure enables:
/// - Tracking which gates occur at which point in the circuit
/// - Pauli frame propagation for T-gate/measurement corrections
/// - Future streaming mode with per-segment windowed decoding
///
/// In the current implementation, `decode_shot` delegates to the
/// full-circuit OSD for maximum accuracy. The segment structure is
/// metadata for frame tracking and streaming (step 5).
pub struct LogicalAlgorithmDecoder {
    /// Full-circuit decoder (OSD on the complete DEM).
    full_decoder: Box<dyn ObservableDecoder + Send + Sync>,
    /// Segment metadata for streaming/frame tracking.
    segments: Vec<SegmentDescriptor>,
    /// Gates at segment boundaries.
    boundary_gates: Vec<Vec<BoundaryGate>>,
    /// Total number of observables.
    _num_observables: usize,
}

impl LogicalAlgorithmDecoder {
    /// Build from a full-circuit decoder and algorithm descriptor.
    ///
    /// The `full_decoder` is typically an `ObservableSubgraphDecoder`
    /// built from the full circuit DEM.
    #[must_use]
    pub fn new(
        full_decoder: Box<dyn ObservableDecoder + Send + Sync>,
        descriptor: AlgorithmDescriptor,
    ) -> Self {
        Self {
            full_decoder,
            segments: descriptor.segments,
            boundary_gates: descriptor.boundary_gates,
            _num_observables: descriptor.num_observables,
        }
    }

    /// Decode one shot using the full-circuit decoder.
    pub fn decode_shot(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.full_decoder.decode_to_observables(syndrome)
    }

    /// Number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Total detectors across all segments.
    #[must_use]
    pub fn total_detectors(&self) -> usize {
        self.segments.iter().map(|s| s.num_detectors).sum()
    }

    /// Apply boundary gate to a Pauli frame.
    /// Used when consuming the frame at logical operations.
    pub fn apply_boundary_gate(frame: &mut u64, gate: &BoundaryGate) {
        match gate {
            BoundaryGate::Hadamard {
                x_obs_bit,
                z_obs_bit,
            } => {
                let x_set = (*frame >> x_obs_bit) & 1;
                let z_set = (*frame >> z_obs_bit) & 1;
                *frame &= !(1u64 << x_obs_bit);
                *frame &= !(1u64 << z_obs_bit);
                *frame |= z_set << x_obs_bit;
                *frame |= x_set << z_obs_bit;
            }
            BoundaryGate::Cnot {
                ctrl_x_bit,
                ctrl_z_bit,
                tgt_x_bit,
                tgt_z_bit,
            } => {
                if (*frame >> ctrl_x_bit) & 1 != 0 {
                    *frame ^= 1u64 << tgt_x_bit;
                }
                if (*frame >> tgt_z_bit) & 1 != 0 {
                    *frame ^= 1u64 << ctrl_z_bit;
                }
            }
            BoundaryGate::SGate {
                x_obs_bit,
                z_obs_bit,
            } => {
                if (*frame >> x_obs_bit) & 1 != 0 {
                    *frame ^= 1u64 << z_obs_bit;
                }
            }
            BoundaryGate::TGateInjection {
                z_obs_bit,
                ancilla_z_bit,
            } => {
                // T-gate teleportation: CX(data, ancilla) + measure ancilla Z.
                // The ancilla Z measurement outcome (corrected by frame)
                // determines whether to apply S correction on data.
                //
                // Frame propagation: the ancilla's Z observable is folded
                // into the data's Z observable. If the ancilla Z bit is
                // set in the frame, flip the data's Z bit.
                if (*frame >> ancilla_z_bit) & 1 != 0 {
                    *frame ^= 1u64 << z_obs_bit;
                }
            }
        }
    }
}

impl ObservableDecoder for LogicalAlgorithmDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_shot(syndrome)
    }
}

// ============================================================================
// Streaming mode
// ============================================================================

/// Streaming wrapper for `LogicalAlgorithmDecoder`.
///
/// Buffers syndrome data round-by-round. The full-circuit OSD decodes
/// the entire accumulated syndrome at `flush()` for maximum accuracy.
///
/// The segment structure tracks which rounds belong to which segment.
/// At each segment boundary, the Pauli frame can be queried and
/// propagated through the boundary gate.
///
/// # Usage
///
/// ```
/// use pecos_decoder_core::{DecoderError, ObservableDecoder};
/// use pecos_decoder_core::logical_algorithm::{
///     AlgorithmDescriptor, LogicalAlgorithmDecoder, SegmentDescriptor, StreamingLogicalDecoder,
/// };
///
/// struct AnyDetectionDecoder;
///
/// impl ObservableDecoder for AnyDetectionDecoder {
///     fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
///         Ok(u64::from(syndrome.iter().any(|&bit| bit != 0)))
///     }
/// }
///
/// let descriptor = AlgorithmDescriptor {
///     segments: vec![SegmentDescriptor {
///         num_detectors: 2,
///         num_observables: 1,
///     }],
///     boundary_gates: vec![],
///     num_observables: 1,
/// };
/// let decoder = LogicalAlgorithmDecoder::new(Box::new(AnyDetectionDecoder), descriptor);
/// let mut stream = StreamingLogicalDecoder::new(decoder);
///
/// // Feed syndrome round by round
/// for sparse_round in [vec![(0, 1)], vec![(1, 0)]] {
///     stream.feed_sparse(&sparse_round);
/// }
///
/// // Decode at the end
/// let obs = stream.flush().unwrap();
/// assert_eq!(obs, 1);
/// ```
pub struct StreamingLogicalDecoder {
    /// The underlying batch decoder (full-circuit OSD).
    inner: LogicalAlgorithmDecoder,
    /// Accumulated syndrome buffer (full circuit size).
    syndrome: Vec<u8>,
    /// Total detectors.
    total_detectors: usize,
    /// Rounds fed so far.
    rounds_fed: usize,
    /// Accumulated observable correction from last flush.
    accumulated_obs: u64,
}

impl StreamingLogicalDecoder {
    /// Create from a `LogicalAlgorithmDecoder`.
    #[must_use]
    pub fn new(decoder: LogicalAlgorithmDecoder) -> Self {
        let total = decoder.total_detectors();
        Self {
            inner: decoder,
            syndrome: vec![0u8; total],
            total_detectors: total,
            rounds_fed: 0,
            accumulated_obs: 0,
        }
    }

    /// Feed one detection event into the syndrome buffer.
    #[inline]
    pub fn feed_detection(&mut self, detector_idx: usize, value: u8) {
        if detector_idx < self.total_detectors {
            self.syndrome[detector_idx] = value;
        }
    }

    /// Feed a dense syndrome slice (all detectors, in order).
    pub fn feed_dense(&mut self, syndrome: &[u8]) {
        let len = syndrome.len().min(self.total_detectors);
        self.syndrome[..len].copy_from_slice(&syndrome[..len]);
    }

    /// Feed sparse detection events: (`detector_index`, value) pairs.
    pub fn feed_sparse(&mut self, detectors: &[(u32, u8)]) {
        for &(det, val) in detectors {
            self.feed_detection(det as usize, val);
        }
        self.rounds_fed += 1;
    }

    /// Decode the accumulated syndrome using the full-circuit OSD.
    ///
    /// Returns the observable correction mask. This is the final
    /// correction to apply to raw measurement outcomes.
    pub fn flush(&mut self) -> Result<u64, DecoderError> {
        let obs = self.inner.decode_shot(&self.syndrome)?;
        self.accumulated_obs = obs;
        Ok(obs)
    }

    /// Decode a full syndrome at once (convenience for batch mode).
    pub fn decode_shot(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.feed_dense(syndrome);
        self.flush()
    }

    /// Current accumulated observable correction.
    #[must_use]
    pub fn accumulated_obs(&self) -> u64 {
        self.accumulated_obs
    }

    /// Number of segments in the algorithm.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.inner.num_segments()
    }

    /// Rounds fed so far.
    #[must_use]
    pub fn rounds_fed(&self) -> usize {
        self.rounds_fed
    }

    /// Access the boundary gates for frame propagation.
    #[must_use]
    pub fn boundary_gates(&self) -> &[Vec<BoundaryGate>] {
        &self.inner.boundary_gates
    }

    /// Apply boundary gate to a Pauli frame (delegates to inner).
    pub fn apply_boundary_gate(frame: &mut u64, gate: &BoundaryGate) {
        LogicalAlgorithmDecoder::apply_boundary_gate(frame, gate);
    }

    /// Reset for the next shot.
    pub fn reset(&mut self) {
        self.syndrome.fill(0);
        self.rounds_fed = 0;
        self.accumulated_obs = 0;
    }
}

/// Simulate streaming decode on a batch of samples.
///
/// For each shot: feeds the dense syndrome, flushes, checks against expected.
/// Returns the number of logical errors. This simulates what a real-time
/// system would do — feed syndromes and flush at the end.
pub fn streaming_decode_count(
    decoder: &mut StreamingLogicalDecoder,
    syndromes: &[Vec<u8>],
    expected_masks: &[u64],
) -> Result<usize, DecoderError> {
    let mut errors = 0;
    for (syn, &expected) in syndromes.iter().zip(expected_masks.iter()) {
        decoder.reset();
        let predicted = decoder.decode_shot(syn)?;
        if predicted != expected {
            errors += 1;
        }
    }
    Ok(errors)
}

// ============================================================================
// Budget-aware logical circuit decoder
// ============================================================================

use crate::decode_budget::{DecodeBudget, DecodeStrategy, DetectorRegion};

/// Budget-aware decoder for logical quantum circuits.
///
/// Composes a `DecodeStrategy` (which handles the decode/commit pattern)
/// with segment tracking and Pauli frame propagation. The strategy is
/// selected based on the hardware's time budget.
///
/// # Decode Modes
///
/// - **Offline** (ion trap / simulation): `FullCircuitStrategy` — buffer
///   everything, decode at end. Maximum accuracy.
/// - **Streaming** (neutral atom): `CommittedOsdStrategy` — decode and
///   commit at segment boundaries. Bounded memory.
/// - **Real-time** (superconducting): windowed UF with ghost protocol
///   (future).
///
/// All modes use the same segment + gate + frame infrastructure.
pub struct LogicalCircuitDecoder {
    /// The decode strategy (owns the inner decoder).
    strategy: Box<dyn DecodeStrategy + Send + Sync>,
    /// Segment metadata.
    segments: Vec<SegmentDescriptor>,
    /// Cumulative detector offsets per segment.
    _segment_offsets: Vec<usize>,
    /// Gates at segment boundaries.
    boundary_gates: Vec<Vec<BoundaryGate>>,
    /// Per-qubit Pauli frames.
    frames: Vec<u64>,
    /// Decode budget.
    budget: DecodeBudget,
    /// Syndrome buffer.
    syndrome: Vec<u8>,
    /// Total detectors.
    total_detectors: usize,
    /// Current segment being fed.
    current_segment: usize,
    /// Detectors fed into the current segment so far.
    current_segment_fed: usize,
}

impl LogicalCircuitDecoder {
    /// Build from an algorithm descriptor, decode strategy, and budget.
    #[must_use]
    pub fn new(
        descriptor: AlgorithmDescriptor,
        strategy: Box<dyn DecodeStrategy + Send + Sync>,
        budget: DecodeBudget,
        num_qubits: usize,
    ) -> Self {
        let mut segment_offsets = Vec::with_capacity(descriptor.segments.len());
        let mut offset = 0;
        for seg in &descriptor.segments {
            segment_offsets.push(offset);
            offset += seg.num_detectors;
        }
        let total_detectors = offset;

        Self {
            strategy,
            segments: descriptor.segments,
            _segment_offsets: segment_offsets,
            boundary_gates: descriptor.boundary_gates,
            frames: vec![0u64; num_qubits],
            budget,
            syndrome: vec![0u8; total_detectors],
            total_detectors,
            current_segment: 0,
            current_segment_fed: 0,
        }
    }

    /// Decode a full shot (batch mode).
    ///
    /// For offline/ion trap budgets: equivalent to full-circuit OSD.
    /// For streaming budgets: decodes and commits each segment.
    pub fn decode_shot(&mut self, full_syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.reset();
        let len = full_syndrome.len().min(self.total_detectors);
        self.syndrome[..len].copy_from_slice(&full_syndrome[..len]);

        // Single decode of the full syndrome. The strategy handles
        // commitment internally if it supports it.
        self.strategy.decode(&self.syndrome)
    }

    /// Batch decode: count logical errors across a batch of shots.
    pub fn decode_count(
        &mut self,
        syndromes: &[Vec<u8>],
        expected_masks: &[u64],
    ) -> Result<usize, DecoderError> {
        let mut errors = 0;
        for (syn, &expected) in syndromes.iter().zip(expected_masks.iter()) {
            let predicted = self.decode_shot(syn)?;
            if predicted != expected {
                errors += 1;
            }
        }
        Ok(errors)
    }

    /// Number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Whether the algorithm has any feed-forward decision points.
    ///
    /// If false, the budget doesn't matter — all corrections are
    /// metadata applied at the end (Clifford-only circuit).
    /// If true, the reaction time budget is meaningful.
    #[must_use]
    pub fn has_decision_points(&self) -> bool {
        self.boundary_gates
            .iter()
            .any(|gates| gates.iter().any(BoundaryGate::is_decision_point))
    }

    /// Number of decision points (T gates, magic state injections).
    #[must_use]
    pub fn num_decision_points(&self) -> usize {
        self.boundary_gates
            .iter()
            .flat_map(|gates| gates.iter())
            .filter(|g| g.is_decision_point())
            .count()
    }

    /// Total detectors.
    #[must_use]
    pub fn total_detectors(&self) -> usize {
        self.total_detectors
    }

    /// Current Pauli frames (per qubit).
    #[must_use]
    pub fn frames(&self) -> &[u64] {
        &self.frames
    }

    /// The decode budget.
    #[must_use]
    pub fn budget(&self) -> &DecodeBudget {
        &self.budget
    }

    /// Reset for next shot.
    pub fn reset(&mut self) {
        self.strategy.reset();
        self.syndrome.fill(0);
        self.frames.fill(0);
        self.current_segment = 0;
        self.current_segment_fed = 0;
    }
}

impl ObservableDecoder for LogicalCircuitDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_shot(syndrome)
    }
}

// ============================================================================
// Strategy: Full Circuit (offline / ion trap)
// ============================================================================

/// Full-circuit decode strategy.
///
/// Buffers the entire syndrome, decodes at flush. Maximum accuracy.
/// Used for offline analysis, ion trap systems, or any budget that
/// allows full-circuit processing.
pub struct FullCircuitStrategy {
    inner: Box<dyn ObservableDecoder + Send + Sync>,
}

impl FullCircuitStrategy {
    /// Wrap any `ObservableDecoder` (typically OSD).
    #[must_use]
    pub fn new(decoder: Box<dyn ObservableDecoder + Send + Sync>) -> Self {
        Self { inner: decoder }
    }
}

impl DecodeStrategy for FullCircuitStrategy {
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.inner.decode_to_observables(syndrome)
    }

    fn commit(&mut self, _region: &DetectorRegion) -> Result<u64, DecoderError> {
        // Full circuit doesn't commit incrementally
        Ok(0)
    }

    fn committed_obs(&self) -> u64 {
        0
    }

    fn reset(&mut self) {
        // No state to reset for full-circuit strategy
    }
}

// ============================================================================
// Strategy: Windowed OSD (neutral atom / medium budget)
// ============================================================================

/// Windowed OSD strategy: per-observable subgraph windowed decoding.
///
/// Each observable's subgraph is graphlike (no hyperedges). A windowed
/// decoder (sandwich or plain PM) runs inside each subgraph with bounded
/// latency. The full matching graph is pre-built; only syndrome routing
/// and per-window matching are per-shot work.
///
/// This achieves bounded-latency streaming with OSD-level accuracy.
pub struct WindowedOsdStrategy {
    /// Per-subgraph decoders (windowed or plain).
    subgraph_decoders: Vec<Box<dyn ObservableDecoder + Send + Sync>>,
    /// Per-subgraph detector maps: `subgraph_detector_maps`[i][local] = global.
    detector_maps: Vec<Vec<usize>>,
    /// Per-subgraph sub-syndrome buffers (reusable).
    sub_syndromes: Vec<Vec<u8>>,
    /// Number of observables.
    _num_observables: usize,
}

impl WindowedOsdStrategy {
    /// Build from pre-extracted subgraph DEMs and detector maps.
    ///
    /// `subgraph_dems`: per-observable DEM strings (graphlike).
    /// `detector_maps`: per-observable local→global detector index maps.
    /// `factory`: creates the inner decoder for each subgraph DEM.
    pub fn new<F>(
        subgraph_dems: Vec<String>,
        detector_maps: Vec<Vec<usize>>,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&str) -> Result<Box<dyn ObservableDecoder + Send + Sync>, DecoderError>,
    {
        let num_observables = subgraph_dems.len();
        let mut decoders = Vec::with_capacity(num_observables);
        let mut sub_syndromes = Vec::with_capacity(num_observables);

        for (i, dem_str) in subgraph_dems.iter().enumerate() {
            let dec = factory(dem_str)?;
            let n = detector_maps.get(i).map_or(0, std::vec::Vec::len);
            sub_syndromes.push(vec![0u8; n]);
            decoders.push(dec);
        }

        Ok(Self {
            subgraph_decoders: decoders,
            detector_maps,
            sub_syndromes,
            _num_observables: num_observables,
        })
    }
}

impl DecodeStrategy for WindowedOsdStrategy {
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let mut obs_mask = 0u64;

        for (i, (dec, dmap)) in self
            .subgraph_decoders
            .iter_mut()
            .zip(self.detector_maps.iter())
            .enumerate()
        {
            let n = dmap.len();
            if n == 0 {
                continue;
            }

            // Route global syndrome to subgraph-local syndrome
            let buf = &mut self.sub_syndromes[i];
            for (local, &global) in dmap.iter().enumerate() {
                buf[local] = if global < syndrome.len() {
                    syndrome[global]
                } else {
                    0
                };
            }

            // Decode this subgraph
            let sub_obs = dec.decode_to_observables(&buf[..n])?;
            if sub_obs & 1 != 0 {
                obs_mask |= 1 << i;
            }
        }

        Ok(obs_mask)
    }

    fn commit(&mut self, _region: &DetectorRegion) -> Result<u64, DecoderError> {
        // Commitment is handled internally by the windowed inner decoders
        Ok(0)
    }

    fn committed_obs(&self) -> u64 {
        0
    }

    fn reset(&mut self) {
        for buf in &mut self.sub_syndromes {
            buf.fill(0);
        }
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
    fn test_single_segment() {
        let desc = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 4,
                num_observables: 2,
            }],
            boundary_gates: vec![],
            num_observables: 2,
        };
        let mut dec = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b01)), desc);
        assert_eq!(dec.decode_shot(&[0, 1, 0, 1]).unwrap(), 0b01);
    }

    #[test]
    fn test_hadamard_frame() {
        let mut frame = 0b01u64; // X correction on bit 0
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame, 0b10); // X became Z
    }

    #[test]
    fn test_cnot_frame() {
        let mut frame = 0b0001u64; // X on control (bit 0)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Cnot {
                ctrl_x_bit: 0,
                ctrl_z_bit: 1,
                tgt_x_bit: 2,
                tgt_z_bit: 3,
            },
        );
        assert_eq!(frame, 0b0101); // X propagated to target
    }

    #[test]
    fn test_logical_circuit_decoder_unlimited() {
        let desc = AlgorithmDescriptor {
            segments: vec![
                SegmentDescriptor {
                    num_detectors: 4,
                    num_observables: 2,
                },
                SegmentDescriptor {
                    num_detectors: 4,
                    num_observables: 2,
                },
            ],
            boundary_gates: vec![vec![BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            }]],
            num_observables: 2,
        };

        let strategy = FullCircuitStrategy::new(Box::new(FixedDecoder(0b01)));
        let budget = DecodeBudget::unlimited();

        let mut dec = LogicalCircuitDecoder::new(desc, Box::new(strategy), budget, 1);
        let result = dec.decode_shot(&[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(result, 0b01);
    }

    #[test]
    fn test_cnot_frame_z_backward() {
        // Z on target should propagate back to control
        let mut frame = 0b1000u64; // Z on target (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Cnot {
                ctrl_x_bit: 0,
                ctrl_z_bit: 1,
                tgt_x_bit: 2,
                tgt_z_bit: 3,
            },
        );
        assert_eq!(frame, 0b1010); // Z propagated back to control Z (bit 1)
    }

    #[test]
    fn test_cnot_frame_both_directions() {
        // X on control + Z on target -> both propagate
        let mut frame = 0b1001u64; // X on ctrl (bit 0), Z on tgt (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Cnot {
                ctrl_x_bit: 0,
                ctrl_z_bit: 1,
                tgt_x_bit: 2,
                tgt_z_bit: 3,
            },
        );
        // X ctrl -> X tgt (bit 2), Z tgt -> Z ctrl (bit 1)
        assert_eq!(frame, 0b1111);
    }

    #[test]
    fn test_sgate_frame_x_induces_z() {
        // S gate: X correction induces Z correction (X -> XZ = Y)
        let mut frame = 0b01u64; // X correction on bit 0
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame, 0b11); // X stays, Z also set
    }

    #[test]
    fn test_sgate_frame_z_unchanged() {
        // S gate: Z correction is unchanged (S commutes with Z)
        let mut frame = 0b10u64; // Z correction on bit 1
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame, 0b10); // Z stays, no X induced
    }

    #[test]
    fn test_sgate_frame_no_correction() {
        let mut frame = 0u64;
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame, 0); // No correction, no change
    }

    #[test]
    fn test_t_injection_frame_ancilla_z_folds() {
        // T injection: ancilla Z bit folds into data Z bit
        let mut frame = 0b1000u64; // ancilla Z set (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::TGateInjection {
                z_obs_bit: 1,     // data Z
                ancilla_z_bit: 3, // ancilla Z
            },
        );
        assert_eq!(frame, 0b1010); // data Z (bit 1) flipped
    }

    #[test]
    fn test_t_injection_frame_ancilla_z_cancels() {
        // If data Z already set and ancilla Z set, they cancel (XOR)
        let mut frame = 0b1010u64; // both data Z (bit 1) and ancilla Z (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::TGateInjection {
                z_obs_bit: 1,
                ancilla_z_bit: 3,
            },
        );
        assert_eq!(frame, 0b1000); // data Z cancelled, ancilla unchanged
    }

    #[test]
    fn test_t_injection_frame_no_ancilla_z() {
        // No ancilla Z -> no change
        let mut frame = 0b0010u64; // data Z set, ancilla Z not set
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::TGateInjection {
                z_obs_bit: 1,
                ancilla_z_bit: 3,
            },
        );
        assert_eq!(frame, 0b0010); // unchanged
    }

    #[test]
    fn test_hadamard_frame_swap_both() {
        // Both X and Z set -> swap
        let mut frame = 0b11u64;
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame, 0b11); // Swap of (1,1) is still (1,1)
    }

    #[test]
    fn test_hadamard_frame_z_to_x() {
        let mut frame = 0b10u64; // Z only
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame, 0b01); // Z became X
    }

    #[test]
    fn test_is_decision_point() {
        assert!(
            BoundaryGate::TGateInjection {
                z_obs_bit: 1,
                ancilla_z_bit: 3,
            }
            .is_decision_point()
        );

        assert!(
            !BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            }
            .is_decision_point()
        );

        assert!(
            !BoundaryGate::Cnot {
                ctrl_x_bit: 0,
                ctrl_z_bit: 1,
                tgt_x_bit: 2,
                tgt_z_bit: 3,
            }
            .is_decision_point()
        );

        assert!(
            !BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            }
            .is_decision_point()
        );
    }

    #[test]
    fn test_budget_windowed_vs_unlimited() {
        use std::time::Duration;
        let windowed = DecodeBudget::from_reaction_time(Duration::from_millis(1), 7);
        assert!(windowed.is_windowed());

        let unlimited = DecodeBudget::unlimited();
        assert!(unlimited.is_unlimited());
    }

    #[test]
    fn test_streaming_feed_dense_and_flush() {
        let desc = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 4,
                num_observables: 2,
            }],
            boundary_gates: vec![],
            num_observables: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b10)), desc);
        let mut streaming = StreamingLogicalDecoder::new(inner);

        // Feed full syndrome at once
        let result = streaming.decode_shot(&[0, 1, 0, 1]).unwrap();
        assert_eq!(result, 0b10);
        assert_eq!(streaming.accumulated_obs(), 0b10);
    }

    #[test]
    fn test_streaming_feed_sparse() {
        let desc = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 4,
                num_observables: 2,
            }],
            boundary_gates: vec![],
            num_observables: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b01)), desc);
        let mut streaming = StreamingLogicalDecoder::new(inner);

        // Feed individual detectors
        streaming.feed_detection(1, 1);
        streaming.feed_detection(3, 1);
        let result = streaming.flush().unwrap();
        assert_eq!(result, 0b01);
    }

    #[test]
    fn test_streaming_reset() {
        let desc = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 4,
                num_observables: 2,
            }],
            boundary_gates: vec![],
            num_observables: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b11)), desc);
        let mut streaming = StreamingLogicalDecoder::new(inner);

        streaming.decode_shot(&[1, 0, 1, 0]).unwrap();
        assert_eq!(streaming.accumulated_obs(), 0b11);

        streaming.reset();
        assert_eq!(streaming.accumulated_obs(), 0);
    }

    #[test]
    fn test_streaming_decode_count() {
        let desc = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 2,
                num_observables: 1,
            }],
            boundary_gates: vec![],
            num_observables: 1,
        };
        let inner = LogicalAlgorithmDecoder::new(
            Box::new(FixedDecoder(0b1)),
            desc, // always predicts obs flip
        );
        let mut streaming = StreamingLogicalDecoder::new(inner);

        let syndromes = vec![vec![0u8, 0], vec![1, 0], vec![0, 1]];
        let expected = vec![0b1, 0b0, 0b1]; // matches on shot 0 and 2

        let errors = streaming_decode_count(&mut streaming, &syndromes, &expected).unwrap();
        assert_eq!(errors, 1); // only shot 1 is wrong (predicted 1, expected 0)
    }
}
