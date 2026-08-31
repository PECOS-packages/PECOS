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
//! - **Full-circuit**: Uses the full DEM's logical-subgraph decoder for maximum accuracy.
//!   Equivalent to `LogicalSubgraphDecoder` on the full circuit.
//! - **Per-segment** (future streaming): Each segment decoded independently
//!   with buffer overlap at gate boundaries.

use crate::ObservableDecoder;
use crate::decode_budget::DecodeStrategy;
use crate::errors::DecoderError;
use crate::obs_mask::ObsMask;

const UNSUPPORTED_DECISION_POINT_MESSAGE: &str = "descriptor contains feed-forward decision points \
    (TGateInjection); no available decode strategy consults them yet — decoding would silently \
    treat the injection as Clifford (issue #596)";

/// Pauli-frame bits for the logical patches in an algorithm descriptor.
///
/// Slot `2 * patch` stores that patch's X frame and slot `2 * patch + 1`
/// stores its Z frame. The storage reuses [`ObsMask`] so frame widths are not
/// limited to one machine word; `num_slots` records the descriptor's logical
/// width independently of the detector error model's observable count.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FrameBits {
    bits: ObsMask,
    num_slots: usize,
}

impl FrameBits {
    /// Create an all-zero frame with `num_slots` addressable slots.
    #[must_use]
    pub fn new(num_slots: usize) -> Self {
        Self {
            bits: ObsMask::new(),
            num_slots,
        }
    }

    /// Return the value of a frame slot.
    #[must_use]
    pub fn get(&self, slot: usize) -> bool {
        assert!(slot < self.num_slots, "frame slot {slot} is out of range");
        self.bits.get(slot)
    }

    /// Toggle a frame slot.
    pub fn flip(&mut self, slot: usize) {
        assert!(slot < self.num_slots, "frame slot {slot} is out of range");
        let mut mask = ObsMask::new();
        mask.set(slot);
        self.bits ^= &mask;
    }

    /// Set a frame slot to `value`.
    pub fn set(&mut self, slot: usize, value: bool) {
        if self.get(slot) != value {
            self.flip(slot);
        }
    }

    /// Number of addressable frame slots.
    #[must_use]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }
}

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

    /// All frame-slot indices this gate references.
    #[must_use]
    pub fn obs_bits(&self) -> Vec<u32> {
        match self {
            Self::Hadamard {
                x_obs_bit,
                z_obs_bit,
            }
            | Self::SGate {
                x_obs_bit,
                z_obs_bit,
            } => vec![*x_obs_bit, *z_obs_bit],
            Self::Cnot {
                ctrl_x_bit,
                ctrl_z_bit,
                tgt_x_bit,
                tgt_z_bit,
            } => vec![*ctrl_x_bit, *ctrl_z_bit, *tgt_x_bit, *tgt_z_bit],
            Self::TGateInjection {
                z_obs_bit,
                ancilla_z_bit,
            } => vec![*z_obs_bit, *ancilla_z_bit],
        }
    }
}

/// Full description of a logical algorithm for decoding.
pub struct AlgorithmDescriptor {
    /// Per-segment descriptors.
    pub segments: Vec<SegmentDescriptor>,
    /// Gates at segment boundaries. `boundary_gates[i]` between segment i and i+1.
    pub boundary_gates: Vec<Vec<BoundaryGate>>,
    /// Number of observables declared by the full detector error model.
    pub num_observables: usize,
    /// Number of Pauli-frame slots (two per logical patch: X then Z).
    pub num_frame_slots: usize,
}

impl AlgorithmDescriptor {
    /// Validate the between-segment boundary schema and all frame references.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError::InvalidConfiguration`] when the descriptor is
    /// empty, its boundary cardinality is not exactly one fewer than its segment
    /// cardinality, or a boundary gate references a slot outside
    /// `num_frame_slots`.
    pub fn validate(&self) -> Result<(), DecoderError> {
        if self.segments.is_empty() {
            return Err(DecoderError::InvalidConfiguration(
                "algorithm descriptor must contain at least one segment".into(),
            ));
        }

        let expected_boundaries = self.segments.len() - 1;
        if self.boundary_gates.len() != expected_boundaries {
            return Err(DecoderError::InvalidConfiguration(format!(
                "algorithm descriptor has {} boundary gate lists and {} segments; expected exactly one boundary list between consecutive segments",
                self.boundary_gates.len(),
                self.segments.len(),
            )));
        }

        for (boundary_index, gates) in self.boundary_gates.iter().enumerate() {
            for (gate_index, gate) in gates.iter().enumerate() {
                if let Some(bit) = gate
                    .obs_bits()
                    .into_iter()
                    .find(|&bit| bit as usize >= self.num_frame_slots)
                {
                    return Err(DecoderError::InvalidConfiguration(format!(
                        "boundary gate at boundary {boundary_index}, gate {gate_index} references frame slot {bit}, but num_frame_slots is {}",
                        self.num_frame_slots,
                    )));
                }
            }
        }

        Ok(())
    }

    fn reject_unsupported_decision_points(&self) -> Result<(), DecoderError> {
        if self.has_decision_points() {
            return Err(DecoderError::InvalidConfiguration(
                UNSUPPORTED_DECISION_POINT_MESSAGE.into(),
            ));
        }
        Ok(())
    }

    /// Whether any boundary requires feed-forward consultation.
    #[must_use]
    pub fn has_decision_points(&self) -> bool {
        self.boundary_gates
            .iter()
            .flatten()
            .any(BoundaryGate::is_decision_point)
    }
}

/// Outcome of consulting an interactive decoder at a feed-forward boundary.
pub enum DecisionStatus {
    /// The frame-corrected ancilla bit is ready and selects the S correction.
    Ready { corrected_bit: bool },
    /// The patience protocol needs more syndrome before it can decide.
    NeedMoreData { extra_rounds: usize },
}

/// Decode strategy capable of consulting a feed-forward decision point.
///
/// In the phase-2 contract, [`Self::consult`] is called at a decision-point
/// boundary *after* the post-injection segment's syndrome has been fed.
/// [`DecisionStatus::Ready`] carries the frame-corrected ancilla bit selecting
/// the S correction; [`DecisionStatus::NeedMoreData`] asks the harness for
/// additional syndrome rounds under the patience protocol. No implementation
/// exists yet, and all current logical decoders reject decision descriptors so
/// they cannot silently bypass consultation (issue #596). Terminal-segment
/// support for gates after final measurement remains tracked by issue #595.
/// See `pecos-docs/design/streaming-transversal-decoding.md`.
pub trait DecisionConsultingStrategy: DecodeStrategy {
    /// Consult the strategy at a T-injection decision boundary.
    fn consult(
        &mut self,
        gate: &BoundaryGate,
        raw_ancilla_outcome: u8,
        frame: &FrameBits,
    ) -> Result<DecisionStatus, DecoderError>;
}

/// Decoder for logical quantum algorithms.
///
/// Wraps a full-circuit decoder (logical-subgraph decoder) with segment metadata. The
/// segment structure enables:
/// - Tracking which gates occur at which point in the circuit
/// - Pauli frame propagation for T-gate/measurement corrections
/// - Future streaming mode with per-segment windowed decoding
///
/// In the current implementation, `decode_shot` delegates to the
/// full-circuit logical-subgraph decoder for maximum accuracy. The segment structure is
/// metadata for frame tracking and streaming (step 5).
pub struct LogicalAlgorithmDecoder {
    /// Full-circuit decoder (logical-subgraph decoder on the complete DEM).
    full_decoder: Box<dyn ObservableDecoder + Send + Sync>,
    /// Validated segment, boundary, observable, and frame metadata.
    descriptor: AlgorithmDescriptor,
}

impl LogicalAlgorithmDecoder {
    /// Build from a full-circuit decoder and algorithm descriptor.
    ///
    /// The `full_decoder` is typically an `LogicalSubgraphDecoder`
    /// built from the full circuit DEM.
    pub fn new(
        full_decoder: Box<dyn ObservableDecoder + Send + Sync>,
        descriptor: AlgorithmDescriptor,
    ) -> Result<Self, DecoderError> {
        descriptor.validate()?;
        descriptor.reject_unsupported_decision_points()?;
        Ok(Self {
            full_decoder,
            descriptor,
        })
    }

    /// Decode one shot using the full-circuit decoder.
    pub fn decode_shot(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.full_decoder.decode_to_observables(syndrome)
    }

    /// Number of segments.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.descriptor.segments.len()
    }

    /// Total detectors across all segments.
    #[must_use]
    pub fn total_detectors(&self) -> usize {
        self.descriptor
            .segments
            .iter()
            .map(|s| s.num_detectors)
            .sum()
    }

    /// Apply boundary gate to a Pauli frame.
    /// Used when consuming the frame at logical operations.
    ///
    pub fn apply_boundary_gate(frame: &mut FrameBits, gate: &BoundaryGate) {
        match gate {
            BoundaryGate::Hadamard {
                x_obs_bit,
                z_obs_bit,
            } => {
                let x_set = frame.get(*x_obs_bit as usize);
                let z_set = frame.get(*z_obs_bit as usize);
                frame.set(*x_obs_bit as usize, z_set);
                frame.set(*z_obs_bit as usize, x_set);
            }
            BoundaryGate::Cnot {
                ctrl_x_bit,
                ctrl_z_bit,
                tgt_x_bit,
                tgt_z_bit,
            } => {
                if frame.get(*ctrl_x_bit as usize) {
                    frame.flip(*tgt_x_bit as usize);
                }
                if frame.get(*tgt_z_bit as usize) {
                    frame.flip(*ctrl_z_bit as usize);
                }
            }
            BoundaryGate::SGate {
                x_obs_bit,
                z_obs_bit,
            } => {
                if frame.get(*x_obs_bit as usize) {
                    frame.flip(*z_obs_bit as usize);
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
                if frame.get(*ancilla_z_bit as usize) {
                    frame.flip(*z_obs_bit as usize);
                }
            }
        }
    }
}

impl ObservableDecoder for LogicalAlgorithmDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_shot(syndrome)
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        self.full_decoder.decode_obs(syndrome)
    }
}

// ============================================================================
// Streaming mode
// ============================================================================

/// Streaming wrapper for `LogicalAlgorithmDecoder`.
///
/// Buffers syndrome data round-by-round. The full-circuit logical-subgraph decoder decodes
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
/// use pecos_decoder_core::obs_mask::ObsMask;
/// use pecos_decoder_core::logical_algorithm::{
///     AlgorithmDescriptor, LogicalAlgorithmDecoder, SegmentDescriptor, StreamingLogicalDecoder,
/// };
///
/// struct AnyDetectionDecoder;
///
/// impl ObservableDecoder for AnyDetectionDecoder {
///     fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
///         Ok(ObsMask::from_u64(u64::from(syndrome.iter().any(|&bit| bit != 0))))
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
///     num_frame_slots: 2,
/// };
/// let decoder = LogicalAlgorithmDecoder::new(Box::new(AnyDetectionDecoder), descriptor).unwrap();
/// let mut stream = StreamingLogicalDecoder::new(decoder).unwrap();
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
    /// The underlying batch decoder (full-circuit logical-subgraph decoder).
    inner: LogicalAlgorithmDecoder,
    /// Accumulated syndrome buffer (full circuit size).
    syndrome: Vec<u8>,
    /// Total detectors.
    total_detectors: usize,
    /// Rounds fed so far.
    rounds_fed: usize,
    /// Accumulated observable correction from last flush (wide; lossless
    /// beyond 64 observables).
    accumulated_obs: ObsMask,
}

impl StreamingLogicalDecoder {
    /// Create from a `LogicalAlgorithmDecoder`.
    pub fn new(decoder: LogicalAlgorithmDecoder) -> Result<Self, DecoderError> {
        decoder.descriptor.validate()?;
        decoder.descriptor.reject_unsupported_decision_points()?;
        let total = decoder.total_detectors();
        Ok(Self {
            inner: decoder,
            syndrome: vec![0u8; total],
            total_detectors: total,
            rounds_fed: 0,
            accumulated_obs: ObsMask::new(),
        })
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

    /// Decode the accumulated syndrome using the full-circuit logical-subgraph decoder.
    ///
    /// Returns the observable correction mask. This is the final
    /// correction to apply to raw measurement outcomes.
    pub fn flush(&mut self) -> Result<u64, DecoderError> {
        self.flush_obs()?.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; use flush_obs() for the wide mask".into(),
            )
        })
    }

    /// Wide variant of [`Self::flush`]: returns an [`ObsMask`] supporting more
    /// than 64 observables (no truncation). Updates the accumulated correction.
    pub fn flush_obs(&mut self) -> Result<ObsMask, DecoderError> {
        let obs = self.inner.decode_obs(&self.syndrome)?;
        self.accumulated_obs = obs.clone();
        Ok(obs)
    }

    /// Wide variant of [`Self::decode_shot`]: feed + flush as an [`ObsMask`].
    pub fn decode_shot_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        self.feed_dense(syndrome);
        self.flush_obs()
    }

    /// Decode a full syndrome at once (convenience for batch mode).
    pub fn decode_shot(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.feed_dense(syndrome);
        self.flush()
    }

    /// Current accumulated observable correction, narrowed to `u64`.
    ///
    /// # Errors
    /// Errors (rather than truncating) if the accumulated mask exceeds 64
    /// observables; use [`Self::accumulated_obs_mask`] for the wide mask.
    pub fn accumulated_obs(&self) -> Result<u64, DecoderError> {
        self.accumulated_obs.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "accumulated mask has more than 64 observables; use accumulated_obs_mask()".into(),
            )
        })
    }

    /// Current accumulated observable correction as a wide mask.
    #[must_use]
    pub fn accumulated_obs_mask(&self) -> &ObsMask {
        &self.accumulated_obs
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
        &self.inner.descriptor.boundary_gates
    }

    /// Apply boundary gate to a Pauli frame (delegates to inner).
    pub fn apply_boundary_gate(frame: &mut FrameBits, gate: &BoundaryGate) {
        LogicalAlgorithmDecoder::apply_boundary_gate(frame, gate);
    }

    /// Reset for the next shot.
    pub fn reset(&mut self) {
        self.syndrome.fill(0);
        self.rounds_fed = 0;
        self.accumulated_obs = ObsMask::new();
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

use crate::decode_budget::{DecodeBudget, DetectorRegion};

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
/// - **Streaming** (neutral atom): `CommittedLogicalSubgraphStrategy` — decode and
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
    /// Reset-only Pauli frame. Phase 2 will update it while decoding boundaries.
    frame: FrameBits,
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
    pub fn new(
        descriptor: AlgorithmDescriptor,
        strategy: Box<dyn DecodeStrategy + Send + Sync>,
        budget: DecodeBudget,
    ) -> Result<Self, DecoderError> {
        descriptor.validate()?;
        descriptor.reject_unsupported_decision_points()?;
        let mut segment_offsets = Vec::with_capacity(descriptor.segments.len());
        let mut offset = 0;
        for seg in &descriptor.segments {
            segment_offsets.push(offset);
            offset += seg.num_detectors;
        }
        let total_detectors = offset;

        let frame = FrameBits::new(descriptor.num_frame_slots);
        Ok(Self {
            strategy,
            segments: descriptor.segments,
            _segment_offsets: segment_offsets,
            boundary_gates: descriptor.boundary_gates,
            frame,
            budget,
            syndrome: vec![0u8; total_detectors],
            total_detectors,
            current_segment: 0,
            current_segment_fed: 0,
        })
    }

    /// Decode a full shot (batch mode).
    ///
    /// For offline/ion trap budgets: equivalent to full-circuit logical-subgraph decoder.
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

    /// Current Pauli frame.
    ///
    /// The frame remains reset-only in phase 0. Boundary-driven updates arrive
    /// with decision plumbing in phase 2.
    #[must_use]
    pub fn frame(&self) -> &FrameBits {
        &self.frame
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
        self.frame = FrameBits::new(self.frame.num_slots());
        self.current_segment = 0;
        self.current_segment_fed = 0;
    }
}

impl ObservableDecoder for LogicalCircuitDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_shot(syndrome)
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        self.reset();
        let len = syndrome.len().min(self.total_detectors);
        self.syndrome[..len].copy_from_slice(&syndrome[..len]);
        self.strategy.decode_obs(&self.syndrome)
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
    /// Wrap any `ObservableDecoder` (typically logical-subgraph decoder).
    #[must_use]
    pub fn new(decoder: Box<dyn ObservableDecoder + Send + Sync>) -> Self {
        Self { inner: decoder }
    }
}

impl DecodeStrategy for FullCircuitStrategy {
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.inner.decode_to_observables(syndrome)
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        self.inner.decode_obs(syndrome)
    }

    fn commit(&mut self, _region: &DetectorRegion) -> Result<u64, DecoderError> {
        // Full circuit doesn't commit incrementally
        Ok(0)
    }

    fn committed_obs(&self) -> Result<u64, DecoderError> {
        Ok(0)
    }

    fn reset(&mut self) {
        // No state to reset for full-circuit strategy
    }
}

// ============================================================================
// Strategy: Windowed logical-subgraph decoding (neutral atom / medium budget)
// ============================================================================

/// Windowed logical-subgraph strategy: per-logical-operator subgraph windowed decoding.
///
/// Each observable's subgraph is graphlike (no hyperedges). A windowed
/// decoder (sandwich or plain PM) runs inside each subgraph with bounded
/// latency. The full matching graph is pre-built; only syndrome routing
/// and per-window matching are per-shot work.
///
/// This achieves bounded-latency streaming with logical-subgraph decoder-level accuracy.
pub struct WindowedLogicalSubgraphStrategy {
    /// Per-subgraph decoders (windowed or plain).
    subgraph_decoders: Vec<Box<dyn ObservableDecoder + Send + Sync>>,
    /// Per-subgraph detector maps: `subgraph_detector_maps`[i][local] = global.
    detector_maps: Vec<Vec<usize>>,
    /// Global observable (logical) index each subgraph decodes. Required because
    /// callers may pass only the non-empty subgraphs (empty-region observables
    /// dropped), so the subgraph's list position is NOT its observable index.
    observable_indices: Vec<usize>,
    /// Per-subgraph sub-syndrome buffers (reusable).
    sub_syndromes: Vec<Vec<u8>>,
}

impl WindowedLogicalSubgraphStrategy {
    /// Build from pre-extracted subgraph DEMs, detector maps, and the global
    /// observable index each subgraph decodes.
    ///
    /// `subgraph_dems`: per-subgraph DEM strings (graphlike).
    /// `detector_maps`: per-subgraph local→global detector index maps.
    /// `observable_indices`: the global observable bit each subgraph flips
    ///   (each subgraph reports its observable as local bit 0). MUST line up
    ///   with `subgraph_dems` — when empty-region observables are filtered out,
    ///   pass the surviving observables' true indices, not `0..n`.
    /// `factory`: creates the inner decoder for each subgraph DEM.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the factory fails, if the three input vectors
    /// disagree in length, or if any observable index is >= 64 (the u64
    /// observable mask cannot hold it).
    pub fn new<F>(
        subgraph_dems: Vec<String>,
        detector_maps: Vec<Vec<usize>>,
        observable_indices: Vec<usize>,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&str) -> Result<Box<dyn ObservableDecoder + Send + Sync>, DecoderError>,
    {
        let num = subgraph_dems.len();
        if detector_maps.len() != num || observable_indices.len() != num {
            return Err(DecoderError::InvalidConfiguration(format!(
                "WindowedLogicalSubgraphStrategy: mismatched inputs (dems={num}, \
                 maps={}, obs={})",
                detector_maps.len(),
                observable_indices.len(),
            )));
        }
        let mut decoders = Vec::with_capacity(num);
        let mut sub_syndromes = Vec::with_capacity(num);
        for (i, dem_str) in subgraph_dems.iter().enumerate() {
            let dec = factory(dem_str)?;
            let n = detector_maps[i].len();
            sub_syndromes.push(vec![0u8; n]);
            decoders.push(dec);
        }

        Ok(Self {
            subgraph_decoders: decoders,
            detector_maps,
            observable_indices,
            sub_syndromes,
        })
    }
}

impl DecodeStrategy for WindowedLogicalSubgraphStrategy {
    /// Narrowing wrapper over [`Self::decode_obs`]; errors above 64 observables.
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_obs(syndrome)?.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; use decode_obs() for the wide mask".into(),
            )
        })
    }

    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        let mut obs_mask = ObsMask::new();

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

            // Decode this subgraph: it reports its observable as local bit 0;
            // map that to the subgraph's *global* observable bit (not its list
            // position `i`, which differs once empty observables are filtered).
            let sub_obs = dec.decode_to_observables(&buf[..n])?;
            if sub_obs & 1 != 0 {
                obs_mask.set(self.observable_indices[i]);
            }
        }

        Ok(obs_mask)
    }

    fn commit(&mut self, _region: &DetectorRegion) -> Result<u64, DecoderError> {
        // NOTE (abstraction caveat): this strategy is currently a *batch* decoder
        // exposed through the streaming `DecodeStrategy` trait. It decodes the
        // whole syndrome in one `decode()` call; per-observable subgraph windowing
        // (when enabled) is handled inside each inner decoder, not via incremental
        // region commits. So `commit()` is intentionally a no-op and
        // `committed_obs()` returns 0. Real streaming commit semantics are a
        // follow-up (see the windowed logical-subgraph proper-solution design).
        Ok(0)
    }

    fn committed_obs(&self) -> Result<u64, DecoderError> {
        Ok(0)
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

    fn frame_from_u64(value: u64, num_slots: usize) -> FrameBits {
        let mut frame = FrameBits::new(num_slots);
        for slot in 0..num_slots.min(64) {
            frame.set(slot, value & (1 << slot) != 0);
        }
        frame
    }

    fn frame_as_u64(frame: &FrameBits) -> u64 {
        (0..frame.num_slots().min(64)).fold(0, |value, slot| {
            value | (u64::from(frame.get(slot)) << slot)
        })
    }

    struct FixedDecoder(u64);
    impl ObservableDecoder for FixedDecoder {
        fn decode_obs(&mut self, _: &[u8]) -> Result<crate::obs_mask::ObsMask, DecoderError> {
            Ok(crate::obs_mask::ObsMask::from_u64(self.0))
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
            num_frame_slots: 2,
        };
        let mut dec = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b01)), desc).unwrap();
        assert_eq!(dec.decode_shot(&[0, 1, 0, 1]).unwrap(), 0b01);
    }

    #[test]
    fn descriptor_rejects_empty_segments() {
        let descriptor = AlgorithmDescriptor {
            segments: vec![],
            boundary_gates: vec![],
            num_observables: 0,
            num_frame_slots: 0,
        };
        let error = descriptor.validate().unwrap_err();
        assert!(error.to_string().contains("at least one segment"));
    }

    #[test]
    fn descriptor_rejects_boundary_cardinality_mismatch() {
        let descriptor = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 1,
                num_observables: 0,
            }],
            boundary_gates: vec![vec![]],
            num_observables: 0,
            num_frame_slots: 0,
        };
        let error = descriptor.validate().unwrap_err();
        let message = error.to_string();
        assert!(message.contains("1 boundary gate lists"));
        assert!(message.contains("1 segments"));
    }

    #[test]
    fn descriptor_rejects_frame_slot_out_of_range() {
        let descriptor = AlgorithmDescriptor {
            segments: vec![
                SegmentDescriptor {
                    num_detectors: 1,
                    num_observables: 1,
                },
                SegmentDescriptor {
                    num_detectors: 1,
                    num_observables: 1,
                },
            ],
            boundary_gates: vec![vec![BoundaryGate::Hadamard {
                x_obs_bit: 4,
                z_obs_bit: 1,
            }]],
            num_observables: 1,
            num_frame_slots: 4,
        };
        let error = descriptor.validate().unwrap_err();
        let message = error.to_string();
        assert!(message.contains("frame slot 4"));
        assert!(message.contains("num_frame_slots is 4"));
    }

    fn decision_descriptor() -> AlgorithmDescriptor {
        AlgorithmDescriptor {
            segments: vec![
                SegmentDescriptor {
                    num_detectors: 1,
                    num_observables: 1,
                },
                SegmentDescriptor {
                    num_detectors: 1,
                    num_observables: 1,
                },
            ],
            boundary_gates: vec![vec![BoundaryGate::TGateInjection {
                z_obs_bit: 1,
                ancilla_z_bit: 3,
            }]],
            num_observables: 1,
            num_frame_slots: 4,
        }
    }

    fn assert_decision_rejection(error: DecoderError) {
        assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
        assert_eq!(
            error.to_string(),
            format!("Invalid configuration: {UNSUPPORTED_DECISION_POINT_MESSAGE}")
        );
    }

    #[test]
    fn logical_algorithm_decoder_rejects_decision_points() {
        let error = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0)), decision_descriptor())
            .err()
            .expect("decision descriptors must be rejected");
        assert_decision_rejection(error);
    }

    #[test]
    fn streaming_logical_decoder_revalidates_decision_points() {
        let valid = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 1,
                num_observables: 1,
            }],
            boundary_gates: vec![],
            num_observables: 1,
            num_frame_slots: 4,
        };
        let mut decoder = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0)), valid).unwrap();
        decoder.descriptor = decision_descriptor();
        let error = StreamingLogicalDecoder::new(decoder)
            .err()
            .expect("decision descriptors must be rejected");
        assert_decision_rejection(error);
    }

    #[test]
    fn logical_circuit_decoder_rejects_decision_points() {
        let strategy = FullCircuitStrategy::new(Box::new(FixedDecoder(0)));
        let error = LogicalCircuitDecoder::new(
            decision_descriptor(),
            Box::new(strategy),
            DecodeBudget::unlimited(),
        )
        .err()
        .expect("decision descriptors must be rejected");
        assert_decision_rejection(error);
    }

    #[test]
    fn frame_bits_support_slots_above_u64() {
        let mut frame = FrameBits::new(130);
        frame.flip(97);
        assert!(frame.get(97));
        frame.set(97, false);
        assert!(!frame.get(97));
        assert_eq!(frame.num_slots(), 130);
    }

    #[test]
    fn windowed_strategy_maps_to_global_observable_index() {
        // Two surviving subgraphs whose true (global) observable indices are
        // NON-contiguous -- as happens when earlier observables had empty
        // regions and were filtered out. Each reports its observable as local
        // bit 0; the strategy must flip the GLOBAL bit, not the list position.
        // (The pre-fix `1 << i` would have produced bits {0,1} = 0b0011.)
        let mut strategy = WindowedLogicalSubgraphStrategy::new(
            vec![
                "error(0.1) D0 L0".to_string(),
                "error(0.1) D0 L0".to_string(),
            ],
            vec![vec![0usize], vec![1usize]],
            vec![1usize, 3usize],
            |_dem| Ok(Box::new(FixedDecoder(1)) as Box<dyn ObservableDecoder + Send + Sync>),
        )
        .unwrap();
        let obs = strategy.decode(&[1, 1]).unwrap();
        assert_eq!(obs, (1u64 << 1) | (1u64 << 3));
    }

    #[test]
    fn windowed_strategy_supports_observable_index_over_63() {
        use crate::decode_budget::DecodeStrategy;
        // Observable index 64 was previously rejected; it now constructs and the
        // wide `decode_obs` represents bit 64 with no truncation.
        let mut s = WindowedLogicalSubgraphStrategy::new(
            vec!["error(0.1) D0 L0".to_string()],
            vec![vec![0usize]],
            vec![64usize],
            |_dem| Ok(Box::new(FixedDecoder(1)) as Box<dyn ObservableDecoder + Send + Sync>),
        )
        .unwrap();
        let wide = s.decode_obs(&[1]).unwrap();
        assert!(wide.get(64));
        assert_eq!(wide.to_u64(), None);
        // The narrowing u64 path errors rather than truncating.
        assert!(s.decode(&[1]).is_err());
    }

    #[test]
    fn windowed_strategy_rejects_mismatched_input_lengths() {
        let r = WindowedLogicalSubgraphStrategy::new(
            vec!["error(0.1) D0 L0".to_string()],
            vec![vec![0usize], vec![1usize]], // 2 maps for 1 dem
            vec![0usize],
            |_dem| Ok(Box::new(FixedDecoder(1)) as Box<dyn ObservableDecoder + Send + Sync>),
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_hadamard_frame() {
        let mut frame = frame_from_u64(0b01, 2); // X correction on bit 0
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b10); // X became Z
    }

    #[test]
    fn test_apply_boundary_gate_supports_frame_slot_ge_64() {
        let mut frame = FrameBits::new(66);
        frame.set(64, true);
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 64,
                z_obs_bit: 65,
            },
        );
        assert!(!frame.get(64));
        assert!(frame.get(65));
    }

    #[test]
    fn test_boundary_gate_obs_bits_cover_all_fields() {
        // Descriptor validation relies on obs_bits() listing every referenced slot.
        assert_eq!(
            BoundaryGate::Hadamard {
                x_obs_bit: 2,
                z_obs_bit: 5
            }
            .obs_bits(),
            vec![2, 5]
        );
        assert_eq!(
            BoundaryGate::Cnot {
                ctrl_x_bit: 1,
                ctrl_z_bit: 2,
                tgt_x_bit: 3,
                tgt_z_bit: 4
            }
            .obs_bits(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            BoundaryGate::SGate {
                x_obs_bit: 7,
                z_obs_bit: 9
            }
            .obs_bits(),
            vec![7, 9]
        );
        assert_eq!(
            BoundaryGate::TGateInjection {
                z_obs_bit: 6,
                ancilla_z_bit: 8
            }
            .obs_bits(),
            vec![6, 8]
        );
    }

    #[test]
    fn test_cnot_frame() {
        let mut frame = frame_from_u64(0b0001, 4); // X on control (bit 0)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Cnot {
                ctrl_x_bit: 0,
                ctrl_z_bit: 1,
                tgt_x_bit: 2,
                tgt_z_bit: 3,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b0101); // X propagated to target
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
            num_frame_slots: 2,
        };

        let strategy = FullCircuitStrategy::new(Box::new(FixedDecoder(0b01)));
        let budget = DecodeBudget::unlimited();

        let mut dec = LogicalCircuitDecoder::new(desc, Box::new(strategy), budget).unwrap();
        assert_eq!(dec.frame().num_slots(), 2);
        let result = dec.decode_shot(&[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(result, 0b01);
    }

    #[test]
    fn test_cnot_frame_z_backward() {
        // Z on target should propagate back to control
        let mut frame = frame_from_u64(0b1000, 4); // Z on target (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Cnot {
                ctrl_x_bit: 0,
                ctrl_z_bit: 1,
                tgt_x_bit: 2,
                tgt_z_bit: 3,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b1010); // Z propagated back to control Z (bit 1)
    }

    #[test]
    fn test_cnot_frame_both_directions() {
        // X on control + Z on target -> both propagate
        let mut frame = frame_from_u64(0b1001, 4); // X on ctrl (bit 0), Z on tgt (bit 3)
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
        assert_eq!(frame_as_u64(&frame), 0b1111);
    }

    #[test]
    fn test_sgate_frame_x_induces_z() {
        // S gate: X correction induces Z correction (X -> XZ = Y)
        let mut frame = frame_from_u64(0b01, 2); // X correction on bit 0
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b11); // X stays, Z also set
    }

    #[test]
    fn test_sgate_frame_z_unchanged() {
        // S gate: Z correction is unchanged (S commutes with Z)
        let mut frame = frame_from_u64(0b10, 2); // Z correction on bit 1
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b10); // Z stays, no X induced
    }

    #[test]
    fn test_sgate_frame_no_correction() {
        let mut frame = FrameBits::new(2);
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::SGate {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0); // No correction, no change
    }

    #[test]
    fn test_t_injection_frame_ancilla_z_folds() {
        // T injection: ancilla Z bit folds into data Z bit
        let mut frame = frame_from_u64(0b1000, 4); // ancilla Z set (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::TGateInjection {
                z_obs_bit: 1,     // data Z
                ancilla_z_bit: 3, // ancilla Z
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b1010); // data Z (bit 1) flipped
    }

    #[test]
    fn test_t_injection_frame_ancilla_z_cancels() {
        // If data Z already set and ancilla Z set, they cancel (XOR)
        let mut frame = frame_from_u64(0b1010, 4); // both data Z (bit 1) and ancilla Z (bit 3)
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::TGateInjection {
                z_obs_bit: 1,
                ancilla_z_bit: 3,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b1000); // data Z cancelled, ancilla unchanged
    }

    #[test]
    fn test_t_injection_frame_no_ancilla_z() {
        // No ancilla Z -> no change
        let mut frame = frame_from_u64(0b0010, 4); // data Z set, ancilla Z not set
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::TGateInjection {
                z_obs_bit: 1,
                ancilla_z_bit: 3,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b0010); // unchanged
    }

    #[test]
    fn test_hadamard_frame_swap_both() {
        // Both X and Z set -> swap
        let mut frame = frame_from_u64(0b11, 2);
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b11); // Swap of (1,1) is still (1,1)
    }

    #[test]
    fn test_hadamard_frame_z_to_x() {
        let mut frame = frame_from_u64(0b10, 2); // Z only
        LogicalAlgorithmDecoder::apply_boundary_gate(
            &mut frame,
            &BoundaryGate::Hadamard {
                x_obs_bit: 0,
                z_obs_bit: 1,
            },
        );
        assert_eq!(frame_as_u64(&frame), 0b01); // Z became X
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
            num_frame_slots: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b10)), desc).unwrap();
        let mut streaming = StreamingLogicalDecoder::new(inner).unwrap();

        // Feed full syndrome at once
        let result = streaming.decode_shot(&[0, 1, 0, 1]).unwrap();
        assert_eq!(result, 0b10);
        assert_eq!(streaming.accumulated_obs().unwrap(), 0b10);
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
            num_frame_slots: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b01)), desc).unwrap();
        let mut streaming = StreamingLogicalDecoder::new(inner).unwrap();

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
            num_frame_slots: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b11)), desc).unwrap();
        let mut streaming = StreamingLogicalDecoder::new(inner).unwrap();

        streaming.decode_shot(&[1, 0, 1, 0]).unwrap();
        assert_eq!(streaming.accumulated_obs().unwrap(), 0b11);

        streaming.reset();
        assert_eq!(streaming.accumulated_obs().unwrap(), 0);
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
            num_frame_slots: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(
            Box::new(FixedDecoder(0b1)),
            desc, // always predicts obs flip
        )
        .unwrap();
        let mut streaming = StreamingLogicalDecoder::new(inner).unwrap();

        let syndromes = vec![vec![0u8, 0], vec![1, 0], vec![0, 1]];
        let expected = vec![0b1, 0b0, 0b1]; // matches on shot 0 and 2

        let errors = streaming_decode_count(&mut streaming, &syndromes, &expected).unwrap();
        assert_eq!(errors, 1); // only shot 1 is wrong (predicted 1, expected 0)
    }
    #[test]
    fn wide_decode_path_updates_accumulated_obs() {
        // Regression: decode_shot_obs (the wide path, used by the Python
        // bindings' decode()) must update the accumulated correction just
        // like the narrow decode_shot path does.
        let desc = AlgorithmDescriptor {
            segments: vec![SegmentDescriptor {
                num_detectors: 4,
                num_observables: 2,
            }],
            boundary_gates: vec![],
            num_observables: 2,
            num_frame_slots: 2,
        };
        let inner = LogicalAlgorithmDecoder::new(Box::new(FixedDecoder(0b10)), desc).unwrap();
        let mut streaming = StreamingLogicalDecoder::new(inner).unwrap();

        let mask = streaming.decode_shot_obs(&[0, 1, 0, 1]).unwrap();
        assert_eq!(mask.to_u64(), Some(0b10));
        assert_eq!(streaming.accumulated_obs().unwrap(), 0b10);
        assert_eq!(streaming.accumulated_obs_mask().to_u64(), Some(0b10));

        streaming.reset();
        assert_eq!(streaming.accumulated_obs().unwrap(), 0);
    }
}
