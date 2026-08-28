//! Influence Map Builder
//!
//! Complete pipeline for building fault influence maps from circuits:
//!
//! 1. **Forward symbolic simulation** (`SymbolicSparseStab`) to determine measurement correlations
//! 2. **Detector definition** from deterministic measurements
//! 3. **Backward propagation** to build the influence map
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::InfluenceBuilder;
//! use pecos_qec::fault_tolerance::dem_builder::DemSampler;
//! use pecos_quantum::DagCircuit;
//!
//! // Build a syndrome extraction circuit
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);
//! dag.cx(&[(0, 2)]);
//! dag.cx(&[(1, 2)]);
//! dag.mz(&[2]);
//!
//! // Build influence map with automatic detector discovery
//! let builder = InfluenceBuilder::new(&dag);
//! let influence_map = builder.build().expect("circuit is replayable");
//!
//! // Build a fast DemSampler from the influence map
//! let num_locations = influence_map.locations.len();
//! let sampler = DemSampler::from_influence_map(&influence_map, &vec![0.001; num_locations]);
//! let stats = sampler.sample_statistics(100, 42);
//! ```

use super::propagator::dag::{DagFaultInfluenceMap, DagSpacetimeLocation, DemOutputMetadata};
use super::propagator::types::{DetectorId, MeasurementId};
use super::propagator::{DagFaultAnalyzer, DagPropagator, Direction, Pauli, apply_gate};
use super::symbolic_replay::{Dispatch, apply_unitary_clifford};
use pecos_core::QubitId;
use pecos_simulators::{PauliProp, SymbolicSparseStab};
use smallvec::SmallVec;
use std::collections::{BTreeSet, BinaryHeap};

struct ObservablePropagationWork<'a> {
    recorder: &'a mut CompoundRecorder,
    visited: &'a mut [bool],
    active_qubits: &'a mut [bool],
    heap: &'a mut BinaryHeap<(usize, usize)>,
}

/// Builder for fault influence maps with proper detector definitions.
///
/// This integrates forward symbolic simulation with backward propagation
/// to create complete influence maps suitable for noisy sampling.
/// Re-export `PauliString` as the type used for Pauli operator tracking.
///
/// All circuit annotations (detectors, observables, tracked Paulis) are Pauli
/// strings tracked for flipping via backward propagation. The difference
/// is role and readout:
///
/// | Kind | Meaning | Readout | API |
/// |------|---------|---------|-----|
/// | Detector | Syndrome parity from measurements | measurement XOR = 0 | `dag.detector(&[...])` |
/// | Observable | Standard `L<n>` output from measurements | measurement XOR | `dag.observable(&[...])` |
/// | Tracked Pauli | User Pauli string annotated at a circuit point | fault anticommutes with tracked Pauli | `dag.tracked_pauli(&[...])` |
///
/// Observables and tracked Paulis both use backward Pauli propagation, but
/// they are not the same concept. Observables are values observed through
/// measurements, are defined by measurement records, and are decoder-visible
/// `L<n>` outputs. Tracked Paulis are not measured and are not applied to the
/// computation; they ask whether a fault would flip the annotated Pauli placed as
/// an annotation in the circuit, such as a logical operator, stabilizer, or
/// other Pauli of interest. They live in a separate PECOS-only namespace.
pub use pecos_core::PauliString;

struct NonDetectorOutputTarget {
    metadata: DemOutputMetadata,
    terms: Vec<PauliPropagationTerm>,
}

struct PauliPropagationTerm {
    pauli: PauliString,
    start_node: Option<usize>,
}

/// Why circuit annotations could not be ingested into the influence map.
///
/// An annotation that loses references silently becomes an output with no
/// propagation terms -- not merely empty, but wrong: it suppresses the
/// record-based fallback downstream. So unresolvable references are errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnotationIngestError {
    /// An observable annotation references a measurement id that does not
    /// resolve against the circuit.
    ObservableRefUnresolved {
        /// Index of the annotation among the circuit's annotations.
        annotation_index: usize,
        /// The unresolved id.
        meas_id: pecos_core::MeasId,
        /// Why resolution failed.
        source: pecos_quantum::MeasResolveError,
    },
    /// A tracked-Pauli annotation has no matching `TrackedPauliMeta` gate.
    TrackedPauliWithoutMetaNode {
        /// Index of the annotation among the circuit's annotations.
        annotation_index: usize,
    },
}

impl std::fmt::Display for AnnotationIngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ObservableRefUnresolved {
                annotation_index,
                meas_id,
                source,
            } => write!(
                f,
                "annotation {annotation_index}: observable references MeasId({}), which \
                 does not resolve against the circuit: {source}",
                meas_id.index()
            ),
            Self::TrackedPauliWithoutMetaNode { annotation_index } => write!(
                f,
                "annotation {annotation_index}: tracked Pauli has no matching \
                 TrackedPauliMeta gate in the circuit"
            ),
        }
    }
}

impl std::error::Error for AnnotationIngestError {}

/// Why the influence map could not be built from the circuit.
///
/// The forward symbolic replay used to skip anything it did not recognize --
/// including mid-circuit resets and non-Clifford rotations -- and to measure
/// only the first qubit of a batched measurement node. All of those produced
/// silently wrong detector analyses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfluenceBuildError {
    /// A gate the symbolic replay cannot represent.
    UnsupportedGate {
        /// The DAG node holding the gate.
        node: usize,
        /// The offending gate type.
        gate_type: pecos_core::gate_type::GateType,
    },
    /// A single node carrying more than one measurement.
    ///
    /// The measurement-info model maps a node to one measurement index, so a
    /// batched node cannot be represented yet (issue #398 tracks per-measurement
    /// mapping). Refused rather than analyzed as if only the first qubit were
    /// measured.
    BatchedMeasurementUnsupported {
        /// The DAG node holding the batched measurement.
        node: usize,
        /// How many measurements it carries.
        count: usize,
    },
}

impl std::fmt::Display for InfluenceBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedGate { node, gate_type } => write!(
                f,
                "symbolic replay cannot represent {gate_type:?} at node {node}; lower it to \
                 supported Cliffords first"
            ),
            Self::BatchedMeasurementUnsupported { node, count } => write!(
                f,
                "node {node} carries {count} measurements in one gate; the influence map \
                 cannot yet represent a batched measurement node"
            ),
        }
    }
}

impl std::error::Error for InfluenceBuildError {}

pub struct InfluenceBuilder<'a> {
    dag: &'a pecos_quantum::DagCircuit,
    /// Non-detector parity outputs to track for flipping.
    ///
    /// This internal list contains both standard observables and PECOS tracked
    /// operators. The metadata kind is the authority for which public namespace
    /// each entry belongs to; callers should not infer that from the raw index.
    ///
    /// Each entry has one metadata item and one or more propagation terms.
    /// Multiple terms accumulate into the same output index, which is needed
    /// for measurement-record observables whose measurements occur at different
    /// circuit positions.
    non_detector_outputs: Vec<NonDetectorOutputTarget>,
}

impl<'a> InfluenceBuilder<'a> {
    /// Create a new influence builder for the given circuit.
    #[must_use]
    pub fn new(dag: &'a pecos_quantum::DagCircuit) -> Self {
        Self {
            dag,
            non_detector_outputs: Vec::new(),
        }
    }

    /// Add a tracked X Pauli (X on all specified qubits).
    #[must_use]
    pub fn with_x(mut self, qubits: &[usize]) -> Self {
        self.push_single_term_output(
            DemOutputMetadata::tracked_pauli(PauliString::xs(qubits)),
            None,
        );
        self
    }

    /// Add a tracked Z Pauli (Z on all specified qubits).
    #[must_use]
    pub fn with_z(mut self, qubits: &[usize]) -> Self {
        self.push_single_term_output(
            DemOutputMetadata::tracked_pauli(PauliString::zs(qubits)),
            None,
        );
        self
    }

    /// Add a tracked Y Pauli (Y on all specified qubits).
    #[must_use]
    pub fn with_y(mut self, qubits: &[usize]) -> Self {
        self.push_single_term_output(
            DemOutputMetadata::tracked_pauli(PauliString::ys(qubits)),
            None,
        );
        self
    }

    /// Add a Pauli check: track whether this Pauli string flips due to faults.
    ///
    /// Unlike observables (`dag.observable()`), a Pauli check
    /// uses backward propagation to detect flips WITHOUT requiring a measurement.
    ///
    /// # Example
    ///
    /// ```
    /// // Check if Y = X_0 * Z_1 * Z_2 flips
    /// use pecos_core::{Pauli, PauliString};
    /// use pecos_qec::fault_tolerance::InfluenceBuilder;
    /// use pecos_quantum::DagCircuit;
    ///
    /// let dag = DagCircuit::new();
    /// let builder = InfluenceBuilder::new(&dag).with_tracked_pauli(
    ///     PauliString::from_paulis(&[Pauli::X, Pauli::Z, Pauli::Z]),
    /// );
    /// let _map = builder.build().expect("circuit is replayable");
    /// ```
    #[must_use]
    pub fn with_tracked_pauli(mut self, pauli: PauliString) -> Self {
        self.push_single_term_output(DemOutputMetadata::tracked_pauli(pauli), None);
        self
    }

    fn push_single_term_output(&mut self, metadata: DemOutputMetadata, start_node: Option<usize>) {
        self.non_detector_outputs.push(NonDetectorOutputTarget {
            terms: vec![PauliPropagationTerm {
                pauli: metadata.pauli.clone(),
                start_node,
            }],
            metadata,
        });
    }

    /// Extract observable and tracked-Pauli annotations from the circuit.
    ///
    /// Observable annotations define logical observables via measurement records.
    /// For backward propagation, each referenced measurement contributes its
    /// own Z-type propagation term starting at that measurement node. The terms
    /// accumulate into the same observable `L<n>` output.
    ///
    /// Tracked-Pauli annotations have a corresponding `TrackedPauliMeta` node
    /// that marks their time position.
    ///
    /// Detector annotations are NOT handled here -- they are processed
    /// by `DemSamplerBuilder::with_circuit_annotations` which maps them
    /// to auto-detected detectors.
    ///
    /// The circuit is `self.dag` -- the one this builder was constructed with.
    /// It used to be a parameter, which let annotations come from a *different*
    /// circuit than the one being analyzed, silently producing nonsense.
    ///
    /// # Errors
    ///
    /// Returns [`AnnotationIngestError`] when an annotation cannot be resolved
    /// against the circuit: an observable referencing a missing node or a
    /// non-measurement gate, or a tracked Pauli with no meta gate. Dropping
    /// such references silently produced outputs with missing propagation
    /// terms.
    pub fn with_circuit_annotations(mut self) -> Result<Self, AnnotationIngestError> {
        let circuit = self.dag;
        // Find TrackedPauliMeta nodes in topological order.
        // The nth meta-gate corresponds to the nth tracked-Pauli annotation.
        let meta_nodes: Vec<usize> = circuit
            .topological_order()
            .into_iter()
            .filter(|&node| circuit.gate(node).is_some_and(|g| g.gate_type.is_meta()))
            .collect();

        let mut operator_idx = 0;
        let mut ingested = Vec::new();
        for (annotation_index, ann) in circuit.annotations().iter().enumerate() {
            match &ann.kind {
                pecos_quantum::AnnotationKind::Observable { measurement_ids } => {
                    let mut terms = Vec::new();
                    for &meas_id in measurement_ids {
                        // Each id names one measurement, so each term is Z on
                        // that measurement's own qubit -- not a Z-spray over
                        // every qubit of a batched node.
                        let mref = circuit.find_measurement(meas_id).map_err(|source| {
                            AnnotationIngestError::ObservableRefUnresolved {
                                annotation_index,
                                meas_id,
                                source,
                            }
                        })?;
                        terms.push(PauliPropagationTerm {
                            pauli: PauliString::zs(&[mref.qubit.index()]),
                            start_node: Some(mref.node),
                        });
                    }
                    ingested.push(NonDetectorOutputTarget {
                        metadata: DemOutputMetadata::observable(ann.pauli.clone())
                            .with_optional_label(ann.label.clone()),
                        terms,
                    });
                }
                pecos_quantum::AnnotationKind::TrackedPauli => {
                    let Some(meta_node) = meta_nodes.get(operator_idx).copied() else {
                        return Err(AnnotationIngestError::TrackedPauliWithoutMetaNode {
                            annotation_index,
                        });
                    };
                    operator_idx += 1;
                    let metadata = DemOutputMetadata::tracked_pauli(ann.pauli.clone())
                        .with_optional_label(ann.label.clone());
                    ingested.push(NonDetectorOutputTarget {
                        terms: vec![PauliPropagationTerm {
                            pauli: metadata.pauli.clone(),
                            start_node: Some(meta_node),
                        }],
                        metadata,
                    });
                }
                pecos_quantum::AnnotationKind::Detector { .. } => {
                    // Detectors handled separately by DemSamplerBuilder
                }
            }
        }
        // Staged so interleaved observables and tracked Paulis keep annotation
        // order in one place. Partial state cannot escape an `Err` regardless:
        // the method consumes `self`, so a failed ingest drops the builder --
        // which also means transactionality here is structural and untestable,
        // not a behavior a test could pin.
        self.non_detector_outputs.extend(ingested);
        Ok(self)
    }

    /// Build the influence map.
    ///
    /// This performs:
    /// 1. Forward symbolic simulation to get measurement correlations
    /// 2. Detector extraction from deterministic measurements
    /// 3. Backward propagation from detectors and DEM outputs
    /// # Errors
    ///
    /// Returns [`InfluenceBuildError`] when the circuit contains a gate the
    /// symbolic replay cannot represent, or a batched measurement node. Both
    /// used to be analyzed silently wrong -- an ignored rotation manufactured
    /// phantom detectors, and a batched node was treated as measuring only its
    /// first qubit.
    pub fn build(&self) -> Result<DagFaultInfluenceMap, InfluenceBuildError> {
        // Step 1: Run forward symbolic simulation
        let measurement_info = self.run_symbolic_simulation()?;

        // Step 2: Extract detectors from deterministic measurements
        let detectors = Self::extract_detectors(&measurement_info);

        // Step 3: Build influence map with backward propagation
        Ok(self.build_influence_map_with_detectors(&measurement_info, &detectors))
    }

    /// Run symbolic simulation to get measurement correlations.
    ///
    /// # Errors
    ///
    /// See [`build`](Self::build).
    pub(crate) fn run_symbolic_simulation(&self) -> Result<MeasurementInfo, InfluenceBuildError> {
        let topo_order = self.dag.topological_order();

        // Determine number of qubits from the circuit
        let max_qubit = topo_order
            .iter()
            .filter_map(|&node| self.dag.gate(node))
            .flat_map(|op| op.qubits.iter())
            .map(pecos_core::QubitId::index)
            .max()
            .unwrap_or(0);

        let num_qubits = max_qubit + 1;
        let mut sim = SymbolicSparseStab::new(num_qubits);

        // Track node -> measurement index mapping
        let node_count = topo_order.iter().copied().max().map_or(0, |node| node + 1);
        let mut node_to_meas_idx: Vec<Option<usize>> = vec![None; node_count];
        let mut meas_idx = 0;

        // Execute circuit symbolically
        for &node in &topo_order {
            if let Some(op) = self.dag.gate(node) {
                let qubits: Vec<usize> = op.qubits.iter().map(pecos_core::QubitId::index).collect();

                // A malformed gate here means the DAG itself is wrong, and
                // this path has no error channel, so report it loudly.
                let dispatch = apply_unitary_clifford(&mut sim, op.gate_type, &qubits)
                    .unwrap_or_else(|err| {
                        panic!(
                            "symbolic simulation got malformed gate {:?} at node {node}: {err:?}",
                            op.gate_type
                        )
                    });
                if dispatch == Dispatch::Applied {
                    continue;
                }

                match op.gate_type {
                    pecos_quantum::GateType::MX
                    | pecos_quantum::GateType::MZ
                    | pecos_quantum::GateType::MeasureFree
                    | pecos_quantum::GateType::MPZ => {
                        if qubits.len() > 1 {
                            return Err(InfluenceBuildError::BatchedMeasurementUnsupported {
                                node,
                                count: qubits.len(),
                            });
                        }
                        if op.gate_type == pecos_quantum::GateType::MX {
                            sim.h(&[qubits[0]]);
                        }
                        sim.mz(&[qubits[0]]);
                        if op.gate_type == pecos_quantum::GateType::MPZ {
                            // Measure-and-prepare resets after the readout.
                            sim.pz(qubits[0]);
                        }
                        node_to_meas_idx[node] = Some(meas_idx);
                        meas_idx += 1;
                    }
                    // A leaked measurement collapses its qubit but consumes no
                    // record -- and this replay cannot express that: `sim.mz`
                    // advances the simulator's internal history index, so a
                    // record-less collapse desyncs symbolic indices from
                    // records and manufactures phantom detectors. Refused,
                    // exactly as `symbolic_measurement_history` refuses it.
                    pecos_quantum::GateType::MeasureLeaked => {
                        return Err(InfluenceBuildError::UnsupportedGate {
                            node,
                            gate_type: op.gate_type,
                        });
                    }
                    // Resets project onto |0>. Skipping them treated a reused
                    // qubit as still carrying its pre-reset correlations.
                    pecos_quantum::GateType::PX
                    | pecos_quantum::GateType::PZ
                    | pecos_quantum::GateType::QAlloc => {
                        for &q in &qubits {
                            sim.pz(q);
                            if op.gate_type == pecos_quantum::GateType::PX {
                                sim.h(&[q]);
                            }
                        }
                    }
                    // No effect on stabilizer correlations.
                    pecos_quantum::GateType::I
                    | pecos_quantum::GateType::Idle
                    | pecos_quantum::GateType::QFree
                    | pecos_quantum::GateType::MeasCrosstalkGlobalPayload
                    | pecos_quantum::GateType::MeasCrosstalkLocalPayload
                    | pecos_quantum::GateType::TrackedPauliMeta => {}
                    gate_type
                        if matches!(
                            gate_type,
                            pecos_quantum::GateType::RX
                                | pecos_quantum::GateType::RY
                                | pecos_quantum::GateType::RZ
                                | pecos_quantum::GateType::RXX
                                | pecos_quantum::GateType::RYY
                                | pecos_quantum::GateType::RZZ
                                | pecos_quantum::GateType::CRZ
                        ) && op.angles.len() == 1
                            && op.angles[0].is_zero() => {}
                    // Anything else would be silently mis-analyzed: an ignored
                    // rotation manufactures detectors its backward propagation
                    // then contradicts.
                    other => {
                        return Err(InfluenceBuildError::UnsupportedGate {
                            node,
                            gate_type: other,
                        });
                    }
                }
            }
        }

        let history = sim.measurement_history().clone();

        Ok(MeasurementInfo {
            history,
            node_to_meas_idx,
            num_measurements: meas_idx,
        })
    }

    /// Extract detectors from deterministic measurements.
    ///
    /// A deterministic measurement `m_i` with outcome `{m_a, m_b, ...}` means
    /// that in the noiseless case: `m_i = m_a XOR m_b XOR ...`
    ///
    /// This defines a detector: `D = m_i XOR m_a XOR m_b XOR ... = 0` always.
    fn extract_detectors(info: &MeasurementInfo) -> Vec<DetectorDef> {
        let mut detectors = Vec::new();

        for result in info.history.iter() {
            if result.is_deterministic {
                // This measurement is deterministic - it defines a detector
                let mut measurement_indices: SmallVec<[usize; 4]> = SmallVec::new();

                // Add the measurement itself
                measurement_indices.push(result.index);

                // Add all its dependencies (the XOR terms)
                for dep_idx in &result.outcome {
                    measurement_indices.push(dep_idx);
                }

                // Account for flip (if flip=true, the detector should be 1, not 0)
                // For now, we treat flipped detectors the same way
                detectors.push(DetectorDef {
                    measurement_indices,
                    expected_value: result.flip,
                });
            }
        }

        detectors
    }

    /// Build influence map with proper detector definitions.
    fn build_influence_map_with_detectors(
        &self,
        info: &MeasurementInfo,
        detectors: &[DetectorDef],
    ) -> DagFaultInfluenceMap {
        let analyzer = DagFaultAnalyzer::new(self.dag);
        let propagator = analyzer.propagator();

        let num_locations = analyzer.propagator().topo_order().len() * 2; // rough estimate
        let mut map = DagFaultInfluenceMap::with_capacity(num_locations);

        // Copy locations from analyzer
        map.locations = Self::extract_locations(propagator);

        // One order per map: measurements, meas_ids, detectors, and the
        // backward-propagation recorder all number measurements by the replay
        // ordinal, so `meas_index_of` is the ordinal the influence data (and
        // the sampler's raw channels) actually use. Re-sorting measurements by
        // id rank here while the detectors kept replay order gave one map two
        // internal orderings, which only agreed while ids were positional.
        let mut measurements = vec![(0usize, 0usize, 0u8); info.num_measurements];
        let mut meas_ids = vec![pecos_core::MeasId::from_raw(usize::MAX); info.num_measurements];
        let mut any_id = false;
        for (node, opt) in info.node_to_meas_idx.iter().enumerate() {
            if let Some(meas_idx) = *opt {
                let gate = self
                    .dag
                    .gate(node)
                    .expect("a replayed measurement node holds a gate");
                // Batched measurement nodes are refused by the replay, so the
                // node holds exactly one measurement.
                let qubit = gate.qubits.first().map_or(0, pecos_core::QubitId::index);
                measurements[meas_idx] = (node, qubit, 0);
                if let Some(&id) = gate.meas_ids.first() {
                    meas_ids[meas_idx] = id;
                    any_id = true;
                }
            }
        }
        map.measurements.clone_from(&measurements);
        map.meas_ids = if any_id { meas_ids } else { Vec::new() };

        // Create DetectorId entries for each detector
        for detector in detectors {
            let meas_ids: SmallVec<[MeasurementId; 2]> = detector
                .measurement_indices
                .iter()
                .filter_map(|&meas_idx| {
                    measurements
                        .get(meas_idx)
                        .map(|&(node, qubit, basis)| MeasurementId {
                            tick: node,
                            qubit,
                            basis,
                        })
                })
                .collect();

            map.detectors.push(DetectorId {
                measurements: meas_ids,
                name: None,
            });
        }

        // Build the influence structure using backward propagation
        let mut recorder = CompoundRecorder::new(map.locations.len());

        // Propagate from each detector
        Self::propagate_detectors(propagator, info, detectors, &mut recorder);

        // Propagate from non-detector DEM outputs.
        self.propagate_non_detector_outputs(propagator, &mut recorder);

        // Convert to SoA format
        map.influences = recorder.into_soa();

        // Store DEM-output labels
        map.dem_output_labels = self
            .non_detector_outputs
            .iter()
            .map(|output| output.metadata.label.clone())
            .collect();
        map.dem_output_metadata = self
            .non_detector_outputs
            .iter()
            .map(|output| output.metadata.clone())
            .collect();

        map
    }

    /// Extract fault locations from the propagator.
    fn extract_locations(propagator: &DagPropagator<'_>) -> Vec<DagSpacetimeLocation> {
        let mut locations = Vec::new();
        let mut prepared_qubits: BTreeSet<QubitId> = BTreeSet::new();

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                // Meta-gates are not physical -- they don't generate faults
                if gate.gate_type.is_meta() {
                    continue;
                }

                let qubits: Vec<QubitId> = gate.qubits.to_vec();

                let is_measurement = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::MX
                        | pecos_quantum::GateType::MZ
                        | pecos_quantum::GateType::MeasureFree
                        | pecos_quantum::GateType::MPZ
                );

                // Standard circuit noise model: one fault location per gate.
                //   Measurement: before. All others: after.
                let before = is_measurement;
                let location_qubits: Vec<QubitId> =
                    if gate.gate_type == pecos_quantum::GateType::MeasCrosstalkGlobalPayload {
                        qubits.iter().copied().for_each(|q| {
                            prepared_qubits.remove(&q);
                        });
                        let victims = prepared_qubits.iter().copied().collect();
                        qubits.iter().copied().for_each(|q| {
                            prepared_qubits.insert(q);
                        });
                        victims
                    } else {
                        qubits.clone()
                    };
                for q in location_qubits {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: vec![q],
                        before,
                        gate_type: gate.gate_type,
                        idle_duration: gate.idle_duration(),
                    });
                }
                if matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::PX
                        | pecos_quantum::GateType::PZ
                        | pecos_quantum::GateType::QAlloc
                ) {
                    prepared_qubits.extend(qubits.iter().copied());
                }
            }
        }

        locations
    }

    /// Propagate backward from all detectors.
    fn propagate_detectors(
        propagator: &DagPropagator<'_>,
        info: &MeasurementInfo,
        detectors: &[DetectorDef],
        recorder: &mut CompoundRecorder,
    ) {
        let max_node = propagator.max_node();
        let max_qubit = propagator.max_qubit();

        let mut visited = vec![false; max_node + 1];
        let mut active_qubits = vec![false; max_qubit + 1];
        let mut heap = BinaryHeap::new();
        let mut work = ObservablePropagationWork {
            recorder,
            visited: &mut visited,
            active_qubits: &mut active_qubits,
            heap: &mut heap,
        };

        for (det_idx, detector) in detectors.iter().enumerate() {
            // One walk per member measurement, each seeded at its own node.
            // A combined seed XORed at the circuit end cancels same-qubit
            // members to identity before the walk starts, erasing every
            // influence between the measurements. The recorder toggles, so
            // influences shared by several members cancel exactly as the
            // detector's XOR readout does.
            for &meas_idx in &detector.measurement_indices {
                if let Some(node) = info
                    .node_to_meas_idx
                    .iter()
                    .position(|&opt| opt == Some(meas_idx))
                    && let Some(gate) = propagator.gate(node)
                {
                    let mut seed = PauliProp::new();
                    for qubit in &gate.qubits {
                        if gate.gate_type == pecos_quantum::GateType::MX {
                            seed.track_x(&[qubit.index()]);
                        } else {
                            seed.track_z(&[qubit.index()]);
                        }
                    }
                    Self::propagate_observable(
                        propagator,
                        &seed,
                        det_idx,
                        true, // is_detector
                        &mut work,
                        Some(propagator.topo_position(node)),
                    );
                }
            }
        }
    }

    /// Propagate backward from non-detector DEM outputs.
    ///
    /// If a propagation term has a corresponding DAG node, propagation starts
    /// from that node's topological position. Otherwise (e.g. operators added
    /// via `with_z`/`with_x` without a circuit annotation), propagation walks
    /// from the circuit end.
    fn propagate_non_detector_outputs(
        &self,
        propagator: &DagPropagator<'_>,
        recorder: &mut CompoundRecorder,
    ) {
        let max_node = propagator.max_node();
        let max_qubit = propagator.max_qubit();

        let mut visited = vec![false; max_node + 1];
        let mut active_qubits = vec![false; max_qubit + 1];
        let mut heap = BinaryHeap::new();
        let mut work = ObservablePropagationWork {
            recorder,
            visited: &mut visited,
            active_qubits: &mut active_qubits,
            heap: &mut heap,
        };

        for (dem_output_idx, output) in self.non_detector_outputs.iter().enumerate() {
            for term in &output.terms {
                let mut prop = PauliProp::new();

                for &(pauli, qubit) in term.pauli.paulis() {
                    use pecos_core::Pauli;
                    let q = qubit.index();
                    match pauli {
                        Pauli::X => prop.track_x(&[q]),
                        Pauli::Y => prop.track_y(&[q]),
                        Pauli::Z => prop.track_z(&[q]),
                        Pauli::I => {}
                    }
                }

                // Resolve the term's node to its topological position.
                // None means no positional bound (walk from circuit end).
                let start_pos = term.start_node.map(|node| propagator.topo_position(node));

                Self::propagate_observable(
                    propagator,
                    &prop,
                    dem_output_idx,
                    false, // is_detector = false (this is a DEM output)
                    &mut work,
                    start_pos,
                );
            }
        }
    }

    /// Propagate a single observable backward and record influences.
    ///
    /// When `start_topo_pos` is `Some(pos)`, only gates at or before that
    /// topological position are considered. This makes Pauli operator
    /// annotations positional: only faults before the meta-gate affect it.
    fn propagate_observable(
        propagator: &DagPropagator<'_>,
        initial_prop: &PauliProp,
        target_idx: usize,
        is_detector: bool,
        work: &mut ObservablePropagationWork<'_>,
        start_topo_pos: Option<usize>,
    ) {
        // Clear work arrays
        work.visited.fill(false);
        work.active_qubits.fill(false);
        work.heap.clear();

        let mut prop = initial_prop.clone();

        // Initialize active qubits from the observable
        for (q, is_active) in work.active_qubits.iter_mut().enumerate() {
            if prop.contains_x(q) || prop.contains_z(q) {
                *is_active = true;

                // Add gates on this qubit to the heap, bounded by start position
                for (topo_pos, node) in propagator.qubit_gates_backward(q) {
                    if start_topo_pos.is_some_and(|max| topo_pos > max) {
                        continue;
                    }
                    if !work.visited[node] {
                        work.visited[node] = true;
                        work.heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Build location index for recording
        let loc_map = Self::build_location_map(propagator);

        // Process gates in reverse topological order
        while let Some((topo_pos, node)) = work.heap.pop() {
            if let Some(gate) = propagator.gate(node) {
                // The seed's own measurement node: the observable represents
                // the readout, which happens before the gate's preparing half,
                // so the crossing must not apply (an MPZ's reset would kill
                // its own seed before the before-location records). Faults
                // after the whole gate cannot flip its record, so only the
                // before-location is recorded.
                if start_topo_pos == Some(topo_pos) && gate.gate_type.consumes_measurement_record()
                {
                    if let Some(qubit_locs) = loc_map.get(&(node, true)) {
                        Self::record_influence(
                            &prop,
                            qubit_locs,
                            target_idx,
                            is_detector,
                            &mut *work.recorder,
                        );
                    }
                    continue;
                }

                // Record per-qubit influences at before=false location
                if let Some(qubit_locs) = loc_map.get(&(node, false)) {
                    Self::record_influence(
                        &prop,
                        qubit_locs,
                        target_idx,
                        is_detector,
                        &mut *work.recorder,
                    );
                }

                // Track which qubits were active before the gate
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() < work.active_qubits.len() {
                        was_active[j] = work.active_qubits[q.index()];
                    }
                }

                // Prep gates (PZ/QAlloc) reset the qubit -- kill the Pauli
                // and mark the qubit inactive. Faults before the prep
                // cannot propagate past it.
                let is_prep = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::PX
                        | pecos_quantum::GateType::PZ
                        | pecos_quantum::GateType::QAlloc
                );
                if is_prep {
                    for q in &gate.qubits {
                        let qi = q.index();
                        // Toggle off X and Z components (XOR to zero)
                        if prop.contains_x(qi) {
                            prop.track_x(&[qi]);
                        }
                        if prop.contains_z(qi) {
                            prop.track_z(&[qi]);
                        }
                        if qi < work.active_qubits.len() {
                            work.active_qubits[qi] = false;
                        }
                    }
                    continue; // don't propagate further on these qubits
                }

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Record per-qubit influences at before=true location
                if let Some(qubit_locs) = loc_map.get(&(node, true)) {
                    Self::record_influence(
                        &prop,
                        qubit_locs,
                        target_idx,
                        is_detector,
                        &mut *work.recorder,
                    );
                }

                // Check if Pauli spread to new qubits
                let node_topo_pos = propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx < work.active_qubits.len() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates
                            work.active_qubits[idx] = true;
                            for (topo_pos, pred_node) in propagator.qubit_gates_backward(idx) {
                                if topo_pos < node_topo_pos && !work.visited[pred_node] {
                                    work.visited[pred_node] = true;
                                    work.heap.push((topo_pos, pred_node));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Build a map from (node, before) to per-qubit location indices.
    fn build_location_map(
        propagator: &DagPropagator<'_>,
    ) -> std::collections::HashMap<(usize, bool), Vec<(usize, usize)>> {
        // (node, before) -> [(qubit_index, loc_idx), ...]
        let mut map: std::collections::HashMap<(usize, bool), Vec<(usize, usize)>> =
            std::collections::HashMap::new();
        let mut loc_idx = 0;
        let mut prepared_qubits: BTreeSet<QubitId> = BTreeSet::new();

        for &node in propagator.topo_order() {
            if let Some(gate) = propagator.gate(node) {
                if gate.gate_type.is_meta() {
                    continue;
                }

                let is_measurement = matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::MX
                        | pecos_quantum::GateType::MZ
                        | pecos_quantum::GateType::MeasureFree
                        | pecos_quantum::GateType::MPZ
                );

                let before = is_measurement;
                let location_qubits: Vec<QubitId> =
                    if gate.gate_type == pecos_quantum::GateType::MeasCrosstalkGlobalPayload {
                        gate.qubits.iter().copied().for_each(|q| {
                            prepared_qubits.remove(&q);
                        });
                        let victims = prepared_qubits.iter().copied().collect();
                        gate.qubits.iter().copied().for_each(|q| {
                            prepared_qubits.insert(q);
                        });
                        victims
                    } else {
                        gate.qubits.to_vec()
                    };
                for q in &location_qubits {
                    let qi = q.index();
                    map.entry((node, before)).or_default().push((qi, loc_idx));
                    loc_idx += 1;
                }
                if matches!(
                    gate.gate_type,
                    pecos_quantum::GateType::PX
                        | pecos_quantum::GateType::PZ
                        | pecos_quantum::GateType::QAlloc
                ) {
                    prepared_qubits.extend(gate.qubits.iter().copied());
                }
            }
        }

        map
    }

    /// Record per-qubit influence of a fault at a gate location.
    fn record_influence(
        prop: &PauliProp,
        qubit_locs: &[(usize, usize)], // [(qubit, loc_idx), ...]
        target_idx: usize,
        is_detector: bool,
        recorder: &mut CompoundRecorder,
    ) {
        for &(qubit, loc_idx) in qubit_locs {
            for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                if Self::fault_anticommutes_qubit(prop, qubit, pauli) {
                    if is_detector {
                        #[allow(clippy::cast_possible_truncation)]
                        recorder.record_detector(loc_idx, pauli, target_idx as u32);
                    } else {
                        #[allow(clippy::cast_possible_truncation)]
                        recorder.record_dem_output(loc_idx, pauli, target_idx as u32);
                    }
                }
            }
        }
    }

    /// Check if a single-qubit fault Pauli anticommutes with the propagated
    /// observable on a specific qubit.
    fn fault_anticommutes_qubit(prop: &PauliProp, qubit: usize, fault: Pauli) -> bool {
        let has_x = prop.contains_x(qubit);
        let has_z = prop.contains_z(qubit);

        match fault {
            Pauli::I => false,
            Pauli::X => has_z,         // X anticommutes with Z
            Pauli::Z => has_x,         // Z anticommutes with X
            Pauli::Y => has_x ^ has_z, // Y anticommutes with X or Z but not both
        }
    }
}

/// Information about measurements from symbolic simulation.
pub(crate) struct MeasurementInfo {
    pub(crate) history: pecos_simulators::symbolic_sparse_stab::MeasurementHistory,
    pub(crate) node_to_meas_idx: Vec<Option<usize>>,
    #[allow(dead_code)]
    pub(crate) num_measurements: usize,
}

/// Definition of a detector as XOR of measurements.
struct DetectorDef {
    /// Measurement indices that XOR together
    measurement_indices: SmallVec<[usize; 4]>,
    /// Expected value (false=0, true=1) in noiseless case
    #[allow(dead_code)]
    expected_value: bool,
}

/// Recorder for compound detector propagation.
struct CompoundRecorder {
    num_locations: usize,

    // Buckets for detector influences [loc_idx][pauli] -> Vec<detector_idx>
    detector_x: Vec<Vec<u32>>,
    detector_y: Vec<Vec<u32>>,
    detector_z: Vec<Vec<u32>>,

    // Buckets for DEM-output influences.
    dem_output_x: Vec<Vec<u32>>,
    dem_output_y: Vec<Vec<u32>>,
    dem_output_z: Vec<Vec<u32>>,
}

impl CompoundRecorder {
    fn new(num_locations: usize) -> Self {
        Self {
            num_locations,
            detector_x: vec![Vec::new(); num_locations],
            detector_y: vec![Vec::new(); num_locations],
            detector_z: vec![Vec::new(); num_locations],
            dem_output_x: vec![Vec::new(); num_locations],
            dem_output_y: vec![Vec::new(); num_locations],
            dem_output_z: vec![Vec::new(); num_locations],
        }
    }

    fn record_detector(&mut self, loc_idx: usize, pauli: Pauli, detector_idx: u32) {
        if loc_idx >= self.num_locations {
            return;
        }
        match pauli {
            Pauli::X => toggle_bucket(&mut self.detector_x[loc_idx], detector_idx),
            Pauli::Y => toggle_bucket(&mut self.detector_y[loc_idx], detector_idx),
            Pauli::Z => toggle_bucket(&mut self.detector_z[loc_idx], detector_idx),
            Pauli::I => {}
        }
    }

    fn record_dem_output(&mut self, loc_idx: usize, pauli: Pauli, dem_output_idx: u32) {
        if loc_idx >= self.num_locations {
            return;
        }
        match pauli {
            Pauli::X => toggle_bucket(&mut self.dem_output_x[loc_idx], dem_output_idx),
            Pauli::Y => toggle_bucket(&mut self.dem_output_y[loc_idx], dem_output_idx),
            Pauli::Z => toggle_bucket(&mut self.dem_output_z[loc_idx], dem_output_idx),
            Pauli::I => {}
        }
    }

    fn into_soa(mut self) -> super::propagator::dag::InfluencesSoA {
        use super::propagator::dag::InfluencesSoA;

        let mut soa = InfluencesSoA::with_capacity(self.num_locations);

        for loc_idx in 0..self.num_locations {
            self.detector_x[loc_idx].sort_unstable();
            self.detector_y[loc_idx].sort_unstable();
            self.detector_z[loc_idx].sort_unstable();
            self.dem_output_x[loc_idx].sort_unstable();
            self.dem_output_y[loc_idx].sort_unstable();
            self.dem_output_z[loc_idx].sort_unstable();

            // Add detector influences
            for &det in &self.detector_x[loc_idx] {
                soa.detectors_x.push(det);
            }
            for &det in &self.detector_y[loc_idx] {
                soa.detectors_y.push(det);
            }
            for &det in &self.detector_z[loc_idx] {
                soa.detectors_z.push(det);
            }

            // Add DEM-output influences
            for &dem_output in &self.dem_output_x[loc_idx] {
                soa.dem_outputs_x.push(dem_output);
            }
            for &dem_output in &self.dem_output_y[loc_idx] {
                soa.dem_outputs_y.push(dem_output);
            }
            for &dem_output in &self.dem_output_z[loc_idx] {
                soa.dem_outputs_z.push(dem_output);
            }

            soa.finish_location();
        }

        soa
    }
}

fn toggle_bucket(bucket: &mut Vec<u32>, value: u32) {
    if let Some(pos) = bucket.iter().position(|&existing| existing == value) {
        bucket.remove(pos);
    } else {
        bucket.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::propagator::DemOutputKind;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_simple_circuit() {
        // Simple circuit: prep, H, measure
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build().expect("circuit is replayable");

        // Should have some locations and at least one detector
        assert!(!map.locations.is_empty());
    }

    #[test]
    fn test_syndrome_extraction() {
        // Simple syndrome extraction: 2 data qubits, 1 ancilla
        let mut dag = DagCircuit::new();

        // Prepare ancilla
        dag.pz(&[2]);

        // CNOT from data to ancilla
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);

        // Measure ancilla
        dag.mz(&[2]);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build().expect("circuit is replayable");

        assert!(!map.locations.is_empty());
        assert!(!map.measurements.is_empty());
    }

    #[test]
    fn test_repeated_syndrome() {
        // Two rounds of syndrome extraction
        let mut dag = DagCircuit::new();

        // Round 1
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        // Round 2
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let builder = InfluenceBuilder::new(&dag);
        let map = builder.build().expect("circuit is replayable");

        // Should have multiple measurements
        assert!(map.measurements.len() >= 2);

        // The second measurement should be deterministic (depends on first)
        // and thus create a proper detector
        assert!(!map.detectors.is_empty());
    }

    #[test]
    fn test_with_tracked_pauli() {
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.mz(&[2]);

        let builder = InfluenceBuilder::new(&dag).with_z(&[0]); // Track Z logical on qubit 0

        let map = builder.build().expect("circuit is replayable");

        // Should track the logical
        assert!(map.influences.max_dem_output_index().is_some());
    }

    #[test]
    fn test_dem_output_metadata_accepts_pauli_string_and_normalizes_phase() {
        use pecos_core::{Pauli, QuarterPhase};

        let pauli =
            PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::X, Pauli::Z]);
        let metadata = DemOutputMetadata::tracked_pauli(pauli).with_label("xz");

        assert_eq!(metadata.kind, DemOutputKind::TrackedPauli);
        assert_eq!(metadata.label.as_deref(), Some("xz"));
        assert_eq!(metadata.pauli.phase(), QuarterPhase::PlusOne);
        assert_eq!(metadata.pauli.to_sparse_str(), "+X0 Z1");
    }

    #[test]
    fn test_circuit_annotation_dem_output_metadata_tracks_observables_and_tracked_paulis() {
        use pecos_core::pauli::X;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        let meas = dag.mz(&[0]);
        dag.observable_labeled("record_obs", &[meas[0]])
            .expect("refs are from this circuit");
        dag.tracked_pauli_labeled("track_x", X(0));

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

        // 1 observable (record_obs) + 1 tracked Pauli (track_x) = 2 DEM outputs
        assert_eq!(map.num_dem_outputs(), 1, "1 observable");
        assert_eq!(map.num_tracked_paulis(), 1, "1 tracked Pauli");
        assert_eq!(map.dem_output_metadata.len(), 2);

        // Observable comes first (annotations are processed in order)
        assert_eq!(map.dem_output_metadata[0].kind, DemOutputKind::Observable);
        assert_eq!(
            map.dem_output_metadata[0].label.as_deref(),
            Some("record_obs")
        );

        // Tracked Pauli second
        assert_eq!(map.dem_output_metadata[1].kind, DemOutputKind::TrackedPauli);
        assert_eq!(map.dem_output_metadata[1].label.as_deref(), Some("track_x"));
        assert_eq!(map.dem_output_metadata[1].pauli.to_sparse_str(), "+X0");
    }

    #[test]
    fn test_observable_measurements_propagate_from_their_own_nodes() {
        use pecos_quantum::GateType;

        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        let early = dag.mz(&[0]);
        dag.h(&[0]);
        let late = dag.mz(&[1]);
        dag.observable_labeled("split_time_obs", &[early[0], late[0]])
            .expect("refs are from this circuit");

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

        assert_eq!(map.num_observables(), 1);

        let post_early_h = map
            .locations
            .iter()
            .enumerate()
            .find(|(_, loc)| {
                loc.gate_type == GateType::H
                    && loc.qubits.first().is_some_and(|q| q.index() == 0)
                    && !loc.before
            })
            .map(|(idx, _)| idx)
            .expect("H fault location after the early measurement");

        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            assert!(
                map.get_observable_indices(post_early_h, pauli.as_u8())
                    .is_empty(),
                "faults after an already-recorded measurement must not flip that record"
            );
        }
    }

    #[test]
    fn test_split_time_observable_fault_before_early_measurement_flips() {
        // Circuit: PZ(0), PZ(1), MZ(0) [early], H(1), MZ(1) [late]
        // Observable = MZ(0) XOR MZ(1)
        //
        // A prep fault on qubit 0 (after PZ(0), before MZ(0)) should flip the
        // observable via the early measurement term.
        use pecos_quantum::GateType;

        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        let early = dag.mz(&[0]);
        dag.h(&[1]);
        let late = dag.mz(&[1]);
        dag.observable_labeled("split_obs", &[early[0], late[0]])
            .expect("refs are from this circuit");

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

        assert_eq!(map.num_observables(), 1);

        // Prep fault on qubit 0 (PZ after-gate location) should flip the
        // observable because it propagates through the early MZ(0).
        let prep_q0 = map
            .locations
            .iter()
            .enumerate()
            .find(|(_, loc)| {
                loc.gate_type == GateType::PZ
                    && loc.qubits.first().is_some_and(|q| q.index() == 0)
                    && !loc.before
            })
            .map(|(idx, _)| idx)
            .expect("PZ(0) fault location");

        // X fault after PZ propagates through MZ as a bit flip
        assert!(
            !map.get_observable_indices(prep_q0, Pauli::X.as_u8())
                .is_empty(),
            "X fault before early measurement should flip observable"
        );
    }

    #[test]
    fn test_split_time_observable_fault_between_measurements_flips_late_only() {
        // Circuit: PZ(0), PZ(1), MZ(0) [early], H(1), MZ(1) [late]
        // Observable = MZ(0) XOR MZ(1)
        //
        // An H fault on qubit 1 (between the two measurements) should flip the
        // observable via the late measurement term only.
        use pecos_quantum::GateType;

        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        let early = dag.mz(&[0]);
        dag.h(&[1]);
        let late = dag.mz(&[1]);
        dag.observable_labeled("split_obs", &[early[0], late[0]])
            .expect("refs are from this circuit");

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

        assert_eq!(map.num_observables(), 1);

        // H(1) fault location — between the two measurements, on qubit 1
        let h_q1 = map
            .locations
            .iter()
            .enumerate()
            .find(|(_, loc)| {
                loc.gate_type == GateType::H
                    && loc.qubits.first().is_some_and(|q| q.index() == 1)
                    && !loc.before
            })
            .map(|(idx, _)| idx)
            .expect("H(1) fault location");

        // X fault after H(1) becomes Z before MZ(1), which does NOT flip MZ.
        // Z fault after H(1) becomes X before MZ(1), which DOES flip MZ.
        assert!(
            !map.get_observable_indices(h_q1, Pauli::X.as_u8())
                .is_empty()
                || !map
                    .get_observable_indices(h_q1, Pauli::Z.as_u8())
                    .is_empty(),
            "at least one Pauli fault between measurements should flip the late term"
        );
    }

    /// An ignored rotation used to manufacture phantom detectors: forward
    /// replay skipped `RZ` while backward propagation simplified it, so the two
    /// halves of the analysis disagreed. Reviewer witness on #408.
    #[test]
    fn an_unrepresentable_gate_is_refused_not_skipped() {
        use pecos_core::Angle64;
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.rz(Angle64::QUARTER_TURN, &[0]);
        dag.mz(&[0]);

        let err = InfluenceBuilder::new(&dag)
            .build()
            .map(|_| ())
            .expect_err("RZ is not representable in the symbolic replay");
        assert!(matches!(err, InfluenceBuildError::UnsupportedGate { .. }));
    }

    #[test]
    fn a_zero_angle_rotation_is_replayed_as_an_identity() {
        use pecos_core::Angle64;
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.rz(Angle64::ZERO, &[0]);
        dag.mz(&[0]);

        InfluenceBuilder::new(&dag)
            .build()
            .expect("an exact zero-angle rotation is representable as an identity");
    }

    /// A batched measurement node used to be analyzed as measuring only its
    /// first qubit, silently undercounting detectors. Refused until the
    /// per-measurement model (#398) can represent it.
    #[test]
    fn a_batched_measurement_node_is_refused_not_undercounted() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        dag.add_gate_auto_wire(pecos_quantum::Gate::mz(&[0usize, 1]));

        let err = InfluenceBuilder::new(&dag)
            .build()
            .map(|_| ())
            .expect_err("a batched measurement node cannot be represented");
        assert!(matches!(
            err,
            InfluenceBuildError::BatchedMeasurementUnsupported { count: 2, .. }
        ));
    }

    /// A leaked measurement cannot be represented in this replay: collapsing
    /// it via `sim.mz` advances the simulator's internal history index without
    /// a matching record, desyncing symbolic indices from records. The witness
    /// circuit manufactured a phantom detector on the visible measurement.
    #[test]
    fn a_leaked_measurement_is_refused_not_desynced() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.add_gate_auto_wire(pecos_quantum::Gate::measure_leaked(&[0usize]));
        dag.mz(&[0]);

        let err = InfluenceBuilder::new(&dag)
            .build()
            .map(|_| ())
            .expect_err("a leaked measurement has no record this replay can express");
        assert!(matches!(err, InfluenceBuildError::UnsupportedGate { .. }));
    }

    /// A mid-circuit reset must reach the simulator: skipping it treated a
    /// reused qubit as still carrying its pre-reset correlations, so the
    /// re-measurement looked like a deterministic repeat instead of fresh.
    #[test]
    fn a_mid_circuit_reset_reaches_the_simulator() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]); // random
        dag.pz(&[0]); // reset -- was silently skipped
        dag.mz(&[0]); // deterministic |0>, NOT a repeat of the first

        let map = InfluenceBuilder::new(&dag)
            .build()
            .expect("all gates representable");
        assert_eq!(
            map.detectors.len(),
            1,
            "the post-reset measurement is deterministic on its own"
        );
        // Skipping the reset ALSO yields one detector -- the parity of both
        // measurements -- so counting alone distinguishes nothing. The
        // detector must contain exactly the post-reset measurement.
        assert_eq!(
            map.detectors[0].measurements.len(),
            1,
            "one measurement, not a two-measurement parity"
        );
        assert_eq!(
            map.detectors[0].measurements[0].qubit, 0,
            "and it is the post-reset measurement on qubit 0"
        );
    }

    /// An observable referencing an unknown id is an error, not a silently
    /// thinner observable. De-aliased: id 99 exists in no space.
    #[test]
    fn an_observable_ref_to_an_unknown_id_is_an_error() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.mz(&[0]);
        dag.add_annotation(pecos_quantum::PauliAnnotation {
            pauli: PauliString::zs(&[0usize]),
            kind: pecos_quantum::AnnotationKind::Observable {
                measurement_ids: vec![pecos_core::MeasId::from_raw(99)],
            },
            label: None,
        });

        let Err(err) = InfluenceBuilder::new(&dag).with_circuit_annotations() else {
            panic!("id 99 was never minted, so ingest must fail");
        };
        assert!(matches!(
            err,
            AnnotationIngestError::ObservableRefUnresolved {
                annotation_index: 0,
                source: pecos_quantum::MeasResolveError::Unknown(_),
                ..
            }
        ));
    }

    /// An id held by a gate that consumes no measurement record cannot
    /// contribute a readout term, so it is an error rather than a Z-spray over
    /// the gate's qubits.
    #[test]
    fn an_observable_ref_to_a_record_less_measurement_is_an_error() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        let mut leaked = pecos_core::Gate::measure_leaked(&[0usize]);
        leaked.meas_ids = smallvec::smallvec![pecos_core::MeasId::from_raw(7)];
        dag.try_add_gate(leaked).expect("gate is valid");
        dag.mz(&[1]);
        dag.add_annotation(pecos_quantum::PauliAnnotation {
            pauli: PauliString::zs(&[0usize]),
            kind: pecos_quantum::AnnotationKind::Observable {
                measurement_ids: vec![pecos_core::MeasId::from_raw(7)],
            },
            label: None,
        });

        let Err(err) = InfluenceBuilder::new(&dag).with_circuit_annotations() else {
            panic!("MeasureLeaked consumes no record, so ingest must fail");
        };
        assert!(matches!(
            err,
            AnnotationIngestError::ObservableRefUnresolved {
                source: pecos_quantum::MeasResolveError::RecordLess { .. },
                ..
            }
        ));
    }

    #[test]
    fn test_duplicate_observable_terms_cancel_in_influence_map() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        let meas = dag.mz(&[0]);
        dag.observable_labeled("duplicate_record_obs", &[meas[0], meas[0]])
            .expect("refs are from this circuit");

        let map = InfluenceBuilder::new(&dag)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

        assert_eq!(map.num_observables(), 1);
        for loc_idx in 0..map.locations.len() {
            for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                assert!(
                    map.get_observable_indices(loc_idx, pauli.as_u8())
                        .is_empty(),
                    "observable record XOR should cancel duplicate measurement terms"
                );
            }
        }
    }
}

#[cfg(test)]
mod batched_node_tests {
    use super::InfluenceBuilder;
    use pecos_core::gates::Gate;
    use pecos_quantum::DagCircuit;

    /// A `DagCircuit` node may carry several gate instances -- `gate_count`
    /// counts them individually, and `Gate::h(&[0, 1])` builds exactly that.
    /// The symbolic simulation used to apply only `qubits[0]` and silently drop
    /// the rest, which disagreed with `DagFaultAnalyzer`'s `apply_gate` (it
    /// passes the whole slice) inside the very same `build_dem_from_circuit`
    /// call. Both now apply every instance.
    #[test]
    fn batched_gate_nodes_apply_to_every_instance() {
        let mut batched = DagCircuit::new();
        batched.pz(&[0, 1]);
        batched.add_gate_auto_wire(Gate::h(&[0usize, 1usize]));
        let _ = batched.mz(&[0, 1]);

        let mut split = DagCircuit::new();
        split.pz(&[0, 1]);
        split.h(&[0, 1]);
        let _ = split.mz(&[0, 1]);

        assert_eq!(
            batched.gate_count(),
            split.gate_count(),
            "the two circuits must describe the same gate count"
        );

        let batched_det: Vec<bool> = InfluenceBuilder::new(&batched)
            .run_symbolic_simulation()
            .expect("circuit is replayable")
            .history
            .iter()
            .map(|result| result.is_deterministic)
            .collect();
        let split_det: Vec<bool> = InfluenceBuilder::new(&split)
            .run_symbolic_simulation()
            .expect("circuit is replayable")
            .history
            .iter()
            .map(|result| result.is_deterministic)
            .collect();

        assert_eq!(
            batched_det, split_det,
            "a batched gate node must replay like the same gates split across nodes"
        );
        assert!(
            batched_det.iter().all(|deterministic| !deterministic),
            "H on both qubits makes both Z measurements random; a deterministic \
             outcome means an instance of the batched gate was dropped"
        );
    }
}
