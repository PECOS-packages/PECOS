// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Deferred measurement: expand a circuit by replacing mid-circuit
//! MZ+PZ with CX to auxiliary qubits, deferring all measurements to the end.
//!
//! After expansion, the circuit is purely Clifford (no mid-circuit
//! measurements), and error generators can be propagated straight through.

use crate::Bm;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_core::{Gate, QubitId};

/// Why an EEG DEM could not be built from the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EegBuildError {
    /// A gate fails `Gate::validate`: for example it repeats a qubit, or its
    /// qubit count is not a multiple of its arity. Batched gates are applied
    /// operand group by operand group, which assumes the groups are disjoint.
    InvalidGate {
        /// Position of the gate in the input circuit.
        index: usize,
        /// The validation message.
        reason: String,
    },
    /// The dense matrix Heisenberg walk does not implement this noise type.
    UnsupportedExactNoise {
        /// The offending EEG type.
        eeg_type: crate::eeg::EegType,
    },
    /// A noise label acts on a qubit outside the expanded circuit.
    ExactLabelOutOfRange {
        /// The highest qubit the label acts on.
        qubit: usize,
        /// The number of qubits in the expanded circuit.
        num_qubits: usize,
    },
    /// The dense matrix Heisenberg walk does not implement this gate adjoint.
    UnsupportedExactGate {
        /// The offending gate type in the expanded circuit.
        gate_type: GateType,
    },
    /// The circuit contains a measurement type the EEG expansion does not
    /// handle. Expansion is `MZ`-only; any other measurement would silently
    /// vanish from the deferred-measurement circuit, taking its record with it.
    UnsupportedMeasurement {
        /// The offending gate type.
        gate_type: pecos_core::gate_type::GateType,
    },
    /// An annotation references a measurement record the circuit does not have.
    UnresolvableAnnotationRecord {
        /// The out-of-range record index.
        record_idx: usize,
        /// How many measurement records the expansion produced.
        num_measurements: usize,
    },
    /// Two measurements carry the same id, so id resolution would be
    /// ambiguous. `TickCircuit` does not enforce uniqueness; this expansion
    /// must, because it resolves annotations by id.
    DuplicateMeasId {
        /// The id held by more than one measurement.
        meas_id: pecos_core::MeasId,
    },
    /// An annotation references a measurement id the expansion never recorded.
    UnresolvableAnnotationId {
        /// The unknown id.
        meas_id: pecos_core::MeasId,
        /// How many measurement records the expansion produced.
        num_measurements: usize,
    },
}

impl std::fmt::Display for EegBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGate { index, reason } => write!(f, "gate {index} is invalid: {reason}"),
            Self::UnsupportedExactNoise { eeg_type } => write!(
                f,
                "matrix Heisenberg does not support {eeg_type:?} injections; only H and S are implemented"
            ),
            Self::ExactLabelOutOfRange { qubit, num_qubits } => write!(
                f,
                "noise label acts on qubit {qubit}, outside the {num_qubits}-qubit expanded circuit"
            ),
            Self::UnsupportedExactGate { gate_type } => write!(
                f,
                "matrix Heisenberg does not support the {gate_type:?} gate adjoint"
            ),
            Self::UnsupportedMeasurement { gate_type } => write!(
                f,
                "circuit contains {gate_type:?}, which the MZ-only EEG expansion cannot \
                 represent; its measurement record would silently vanish"
            ),
            Self::UnresolvableAnnotationRecord {
                record_idx,
                num_measurements,
            } => write!(
                f,
                "annotation references measurement record {record_idx}, but the expansion \
                 produced only {num_measurements}"
            ),
            Self::DuplicateMeasId { meas_id } => write!(
                f,
                "two measurements carry MeasId({}); annotation resolution by id \
                 requires each measurement to hold a unique id",
                meas_id.index()
            ),
            Self::UnresolvableAnnotationId {
                meas_id,
                num_measurements,
            } => write!(
                f,
                "annotation references MeasId({}), which the expansion never recorded \
                 ({num_measurements} measurement records exist)",
                meas_id.index()
            ),
        }
    }
}

impl std::error::Error for EegBuildError {}

/// Result of circuit expansion.
pub struct ExpandedCircuit {
    /// The expanded gate sequence (purely Clifford, no mid-circuit measurements).
    pub gates: Vec<Gate>,
    /// Flags parallel to `gates`: true for the `QAlloc`, `CX` and `PZ` this
    /// expansion inserted, which carry no physical noise. User gates and the
    /// final auxiliary measurements are false.
    pub expansion_gates: Vec<bool>,
    /// Total number of qubits (original + auxiliary).
    pub num_qubits: usize,
    /// Number of original qubits.
    pub num_original_qubits: usize,
    /// Mapping: measurement record index → auxiliary qubit index.
    /// measurement_qubit[k] = the auxiliary qubit whose Z-measurement at
    /// the end gives the k-th measurement record.
    pub measurement_qubit: Vec<usize>,
    /// Mapping: measurement record index → original qubit that was measured.
    /// original_measured_qubit[k] = the qubit in the original circuit that
    /// the k-th MZ gate acted on.
    pub original_measured_qubit: Vec<usize>,
    /// Expansion rank of each stamped measurement id.
    ///
    /// `meas_id_rank[&id] = k` means the measurement holding `id` is the k-th
    /// one this expansion recorded -- an index into `measurement_qubit` and
    /// `original_measured_qubit`.
    ///
    /// `MZ`, `MeasureFree`, and `MPZ` all appear -- `MeasureFree` lowers to `MZ` in
    /// this expansion, since the free has no stabilizer effect and its record
    /// is real. `MeasureLeaked` is refused at the entrance, so an absent id
    /// here means unknown. Id-less legacy circuits yield an empty map.
    ///
    /// This is eeg's private ordinal. It is expansion order, not id-rank order
    /// and not any other component's ordering; resolve ids through this map
    /// rather than assuming an id's numeric value indexes anything.
    pub meas_id_rank: std::collections::BTreeMap<pecos_core::MeasId, usize>,
}

/// Expand a circuit by deferring mid-circuit measurements.
///
/// Each measurement of qubit q (`MZ`, `MeasureFree`, `MPZ`) is replaced by:
/// 1. QAlloc(aux) -- a fresh auxiliary qubit
/// 2. CX(q, aux) -- copy q's Z value onto it
/// 3. PZ(q) -- for an ancilla (a qubit reset later) or an `MPZ`, the
///    projection the measurement performs
///
/// All auxiliary qubits are measured at the end via MZ. Final data
/// measurements are deferred the same way, for uniformity.
///
/// The inserted gates are virtual and carry no physical noise.
/// [`ExpandedCircuit::expansion_gates`] marks exactly them; user gates and the
/// final auxiliary measurements are not marked. Consumers take these flags
/// rather than inferring inserted gates from gate types, which a user
/// `QAlloc` followed by a `CX` into it would fool.
///
/// # Errors
///
/// Returns [`EegBuildError::UnsupportedMeasurement`] when the circuit contains
/// a measurement type this MZ-only expansion cannot represent -- passing it
/// through would silently delete its measurement record. This is THE checked
/// entrance: every consumer (builder, simulator, bindings) routes through it,
/// so the check cannot drift across copies.
pub fn expand_circuit(gates: &[Gate]) -> Result<ExpandedCircuit, EegBuildError> {
    for (index, gate) in gates.iter().enumerate() {
        gate.validate()
            .map_err(|reason| EegBuildError::InvalidGate { index, reason })?;
        // `MeasureFree` is record-bearing and lowers to `MZ` below -- the
        // "free" is resource bookkeeping with no stabilizer effect, and a
        // reused qubit reappears behind an explicit prep the expansion keeps.
        // Only `MeasureLeaked` is refused: it consumes no record, so this
        // record-aligned expansion cannot represent it.
        if gate.gate_type == GateType::MeasureLeaked {
            return Err(EegBuildError::UnsupportedMeasurement {
                gate_type: gate.gate_type,
            });
        }
    }
    // First pass: find the max qubit index to know where auxiliaries start
    let max_qubit = gates
        .iter()
        .flat_map(|g| g.qubits.iter())
        .map(pecos_core::QubitId::index)
        .max()
        .unwrap_or(0);
    let num_original = max_qubit + 1;
    let mut next_aux = num_original;

    let mut expanded = Vec::with_capacity(gates.len() * 2);
    let mut expansion_gates = Vec::with_capacity(gates.len() * 2);
    let mut meas_id_rank = std::collections::BTreeMap::new();
    let mut measurement_qubit = Vec::new();
    let mut original_measured_qubit = Vec::new();

    // Identify which MZ gates are mid-circuit (followed by PZ on same qubit)
    // vs final (not followed by any operation on that qubit, or followed by
    // a different operation).
    //
    // Track ancilla qubits: those with a PZ after the first non-PZ gate.
    // Only ancilla MZ gets a post-expansion PZ (measurement projection).
    let mut ancilla_qubits = std::collections::HashSet::new();
    {
        let mut past_init = false;
        for g in gates {
            if past_init && (g.gate_type == GateType::PZ || g.gate_type == GateType::QAlloc) {
                for q in &g.qubits {
                    ancilla_qubits.insert(q.index());
                }
            }
            if g.gate_type != GateType::PZ && g.gate_type != GateType::QAlloc {
                past_init = true;
            }
        }
    }

    // Strategy: walk gates, when we see MZ(q):
    //   - Replace with CX(q, aux_new) where aux_new is a fresh auxiliary
    //   - For ancilla qubits: add PZ(q) to model measurement projection
    //   - Record that measurement k maps to aux_new
    //   - The auxiliary qubit is measured at the end

    let mut i = 0;
    while i < gates.len() {
        let gate = &gates[i];

        match gate.gate_type {
            GateType::MZ | GateType::MeasureFree | GateType::MPZ => {
                // For each qubit in this MZ gate, create CX to auxiliary
                for (position, q) in gate.qubits.iter().enumerate() {
                    let q_idx = q.index();
                    if let Some(&id) = gate.meas_ids.get(position)
                        && meas_id_rank.insert(id, measurement_qubit.len()).is_some()
                    {
                        // Last-wins would silently rebind every annotation
                        // naming this id to the later measurement.
                        return Err(EegBuildError::DuplicateMeasId { meas_id: id });
                    }
                    let aux = next_aux;
                    next_aux += 1;

                    // Initialize auxiliary: QAlloc(aux)
                    expanded.push(make_gate(GateType::QAlloc, &[aux]));
                    expansion_gates.push(true);

                    // CX(q, aux) — copy measurement info to auxiliary
                    expanded.push(make_gate(GateType::CX, &[q_idx, aux]));
                    expansion_gates.push(true);

                    // For ancilla qubits: add PZ to model measurement projection.
                    // MZ projects to a Z eigenstate, destroying X/Y coherences.
                    // Without this, the last round's syndrome generators retain
                    // X on the measured ancilla, creating spurious correlations.
                    // For intermediate rounds this is redundant (circuit PZ follows).
                    //
                    // Data readout MZ does NOT get PZ: data qubit Z components
                    // must persist for correct generator labels (Z errors are
                    // invisible to Z-basis readout and must not be cleared).
                    // MPZ carries its own reset, so the projection PZ is
                    // unconditional for it.
                    if ancilla_qubits.contains(&q_idx) || gate.gate_type == GateType::MPZ {
                        expanded.push(make_gate(GateType::PZ, &[q_idx]));
                        expansion_gates.push(true);
                    }

                    // Record: this measurement maps to the auxiliary qubit
                    measurement_qubit.push(aux);
                    original_measured_qubit.push(q_idx);
                }
            }
            GateType::PZ | GateType::QAlloc => {
                // Keep resets — they re-initialize the qubit for the next round
                expanded.push(gate.clone());
                expansion_gates.push(false);
            }
            _ => {
                // All other gates pass through unchanged
                expanded.push(gate.clone());
                expansion_gates.push(false);
            }
        }

        i += 1;
    }

    // Add final measurements of all auxiliary qubits at the end
    for &aux in &measurement_qubit {
        expanded.push(make_gate(GateType::MZ, &[aux]));
        expansion_gates.push(false);
    }

    Ok(ExpandedCircuit {
        gates: expanded,
        expansion_gates,
        num_qubits: next_aux,
        num_original_qubits: num_original,
        measurement_qubit,
        original_measured_qubit,
        meas_id_rank,
    })
}

impl ExpandedCircuit {
    /// Map an expanded-circuit Pauli back to the original circuit frame.
    ///
    /// X on auxiliary qubit `aux_k` → X on `original_measured_qubit[k]`
    /// (because `CX(q, aux)` copies X from control to target: X on aux
    /// in the expanded circuit corresponds to X on q in the original).
    ///
    /// Z on auxiliary qubits is dropped (doesn't correspond to original).
    /// Components on original qubits pass through unchanged.
    #[must_use]
    pub fn map_to_original_frame(&self, p: &Bm) -> Bm {
        let mut result = Bm::default();

        // Copy components on original qubits directly
        for q in 0..self.num_original_qubits {
            if p.has_x(q) {
                result.x_bits.set_bit(q);
            }
            if p.has_z(q) {
                result.z_bits.set_bit(q);
            }
        }

        // Map X on auxiliary qubits to X on original measured qubits
        for (meas_idx, &aux_q) in self.measurement_qubit.iter().enumerate() {
            if p.has_x(aux_q) {
                let orig_q = self.original_measured_qubit[meas_idx];
                result.x_bits.xor_bit(orig_q); // XOR because same qubit may be measured multiple times
            }
            // Z on aux is ignored (measurement projection absorbs Z)
        }

        result
    }
}

/// Precomputed qubit-to-gate index for sparse backward traversal.
///
/// For each qubit, stores the gate indices (in the flat gate list) that
/// touch it through the gate itself or through the gate's exact noise,
/// sorted in ascending order. This enables the backward walk to visit only
/// gates on active qubits instead of scanning all gates.
pub struct GateIndex {
    /// qubit_gates[q] = sorted Vec of gate indices touching qubit q.
    qubit_gates: Vec<Vec<u32>>,
    /// Which gates are expansion gates (no physical noise).
    pub expansion_gates: Vec<bool>,
}

impl GateIndex {
    /// Build the index from a gate list (typically the expanded circuit) and
    /// the noise the sparse walks will apply. Noise may act outside its gate's
    /// qubits, so the gate is also indexed under every qubit its noise acts on.
    /// Explicit provenance flags suppress noise on inserted expansion gates.
    ///
    /// # Panics
    /// Panics if the provenance flags do not match the gate count.
    #[must_use]
    pub fn build(
        gates: &[Gate],
        num_qubits: usize,
        noise: &dyn crate::noise::NoiseSpec,
        expansion_gates: &[bool],
    ) -> Self {
        assert_one_per_gate("expansion_gates", expansion_gates.len(), gates.len());

        let mut qubit_gates = vec![Vec::new(); num_qubits];

        for (i, gate) in gates.iter().enumerate() {
            let gate_qubits: Vec<usize> = gate.qubits.iter().map(QubitId::index).collect();
            let mut touched = gate_qubits.clone();
            if !expansion_gates[i] {
                touched.extend(
                    noise
                        .exact_noise_after_gate(i, gate.gate_type, &gate_qubits)
                        .qubits(),
                );
            }
            touched.sort_unstable();
            touched.dedup();
            for q in touched {
                if q >= qubit_gates.len() {
                    qubit_gates.resize_with(q + 1, Vec::new);
                }
                qubit_gates[q].push(i as u32);
            }
        }

        Self {
            qubit_gates,
            expansion_gates: expansion_gates.to_vec(),
        }
    }

    /// Gate indices touching qubit `q` in reverse order (for backward walk).
    pub fn gates_on_qubit_rev(&self, q: usize) -> impl Iterator<Item = u32> + '_ {
        self.qubit_gates
            .get(q)
            .into_iter()
            .flat_map(|v| v.iter().copied().rev())
    }

    /// Is this gate an expansion gate (no physical noise)?
    ///
    /// # Panics
    /// Panics if `gate_idx` is outside the indexed gate list.
    #[inline]
    #[must_use]
    pub fn is_expansion(&self, gate_idx: usize) -> bool {
        self.expansion_gates[gate_idx]
    }
}

/// Panic unless a per-gate list (`what`, such as expansion flags or a noise
/// map) has exactly one entry per gate.
pub(crate) fn assert_one_per_gate(what: &str, len: usize, num_gates: usize) {
    assert_eq!(
        len, num_gates,
        "{what} length {len} must equal gates length {num_gates}"
    );
}

/// Construct an unparameterized expansion gate.
///
/// # Panics
/// Panics if the gate type requires angles.
#[must_use]
pub fn make_gate(gt: GateType, qubits: &[usize]) -> Gate {
    Gate::simple(
        gt,
        qubits
            .iter()
            .map(|&q| QubitId(q))
            .collect::<pecos_core::GateQubits>(),
    )
}

impl ExpandedCircuit {
    /// The auxiliary qubit whose final Z-measurement carries `record_idx`.
    ///
    /// The one implementation of record resolution: an out-of-range record is
    /// an error naming both sides of the mismatch, never a silent skip.
    ///
    /// # Errors
    ///
    /// Returns [`EegBuildError::UnresolvableAnnotationRecord`] when the record
    /// does not exist.
    pub fn aux_qubit_for_record(&self, record_idx: usize) -> Result<usize, EegBuildError> {
        self.measurement_qubit.get(record_idx).copied().ok_or(
            EegBuildError::UnresolvableAnnotationRecord {
                record_idx,
                num_measurements: self.measurement_qubit.len(),
            },
        )
    }

    /// The auxiliary qubit whose final Z-measurement carries the measurement
    /// named by `meas_id`.
    ///
    /// Resolution goes through `meas_id_rank`, so it is correct for external
    /// (non-positional) ids -- an id's numeric value is never used as an index.
    ///
    /// # Errors
    ///
    /// Returns [`EegBuildError::UnresolvableAnnotationId`] when the expansion
    /// never recorded the id.
    pub fn aux_qubit_for_id(&self, meas_id: pecos_core::MeasId) -> Result<usize, EegBuildError> {
        let rank = self.meas_id_rank.get(&meas_id).copied().ok_or(
            EegBuildError::UnresolvableAnnotationId {
                meas_id,
                num_measurements: self.measurement_qubit.len(),
            },
        )?;
        self.aux_qubit_for_record(rank)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn expansion_flags_are_exact() {
        let gates = [
            make_gate(GateType::PZ, &[0]),
            make_gate(GateType::H, &[0]),
            make_gate(GateType::PZ, &[1]),
            make_gate(GateType::CX, &[0, 1]),
            make_gate(GateType::MZ, &[1]),
            make_gate(GateType::QAlloc, &[2]),
            make_gate(GateType::CX, &[0, 2]),
            make_gate(GateType::PZ, &[0]),
            make_gate(GateType::MZ, &[2, 3]),
        ];
        let expanded = expand_circuit(&gates).unwrap();
        assert_eq!(expanded.expansion_gates.len(), expanded.gates.len());
        assert_eq!(
            expanded.expansion_gates,
            [
                false, false, false, false, // original gates
                true, true, true, // deferred ancilla measurement
                false, false, false, // user allocation, CX and reset
                true, true, true, // deferred measurement of qubit 2
                true, true, // final data measurement of qubit 3
                false, false, false, // final auxiliary measurements
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Gate RZ expected 1 angle parameters, got 0")]
    fn make_gate_refuses_rotation_without_angles() {
        let _ = make_gate(GateType::RZ, &[0]);
    }

    /// `meas_id_rank` records expansion order, keyed by the id the gate holds.
    ///
    /// De-aliased: the ids arrive in descending order (9 before 3), so
    /// expansion rank disagrees with numeric id order and with the id values
    /// themselves -- ranking by anything except the walk fails.
    #[test]
    fn meas_id_rank_follows_expansion_order_not_id_order() {
        use pecos_core::MeasId;
        // One BATCHED MZ: two measurements inside a single gate, ids scrambled
        // (9 then 3) so rank disagrees with id order, and id values (9, 3)
        // coincide with no gate index (0..=1) or rank (0..=1). Ranking by the
        // first id, by id value, or by anything except the per-position walk
        // fails.
        let mut batch = super::make_gate(super::GateType::MZ, &[0, 1]);
        batch.meas_ids.push(MeasId::from_raw(9));
        batch.meas_ids.push(MeasId::from_raw(3));
        let gates = vec![
            super::make_gate(super::GateType::PZ, &[0]),
            super::make_gate(super::GateType::PZ, &[1]),
            batch,
        ];

        let expanded = super::expand_circuit(&gates).expect("MZ-only circuit");
        assert_eq!(
            expanded.meas_id_rank.get(&MeasId::from_raw(9)),
            Some(&0),
            "first batch member is expansion rank 0"
        );
        assert_eq!(
            expanded.meas_id_rank.get(&MeasId::from_raw(3)),
            Some(&1),
            "second batch member is rank 1 -- per-position, not first-id"
        );
        assert_eq!(expanded.meas_id_rank.len(), 2);
    }

    use super::*;

    fn gate(gt: GateType, qubits: &[usize]) -> Gate {
        make_gate(gt, qubits)
    }

    #[test]
    fn test_expand_simple_mcm() {
        // PZ(0), H(0), MZ(0), PZ(0), H(0), MZ(0)
        // Two rounds: measure, reset, measure again
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]), // → CX(0, aux0)
            gate(GateType::PZ, &[0]), // reset
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]), // → CX(0, aux1)
        ];

        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        // Should have 3 qubits: original 0, aux 1, aux 2
        assert_eq!(expanded.num_original_qubits, 1);
        assert_eq!(expanded.num_qubits, 3);
        assert_eq!(expanded.measurement_qubit.len(), 2);

        // No MZ in the middle — only at the end
        let mid_mz = expanded.gates[..expanded.gates.len() - 2]
            .iter()
            .filter(|g| g.gate_type == GateType::MZ)
            .count();
        assert_eq!(mid_mz, 0, "No mid-circuit MZ in expanded circuit");

        // Two MZ at the end (one per auxiliary)
        let end_mz = expanded
            .gates
            .iter()
            .rev()
            .take_while(|g| g.gate_type == GateType::MZ)
            .count();
        assert_eq!(end_mz, 2);
    }

    #[test]
    fn test_expand_preserves_cliffords() {
        let gates = vec![
            gate(GateType::PZ, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
        ];

        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        // No measurements → no expansion needed
        assert_eq!(expanded.num_qubits, 2);
        assert_eq!(expanded.measurement_qubit.len(), 0);
        assert_eq!(expanded.gates.len(), 3); // same gates
    }

    #[test]
    fn test_measurement_qubit_mapping() {
        // 2 qubits, measure both
        let gates = vec![
            gate(GateType::PZ, &[0, 1]),
            gate(GateType::H, &[0]),
            gate(GateType::CX, &[0, 1]),
            gate(GateType::MZ, &[0]), // meas record 0 → aux 2
            gate(GateType::MZ, &[1]), // meas record 1 → aux 3
        ];

        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        assert_eq!(expanded.measurement_qubit, vec![2, 3]);
        assert_eq!(expanded.num_qubits, 4);
    }

    #[test]
    fn test_map_to_original_frame_x_on_aux() {
        // X on auxiliary → X on original measured qubit
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]), // meas 0 → aux 1
        ];
        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        // X on aux 1 maps to X on original qubit 0
        let p = Bm::x(1); // aux qubit
        let mapped = expanded.map_to_original_frame(&p);
        assert_eq!(mapped, Bm::x(0));
    }

    #[test]
    fn test_map_to_original_frame_z_on_aux_dropped() {
        // Z on auxiliary is dropped (measurement projection absorbs it)
        let gates = vec![gate(GateType::PZ, &[0]), gate(GateType::MZ, &[0])];
        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        let p = Bm::z(1); // Z on aux
        let mapped = expanded.map_to_original_frame(&p);
        assert!(mapped.is_identity(), "Z on aux should be dropped");
    }

    #[test]
    fn test_map_to_original_frame_original_passthrough() {
        // Components on original qubits pass through unchanged
        let gates = vec![gate(GateType::PZ, &[0, 1]), gate(GateType::MZ, &[0])];
        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        let p = Bm::x(0).multiply(&Bm::z(1)); // X0 Z1
        let mapped = expanded.map_to_original_frame(&p);
        assert_eq!(mapped, Bm::x(0).multiply(&Bm::z(1)));
    }

    #[test]
    fn test_expand_final_only_mz() {
        // Circuit with only final MZ (no mid-circuit measurement)
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::H, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        // Still creates one aux qubit for the final MZ
        assert_eq!(expanded.num_qubits, 2);
        assert_eq!(expanded.measurement_qubit.len(), 1);
    }

    #[test]
    fn test_expansion_pz_for_ancilla_mz() {
        // Circuit: PZ(0,1), H(1), CX(1,0), MZ(1), PZ(1), H(1), CX(1,0), MZ(1), MZ(0)
        // Qubit 1 is ancilla (has mid-circuit PZ). Qubit 0 is data.
        // Last-round MZ(1) should get expansion PZ(1).
        // Final MZ(0) should NOT get expansion PZ.
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::PZ, &[1]),
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::MZ, &[1]), // round 1 syndrome
            gate(GateType::PZ, &[1]), // reset
            gate(GateType::H, &[1]),
            gate(GateType::CX, &[1, 0]),
            gate(GateType::MZ, &[1]), // round 2 syndrome (last round, no PZ after)
            gate(GateType::MZ, &[0]), // data readout
        ];

        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        // Count PZ/QAlloc gates on qubit 1 in the expanded circuit
        let resets_on_1: Vec<_> = expanded
            .gates
            .iter()
            .filter(|g| {
                (g.gate_type == GateType::PZ || g.gate_type == GateType::QAlloc)
                    && g.qubits.iter().any(|q| q.index() == 1)
            })
            .collect();

        // Should have: original PZ(1) init + expansion PZ(1) round 1 + circuit PZ(1) reset
        //            + expansion PZ(1) round 2 = 4 reset gates on qubit 1
        eprintln!("Resets on qubit 1: {} gates", resets_on_1.len());
        assert!(
            resets_on_1.len() >= 4,
            "Should have expansion PZ for last-round MZ(1): got {} on q1",
            resets_on_1.len()
        );

        // Count resets on qubit 0 in expanded circuit
        let resets_on_0: Vec<_> = expanded
            .gates
            .iter()
            .filter(|g| {
                (g.gate_type == GateType::PZ || g.gate_type == GateType::QAlloc)
                    && g.qubits.iter().any(|q| q.index() == 0)
            })
            .collect();
        // Should have only: original PZ(0) init = 1
        eprintln!("Resets on qubit 0: {} gates", resets_on_0.len());
        assert_eq!(
            resets_on_0.len(),
            1,
            "Data qubit should NOT get expansion PZ"
        );
    }

    #[test]
    fn test_expand_multi_round_tracks_original_qubits() {
        // Two rounds measuring qubit 0: both aux should map back to qubit 0
        let gates = vec![
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]),
            gate(GateType::PZ, &[0]),
            gate(GateType::MZ, &[0]),
        ];
        let expanded = expand_circuit(&gates).expect("MZ-only circuit");

        assert_eq!(expanded.measurement_qubit.len(), 2);
        assert_eq!(expanded.original_measured_qubit, vec![0, 0]);
    }
}
