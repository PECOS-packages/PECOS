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

//! Round-boundary noise compression.
//!
//! Propagates mid-round fault locations to round boundaries, producing
//! effective noise sources with accumulated probabilities/amplitudes.
//!
//! For stochastic Pauli noise: exact (Paulis compose deterministically).
//! For coherent noise: accumulates within-round angles exactly.
//!
//! This dramatically reduces the number of noise sources:
//! ~60 mid-round faults per round → ~17 boundary faults (9 data + 8 meas).

use crate::Bm;
use crate::eeg::EegType;
use crate::noise::{NoiseInjection, NoiseSpec};
use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use smallvec::SmallVec;
use std::collections::BTreeMap;

/// An effective noise source at a round boundary.
#[derive(Debug, Clone)]
pub struct BoundaryNoise {
    /// The effective Pauli label at the boundary.
    pub label: Bm,
    /// EEG type (H or S).
    pub eeg_type: EegType,
    /// Accumulated rate (S-type) or amplitude (H-type).
    pub value: f64,
    /// Gate index of the boundary (for position tracking).
    pub boundary_gate: usize,
}

/// Result of noise compression.
pub struct CompressedNoise {
    /// Effective noise sources at round boundaries.
    pub boundary_sources: Vec<BoundaryNoise>,
    /// Measurement noise (kept as-is, not compressed).
    pub measurement_sources: Vec<(usize, NoiseInjection)>,
    /// Preparation noise (kept as-is).
    pub preparation_sources: Vec<(usize, NoiseInjection)>,
    /// Number of original noise sources before compression.
    pub original_count: usize,
    /// Number of compressed sources.
    pub compressed_count: usize,
}

/// Compress mid-round noise to round boundaries.
///
/// Identifies rounds (PZ/QAlloc → gates → MZ), propagates each
/// mid-round noise source's Pauli label forward through remaining
/// gates to the next boundary, and accumulates.
///
/// Gate noise (p1, p2) is compressed. Measurement (p_meas) and
/// preparation (p_prep) noise is kept at its original position.
pub fn compress_noise_to_boundaries(
    gates: &[Gate],
    noise: &dyn NoiseSpec,
    expansion_gates: &[bool],
) -> CompressedNoise {
    let n_gates = gates.len();
    let max_qubit = gates
        .iter()
        .flat_map(|g| g.qubits.iter())
        .map(pecos_core::QubitId::index)
        .max()
        .unwrap_or(0);

    // Step 1: Collect all noise sources
    let mut all_noise: Vec<(usize, NoiseInjection)> = Vec::new();
    for (gate_idx, gate) in gates.iter().enumerate() {
        if gate_idx < expansion_gates.len() && expansion_gates[gate_idx] {
            continue;
        }
        let qubits: SmallVec<[usize; 4]> =
            gate.qubits.iter().map(pecos_core::QubitId::index).collect();
        let injections = noise.noise_after_gate(gate_idx, gate.gate_type, &qubits);
        for inj in injections {
            all_noise.push((gate_idx, inj));
        }
    }

    let original_count = all_noise.len();

    // Step 2: Identify round boundaries.
    // A boundary is an MZ gate (end of round) or PZ/QAlloc (start of round).
    // We propagate gate noise forward to the NEXT MZ on the same qubit.
    let mut measurement_sources = Vec::new();
    let mut preparation_sources = Vec::new();
    let mut gate_noise: Vec<(usize, NoiseInjection)> = Vec::new();

    for (gate_idx, inj) in &all_noise {
        let gate = &gates[*gate_idx];
        match gate.gate_type {
            GateType::MZ | GateType::MeasureFree => {
                measurement_sources.push((*gate_idx, inj.clone()));
            }
            GateType::PZ | GateType::QAlloc => {
                preparation_sources.push((*gate_idx, inj.clone()));
            }
            _ => {
                gate_noise.push((*gate_idx, inj.clone()));
            }
        }
    }

    // Step 3: For each gate noise source, propagate its label forward
    // through subsequent gates until we hit a round boundary (MZ or PZ).
    // Group by (boundary_gate, effective_label, eeg_type).
    let mut boundary_groups: BTreeMap<(usize, Bm, EegType), f64> = BTreeMap::new();

    for (gate_idx, inj) in &gate_noise {
        let mut label = inj.label.clone();

        // Propagate forward through subsequent gates until we hit a boundary.
        // The effective noise lives just BEFORE the boundary gate, so we
        // inject it "after" the last non-boundary gate before it.
        let mut inject_at = *gate_idx; // default: stay at original position
        for g in (*gate_idx + 1)..n_gates {
            match gates[g].gate_type {
                GateType::MZ | GateType::MeasureFree | GateType::PZ | GateType::QAlloc => {
                    let noise_qubits: Vec<usize> = label_qubits(&label, max_qubit);
                    let boundary_qubits: Vec<usize> = gates[g]
                        .qubits
                        .iter()
                        .map(pecos_core::QubitId::index)
                        .collect();
                    if noise_qubits.iter().any(|q| boundary_qubits.contains(q)) {
                        // Inject at the gate just before the boundary
                        // (inject_at was set to g-1 by the last non-boundary gate)
                        break;
                    }
                }
                _ => {
                    // Propagate the label forward through this gate
                    forward_conjugate_label(&mut label, &gates[g]);
                    // Only update inject_at for non-expansion gates.
                    // Expansion gates are invisible to the noise map —
                    // injecting there would be silently dropped.
                    if !(g < expansion_gates.len() && expansion_gates[g]) {
                        inject_at = g;
                    }
                }
            }
        }

        let key = (inject_at, label.clone(), inj.eeg_type);
        *boundary_groups.entry(key).or_insert(0.0) += inj.rate;
    }

    // Step 4: Convert groups to boundary noise sources
    let boundary_sources: Vec<BoundaryNoise> = boundary_groups
        .into_iter()
        .filter(|(_, value)| value.abs() > 1e-20)
        .map(|((boundary_gate, label, eeg_type), value)| BoundaryNoise {
            label,
            eeg_type,
            value,
            boundary_gate,
        })
        .collect();

    let compressed_count =
        boundary_sources.len() + measurement_sources.len() + preparation_sources.len();

    CompressedNoise {
        boundary_sources,
        measurement_sources,
        preparation_sources,
        original_count,
        compressed_count,
    }
}

/// Extract qubits that a Pauli label acts on (up to max_qubit).
fn label_qubits(label: &Bm, max_qubit: usize) -> Vec<usize> {
    let mut qubits = Vec::new();
    for q in 0..=max_qubit {
        if label.has_x(q) || label.has_z(q) {
            qubits.push(q);
        }
    }
    qubits
}

/// Forward-conjugate a Pauli label through a gate (Schrödinger picture).
///
/// This is the FORWARD direction: P → U P U†.
/// For non-self-adjoint gates, we use the forward conjugation directly
/// (not the adjoint swap used in backward walks).
fn forward_conjugate_label(label: &mut Bm, gate: &Gate) {
    use crate::heisenberg::{SparsePauli, sparse_conjugate};

    match gate.gate_type {
        GateType::PZ
        | GateType::QAlloc
        | GateType::QFree
        | GateType::MZ
        | GateType::MeasureFree
        | GateType::MeasureLeaked
        | GateType::I
        | GateType::Idle => return,
        _ => {}
    }

    // sparse_conjugate uses backward (Heisenberg) convention.
    // For forward propagation of a Pauli label, we need U P U†.
    // For self-adjoint gates: same as backward.
    // For non-self-adjoint: we need to swap the gate to its adjoint
    // before calling sparse_conjugate (which already swaps for backward).
    // Two swaps cancel → just call sparse_conjugate on the adjoint gate.
    //
    // Simpler: for forward, swap S↔Sdg BEFORE calling sparse_conjugate
    // (which swaps again for backward), giving net: forward conjugation.
    //
    // Actually, sparse_conjugate applies backward convention (swaps non-self-adjoint).
    // For forward conjugation, we need the opposite swap.
    // Forward of SZ: SZ P SZdg → same as backward of SZdg.
    // So forward_conjugate(P, SZ) = sparse_conjugate(P, SZdg).
    //
    // For self-adjoint gates: no difference.
    // For non-self-adjoint: pass the adjoint gate type.

    let adjoint_type = match gate.gate_type {
        GateType::SZ => GateType::SZdg,
        GateType::SZdg => GateType::SZ,
        GateType::SX => GateType::SXdg,
        GateType::SXdg => GateType::SX,
        GateType::SY => GateType::SYdg,
        GateType::SYdg => GateType::SY,
        GateType::SZZ => GateType::SZZdg,
        GateType::SZZdg => GateType::SZZ,
        GateType::SXX => GateType::SXXdg,
        GateType::SXXdg => GateType::SXX,
        GateType::SYY => GateType::SYYdg,
        GateType::SYYdg => GateType::SYY,
        other => other, // self-adjoint
    };

    // Build a temporary gate with the adjoint type
    let adj_gate = Gate {
        gate_type: adjoint_type,
        qubits: gate.qubits.clone(),
        angles: gate.angles.clone(),
        params: gate.params.clone(),
        meas_ids: gate.meas_ids.clone(),
        channel: None,
    };

    let mut sp = SparsePauli::from_bm(label);
    let _sign = sparse_conjugate(&mut sp, &adj_gate);
    *label = sp.to_bm();
}

/// A `NoiseSpec` adapter that returns compressed boundary noise.
///
/// Call `noise_after_gate()` on each gate just like the original noise model,
/// but mid-round gate noise is empty — all accumulated at boundaries.
pub struct CompressedNoiseSpec {
    /// Gate index → noise injections at that gate.
    gate_noise: BTreeMap<usize, Vec<NoiseInjection>>,
}

impl CompressedNoiseSpec {
    /// Build from compressed noise result.
    #[must_use]
    pub fn from_compressed(compressed: &CompressedNoise) -> Self {
        let mut gate_noise: BTreeMap<usize, Vec<NoiseInjection>> = BTreeMap::new();

        // Boundary sources → noise at boundary gate
        for bn in &compressed.boundary_sources {
            gate_noise
                .entry(bn.boundary_gate)
                .or_default()
                .push(NoiseInjection {
                    eeg_type: bn.eeg_type,
                    label: bn.label.clone(),
                    label2: None,
                    rate: bn.value,
                });
        }

        // Measurement and prep sources stay at original positions
        for (gate_idx, inj) in &compressed.measurement_sources {
            gate_noise.entry(*gate_idx).or_default().push(inj.clone());
        }
        for (gate_idx, inj) in &compressed.preparation_sources {
            gate_noise.entry(*gate_idx).or_default().push(inj.clone());
        }

        Self { gate_noise }
    }
}

impl NoiseSpec for CompressedNoiseSpec {
    fn noise_after_gate(
        &self,
        gate_index: usize,
        _gate_type: GateType,
        _qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        self.gate_noise
            .get(&gate_index)
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::UniformNoise;

    #[test]
    fn test_compression_reduces_count() {
        // Simple circuit: PZ(0,1), CX(0,1), H(1), CX(0,1), MZ(0,1)
        let gates = vec![
            crate::expand::make_gate(GateType::PZ, &[0]),
            crate::expand::make_gate(GateType::PZ, &[1]),
            crate::expand::make_gate(GateType::CX, &[0, 1]),
            crate::expand::make_gate(GateType::H, &[1]),
            crate::expand::make_gate(GateType::CX, &[0, 1]),
            crate::expand::make_gate(GateType::MZ, &[0]),
            crate::expand::make_gate(GateType::MZ, &[1]),
        ];
        let noise = UniformNoise {
            idle_rz: 0.0,
            p1: 0.001,
            p2: 0.01,
            p_meas: 0.01,
            p_prep: 0.01,
        };
        let expansion = vec![false; gates.len()];

        let result = compress_noise_to_boundaries(&gates, &noise, &expansion);
        assert!(
            result.compressed_count < result.original_count,
            "compressed {} should be < original {}",
            result.compressed_count,
            result.original_count
        );
    }
}
