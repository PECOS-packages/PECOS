// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Transversal-CCZ magic-state benchmark on three bivariate tricycle blocks.
//!
//! The codes are the `[[27,3,3]]` and `[[81,3,5]]` BT codes that arXiv:2610.06811,
//! Section 7, benchmarks (from arXiv:2510.07269). This circuit is an ideal encoder,
//! one depolarizing layer, the cup-product CCZ of arXiv:2508.10714, Proposition
//! (`fccz_def`), and an ideal decoder; it contains no 2D code and no code switch.
//! The systematic encoder is this module's own construction (see
//! [`magic_state_circuit`]).
//!
//! Every ideal X measurement is deterministic, including target scoring after
//! undoing the logical CCZ tensor. Thus arXiv:2610.06811, Section 6, guarantees
//! phase-polynomial compatibility under Pauli noise. Measuring logical X before
//! undoing that tensor would instead be incompatible.

use pecos_core::{channel, gate_type::GateType};
use pecos_qec::BivariateTricycleCode;
use pecos_quantum::{F2Matrix, TickCircuit};

use crate::distillation::validate_probability;

#[cfg(test)]
mod tests;

/// Build an ideal encoder, data depolarization, physical CCZ, and inverse encoder.
///
/// Blocks occupy `b*n..(b+1)*n`. Detectors are ordered by block, then X-check
/// pivots in increasing column order, then non-pivot Z measurements in increasing
/// column order. Each measurement defines its own detector. When `target_scoring`
/// is true, the logical CCZ tensor is undone and each logical pivot is measured in
/// X, defining observables in block order then increasing logical-pivot order.
/// Otherwise logical pivots remain unmeasured.
///
/// The logical tensor is the trilinear form of arXiv:2508.10714, Proposition
/// (`fccz_def`), evaluated in the encoder's reduced logical basis.
///
/// # Errors
/// Rejects non-finite `p` or `p` outside `[0,1]`, circuit-size overflow, or invalid
/// internally generated detector/observable annotations.
pub fn magic_state_circuit(
    code: &BivariateTricycleCode,
    p: f64,
    target_scoring: bool,
) -> Result<TickCircuit, String> {
    build(code, p, target_scoring, &[])
}

struct Encoder {
    stabilizers: Vec<Vec<u8>>,
    stabilizer_pivots: Vec<usize>,
    logicals: Vec<Vec<u8>>,
    logical_pivots: Vec<usize>,
    non_pivots: Vec<usize>,
}

impl Encoder {
    // Systematic CSS preparation. Reduce S first, remove its pivots from L, and
    // only then reduce L within its quotient basis.
    fn new(code: &BivariateTricycleCode) -> Self {
        let (reduced, stabilizer_pivots) = F2Matrix::from_rows(code.hx().rows()).row_reduce();
        let mut stabilizers = reduced.rows();
        stabilizers.truncate(stabilizer_pivots.len());
        let mut logicals = code.logical_x().rows();
        for logical in &mut logicals {
            for (row, &pivot) in stabilizers.iter().zip(&stabilizer_pivots) {
                if logical[pivot] == 1 {
                    for (bit, &check) in logical.iter_mut().zip(row) {
                        *bit ^= check;
                    }
                }
            }
        }
        let (reduced, logical_pivots) = F2Matrix::from_rows(logicals).row_reduce();
        let mut logicals = reduced.rows();
        logicals.truncate(logical_pivots.len());
        let non_pivots = (0..code.num_qubits())
            .filter(|q| !stabilizer_pivots.contains(q) && !logical_pivots.contains(q))
            .collect();
        Self {
            stabilizers,
            stabilizer_pivots,
            logicals,
            logical_pivots,
            non_pivots,
        }
    }

    fn prepare(&self, circuit: &mut TickCircuit, n: usize) {
        let tick = circuit.num_ticks();
        circuit.tick();
        for block in 0..3 {
            let x = self
                .stabilizer_pivots
                .iter()
                .chain(&self.logical_pivots)
                .map(|&q| block * n + q)
                .collect::<Vec<_>>();
            let z = self
                .non_pivots
                .iter()
                .map(|&q| block * n + q)
                .collect::<Vec<_>>();
            circuit.tick_at(tick).px(&x);
            circuit.tick_at(tick).pz(&z);
        }
    }

    fn detectors(&self, circuit: &mut TickCircuit, n: usize) -> Result<(), String> {
        for block in 0..3 {
            let x = self
                .stabilizer_pivots
                .iter()
                .map(|&q| block * n + q)
                .collect::<Vec<_>>();
            let z = self
                .non_pivots
                .iter()
                .map(|&q| block * n + q)
                .collect::<Vec<_>>();
            let tick = circuit.num_ticks();
            let mut refs = circuit.tick().mx(&x);
            refs.extend(circuit.tick_at(tick).mz(&z));
            for measurement in refs {
                circuit
                    .detector(&[measurement])
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }

    fn score(
        &self,
        circuit: &mut TickCircuit,
        n: usize,
        gates: &[[usize; 3]],
    ) -> Result<(), String> {
        // Evaluate the trilinear form in the encoder's basis, not the original
        // logical-discovery basis (arXiv:2508.10714, Proposition (fccz_def)).
        for (r, left) in self.logicals.iter().enumerate() {
            for (s, middle) in self.logicals.iter().enumerate() {
                for (t, right) in self.logicals.iter().enumerate() {
                    let parity = gates.iter().fold(0, |value, &[a, b, c]| {
                        value ^ (left[a] & middle[b] & right[c])
                    });
                    if parity == 1 {
                        circuit.tick().ccz(&[(
                            self.logical_pivots[r],
                            n + self.logical_pivots[s],
                            2 * n + self.logical_pivots[t],
                        )]);
                    }
                }
            }
        }
        for block in 0..3 {
            let qubits = self
                .logical_pivots
                .iter()
                .map(|&q| block * n + q)
                .collect::<Vec<_>>();
            let refs = circuit.tick().mx(&qubits);
            for measurement in refs {
                circuit
                    .observable(&[measurement])
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }
}

fn stage(circuit: &mut TickCircuit, n: usize, rows: &[Vec<u8>], pivots: &[usize]) {
    // Unit pivot columns ensure no stage control is a target in this stage.
    // All these CNOTs commute; put each pair on a separate tick, parallel over blocks.
    for (row, &pivot) in rows.iter().zip(pivots) {
        for (target, &bit) in row.iter().enumerate() {
            if bit == 1 && target != pivot {
                circuit.tick().cx(&[
                    (pivot, target),
                    (n + pivot, n + target),
                    (2 * n + pivot, 2 * n + target),
                ]);
            }
        }
    }
}

fn build(
    code: &BivariateTricycleCode,
    p: f64,
    target_scoring: bool,
    faults: &[(usize, GateType)],
) -> Result<TickCircuit, String> {
    validate_probability(p)?;
    let n = code.num_qubits();
    let total = n
        .checked_mul(3)
        .ok_or("three-block circuit size overflows usize")?;
    let encoder = Encoder::new(code);
    let mut circuit = TickCircuit::new();
    encoder.prepare(&mut circuit, n);
    // After L, the state is uniform over span(L), with P_S still |+>. S adds
    // span(S). X(P_S) maps to X(S), X(P_L) to X(L), and Z on each non-pivot
    // maps to a Z stabilizer. L must precede S: stage S may target L pivots.
    stage(&mut circuit, n, &encoder.logicals, &encoder.logical_pivots);
    stage(
        &mut circuit,
        n,
        &encoder.stabilizers,
        &encoder.stabilizer_pivots,
    );
    let mut noise = circuit.tick();
    for q in 0..total {
        noise.channel(channel::Depolarizing(p, q));
    }
    // Forced Paulis are consecutive ticks in the same pre-CCZ noise region.
    for &(q, kind) in faults {
        if q >= total {
            return Err("forced fault qubit outside the three blocks".into());
        }
        match kind {
            GateType::X => {
                circuit.tick().x(&[q]);
            }
            GateType::Y => {
                circuit.tick().y(&[q]);
            }
            GateType::Z => {
                circuit.tick().z(&[q]);
            }
            _ => return Err("forced fault must be X, Y, or Z".into()),
        }
    }
    let layers = code.cup_product_ccz();
    for layer in &layers {
        let triples = layer
            .iter()
            .map(|&[a, b, c]| (a, n + b, 2 * n + c))
            .collect::<Vec<_>>();
        circuit.tick().ccz(&triples);
    }
    stage(
        &mut circuit,
        n,
        &encoder.stabilizers,
        &encoder.stabilizer_pivots,
    );
    stage(&mut circuit, n, &encoder.logicals, &encoder.logical_pivots);
    encoder.detectors(&mut circuit, n)?;
    if target_scoring {
        encoder.score(&mut circuit, n, &layers.concat())?;
    }
    Ok(circuit)
}
