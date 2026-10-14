// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::{
    IncompatibleMeasurement, Mask, PecosRng, PhasePoly, QuantumSimulator, QubitId, ShotResult,
    TriorthogonalMatrix, logical_paulis, validate_probability,
};
use std::collections::BTreeSet;
use std::fmt::Write;

/// Circuit operations on physical qubits. A generated circuit prepares all
/// qubits, applies T/noise and U, measures all syndromes, then checks outputs.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// Prepare zero.
    PZ(QubitId),
    /// Prepare plus.
    PX(QubitId),
    /// Controlled X, control first.
    CX(QubitId, QubitId),
    /// Controlled Z.
    CZ(QubitId, QubitId),
    /// Phase S.
    S(QubitId),
    /// Inverse S.
    Sdg(QubitId),
    /// Pauli Z.
    Z(QubitId),
    /// Phase T.
    T(QubitId),
    /// Independently sampled Z error with the supplied probability.
    ZError(QubitId, f64),
    /// Measure an X string; false denotes +1.
    MeasureX(Vec<QubitId>),
    /// Query logical W on an odd-weight original row, on accepted shots only.
    /// Uses the signed Y product `i (-i)^weight Y(support)`.
    ExpectW(Vec<QubitId>),
}

impl TriorthogonalMatrix {
    /// Build the encoder, noisy transversal T, Clifford U, syndrome, and outputs.
    /// Qubits are `0..n`; measurements and outputs retain original row order.
    ///
    /// # Errors
    /// Requires finite `p` in `[0,1]`.
    pub fn circuit(&self, p: f64) -> Result<Vec<Op>, String> {
        validate_probability(p)?;
        let (reduced, transform, pivots) = self.reduced();
        let mut ops: Vec<_> = (0..self.n).map(|q| Op::PZ(QubitId(q))).collect();
        ops.extend(pivots.iter().map(|&p| Op::PX(QubitId(p))));
        for (row, &pivot) in reduced.iter().zip(&pivots) {
            ops.extend(
                row.ones()
                    .filter(|&j| j != pivot)
                    .map(|j| Op::CX(QubitId(pivot), QubitId(j))),
            );
        }
        for j in 0..self.n {
            ops.push(Op::T(QubitId(j)));
            ops.push(Op::ZError(QubitId(j), p));
        }
        self.correction(&transform, &pivots, &mut ops);
        ops.extend(
            self.rows[self.k..]
                .iter()
                .map(|row| Op::MeasureX(row.ones().map(QubitId).collect())),
        );
        ops.extend(
            self.rows[..self.k]
                .iter()
                .map(|row| Op::ExpectW(row.ones().map(QubitId).collect())),
        );
        Ok(ops)
    }

    /// Build the same encoder, noisy T layer, correction and syndromes as
    /// [`Self::circuit`], using typed channels and one detector per syndrome.
    /// X strings use CX ladders around a single MX; logical W expectations
    /// are omitted because they are queries rather than measurements.
    ///
    /// # Errors
    /// Requires finite `p` in `[0,1]`, as in [`Self::circuit`].
    pub fn tick_circuit(&self, p: f64) -> Result<pecos_quantum::TickCircuit, String> {
        use pecos_core::{Gate, channel, gate_type::GateType};
        let mut circuit = pecos_quantum::TickCircuit::new();
        for op in self.circuit(p)? {
            let (kind, qubits) = match op {
                Op::PZ(q) => (GateType::PZ, vec![q]),
                Op::PX(q) => (GateType::PX, vec![q]),
                Op::CX(a, b) => (GateType::CX, vec![a, b]),
                Op::CZ(a, b) => (GateType::CZ, vec![a, b]),
                Op::S(q) => (GateType::SZ, vec![q]),
                Op::Sdg(q) => (GateType::SZdg, vec![q]),
                Op::Z(q) => (GateType::Z, vec![q]),
                Op::T(q) => (GateType::T, vec![q]),
                Op::ZError(q, rate) => {
                    circuit.tick().channel(channel::Dephasing(rate, q));
                    continue;
                }
                Op::MeasureX(support) => {
                    // Validated independent matrix rows are nonzero.
                    let first = support[0];
                    for &other in &support[1..] {
                        circuit.tick().cx(&[(first, other)]);
                    }
                    let refs = circuit.tick().mx(&[first]);
                    circuit.detector(&refs).map_err(|e| e.to_string())?;
                    for &other in support[1..].iter().rev() {
                        circuit.tick().cx(&[(first, other)]);
                    }
                    continue;
                }
                Op::ExpectW(_) => continue,
            };
            let tick = circuit.num_ticks();
            circuit.tick();
            circuit.ticks_mut()[tick].add_gate(Gate::simple(kind, qubits));
        }
        Ok(circuit)
    }

    // R = A G; Gaussian elimination keeps A alongside R. Every pivot column
    // is a unit vector, hence f[p_b] = y_b and x_a = XOR_b A[b,a] f[p_b].
    fn reduced(&self) -> (Vec<Mask>, Vec<Mask>, Vec<usize>) {
        let mut rows = self.rows.clone();
        let mut transform: Vec<_> = (0..self.m())
            .map(|a| {
                let mut mask = Mask::zero(self.m());
                mask.toggle(a);
                mask
            })
            .collect();
        let mut pivots = Vec::new();
        for j in 0..self.n {
            let b = pivots.len();
            let Some(next) = (b..self.m()).find(|&a| rows[a].get(j)) else {
                continue;
            };
            rows.swap(b, next);
            transform.swap(b, next);
            let pivot_row = rows[b].clone();
            let pivot_transform = transform[b].clone();
            for a in 0..self.m() {
                if a != b && rows[a].get(j) {
                    rows[a].xor(&pivot_row);
                    transform[a].xor(&pivot_transform);
                }
            }
            pivots.push(j);
        }
        (rows, transform, pivots)
    }

    fn correction(&self, transform: &[Mask], pivots: &[usize], ops: &mut Vec<Op>) {
        let sets: Vec<Vec<_>> = (0..self.m())
            .map(|a| {
                transform
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| row.get(a))
                    .map(|(b, _)| pivots[b])
                    .collect()
            })
            .collect();
        let mut powers = vec![0; self.n];
        let mut pairs = BTreeSet::new();
        // Inclusion-exclusion for |f| holds modulo 8; all fourth/higher
        // intersections vanish modulo 8, and triorthogonality kills the cubic.
        // U contributes -sum c_a x_a + 4 sum h_ab x_a x_b (mod 8).
        for (a, set) in sets.iter().enumerate() {
            let half_c = (self.rows[a].ones().count() - usize::from(a < self.k)) / 2;
            for &p in set {
                powers[p] = (powers[p] + 4 - half_c % 4) % 4;
            }
            if half_c % 2 == 1 {
                for (i, &p) in set.iter().enumerate() {
                    for &q in &set[i + 1..] {
                        toggle_pair(&mut pairs, p, q);
                    }
                }
            }
            for (b, other) in sets.iter().enumerate().skip(a + 1) {
                let half_overlap = self.rows[a].ones().filter(|&j| self.rows[b].get(j)).count() / 2;
                if half_overlap % 2 == 1 {
                    for &p in set {
                        for &q in other {
                            if p == q {
                                powers[p] = (powers[p] + 2) % 4;
                            } else {
                                toggle_pair(&mut pairs, p, q);
                            }
                        }
                    }
                }
            }
        }
        for (q, power) in powers.into_iter().enumerate() {
            let q = QubitId(q);
            match power {
                1 => ops.push(Op::S(q)),
                2 => ops.push(Op::Z(q)),
                3 => ops.push(Op::Sdg(q)),
                _ => {}
            }
        }
        ops.extend(
            pairs
                .into_iter()
                .map(|(p, q)| Op::CZ(QubitId(p), QubitId(q))),
        );
    }
}

fn toggle_pair(pairs: &mut BTreeSet<(usize, usize)>, p: usize, q: usize) {
    let pair = (p.min(q), p.max(q));
    if !pairs.insert(pair) {
        pairs.remove(&pair);
    }
}

/// Execute a complete generated circuit, resetting the simulator first.
/// `errors(op_index, qubit_index, p)` is called exactly once per `ZError`, so
/// callers can force arbitrary patterns independently of measurement randomness.
/// Output checks must follow every syndrome measurement, as in `circuit()`.
///
/// # Errors
/// Propagates unsupported X measurements; a generated circuit has deterministic
/// syndrome measurements for every Z-error pattern.
///
/// # Panics
/// Indices must fit the simulator; output supports must have odd weight.
pub fn run_shot<F>(
    sim: &mut PhasePoly,
    ops: &[Op],
    errors: &mut F,
) -> Result<ShotResult, IncompatibleMeasurement>
where
    F: FnMut(usize, usize, f64) -> bool,
{
    sim.reset();
    let mut syndrome = Vec::new();
    let mut accepted = true;
    let mut logical_w = Vec::new();
    for (index, op) in ops.iter().enumerate() {
        match op {
            Op::PZ(q) => {
                sim.pz(&[*q]);
            }
            Op::PX(q) => {
                sim.px(&[*q]);
            }
            Op::CX(a, b) => {
                sim.cx(&[(*a, *b)]);
            }
            Op::CZ(a, b) => {
                sim.cz(&[(*a, *b)]);
            }
            Op::S(q) => {
                sim.sz(&[*q]);
            }
            Op::Sdg(q) => {
                sim.szdg(&[*q]);
            }
            Op::Z(q) => {
                sim.z(&[*q]);
            }
            Op::T(q) => {
                sim.t(&[*q]);
            }
            Op::ZError(q, p) => {
                if errors(index, q.0, *p) {
                    sim.z(&[*q]);
                }
            }
            Op::MeasureX(support) => {
                let result = sim.mx_string(support)?;
                accepted &= !result.outcome;
                syndrome.push(result);
            }
            Op::ExpectW(support) => {
                assert!(
                    support.len() % 2 == 1,
                    "logical output must have odd weight"
                );
                if accepted {
                    let (x, y) = logical_paulis(support);
                    logical_w.push(
                        (sim.expectation(&x) + sim.expectation(&y))
                            * std::f64::consts::FRAC_1_SQRT_2,
                    );
                }
            }
        }
    }
    Ok(ShotResult {
        syndrome,
        accepted,
        logical_w: accepted.then_some(logical_w),
    })
}

/// Run one shot with independently sampled Z errors from a caller-owned RNG.
///
/// # Errors
/// Propagates unsupported X measurements, as in [`run_shot`].
///
/// # Panics
/// Requires valid circuit indices, odd output supports, and probabilities in `[0,1]`.
pub fn run_shot_sampled(
    sim: &mut PhasePoly,
    ops: &[Op],
    rng: &mut PecosRng,
) -> Result<ShotResult, IncompatibleMeasurement> {
    run_shot(sim, ops, &mut |_, _, p| rng.random_bool(p))
}

/// Serialize a generated circuit as Stim-format text with T and `EXP_VAL` extensions.
/// `EXP_VAL` reports each listed product's expectation separately, without
/// measurement or collapse. Each line lists `X(f)` and signed `Y(f)` for one logical;
/// its W check is their sum over sqrt(2), meaningful when all detectors are zero.
/// A leading `!` on a product's first factor negates that entire product.
///
/// # Panics
/// Output supports must have odd weight; identity syndrome strings are unsupported.
#[must_use]
pub fn to_stim(ops: &[Op]) -> String {
    let mut text = String::from(
        "# Stim dialect: T = diag(1,exp(i*pi/4)).\n# EXP_VAL P Q: expectations of P and Q, reported separately; no collapse.\n# Logical W check = (<P> + <Q>)/sqrt(2) on shots with all DETECTOR values zero.\n# ! on the first factor negates a product. Acceptance: all DETECTOR values zero.\n",
    );
    for op in ops {
        let line = match op {
            Op::PZ(q) => format!("R {}", q.0),
            Op::PX(q) => format!("RX {}", q.0),
            Op::CX(a, b) => format!("CX {} {}", a.0, b.0),
            Op::CZ(a, b) => format!("CZ {} {}", a.0, b.0),
            Op::S(q) => format!("S {}", q.0),
            Op::Sdg(q) => format!("S_DAG {}", q.0),
            Op::Z(q) => format!("Z {}", q.0),
            Op::T(q) => format!("T {}", q.0),
            Op::ZError(q, p) => format!("Z_ERROR({p}) {}", q.0),
            Op::MeasureX(support) => {
                assert!(!support.is_empty(), "syndrome string must be nonempty");
                format!("MPP {}\nDETECTOR rec[-1]", product(support, 'X'))
            }
            Op::ExpectW(support) => {
                assert!(
                    support.len() % 2 == 1,
                    "logical output must have odd weight"
                );
                let sign = if support.len() % 4 == 1 { "" } else { "!" };
                format!(
                    "EXP_VAL {} {sign}{}",
                    product(support, 'X'),
                    product(support, 'Y')
                )
            }
        };
        writeln!(text, "{line}").expect("writing to String cannot fail");
    }
    text
}

fn product(support: &[QubitId], pauli: char) -> String {
    support
        .iter()
        .map(|q| format!("{pauli}{}", q.0))
        .collect::<Vec<_>>()
        .join("*")
}
