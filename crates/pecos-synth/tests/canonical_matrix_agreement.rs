//! Ties this crate's exact `D[omega]` generators to PECOS's canonical
//! single-qubit matrix table.
//!
//! `pecos-synth` must define its generators exactly, in ring arithmetic, and so
//! cannot reuse the `f64` table in `pecos-core`. That leaves the convention as a
//! duplicated notion with two homes, which is exactly the situation that drifts
//! silently. This test evaluates the exact matrices numerically and requires
//! them to equal the canonical table entry for the corresponding `GateType`.
//!
//! If this fails, do not "fix" it by editing the expectation: one of the two
//! definitions has moved, and which one is correct is a deliberate decision.

use pecos_core::gate_type::GateType;
use pecos_synth::matrix::{Gate, Matrix};
use pecos_synth::ring::DOmega;

/// Numeric value of `omega^j` where `omega = exp(i*pi/4)`.
fn omega_pow(j: u32) -> (f64, f64) {
    let angle = std::f64::consts::FRAC_PI_4 * f64::from(j);
    (angle.cos(), angle.sin())
}

/// Evaluate one exact `D[omega]` entry as a complex number.
fn evaluate(entry: &DOmega) -> (f64, f64) {
    let coords = entry.numerator().coordinates();
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (j, c) in coords.iter().enumerate() {
        let c: f64 = c.to_string().parse().expect("coordinate fits in f64");
        let (cr, ci) = omega_pow(u32::try_from(j).expect("index fits in u32"));
        re += c * cr;
        im += c * ci;
    }
    let scale = 2f64.powf(-f64::from(entry.least_denominator_exponent()) / 2.0);
    (re * scale, im * scale)
}

#[test]
fn exact_generators_match_the_canonical_single_qubit_table() {
    const TOL: f64 = 1e-12;
    let cases = [
        (Gate::I, GateType::I),
        (Gate::X, GateType::X),
        (Gate::Y, GateType::Y),
        (Gate::Z, GateType::Z),
        (Gate::H, GateType::H),
        (Gate::S, GateType::SZ),
        (Gate::Sdg, GateType::SZdg),
        (Gate::T, GateType::T),
        (Gate::Tdg, GateType::Tdg),
    ];
    for (gate, gate_type) in cases {
        let canonical = gate_type
            .canonical_1q_matrix()
            .unwrap_or_else(|| panic!("{gate_type:?} has no canonical matrix"));
        let exact = Matrix::from_gate(gate);
        for row in 0..2 {
            for col in 0..2 {
                let (re, im) = evaluate(&exact.entries()[row][col]);
                let idx = (row * 2 + col) * 2;
                let (cre, cim) = (canonical[idx], canonical[idx + 1]);
                assert!(
                    (re - cre).abs() < TOL && (im - cim).abs() < TOL,
                    "{gate:?} vs {gate_type:?} entry [{row}][{col}]: \
                     exact ring gives ({re}, {im}), canonical table gives ({cre}, {cim})"
                );
            }
        }
    }
}
