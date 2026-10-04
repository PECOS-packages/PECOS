//! Caller-side decomposition for every Selene runtime route, in execution order.

use crate::runtime::{Result, RuntimeError};
use pecos_qis_ffi_types::QuantumOp;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// The only quantum operations with a Selene runtime entry point.
#[derive(Debug, Clone)]
pub(crate) enum NativeOp {
    Rxy(f64, f64, usize),
    Rz(f64, usize),
    Rzz(f64, usize, usize),
    Rpp(f64, f64, usize, usize),
    Reset(usize),
    Measure(usize, usize),
    MeasureLeaked(usize, usize),
}

impl NativeOp {
    pub(crate) fn quantum_op(&self) -> QuantumOp {
        match *self {
            Self::Rxy(a, b, q) => QuantumOp::RXY(a, b, q),
            Self::Rz(a, q) => QuantumOp::RZ(a, q),
            Self::Rzz(a, q, r) => QuantumOp::RZZ(a, q, r),
            Self::Rpp(a, b, q, r) => QuantumOp::RXYXY2Q(a, b, q, r),
            Self::Reset(q) => QuantumOp::Reset(q),
            Self::Measure(q, r) => QuantumOp::Measure(q, r),
            Self::MeasureLeaked(q, r) => QuantumOp::MeasureLeaked(q, r),
        }
    }
}

pub(crate) enum RuntimeInput {
    Native(NativeOp),
    Decomposed(Vec<NativeOp>),
    Idle { duration: f64, qubit: usize },
}

/// Exhaustive ingress classification: a new `QuantumOp` must be classified here.
/// References below are to Quantinuum qir-qis/src/decompose.rs.
pub(crate) fn classify(op: &QuantumOp) -> Result<RuntimeInput> {
    use NativeOp::{Rxy, Rz, Rzz};
    Ok(RuntimeInput::Decomposed(match *op {
        // define_h_gate, lines 162-202.
        QuantumOp::H(q) => vec![Rxy(FRAC_PI_2, -FRAC_PI_2, q), Rz(PI, q)],
        // define_x_gate, lines 205-237.
        QuantumOp::X(q) => vec![Rxy(PI, 0.0, q)],
        // define_y_gate, lines 240-272.
        QuantumOp::Y(q) => vec![Rxy(PI, FRAC_PI_2, q)],
        // define_z_gate, lines 275-306.
        QuantumOp::Z(q) => vec![Rz(PI, q)],
        // define_s_gate, lines 309-340.
        QuantumOp::S(q) => vec![Rz(FRAC_PI_2, q)],
        // define_s_adj_gate, lines 343-374.
        QuantumOp::Sdg(q) => vec![Rz(-FRAC_PI_2, q)],
        // define_t_gate, lines 377-408.
        QuantumOp::T(q) => vec![Rz(FRAC_PI_4, q)],
        // define_t_adj_gate, lines 411-442.
        QuantumOp::Tdg(q) => vec![Rz(-FRAC_PI_4, q)],
        // define_rx_gate, lines 445-479.
        QuantumOp::RX(theta, q) => vec![Rxy(theta, 0.0, q)],
        // define_ry_gate, lines 482-520.
        QuantumOp::RY(theta, q) => vec![Rxy(theta, FRAC_PI_2, q)],
        // define_cz_gate, lines 523-572.
        QuantumOp::CZ(c, t) => vec![Rzz(FRAC_PI_2, c, t), Rz(-FRAC_PI_2, t), Rz(-FRAC_PI_2, c)],
        // define_cx_gate, lines 575-637.
        QuantumOp::CX(c, t) => cx(c, t),
        // Derived: CY = S(target) CX Sdg(target), in matrix order.
        QuantumOp::CY(c, t) => {
            let mut sequence = vec![Rz(-FRAC_PI_2, t)];
            sequence.extend(cx(c, t));
            sequence.push(Rz(FRAC_PI_2, t));
            sequence
        }
        // Derived: H = RY(-pi/4) X RY(pi/4), in matrix order.
        QuantumOp::CH(c, t) => {
            let mut sequence = vec![Rxy(FRAC_PI_4, FRAC_PI_2, t)];
            sequence.extend(cx(c, t));
            sequence.push(Rxy(-FRAC_PI_4, FRAC_PI_2, t));
            sequence
        }
        // Reuse controlled_rotations.rs:62-73, including its 4pi wrap parity.
        QuantumOp::CRZ(theta, c, t) => {
            pecos_core::controlled_rotations::lower_crz(theta, c.into(), t.into())
                .into_iter()
                .map(|gate| crz_native_gate(&gate))
                .collect::<Result<Vec<_>>>()?
        }
        // define_ccx_gate, lines 644-773.
        QuantumOp::CCX(a, b, t) => vec![
            Rxy(PI, -FRAC_PI_2, t),
            Rzz(FRAC_PI_2, b, t),
            Rxy(FRAC_PI_4, FRAC_PI_2, t),
            Rzz(FRAC_PI_2, a, t),
            Rxy(FRAC_PI_4, 0.0, t),
            Rzz(FRAC_PI_2, b, t),
            Rxy(FRAC_PI_4, -FRAC_PI_2, t),
            Rzz(FRAC_PI_2, a, t),
            Rxy(PI, FRAC_PI_4, a),
            Rxy(-3.0 * FRAC_PI_4, PI, t),
            Rzz(FRAC_PI_4, a, b),
            Rz(PI, t),
            Rxy(PI, -FRAC_PI_4, a),
            Rz(-3.0 * FRAC_PI_4, b),
            Rz(FRAC_PI_4, a),
        ],
        // QIS ZZ means SZZ (ccengine.rs), diag(1,i,i,1), hence Rzz(pi/2)
        // up to global phase; see CliffordGateable::szz's matrix documentation.
        QuantumOp::ZZ(a, b) => vec![Rzz(FRAC_PI_2, a, b)],
        QuantumOp::RXY(a, b, q) => return Ok(RuntimeInput::Native(Rxy(a, b, q))),
        QuantumOp::RZ(a, q) => return Ok(RuntimeInput::Native(Rz(a, q))),
        QuantumOp::RZZ(a, q, r) => return Ok(RuntimeInput::Native(Rzz(a, q, r))),
        QuantumOp::RXYXY2Q(a, b, q, r) => {
            return Ok(RuntimeInput::Native(NativeOp::Rpp(a, b, q, r)));
        }
        QuantumOp::Reset(q) => return Ok(RuntimeInput::Native(NativeOp::Reset(q))),
        QuantumOp::Measure(q, r) => return Ok(RuntimeInput::Native(NativeOp::Measure(q, r))),
        QuantumOp::MeasureLeaked(q, r) => {
            return Ok(RuntimeInput::Native(NativeOp::MeasureLeaked(q, r)));
        }
        QuantumOp::Idle(duration, qubit) => return Ok(RuntimeInput::Idle { duration, qubit }),
    }))
}

fn crz_native_gate(gate: &pecos_core::Gate) -> Result<NativeOp> {
    use pecos_core::gate_type::GateType;
    match (
        gate.gate_type,
        gate.qubits.as_slice(),
        gate.angles.as_slice(),
    ) {
        (GateType::Z, [q], []) => Ok(NativeOp::Rz(PI, q.index())),
        (GateType::RZ, [q], [angle]) => Ok(NativeOp::Rz(angle.to_radians_signed(), q.index())),
        (GateType::RZZ, [q, r], [angle]) => Ok(NativeOp::Rzz(
            angle.to_radians_signed(),
            q.index(),
            r.index(),
        )),
        other => Err(RuntimeError::ExecutionError(format!(
            "CRZ lowering emitted unsupported gate {other:?}"
        ))),
    }
}

fn cx(c: usize, t: usize) -> Vec<NativeOp> {
    use NativeOp::{Rxy, Rz, Rzz};
    vec![
        Rxy(-FRAC_PI_2, FRAC_PI_2, t),
        Rzz(FRAC_PI_2, c, t),
        Rz(-FRAC_PI_2, c),
        Rxy(FRAC_PI_2, PI, t),
        Rz(-FRAC_PI_2, t),
    ]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use num_complex::Complex64;
    use pecos_core::{Angle64, QubitId};
    use pecos_random::PecosRng;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecAoS};

    pub(crate) fn apply(sim: &mut StateVecAoS, op: &QuantumOp) {
        let angle = Angle64::from_radians;
        match *op {
            QuantumOp::H(q) => {
                sim.h(&[QubitId(q)]);
            }
            QuantumOp::X(q) => {
                sim.x(&[QubitId(q)]);
            }
            QuantumOp::Y(q) => {
                sim.y(&[QubitId(q)]);
            }
            QuantumOp::Z(q) => {
                sim.z(&[QubitId(q)]);
            }
            QuantumOp::S(q) => {
                sim.sz(&[QubitId(q)]);
            }
            QuantumOp::Sdg(q) => {
                sim.szdg(&[QubitId(q)]);
            }
            QuantumOp::T(q) => {
                sim.t(&[QubitId(q)]);
            }
            QuantumOp::Tdg(q) => {
                sim.tdg(&[QubitId(q)]);
            }
            QuantumOp::RX(a, q) => {
                sim.rx(angle(a), &[QubitId(q)]);
            }
            QuantumOp::RY(a, q) => {
                sim.ry(angle(a), &[QubitId(q)]);
            }
            QuantumOp::RZ(a, q) => {
                sim.rz(angle(a), &[QubitId(q)]);
            }
            QuantumOp::RXY(a, b, q) => {
                sim.rxy1q(angle(a), angle(b), &[QubitId(q)]);
            }
            QuantumOp::RZZ(a, q, r) => {
                sim.rzz(angle(a), &[(QubitId(q), QubitId(r))]);
            }
            QuantumOp::CX(q, r) => {
                sim.cx(&[(QubitId(q), QubitId(r))]);
            }
            QuantumOp::CY(q, r) => {
                sim.cy(&[(QubitId(q), QubitId(r))]);
            }
            QuantumOp::CZ(q, r) => {
                sim.cz(&[(QubitId(q), QubitId(r))]);
            }
            QuantumOp::ZZ(q, r) => {
                sim.szz(&[(QubitId(q), QubitId(r))]);
            }
            // StateVecAoS has no named CH/CRZ/CCX methods. Use their defining
            // controlled matrices/truth table, independent of the native table.
            QuantumOp::CH(q, r) => {
                let mut matrix = [[Complex64::default(); 4]; 4];
                matrix[0][0] = 1.0.into();
                matrix[1][1] = 1.0.into();
                let h = std::f64::consts::FRAC_1_SQRT_2;
                matrix[2][2] = h.into();
                matrix[2][3] = h.into();
                matrix[3][2] = h.into();
                matrix[3][3] = (-h).into();
                sim.two_qubit_unitary(q, r, matrix);
            }
            QuantumOp::CRZ(a, q, r) => {
                let mut matrix = [[Complex64::default(); 4]; 4];
                matrix[0][0] = 1.0.into();
                matrix[1][1] = 1.0.into();
                matrix[2][2] = Complex64::from_polar(1.0, -a / 2.0);
                matrix[3][3] = Complex64::from_polar(1.0, a / 2.0);
                sim.two_qubit_unitary(q, r, matrix);
            }
            QuantumOp::CCX(a, b, t) => {
                let mut state = sim.state().to_vec();
                for (i, amplitude) in sim.state().iter().enumerate() {
                    let j = if i & (1 << a) != 0 && i & (1 << b) != 0 {
                        i ^ (1 << t)
                    } else {
                        i
                    };
                    state[j] = *amplitude;
                }
                *sim = StateVecAoS::from_state(state, PecosRng::seed_from_u64(4));
            }
            QuantumOp::Reset(q) => {
                sim.pz(&[QubitId(q)]);
            }
            QuantumOp::Idle(..) => {}
            _ => panic!("unexpected oracle operation {op:?}"),
        }
    }

    #[test]
    fn unexpected_crz_output_is_an_error_not_a_panic() {
        let error = crz_native_gate(&pecos_core::Gate::h(&[0]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("CRZ") && error.contains('H'), "{error}");
        let mut malformed = pecos_core::Gate::rz(Angle64::ZERO, &[0]);
        malformed.angles.clear();
        assert!(
            crz_native_gate(&malformed)
                .unwrap_err()
                .to_string()
                .contains("RZ")
        );
    }

    #[test]
    fn every_native_decomposition_matches_original_with_one_global_phase() {
        let mut gates = vec![
            QuantumOp::H(0),
            QuantumOp::X(0),
            QuantumOp::Y(0),
            QuantumOp::Z(0),
            QuantumOp::S(0),
            QuantumOp::Sdg(0),
            QuantumOp::T(0),
            QuantumOp::Tdg(0),
        ];
        for (a, b) in [(0, 1), (1, 0), (2, 0)] {
            gates.extend([
                QuantumOp::CX(a, b),
                QuantumOp::CY(a, b),
                QuantumOp::CZ(a, b),
                QuantumOp::CH(a, b),
                QuantumOp::ZZ(a, b),
            ]);
        }
        for (a, b, c) in [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ] {
            gates.push(QuantumOp::CCX(a, b, c));
        }
        for theta in [-7.3, -2.0 * PI, -PI, -0.73, 0.0, 0.41, PI, 2.0 * PI, 9.1] {
            gates.extend([QuantumOp::RX(theta, 0), QuantumOp::RY(theta, 0)]);
            for (a, b) in [(0, 1), (1, 0), (2, 0)] {
                gates.push(QuantumOp::CRZ(theta, a, b));
            }
        }
        for gate in gates {
            let RuntimeInput::Decomposed(sequence) = classify(&gate).unwrap() else {
                panic!("expected decomposition for {gate:?}");
            };
            let mut phase = None;
            // All eight basis states (including a spectator for smaller gates),
            // plus three fixed, complex, entangled superpositions.
            for input in 0..11 {
                let mut original = StateVecAoS::new(3);
                if input < 8 {
                    original.prepare_computational_basis(input);
                } else {
                    let offset = f64::from(u32::try_from(input - 8).unwrap());
                    for q in 0..3 {
                        let f = f64::from(u32::try_from(q).unwrap());
                        original
                            .ry(
                                Angle64::from_radians(0.37 + f * 0.43 + offset * 0.19),
                                &[QubitId(q)],
                            )
                            .rz(
                                Angle64::from_radians(-0.61 + f * 0.29 - offset * 0.23),
                                &[QubitId(q)],
                            );
                    }
                    original.cx(&[(QubitId(0), QubitId(1)), (QubitId(1), QubitId(2))]);
                }
                let mut native = original.clone();
                apply(&mut original, &gate);
                for op in &sequence {
                    apply(&mut native, &op.quantum_op());
                }
                // Angle64 stores rotations modulo 2pi. RZ/RZZ use signed half
                // angles (state_vec_aos.rs:953,1213); a wrap can change only a
                // global sign. Fix ONE phase across every input for this gate.
                let phase = *phase.get_or_insert_with(|| {
                    let index = original
                        .state()
                        .iter()
                        .position(|a| a.norm() > 1e-8)
                        .unwrap();
                    native.state()[index] / original.state()[index]
                });
                assert!((phase.norm() - 1.0).abs() < 1e-10);
                for (a, b) in original.state().iter().zip(native.state()) {
                    assert!(
                        (*a * phase - b).norm() < 1e-10,
                        "{gate:?}, input {input}: {a} vs {b}, phase {phase}"
                    );
                }
            }
        }
    }
}
