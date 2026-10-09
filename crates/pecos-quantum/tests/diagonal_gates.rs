//! Dense regression oracles for named diagonal gates and gate adjoints.
use pecos_core::gate_type::GateType;
use pecos_core::unitary_rep::RotationType;
use pecos_core::{Angle64, Gate, PhaseGateError, Unitary, UnitaryRep};
use pecos_quantum::unitary_matrix::{ToMatrix, to_matrix_with_size};
use pecos_random::{PecosRng, RngExt};

fn gate_rep(gate: &Gate) -> Option<UnitaryRep> {
    let a = &gate.angles;
    let rotation = match gate.gate_type {
        GateType::RX => Some(RotationType::RX),
        GateType::RY => Some(RotationType::RY),
        GateType::RZ => Some(RotationType::RZ),
        GateType::RXX => Some(RotationType::RXX),
        GateType::RYY => Some(RotationType::RYY),
        GateType::RZZ => Some(RotationType::RZZ),
        _ => None,
    };
    let unitary = if let Some(rotation_type) = rotation {
        Unitary::Rotation {
            rotation_type,
            angle: a[0],
        }
    } else {
        match gate.gate_type {
            GateType::RXY1Q => Unitary::RXY1Q {
                theta: a[0],
                phi: a[1],
            },
            GateType::U => Unitary::U3 {
                theta: a[0],
                phi: a[1],
                lambda: a[2],
            },
            GateType::RXXRYYRZZ => Unitary::RXXRYYRZZ {
                alpha: a[0],
                beta: a[1],
                gamma: a[2],
            },
            GateType::U2q => Unitary::U2q {
                before: [[a[0], a[1], a[2]], [a[3], a[4], a[5]]],
                interaction: [a[6], a[7], a[8]],
                after: [[a[9], a[10], a[11]], [a[12], a[13], a[14]]],
            },
            // This gate has no Unitary descriptor; use its exact axis conjugation.
            GateType::RXYXY2Q => {
                let parts = [
                    Gate::rz(-a[1], &gate.qubits),
                    Gate::rxx(a[0], &[(gate.qubits[0], gate.qubits[1])]),
                    Gate::rz(a[1], &gate.qubits),
                ];
                let mut reps = Vec::new();
                for part in parts {
                    for operands in part.qubits.chunks(part.gate_type.quantum_arity()) {
                        reps.push(gate_rep(&Gate::with_angles(
                            part.gate_type,
                            part.angles.clone(),
                            operands.to_vec(),
                        ))?);
                    }
                }
                return Some(UnitaryRep::Compose(reps));
            }
            GateType::MX
            | GateType::MZ
            | GateType::MeasureLeaked
            | GateType::MeasureFree
            | GateType::MPZ
            | GateType::PX
            | GateType::PZ
            | GateType::QAlloc
            | GateType::QFree
            | GateType::Idle
            | GateType::TrackedPauliMeta
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::Channel
            | GateType::Custom => return None,
            other => {
                assert_eq!(other.angle_arity(), 0, "add the matrix oracle for {other}");
                Unitary::named(other)
            }
        }
    };
    Some(UnitaryRep::Gate(
        unitary,
        gate.qubits.iter().map(pecos_core::QubitId::index).collect(),
    ))
}

fn adjoint_angle_cases(gt: GateType, rng: &mut PecosRng) -> Vec<Vec<Angle64>> {
    let count = gt.angle_arity();
    if count == 0 {
        return vec![vec![]];
    }
    let mut cases: Vec<Vec<_>> = (0..8)
        .map(|_| {
            (0..count)
                .map(|_| Angle64::from_radians(rng.random_range(0.1..6.1)))
                .collect()
        })
        .collect();
    let base = cases[0].clone();
    for angle in [
        Angle64::ZERO,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ] {
        cases.push(vec![angle; count]);
        for slot in 0..count {
            let mut angles = base.clone();
            angles[slot] = angle;
            cases.push(angles);
        }
    }
    // Include every pair to distinguish rotation-slot parity from phase/axis slots.
    for first in 0..count {
        for second in first + 1..count {
            let mut angles = base.clone();
            angles[first] = Angle64::HALF_TURN;
            angles[second] = Angle64::HALF_TURN;
            cases.push(angles);
        }
    }
    cases
}

fn expected_adjoint_phase(gate: &Gate) -> bool {
    // These are the half-angle slots in the dense matrix definitions. The other
    // slots enter through exp(i*angle), sin(angle), or cos(angle), with period 2pi.
    let slots: &[usize] = match gate.gate_type {
        GateType::RX
        | GateType::RY
        | GateType::RZ
        | GateType::RXX
        | GateType::RYY
        | GateType::RZZ
        | GateType::RXY1Q
        | GateType::RXYXY2Q
        | GateType::U => &[0],
        GateType::RXXRYYRZZ => &[0, 1, 2],
        GateType::U2q => &[0, 3, 6, 7, 8, 9, 12],
        _ => {
            assert!(
                gate.angles.is_empty(),
                "classify the angle slots of {}",
                gate.gate_type
            );
            &[]
        }
    };
    slots
        .iter()
        .filter(|&&slot| gate.angles[slot] == Angle64::HALF_TURN)
        .count()
        % 2
        == 1
}

#[test]
fn every_unitary_gate_adjoint_matches_dense_matrix() {
    let mut rng = PecosRng::seed_from_u64(0xcc2);
    let mut failures = Vec::new();
    for id in 0..=u8::MAX {
        let Ok(gt) = GateType::try_from(id) else {
            continue;
        };
        for (draw, angles) in adjoint_angle_cases(gt, &mut rng).into_iter().enumerate() {
            let gate = Gate::with_angles(
                gt,
                angles,
                (0..gt.quantum_arity())
                    .map(pecos_core::QubitId)
                    .collect::<Vec<_>>(),
            );
            let Some(rep) = gate_rep(&gate) else { continue };
            let needs_phase = expected_adjoint_phase(&gate);
            let expected = rep.to_matrix().adjoint();
            assert!((&*rep.dg().to_matrix() - &*expected).norm() < 1e-13);
            let mut adjoints = vec![("Adjoint", UnitaryRep::Adjoint(Box::new(rep.clone())))];
            // RXYXY2Q has only a composite representation; native descriptors
            // must also agree with dg()'s phase-carrying representation.
            if matches!(rep, UnitaryRep::Gate(..)) {
                adjoints.push(("dg", rep.dg()));
            }
            for (path, adjoint) in adjoints {
                match adjoint.try_decompose() {
                    Ok(gates) => {
                        let actual = to_matrix_with_size(
                            &UnitaryRep::Compose(
                                gates.iter().map(|g| gate_rep(g).unwrap()).collect(),
                            ),
                            gt.quantum_arity(),
                        );
                        let error = (&*actual - &*expected).norm();
                        if needs_phase || error >= 1e-13 {
                            let pi_slots: Vec<_> = gate
                                .angles
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &a)| (a == Angle64::HALF_TURN).then_some(i))
                                .collect();
                            failures.push(format!(
                                "{gt} {path} draw {draw}: pi slots={pi_slots:?}, expected phase error={needs_phase}, matrix error={error:.6}, error against -adjoint={:.6}",
                                (&*actual + &*expected).norm()
                            ));
                        }
                    }
                    Err(error) => {
                        if !needs_phase
                            || error
                                != (PhaseGateError::UnrepresentableGlobalPhase {
                                    phase: Angle64::HALF_TURN,
                                })
                        {
                            failures.push(format!("{gt} {path} draw {draw}: unexpected {error:?}"));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn diagonal_matrices_recognition_and_identities() {
    for (gt, angle) in [
        (GateType::CS, Angle64::QUARTER_TURN),
        (GateType::CSdg, -Angle64::QUARTER_TURN),
        (GateType::CCZ, Angle64::HALF_TURN),
    ] {
        let n = gt.quantum_arity();
        let named = Unitary::named(gt);
        let matrix = named.to_matrix();
        let oracle = Unitary::Phase {
            gamma: angle,
            num_qubits: n,
        }
        .to_matrix();
        assert!((&*matrix - &*oracle).norm() < 1e-14);
        assert_eq!(matrix.try_to_unitary(), Some(named));
        assert_eq!(
            Unitary::Phase {
                gamma: angle,
                num_qubits: n
            }
            .to_gate_type(),
            Some(gt)
        );
        assert!(!named.is_clifford());
        let rep = UnitaryRep::gate(gt, (0..n).collect::<Vec<_>>());
        assert_eq!(rep.is_hermitian(), gt == GateType::CCZ);
        assert!(!rep.is_clifford());
        assert!(rep.to_ascii(n).contains(&gt.to_string()));
        let reversed = UnitaryRep::gate(gt, (0..n).rev().collect::<Vec<_>>());
        assert!((&*matrix - &*reversed.to_matrix()).norm() < 1e-14);
    }
    let cs = Unitary::named(GateType::CS).to_matrix();
    let csdg = Unitary::named(GateType::CSdg).to_matrix();
    let ccz = Unitary::named(GateType::CCZ).to_matrix();
    assert!((&*(&cs * &csdg) - nalgebra::DMatrix::identity(4, 4)).norm() < 1e-14);
    assert!((&*(&ccz * &ccz) - nalgebra::DMatrix::identity(8, 8)).norm() < 1e-14);
    assert!((&*(&cs * &cs) - &*Unitary::named(GateType::CZ).to_matrix()).norm() < 1e-14);
}

#[test]
fn diagonal_operator_simplification_and_parsing() {
    for (name, gt, inverse) in [
        ("CS", GateType::CS, GateType::CSdg),
        ("CSdg", GateType::CSdg, GateType::CS),
        ("CCZ", GateType::CCZ, GateType::CCZ),
    ] {
        let n = gt.quantum_arity();
        let operands = (0..n).map(|q| q.to_string()).collect::<Vec<_>>().join(" ");
        let rep: UnitaryRep = format!("{name} {operands}").parse().unwrap();
        assert_eq!(rep, UnitaryRep::gate(gt, (0..n).collect::<Vec<_>>()));
        assert_eq!(
            rep.dg(),
            UnitaryRep::gate(inverse, (0..n).collect::<Vec<_>>())
        );
        let reversed = UnitaryRep::gate(inverse, (0..n).rev().collect::<Vec<_>>());
        assert!((rep * reversed).simplify().is_identity());
    }
}

#[test]
fn cz_products_keep_dev_composition_and_operand_support() {
    for right_qubits in [vec![0, 1], vec![1, 0]] {
        let left = UnitaryRep::gate(GateType::CZ, vec![0, 1]);
        let right = UnitaryRep::gate(GateType::CZ, right_qubits);
        let product = UnitaryRep::Compose(vec![left.clone(), right.clone()]);
        assert_eq!(product.simplify(), product);
        assert_eq!((right * left).simplify(), product);
        assert_eq!(product.simplify().qubits(), vec![0, 1]);
    }
    let cs = UnitaryRep::gate(GateType::CS, vec![0, 1]);
    assert_eq!(
        (cs.clone() * cs).simplify(),
        UnitaryRep::gate(GateType::CZ, vec![0, 1])
    );
}
