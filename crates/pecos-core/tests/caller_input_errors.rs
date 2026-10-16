use pecos_core::{ParsePauliStringError, Pauli, PauliString, QuarterPhase, QubitId};

#[test]
fn duplicate_constructor_returns_error() {
    let result = PauliString::with_phase_and_paulis(
        QuarterPhase::PlusOne,
        vec![(Pauli::X, QubitId(7)), (Pauli::Z, QubitId(7))],
    );
    assert_eq!(
        result,
        Err(ParsePauliStringError::DuplicateQubit { qubit: 7 })
    );
}

#[test]
fn duplicate_decomposed_returns_error() {
    assert_eq!(
        PauliString::from_decomposed(QuarterPhase::PlusOne, [7], [], [7]),
        Err(ParsePauliStringError::DuplicateQubit { qubit: 7 })
    );
}

#[test]
fn explicit_width_returns_error_and_natural_width_is_inclusive() {
    let pauli = PauliString::x(1_000);
    assert_eq!(pauli.to_dense_str_width(1_000).unwrap_err().qubit, 1_000);
    assert_eq!(pauli.pauli_str().len(), 1_001);
    assert_eq!(pauli.to_dense_str().len(), 1_002);
}

#[test]
fn clifford_image_width_rejected_before_application() {
    let mut clifford = pecos_core::clifford_rep::CliffordRep::identity(1);
    assert_eq!(
        clifford
            .set_x_image(0, PauliString::x(7))
            .unwrap_err()
            .qubit,
        7
    );
    assert_eq!(
        clifford
            .set_z_image(7, PauliString::z(0))
            .unwrap_err()
            .qubit,
        7
    );
    assert_eq!(
        clifford.apply(&PauliString::xs(&[0, 7])),
        PauliString::xs(&[0, 7])
    );
}

#[test]
fn inverse_trig_reports_the_original_invalid_input() {
    use pecos_core::Angle64;
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for result in [
            Angle64::try_asin(value),
            Angle64::try_acos(value),
            Angle64::try_atan(value),
            Angle64::try_atan2(value, 1.0),
            Angle64::try_atan2(1.0, value),
        ] {
            let error = result.unwrap_err();
            assert_eq!(error.value.to_bits(), value.to_bits());
            assert!(error.to_string().contains(&value.to_string()));
        }
    }
    for value in [-2.0, 2.0] {
        for result in [Angle64::try_asin(value), Angle64::try_acos(value)] {
            let message = result.unwrap_err().to_string();
            assert!(message.contains(&value.to_string()) && message.contains("-1..=1"));
        }
    }
}

#[test]
fn controlled_lowerings_reject_nan_instead_of_panicking() {
    use pecos_core::{QubitId, controlled_rotations::*};
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let c = QubitId(0);
        let t = QubitId(1);
        let messages = [
            lower_crz(value, c, t).unwrap_err().to_string(),
            lower_crx(value, c, t).unwrap_err().to_string(),
            lower_cry(value, c, t).unwrap_err().to_string(),
            lower_cphase(value, c, t).unwrap_err().to_string(),
            lower_phase(value, &[c, t]).unwrap_err().to_string(),
        ];
        for message in messages {
            assert!(
                message.contains(&value.to_string()) && message.contains("radians"),
                "{message}"
            );
        }
    }
}
