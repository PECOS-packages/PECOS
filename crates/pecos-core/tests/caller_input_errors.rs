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
