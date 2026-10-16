use pecos_core::PauliString;
use pecos_quantum::{PauliGroup, PauliSequence, PauliStabilizerGroup, SymplecticMatrix};

#[test]
fn generator_index_returns_error() {
    let group = PauliGroup::new(vec![PauliString::z(0)]).unwrap();
    assert!(
        group
            .generator_order(1)
            .unwrap_err()
            .to_string()
            .contains("index 1")
    );
}

#[test]
fn group_order_overflow_returns_error() {
    let group = PauliGroup::new((0..64).map(PauliString::z).collect()).unwrap();
    assert!(
        group
            .group_order()
            .unwrap_err()
            .to_string()
            .contains("2^64")
    );
}

#[test]
fn enumeration_limit_returns_error() {
    let group = PauliGroup::new((0..31).map(PauliString::z).collect()).unwrap();
    assert!(group.elements().is_err());
}

#[test]
fn stabilizer_mask_returns_error() {
    let group = PauliStabilizerGroup::new(vec![PauliString::z(0)]).unwrap();
    assert!(group.element(2).unwrap_err().to_string().contains("mask 2"));
}

#[test]
fn centralizer_width_returns_error() {
    let sequence = PauliSequence::new(vec![PauliString::z(5)]);
    assert!(
        sequence
            .centralizer_in(3)
            .unwrap_err()
            .to_string()
            .contains("qubit 5")
    );
}

#[test]
fn centralizer_x_dimensions_reject_overflow() {
    let sequence = PauliSequence::new(vec![PauliString::x(0)]);
    let width = 1 << (usize::BITS - 1);
    assert!(sequence.centralizer_in(width).is_err());
}

#[test]
fn centralizer_z_dimensions_reject_overflow() {
    let sequence = PauliSequence::new(vec![PauliString::z(0)]);
    let width = 1 << (usize::BITS - 1);
    assert!(sequence.centralizer_in(width).is_err());
}

#[test]
fn symplectic_x_dimensions_reject_overflow() {
    let sequence = PauliSequence::new(vec![PauliString::x(0)]);
    let width = 1 << (usize::BITS - 1);
    assert!(SymplecticMatrix::from_pauli_sequence_ignoring_phase(&sequence, width).is_err());
}

#[test]
fn symplectic_z_dimensions_reject_overflow() {
    let sequence = PauliSequence::new(vec![PauliString::z(0)]);
    let width = 1 << (usize::BITS - 1);
    assert!(SymplecticMatrix::from_pauli_sequence_ignoring_phase(&sequence, width).is_err());
}

#[test]
fn dag_crz_rejects_nan_even_without_targets() {
    let mut circuit = pecos_quantum::DagCircuit::new();
    for pairs in [&[][..], &[(0_usize, 1_usize)][..]] {
        let error = circuit.crz(f64::NAN, pairs).unwrap_err();
        assert!(error.to_string().contains("NaN radians"));
        assert_eq!(circuit.gate_count(), 0);
    }
}

#[test]
fn nonfinite_matrices_cannot_be_mistaken_for_named_gates() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let matrix = pecos_quantum::UnitaryMatrix::from(nalgebra::DMatrix::from_element(
            2,
            2,
            num_complex::Complex64::new(value, 0.0),
        ));
        assert!(matrix.try_to_unitary().is_none());
    }
}

#[test]
fn generator_mutations_reject_bad_indices_without_changing_group() {
    let mut group = PauliGroup::new(vec![PauliString::z(0)]).unwrap();
    let mut stabilizers = PauliStabilizerGroup::new(vec![PauliString::z(0)]).unwrap();
    for index in [1, usize::MAX] {
        for error in [
            group
                .multiply_by(index, &PauliString::x(0))
                .unwrap_err()
                .to_string(),
            group.remove_generator(index).unwrap_err().to_string(),
            stabilizers
                .multiply_by(index, &PauliString::x(0))
                .unwrap_err()
                .to_string(),
            stabilizers.remove_generator(index).unwrap_err().to_string(),
        ] {
            assert!(error.contains(&format!("index {index}")), "{error}");
        }
        assert_eq!(group.num_generators(), 1);
        assert_eq!(stabilizers.num_generators(), 1);
    }
}

#[test]
fn tick_crz_rejects_nan_without_mutation() {
    let mut circuit = pecos_quantum::TickCircuit::new();
    let mut tick = circuit.tick();
    let error = tick.crz(f64::NAN, &[(0_usize, 1_usize)]).err().unwrap();
    assert!(error.to_string().contains("NaN radians"));
    assert_eq!(circuit.gate_count(), 0);
}

#[test]
fn merge_still_validates_phases_after_custom_clifford_images() {
    let source = PauliStabilizerGroup::new(vec![PauliString::x(0)]).unwrap();
    let mut image = PauliString::x(0);
    image.set_phase(pecos_core::QuarterPhase::PlusI);
    let mut clifford = pecos_core::clifford_rep::CliffordRep::identity(1);
    clifford.set_x_image(0, image).unwrap();
    let transformed = source.apply_clifford(&clifford);
    let mut target = PauliStabilizerGroup::new(vec![PauliString::z(1)]).unwrap();
    assert!(matches!(
        target.merge(&transformed),
        Err(pecos_quantum::PauliStabilizerGroupError::NonRealPhase { .. })
    ));
    assert_eq!(target.num_generators(), 1);
}
