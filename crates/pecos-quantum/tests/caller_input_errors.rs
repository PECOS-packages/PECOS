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
fn explicit_matrix_columns_reject_overflow() {
    assert!(SymplecticMatrix::zeros(0, 1 << (usize::BITS - 1)).is_err());
}

#[test]
fn explicit_matrix_rows_reject_capacity_overflow() {
    assert!(
        SymplecticMatrix::zeros(usize::MAX, 0)
            .unwrap_err()
            .to_string()
            .contains(&usize::MAX.to_string())
    );
}
