//! Stabilizer phase validation belongs to verification, not construction.

use pecos_core::pauli::algebra::i;
use pecos_core::pauli::*;
use pecos_qec::StabilizerCodeSpec;

#[test]
fn imaginary_stabilizer_is_rejected_during_verification() {
    let spec = StabilizerCodeSpec::new(2, vec![i * Z(0)], vec![Z(1)], vec![X(1)]).unwrap();
    let expected = "Stabilizer 0 has non-real phase; its square is -I (empty code space)";
    assert_eq!(spec.verify().unwrap_err().to_string(), expected);
    assert_eq!(
        spec.verify_as_complete_code().unwrap_err().to_string(),
        expected
    );
    assert_eq!(
        StabilizerCodeSpec::builder(2)
            .stabilizer_pauli(i * Z(0))
            .build_verified()
            .unwrap_err()
            .to_string(),
        expected
    );
}
