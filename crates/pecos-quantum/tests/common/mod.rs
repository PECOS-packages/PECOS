use num_complex::Complex64;
use pecos_core::UnitaryRep;
use pecos_quantum::unitary_matrix::ToMatrix;

const PHASE_TOLERANCE: f64 = 1e-9;

/// Assert `lhs == exp(i * expected_phase) * rhs`, including the residual phase.
pub fn assert_residual_phase(
    lhs: &UnitaryRep,
    rhs: &UnitaryRep,
    expected_phase: f64,
    relationship: &str,
) {
    let lhs = lhs.to_matrix().into_inner();
    let rhs = rhs.to_matrix().into_inner();
    assert_eq!(
        lhs.shape(),
        rhs.shape(),
        "{relationship}: matrix dimensions differ"
    );
    assert!(
        lhs.iter()
            .chain(rhs.iter())
            .all(|entry| entry.re.is_finite() && entry.im.is_finite()),
        "{relationship}: matrix contains a non-finite entry"
    );

    let pivot = (0..lhs.nrows())
        .flat_map(|row| (0..lhs.ncols()).map(move |column| (row, column)))
        .find(|&(row, column)| {
            lhs[(row, column)].norm() > PHASE_TOLERANCE
                && rhs[(row, column)].norm() > PHASE_TOLERANCE
        })
        .unwrap_or_else(|| panic!("{relationship}: matrices have no jointly nonzero entry"));
    let residual = lhs[pivot] / rhs[pivot];
    assert!(
        (residual.norm() - 1.0).abs() <= PHASE_TOLERANCE,
        "{relationship}: residual {residual} is not unit modulus"
    );

    let rebuilt = &rhs * residual;
    let entrywise_error = (&lhs - rebuilt)
        .iter()
        .map(|entry| entry.norm())
        .fold(0.0, f64::max);
    assert!(
        entrywise_error <= PHASE_TOLERANCE,
        "{relationship}: matrices differ by more than a global phase; \
         max entrywise error is {entrywise_error:e}"
    );

    let expected = Complex64::from_polar(1.0, expected_phase);
    let phase_error = (residual - expected).norm();
    assert!(
        phase_error <= PHASE_TOLERANCE,
        "{relationship}: expected residual phase {expected_phase} rad, \
         observed {} rad (complex-plane error {phase_error:e})",
        residual.arg()
    );
}
