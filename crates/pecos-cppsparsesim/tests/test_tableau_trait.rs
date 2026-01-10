// Test the StabilizerTableauSimulator trait implementation for CppSparseStab

use pecos_core::{qid, qid2};
use pecos_cppsparsesim::CppSparseStab;
use pecos_qsim::{CliffordGateable, StabilizerTableauSimulator};

#[test]
fn test_cpp_sparse_stab_tableau_trait() {
    let mut sim = CppSparseStab::new(2);

    // Apply Bell state preparation
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));

    // Test that we can access tableaux through the trait
    let stab = sim.stab_tableau();
    let destab = sim.destab_tableau();
    let full = sim.full_tableau();

    // Verify the stabilizers contain expected patterns
    assert!(stab.contains("XX")); // Bell state stabilizer
    assert!(stab.contains("ZZ")); // Bell state stabilizer

    // Verify destabilizers
    assert!(destab.contains('X'));
    assert!(destab.contains('Z'));

    // Verify full tableau contains both sections
    assert!(full.contains("Stabilizers:"));
    assert!(full.contains("Destabilizers:"));

    // Test num_qubits through the trait
    assert_eq!(sim.num_qubits(), 2);
}

#[test]
fn test_tableau_trait_generic_function() {
    fn generic_tableau_test<T>(mut sim: T)
    where
        T: StabilizerTableauSimulator + CliffordGateable,
    {
        sim.x(&qid(0));
        let stab = sim.stab_tableau();
        assert!(stab.contains('Z')); // X gate changes Z stabilizer
        assert_eq!(sim.num_qubits(), 1);
    }

    let sim = CppSparseStab::new(1);
    generic_tableau_test(sim);
}
