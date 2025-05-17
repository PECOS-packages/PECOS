#[path = "integration/small_circuits.rs"]
pub mod small_circuits;

#[path = "integration/large_circuits.rs"]
pub mod large_circuits;

#[path = "integration/algorithm_tests.rs"]
pub mod algorithm_tests;

#[path = "integration/library_tests.rs"]
pub mod library_tests;

// Single test files
#[path = "integration/large_quantum_circuit_test.rs"]
pub mod large_quantum_circuit_test;

#[path = "integration/nine_qubit_circuit_test.rs"]
pub mod nine_qubit_circuit_test;

#[path = "integration/hqslib1_rzz_test.rs"]
pub mod hqslib1_rzz_test;

#[path = "integration/x_gate_measure_test.rs"]
pub mod x_gate_measure_test;

#[path = "integration/comprehensive_comparisons_test.rs"]
pub mod comprehensive_comparisons_test;

#[path = "integration/comprehensive_qasm_examples.rs"]
pub mod comprehensive_qasm_examples;

#[path = "integration/simulation_validation_test.rs"]
pub mod simulation_validation_test;
