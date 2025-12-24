//! `PyMatching` decoder integration tests
//!
//! This file includes all `PyMatching`-specific tests from the pymatching/ subdirectory.

#[path = "pymatching/pymatching_tests.rs"]
mod pymatching_tests;

#[path = "pymatching/pymatching_comprehensive_tests.rs"]
mod pymatching_comprehensive_tests;

#[path = "pymatching/pymatching_core_tests.rs"]
mod pymatching_core_tests;

#[path = "pymatching/pymatching_integration_tests.rs"]
mod pymatching_integration_tests;

#[path = "pymatching/pymatching_noise_tests.rs"]
mod pymatching_noise_tests;

#[path = "pymatching/pymatching_petgraph_tests.rs"]
mod pymatching_petgraph_tests;

#[path = "pymatching/pymatching_edge_case_tests.rs"]
mod pymatching_edge_case_tests;

#[path = "pymatching/surface_code_tests.rs"]
mod surface_code_tests;

#[path = "pymatching/pymatching_check_matrix_tests.rs"]
mod pymatching_check_matrix_tests;

#[path = "pymatching/pymatching_bit_packed_tests.rs"]
mod pymatching_bit_packed_tests;

#[path = "pymatching/pymatching_stim_tests.rs"]
mod pymatching_stim_tests;

#[path = "pymatching/pymatching_fault_id_tests.rs"]
mod pymatching_fault_id_tests;
