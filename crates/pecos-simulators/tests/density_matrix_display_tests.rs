use num_complex::Complex64;
use pecos_core::{qid, qid2};
use pecos_simulators::CliffordGateable;
use pecos_simulators::DensityMatrix;

#[test]
fn test_get_density_matrix() {
    // Create a Bell state
    let mut state = DensityMatrix::new(2);
    state.h(&qid(0)).cx(&qid2(0, 1));

    // Get the density matrix
    let rho = state.get_density_matrix();

    // Bell state density matrix should be 1/2 * (|00⟩⟨00| + |00⟩⟨11| + |11⟩⟨00| + |11⟩⟨11|)
    // Which means entries at [0,0], [0,3], [3,0], and [3,3] should be 0.5
    assert!((rho[0][0].re - 0.5).abs() < 1e-10);
    assert!((rho[0][3].re - 0.5).abs() < 1e-10);
    assert!((rho[3][0].re - 0.5).abs() < 1e-10);
    assert!((rho[3][3].re - 0.5).abs() < 1e-10);

    // All other entries should be zero
    for (i, rho_row) in rho.iter().enumerate() {
        for (j, rho_ij) in rho_row.iter().enumerate() {
            let is_bell_element = (i == 0 || i == 3) && (j == 0 || j == 3);
            if !is_bell_element {
                assert!(rho_ij.norm() < 1e-10);
            }
        }
    }
}

#[test]
fn test_density_matrix_to_string() {
    // Create a simple state
    let mut state = DensityMatrix::new(1);
    state.h(&qid(0));

    // Get the string representation
    let matrix_str = state.density_matrix_to_string(2, 1e-10);

    // The string should represent a density matrix for |+⟩
    assert!(matrix_str.contains("0.50"));
    assert!(matrix_str.contains("[0.50, 0.50]"));
    assert!(matrix_str.contains("[0.50, 0.50]"));
}

#[test]
fn test_mixed_state_representation() {
    // Create a maximally mixed state
    let mut state = DensityMatrix::new(2);
    state.prepare_maximally_mixed();

    // Get the density matrix
    let rho = state.get_density_matrix();

    // A maximally mixed state should have 0.25 on the diagonal
    // and zeros elsewhere
    for (i, rho_row) in rho.iter().enumerate() {
        for (j, rho_ij) in rho_row.iter().enumerate() {
            if i == j {
                assert!((rho_ij.re - 0.25).abs() < 1e-10);
            } else {
                assert!(rho_ij.norm() < 1e-10);
            }
        }
    }

    // Check the purity
    let purity = state.purity();
    assert!((purity - 0.25).abs() < 1e-10); // Should be 1/2^n = 1/4

    // Create a pure state
    let mut pure_state = DensityMatrix::new(2);
    pure_state.prepare_computational_basis(0);

    // Check purity
    let pure_purity = pure_state.purity();
    assert!((pure_purity - 1.0).abs() < 1e-10); // Should be 1 for pure states
}

#[test]
fn test_real_world_circuit() {
    // Create a quantum circuit with a mix of gates
    let mut state = DensityMatrix::new(2);

    // Apply a sequence of gates
    state
        .h(&qid(0)) // Put qubit 0 in superposition
        .cx(&qid2(0, 1)) // Entangle qubits 0 and 1
        .z(&qid(0)) // Apply phase flip to qubit 0
        .h(&qid(1)); // Apply Hadamard to qubit 1

    // Get the density matrix
    let rho = state.get_density_matrix();

    // Print the density matrix for debugging
    let _matrix_str = state.density_matrix_to_string(4, 1e-10);
    // println!("{}", matrix_str);

    // Trace should be 1
    let trace: Complex64 = rho.iter().enumerate().map(|(i, row)| row[i]).sum();
    assert!((trace.re - 1.0).abs() < 1e-10);
    assert!(trace.im.abs() < 1e-10);

    // Density matrix should be Hermitian (ρ† = ρ)
    for (i, rho_row) in rho.iter().enumerate() {
        for (j, rho_ij) in rho_row.iter().enumerate() {
            assert!((rho_ij - rho[j][i].conj()).norm() < 1e-10);
        }
    }
}

#[test]
fn test_export_formats() {
    // Create a Bell state
    let mut state = DensityMatrix::new(2);
    state.h(&qid(0)).cx(&qid2(0, 1));

    // Test flattened representation
    let flat = state.get_flattened_density_matrix();
    assert_eq!(flat.len(), 16); // 4x4 matrix = 16 elements

    // Bell state should have values at positions 0, 3, 12, and 15
    assert!((flat[0].re - 0.5).abs() < 1e-10);
    assert!((flat[3].re - 0.5).abs() < 1e-10);
    assert!((flat[12].re - 0.5).abs() < 1e-10);
    assert!((flat[15].re - 0.5).abs() < 1e-10);

    // All other elements should be zero
    for (i, flat_i) in flat.iter().enumerate() {
        if i != 0 && i != 3 && i != 12 && i != 15 {
            assert!(flat_i.norm() < 1e-10);
        }
    }

    // Test to_string
    let default_str = state.to_string();
    assert!(default_str.contains("Density matrix (ρ):"));
    assert!(default_str.contains("0.5000"));
}
