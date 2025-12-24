use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, DensityMatrix};
use std::f64::consts::PI;

fn main() {
    println!("Density Matrix Simulator Example");
    println!("================================\n");

    // Create a 2-qubit system
    let mut dm = DensityMatrix::new(2);
    println!("Initial state |00⟩:");
    println!("{dm}");

    // Create a Bell state
    dm.h(0).cx(0, 1);
    println!("\nBell state (|00⟩ + |11⟩)/√2:");
    println!("{dm}");

    // Apply some more gates
    dm.sz(0).h(1);
    println!("\nAfter applying S to qubit 0 and H to qubit 1:");
    println!("{dm}");

    // Create a noisy state with depolarizing noise
    let mut noisy_dm = DensityMatrix::new(1);
    noisy_dm.h(0);
    println!("\nSingle qubit in |+⟩ state:");
    println!("{noisy_dm}");

    // Apply depolarizing noise
    noisy_dm.apply_depolarizing_noise(0, 0.5);
    println!("\nAfter 50% depolarizing noise:");
    println!("{noisy_dm}");

    // Create a maximally mixed state
    let mut mixed = DensityMatrix::new(2);
    mixed.prepare_maximally_mixed();
    println!("\nMaximally mixed 2-qubit state:");
    println!("{mixed}");

    // Non-Clifford gates (rotations)
    let mut rotated = DensityMatrix::new(1);
    rotated.rx(PI / 4.0, 0);
    println!("\nState after Rx(π/4):");
    println!("{rotated}");

    // Show the purity
    println!("\nPurity of pure state: {}", rotated.purity());
    println!("Purity of mixed state: {}", mixed.purity());

    // Export to different formats
    println!("\n\nExporting Density Matrix in Different Formats");
    println!("=============================================\n");

    // Create a simple Bell state for demonstration
    let mut bell = DensityMatrix::new(2);
    bell.h(0).cx(0, 1);

    // Get the 2D density matrix
    let rho_2d = bell.get_density_matrix();
    println!("2D Vector representation of Bell state:");
    println!("- Dimension: {}x{}", rho_2d.len(), rho_2d[0].len());
    println!("- Element [0,0]: {}", rho_2d[0][0]);
    println!("- Element [0,3]: {}", rho_2d[0][3]);
    println!("- Element [3,0]: {}", rho_2d[3][0]);
    println!("- Element [3,3]: {}", rho_2d[3][3]);

    // Get the flattened density matrix
    let rho_flat = bell.get_flattened_density_matrix();
    println!("\nFlattened row-major representation of Bell state:");
    println!("- Size: {}", rho_flat.len());
    println!("- Element [0]: {}", rho_flat[0]);
    println!("- Element [3]: {}", rho_flat[3]);
    println!("- Element [12]: {}", rho_flat[12]);
    println!("- Element [15]: {}", rho_flat[15]);

    // Usage examples in other languages
    println!("\nUsage example in Python:");
    println!("```python");
    println!("# Using the density matrix in NumPy:");
    println!("import numpy as np");
    println!("# Assuming we've received the density matrix from Rust as a nested list");
    println!("density_matrix = [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]");
    println!("rho = np.array(density_matrix, dtype=complex)");
    println!("eigenvalues = np.linalg.eigvals(rho)");
    println!("print(\"Eigenvalues:\", eigenvalues)");
    println!("```");
}
