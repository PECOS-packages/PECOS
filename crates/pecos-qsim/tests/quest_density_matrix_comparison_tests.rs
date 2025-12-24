//! Comparison tests between `DensityMatrix` and `QuEST`'s `QuestDensityMatrix`
//!
//! These tests verify that our `DensityMatrix` implementation produces the same
//! results as the reference `QuEST` density matrix simulator.
//!
//! NOTE: `QuEST` has thread safety issues - run with --test-threads=1

use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, DensityMatrix, QuantumSimulator};
use pecos_quest::QuestDensityMatrix;
use pecos_rng::PecosRng;
use std::f64::consts::PI;

const TOLERANCE: f64 = 1e-10;

fn assert_close(a: f64, b: f64, msg: &str) {
    assert!(
        (a - b).abs() < TOLERANCE,
        "{}: {} vs {} (diff: {})",
        msg,
        a,
        b,
        (a - b).abs()
    );
}

/// Compare probabilities for all computational basis states between simulators
fn compare_probabilities(
    dm: &DensityMatrix,
    qdm: &QuestDensityMatrix<PecosRng>,
    num_qubits: usize,
) {
    for i in 0..(1 << num_qubits) {
        let dm_prob = dm.probability(i);
        let qdm_prob = qdm.probability(i);
        assert_close(dm_prob, qdm_prob, &format!("probability({i})"));
    }
}

/// Compare purity between simulators
fn compare_purity(dm: &DensityMatrix, qdm: &QuestDensityMatrix<PecosRng>) {
    let dm_purity = dm.purity();
    let qdm_purity = qdm.purity();
    assert_close(dm_purity, qdm_purity, "purity");
}

#[test]
fn test_initial_state() {
    let num_qubits = 2;
    let dm = DensityMatrix::new(num_qubits);
    let qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_x_gate() {
    let num_qubits = 2;
    let seed = 42;

    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::with_seed(num_qubits, seed);

    dm.x(0);
    qdm.x(0);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);

    dm.x(1);
    qdm.x(1);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_y_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    dm.y(0);
    qdm.y(0);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_z_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Z on |0> should leave it unchanged
    dm.z(0);
    qdm.z(0);

    compare_probabilities(&dm, &qdm, num_qubits);

    // Create superposition first, then apply Z
    dm.h(0);
    qdm.h(0);

    dm.z(0);
    qdm.z(0);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_hadamard_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    dm.h(0);
    qdm.h(0);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);

    dm.h(1);
    qdm.h(1);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_s_gate() {
    let num_qubits = 1;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create |+> then apply S
    dm.h(0);
    qdm.h(0);

    dm.sz(0);
    qdm.sz(0);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_sdg_gate() {
    let num_qubits = 1;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    dm.h(0);
    qdm.h(0);

    dm.szdg(0);
    qdm.szdg(0);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_cx_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create Bell state
    dm.h(0);
    qdm.h(0);

    dm.cx(0, 1);
    qdm.cx(0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_cz_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Put both qubits in superposition
    dm.h(0);
    dm.h(1);
    qdm.h(0);
    qdm.h(1);

    dm.cz(0, 1);
    qdm.cz(0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_cy_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Set control to |1>
    dm.x(0);
    qdm.x(0);

    dm.cy(0, 1);
    qdm.cy(0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_swap_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Put qubit 0 in |1>
    dm.x(0);
    qdm.x(0);

    dm.swap(0, 1);
    qdm.swap(0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_rx_gate() {
    let num_qubits = 1;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    dm.rx(PI / 4.0, 0);
    qdm.rx(PI / 4.0, 0);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_ry_gate() {
    let num_qubits = 1;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    dm.ry(PI / 3.0, 0);
    qdm.ry(PI / 3.0, 0);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_ry_in_entangled_system() {
    // Test RY on qubit 0 after creating entanglement
    let num_qubits = 3;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create entanglement first
    dm.h(0);
    dm.h(1);
    dm.cx(0, 1);
    dm.h(2);
    dm.cx(1, 2);

    qdm.h(0);
    qdm.h(1);
    qdm.cx(0, 1);
    qdm.h(2);
    qdm.cx(1, 2);

    compare_probabilities(&dm, &qdm, num_qubits);

    // Now apply RY
    dm.ry(PI / 5.0, 0);
    qdm.ry(PI / 5.0, 0);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_rz_gate() {
    let num_qubits = 1;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create superposition first
    dm.h(0);
    qdm.h(0);

    dm.rz(PI / 6.0, 0);
    qdm.rz(PI / 6.0, 0);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_rzz_gate() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create superposition on both qubits
    dm.h(0);
    dm.h(1);
    qdm.h(0);
    qdm.h(1);

    dm.rzz(PI / 4.0, 0, 1);
    qdm.rzz(PI / 4.0, 0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_bell_state() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    dm.h(0);
    dm.cx(0, 1);
    qdm.h(0);
    qdm.cx(0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);

    // Should be a pure state
    assert_close(dm.purity(), 1.0, "Bell state purity");
}

#[test]
fn test_ghz_state() {
    let num_qubits = 3;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create GHZ state (|000> + |111>)/sqrt(2)
    dm.h(0);
    dm.cx(0, 1);
    dm.cx(1, 2);
    qdm.h(0);
    qdm.cx(0, 1);
    qdm.cx(1, 2);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_complex_circuit() {
    let num_qubits = 3;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Apply a complex sequence of gates
    dm.h(0);
    dm.h(1);
    dm.cx(0, 2);
    dm.rz(PI / 4.0, 1);
    dm.cy(1, 0);
    dm.rx(PI / 3.0, 2);
    dm.cz(0, 1);

    qdm.h(0);
    qdm.h(1);
    qdm.cx(0, 2);
    qdm.rz(PI / 4.0, 1);
    qdm.cy(1, 0);
    qdm.rx(PI / 3.0, 2);
    qdm.cz(0, 1);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_reset() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create some state
    dm.h(0);
    dm.cx(0, 1);
    qdm.h(0);
    qdm.cx(0, 1);

    // Reset
    dm.reset();
    qdm.reset();

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_measurement_deterministic() {
    let num_qubits = 1;
    let seed = 42;

    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::with_seed(num_qubits, seed);

    // Deterministic measurement on |0>
    let dm_result = dm.mz(0);
    let qdm_result = qdm.mz(0);

    assert_eq!(
        dm_result.outcome, qdm_result.outcome,
        "measurement outcome mismatch"
    );
    assert_eq!(
        dm_result.is_deterministic, qdm_result.is_deterministic,
        "determinism mismatch"
    );

    compare_probabilities(&dm, &qdm, num_qubits);
}

#[test]
fn test_measurement_superposition() {
    // For superposition states, we can't guarantee same outcomes without same RNG
    // But we can verify post-measurement states are valid
    let num_qubits = 1;
    let seed = 12345;

    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::with_seed(num_qubits, seed);

    // Create superposition
    dm.h(0);
    qdm.h(0);

    // Both should report 50/50 probabilities before measurement
    assert_close(dm.probability(0), 0.5, "pre-measurement prob 0");
    assert_close(dm.probability(1), 0.5, "pre-measurement prob 1");
    assert_close(qdm.probability(0), 0.5, "quest pre-measurement prob 0");
    assert_close(qdm.probability(1), 0.5, "quest pre-measurement prob 1");

    // After measurement, state should be collapsed
    let _dm_result = dm.mz(0);
    let _qdm_result = qdm.mz(0);

    // Both should be in a definite state after measurement
    let dm_prob0 = dm.probability(0);
    let dm_prob1 = dm.probability(1);
    let qdm_prob0 = qdm.probability(0);
    let qdm_prob1 = qdm.probability(1);

    // One probability should be ~1, other ~0
    assert!(
        (dm_prob0 > 0.99 && dm_prob1 < 0.01) || (dm_prob0 < 0.01 && dm_prob1 > 0.99),
        "DensityMatrix not collapsed: p0={dm_prob0}, p1={dm_prob1}"
    );
    assert!(
        (qdm_prob0 > 0.99 && qdm_prob1 < 0.01) || (qdm_prob0 < 0.01 && qdm_prob1 > 0.99),
        "QuestDensityMatrix not collapsed: p0={qdm_prob0}, p1={qdm_prob1}"
    );
}

#[test]
fn test_purity_pure_state() {
    let num_qubits = 2;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Various pure states should all have purity 1
    compare_purity(&dm, &qdm);
    assert_close(dm.purity(), 1.0, "initial purity");

    dm.h(0);
    qdm.h(0);
    compare_purity(&dm, &qdm);
    assert_close(dm.purity(), 1.0, "superposition purity");

    dm.cx(0, 1);
    qdm.cx(0, 1);
    compare_purity(&dm, &qdm);
    assert_close(dm.purity(), 1.0, "entangled purity");
}

#[test]
fn test_rotation_angles() {
    let num_qubits = 1;

    // Test various rotation angles
    let angles = [
        0.0,
        PI / 8.0,
        PI / 4.0,
        PI / 2.0,
        PI,
        3.0 * PI / 2.0,
        2.0 * PI,
    ];

    for &theta in &angles {
        let mut dm = DensityMatrix::new(num_qubits);
        let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

        dm.rx(theta, 0);
        qdm.rx(theta, 0);

        compare_probabilities(&dm, &qdm, num_qubits);
    }
}

#[test]
fn test_larger_system_4_qubits() {
    let num_qubits = 4;

    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Create a complex entangled state
    dm.h(0);
    dm.cx(0, 1);
    dm.h(2);
    dm.cx(2, 3);
    dm.cz(1, 2);
    dm.rx(PI / 3.0, 0);
    dm.ry(PI / 4.0, 3);

    qdm.h(0);
    qdm.cx(0, 1);
    qdm.h(2);
    qdm.cx(2, 3);
    qdm.cz(1, 2);
    qdm.rx(PI / 3.0, 0);
    qdm.ry(PI / 4.0, 3);

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_density_matrix_trace_is_one() {
    let num_qubits = 2;
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply various operations
    dm.h(0);
    dm.cx(0, 1);
    dm.rz(PI / 5.0, 0);

    // Check trace = sum of probabilities = 1
    let mut trace = 0.0;
    for i in 0..(1 << num_qubits) {
        trace += dm.probability(i);
    }
    assert_close(trace, 1.0, "trace should be 1");
}

#[test]
fn test_density_matrix_is_hermitian() {
    let num_qubits = 2;
    let mut dm = DensityMatrix::new(num_qubits);

    dm.h(0);
    dm.cx(0, 1);
    dm.sz(1);

    let rho = dm.get_density_matrix();

    // Check rho[i][j] == rho[j][i].conj()
    for (i, rho_row) in rho.iter().enumerate() {
        for (j, rho_ij) in rho_row.iter().enumerate() {
            let diff = (rho_ij - rho[j][i].conj()).norm();
            assert!(
                diff < TOLERANCE,
                "Not Hermitian at ({},{}): {} vs {}",
                i,
                j,
                rho_ij,
                rho[j][i].conj()
            );
        }
    }
}

#[test]
fn test_density_matrix_probabilities_sum_to_one() {
    let num_qubits = 3;
    let mut dm = DensityMatrix::new(num_qubits);

    // Create GHZ-like state
    dm.h(0);
    dm.cx(0, 1);
    dm.cx(1, 2);

    let mut sum = 0.0;
    for i in 0..(1 << num_qubits) {
        let prob = dm.probability(i);
        assert!(prob >= -TOLERANCE, "Negative probability at {i}: {prob}");
        sum += prob;
    }
    assert_close(sum, 1.0, "probabilities should sum to 1");
}

#[test]
fn test_random_circuit_comparison() {
    // Test a pseudo-random circuit to catch edge cases
    let num_qubits = 3;
    let mut dm = DensityMatrix::new(num_qubits);
    let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

    // Sequence of gates that exercises many code paths
    let ops: Vec<(&str, usize, usize)> = vec![
        ("h", 0, 0),
        ("h", 1, 0),
        ("cx", 0, 1),
        ("rz", 2, 0),
        ("h", 2, 0),
        ("cx", 1, 2),
        ("ry", 0, 0),
        ("cz", 0, 2),
        ("rx", 1, 0),
        ("swap", 0, 1),
        ("cy", 1, 2),
    ];

    for (op, q1, q2) in &ops {
        match *op {
            "h" => {
                dm.h(*q1);
                qdm.h(*q1);
            }
            "cx" => {
                dm.cx(*q1, *q2);
                qdm.cx(*q1, *q2);
            }
            "cy" => {
                dm.cy(*q1, *q2);
                qdm.cy(*q1, *q2);
            }
            "cz" => {
                dm.cz(*q1, *q2);
                qdm.cz(*q1, *q2);
            }
            "swap" => {
                dm.swap(*q1, *q2);
                qdm.swap(*q1, *q2);
            }
            "rx" => {
                dm.rx(PI / 7.0, *q1);
                qdm.rx(PI / 7.0, *q1);
            }
            "ry" => {
                dm.ry(PI / 5.0, *q1);
                qdm.ry(PI / 5.0, *q1);
            }
            "rz" => {
                dm.rz(PI / 3.0, *q1);
                qdm.rz(PI / 3.0, *q1);
            }
            _ => {}
        }
    }

    compare_probabilities(&dm, &qdm, num_qubits);
    compare_purity(&dm, &qdm);
}

#[test]
fn test_all_single_qubit_gates() {
    // Comprehensive test of all single qubit gates
    let num_qubits = 1;

    let gates: Vec<&str> = vec!["x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "sy", "sydg"];

    for gate in gates {
        let mut dm = DensityMatrix::new(num_qubits);
        let mut qdm: QuestDensityMatrix<PecosRng> = QuestDensityMatrix::new(num_qubits);

        // Start from |+> state for more interesting results
        dm.h(0);
        qdm.h(0);

        match gate {
            "x" => {
                dm.x(0);
                qdm.x(0);
            }
            "y" => {
                dm.y(0);
                qdm.y(0);
            }
            "z" => {
                dm.z(0);
                qdm.z(0);
            }
            "h" => {
                dm.h(0);
                qdm.h(0);
            }
            "s" => {
                dm.sz(0);
                qdm.sz(0);
            }
            "sdg" => {
                dm.szdg(0);
                qdm.szdg(0);
            }
            "sx" => {
                dm.sx(0);
                qdm.sx(0);
            }
            "sxdg" => {
                dm.sxdg(0);
                qdm.sxdg(0);
            }
            "sy" => {
                dm.sy(0);
                qdm.sy(0);
            }
            "sydg" => {
                dm.sydg(0);
                qdm.sydg(0);
            }
            _ => {}
        }

        compare_probabilities(&dm, &qdm, num_qubits);
    }
}
