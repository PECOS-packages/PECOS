// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! End-to-end QEC code demo using STN.
//!
//! Demonstrates the toolchain on three increasingly complex stabilizer codes:
//!
//!   - 3-qubit bit-flip code (2 stabilizers): protects against single X errors.
//!   - 8-qubit Z-rep code (7 stabilizers): scales to larger generator counts.
//!   - GHZ state with global parity (1 stabilizer of all Zs): tests
//!     entanglement-aware fidelity.
//!
//! For each code:
//!   - Prepare |`0_L`⟩ via Clifford circuit.
//!   - Verify codespace fidelity = 1.0 using `StabMps::code_state_fidelity`.
//!   - Inject per-qubit depolarizing noise via
//!     `StabMps::apply_depolarizing_all`, observe the fidelity drop.
//!   - Confirm `for_qec()` preset and `auto_grow_bond_dim` work.
//!
//! Now also includes a correct Steane [[7, 1, 3]] CSS code encoder +
//! codespace fidelity verification.

use pecos_core::QubitId;
use pecos_simulators::CliffordGateable;
use pecos_stab_tn::stab_mps::{PauliKind, StabMps};
use std::time::Instant;

/// 3-qubit bit-flip code: |`0_L`⟩ = |000⟩, |`1_L`⟩ = |111⟩.
/// Stabilizers: `Z_0Z_1`, `Z_1Z_2`.
fn bit_flip_3q_stabilizers() -> Vec<Vec<(usize, PauliKind)>> {
    vec![
        vec![(0, PauliKind::Z), (1, PauliKind::Z)],
        vec![(1, PauliKind::Z), (2, PauliKind::Z)],
    ]
}

/// N-qubit Z-repetition code: stabilizers `Z_iZ`_{i+1} for i = 0..n-2.
fn z_rep_stabilizers(n: usize) -> Vec<Vec<(usize, PauliKind)>> {
    (0..n - 1)
        .map(|i| vec![(i, PauliKind::Z), (i + 1, PauliKind::Z)])
        .collect()
}

/// Steane [[7, 1, 3]] CSS code stabilizers. Based on Hamming [7,4,3]
/// parity-check matrix
///   H = [[0,0,0,1,1,1,1],
///        [0,1,1,0,0,1,1],
///        [1,0,1,0,1,0,1]]
/// X-stabilizers = {g1, g2, g3} from each row's X-support; Z-stabilizers
/// = {g4, g5, g6} from each row's Z-support (self-dual CSS).
fn steane_stabilizers() -> Vec<Vec<(usize, PauliKind)>> {
    vec![
        // X-type
        vec![
            (3, PauliKind::X),
            (4, PauliKind::X),
            (5, PauliKind::X),
            (6, PauliKind::X),
        ],
        vec![
            (1, PauliKind::X),
            (2, PauliKind::X),
            (5, PauliKind::X),
            (6, PauliKind::X),
        ],
        vec![
            (0, PauliKind::X),
            (2, PauliKind::X),
            (4, PauliKind::X),
            (6, PauliKind::X),
        ],
        // Z-type
        vec![
            (3, PauliKind::Z),
            (4, PauliKind::Z),
            (5, PauliKind::Z),
            (6, PauliKind::Z),
        ],
        vec![
            (1, PauliKind::Z),
            (2, PauliKind::Z),
            (5, PauliKind::Z),
            (6, PauliKind::Z),
        ],
        vec![
            (0, PauliKind::Z),
            (2, PauliKind::Z),
            (4, PauliKind::Z),
            (6, PauliKind::Z),
        ],
    ]
}

/// Prepare the logical |`0_L`⟩ of the Steane [[7, 1, 3]] CSS code using a
/// standard CX-cascade encoder (no ancillas). Pivots are chosen as
/// qubits {0, 1, 3} — each belongs to exactly one X-stabilizer, so
/// they can Hadamard independently and CX outward without cross-
/// contamination.
fn prepare_steane_logical_zero(stn: &mut StabMps) {
    // H on pivots.
    stn.h(&[QubitId(0), QubitId(1), QubitId(3)]);
    // g1 = X_3X_4X_5X_6: CX from pivot 3 to 4, 5, 6.
    stn.cx(&[(QubitId(3), QubitId(4))]);
    stn.cx(&[(QubitId(3), QubitId(5))]);
    stn.cx(&[(QubitId(3), QubitId(6))]);
    // g2 = X_1X_2X_5X_6: CX from pivot 1 to 2, 5, 6.
    stn.cx(&[(QubitId(1), QubitId(2))]);
    stn.cx(&[(QubitId(1), QubitId(5))]);
    stn.cx(&[(QubitId(1), QubitId(6))]);
    // g3 = X_0X_2X_4X_6: CX from pivot 0 to 2, 4, 6.
    stn.cx(&[(QubitId(0), QubitId(2))]);
    stn.cx(&[(QubitId(0), QubitId(4))]);
    stn.cx(&[(QubitId(0), QubitId(6))]);
}

/// Extract the Steane syndrome via one ancilla per stabilizer generator
/// (6 ancillas total for 7 data + 6 ancilla = 13 qubit layout). For each
/// X-stabilizer, prep ancilla in |+⟩ via H, apply CX from ancilla to
/// each qubit in the stabilizer support, then measure ancilla in X
/// basis (H + Z-measurement). For each Z-stabilizer, prep ancilla in
/// |0⟩, apply CX from each data qubit in support to ancilla, measure
/// ancilla in Z basis.
///
/// Returns the 6-bit syndrome (MSB first: g1..g6 in order of
/// `steane_stabilizers()`).
fn steane_syndrome_extraction(stn: &mut StabMps, ancilla_base: usize) -> [bool; 6] {
    let mut syndrome = [false; 6];
    let stabs = steane_stabilizers();
    for (i, generator) in stabs.iter().enumerate() {
        let anc = QubitId(ancilla_base + i);
        let is_x_type = generator.iter().all(|&(_, k)| k == PauliKind::X);
        if is_x_type {
            // Prep |+⟩ ancilla, CX(anc, data) for each data in support,
            // measure in X basis (H + mz).
            stn.h(&[anc]);
            for &(q, _) in generator {
                stn.cx(&[(anc, QubitId(q))]);
            }
            stn.h(&[anc]);
            syndrome[i] = stn.mz(&[anc])[0].outcome;
        } else {
            // Z-type: prep |0⟩, CX(data, anc) for each data, measure Z.
            for &(q, _) in generator {
                stn.cx(&[(QubitId(q), anc)]);
            }
            syndrome[i] = stn.mz(&[anc])[0].outcome;
        }
    }
    syndrome
}

/// Run a Steane prep + noise + syndrome extraction cycle across many shots.
/// Returns (`detection_count`, `any_errors_count)`:
/// - `detection_count`: shots where the syndrome is non-zero.
/// - `any_errors_count`: shots where depolarizing injected at least one non-I error.
fn steane_syndrome_detection_rate(p_noise: f64, num_shots: u64) -> (usize, usize) {
    let mut detections = 0;
    let mut any_errors = 0;
    for shot in 0..num_shots {
        // 7 data + 6 ancillas = 13 qubits.
        let mut stn = StabMps::builder(13).seed(shot).for_qec().build();
        prepare_steane_logical_zero(&mut stn);
        // Inject per-data-qubit depolarizing.
        let data_qubits: Vec<QubitId> = (0..7).map(QubitId).collect();
        let mut had_error = false;
        for &q in &data_qubits {
            if stn.apply_depolarizing(q, p_noise).is_some() {
                had_error = true;
            }
        }
        if had_error {
            any_errors += 1;
        }
        let syndrome = steane_syndrome_extraction(&mut stn, 7);
        let syndrome_nonzero = syndrome.iter().any(|&b| b);
        if had_error && syndrome_nonzero {
            detections += 1;
        }
    }
    (detections, any_errors)
}

/// GHZ state stabilizers: `X_0X_1...X`_{n-1}, `Z_iZ`_{i+1} for each pair.
fn ghz_stabilizers(n: usize) -> Vec<Vec<(usize, PauliKind)>> {
    let mut stabs: Vec<Vec<(usize, PauliKind)>> = (0..n - 1)
        .map(|i| vec![(i, PauliKind::Z), (i + 1, PauliKind::Z)])
        .collect();
    stabs.push((0..n).map(|i| (i, PauliKind::X)).collect());
    stabs
}

fn run_code_scenario(
    name: &str,
    num_qubits: usize,
    stabs: &[Vec<(usize, PauliKind)>],
    prep: impl Fn(&mut StabMps),
    p_noise: f64,
    num_shots: usize,
) {
    println!();
    println!(
        "=== {name} ({num_qubits} qubits, {} stabilizer generators) ===",
        stabs.len()
    );

    // Phase 1: noiseless prep + codespace fidelity check.
    let start = Instant::now();
    let mut stn = StabMps::builder(num_qubits).seed(42).for_qec().build();
    prep(&mut stn);
    let prep_time = start.elapsed().as_secs_f64();
    let f_clean = stn.code_state_fidelity(stabs);
    println!("Phase 1: noiseless prep");
    println!("  prep + fidelity time: {prep_time:.4} s");
    println!("  fidelity:             {f_clean:.6} (expected 1.0)");
    if (f_clean - 1.0).abs() > 1e-9 {
        println!("  WARNING: prep circuit does not produce |0_L⟩");
        return;
    }

    // Phase 2: prep + depolarizing noise, average across shots.
    let start = Instant::now();
    let mut total_fidelity = 0.0;
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
    for shot in 0..num_shots {
        let mut stn_noisy = StabMps::builder(num_qubits)
            .seed(100 + shot as u64)
            .for_qec()
            .build();
        prep(&mut stn_noisy);
        stn_noisy.apply_depolarizing_all(&qubits, p_noise);
        total_fidelity += stn_noisy.code_state_fidelity(stabs);
    }
    let avg_fidelity = total_fidelity / num_shots as f64;
    let noisy_time = start.elapsed().as_secs_f64();
    println!("Phase 2: prep + per-qubit depolarizing (p = {p_noise:.3})");
    println!(
        "  total time:    {noisy_time:.4} s ({:.2} ms/shot)",
        noisy_time * 1000.0 / num_shots as f64
    );
    println!("  avg fidelity:  {avg_fidelity:.6}");
    println!(
        "  drop:          {:.4} (1.0 - avg_fidelity)",
        1.0 - avg_fidelity
    );
}

fn main() {
    println!("End-to-end QEC code demo using STN");
    println!("{:-<70}", "");
    println!(
        "Toolchain: StabMps::builder().for_qec() + apply_depolarizing_all + code_state_fidelity"
    );

    // 3-qubit bit-flip code: trivial prep (|000⟩ already in code).
    run_code_scenario(
        "3-qubit bit-flip code (Z-rep, k=2)",
        3,
        &bit_flip_3q_stabilizers(),
        |_stn| { /* |000⟩ default state */ },
        0.05,
        500,
    );

    // 8-qubit Z-repetition code: |0^N⟩ prep is trivial.
    run_code_scenario(
        "8-qubit Z-repetition code (k=7)",
        8,
        &z_rep_stabilizers(8),
        |_stn| {},
        0.02,
        200,
    );

    // GHZ state on 6 qubits: H + CX cascade.
    run_code_scenario(
        "6-qubit GHZ state (k=6: 5 ZZ + 1 XX...X)",
        6,
        &ghz_stabilizers(6),
        |stn| {
            stn.h(&[QubitId(0)]);
            for q in 0..5 {
                stn.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
        },
        0.01,
        200,
    );

    // Steane [[7, 1, 3]] code: standard CSS encoder.
    run_code_scenario(
        "Steane [[7, 1, 3]] CSS code (k=6: 3 X-type + 3 Z-type)",
        7,
        &steane_stabilizers(),
        prepare_steane_logical_zero,
        0.01,
        200,
    );

    // --- Steane syndrome extraction with ancillas ---
    println!();
    println!("{:-<70}", "");
    println!("Steane syndrome extraction cycle (prep + noise + ancilla syndrome):");
    println!("  Detection ratio (syndrome non-zero / noise injected):");
    let num_cycles = 500;
    for &p in &[0.001_f64, 0.005, 0.01, 0.02, 0.05] {
        let (detections, any_errors) = steane_syndrome_detection_rate(p, num_cycles);
        let detection_ratio = if any_errors == 0 {
            0.0
        } else {
            detections as f64 / any_errors as f64
        };
        println!(
            "    p={p:.3}: {detections}/{any_errors} noisy cycles triggered syndrome ({:.1}%)",
            detection_ratio * 100.0
        );
    }

    // --- Pauli frame tracking: noise-injection speedup at scale ---
    println!();
    println!("{:-<70}", "");
    println!("Pauli frame tracking: noise-injection scaling (n=32, 10k injections):");
    let n_large = 32;
    let num_injects = 10_000;

    let start = Instant::now();
    let mut stn_eager = StabMps::builder(n_large).seed(42).build();
    for _ in 0..num_injects {
        stn_eager.apply_depolarizing(QubitId(0), 1.0);
    }
    let t_eager = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let mut stn_frame = StabMps::builder(n_large)
        .seed(42)
        .pauli_frame_tracking(true)
        .build();
    for _ in 0..num_injects {
        stn_frame.apply_depolarizing(QubitId(0), 1.0);
    }
    let t_frame = start.elapsed().as_secs_f64();

    println!("  eager (apply to tableau):       {t_eager:.4} s");
    println!("  frame tracking (O(1) per inj):  {t_frame:.4} s");
    println!(
        "  speedup:                        {:.2}×",
        t_eager / t_frame
    );

    println!();
    println!("{:-<70}", "");
    println!("Demo complete. The StabMps::for_qec() preset plus apply_depolarizing_all");
    println!("+ code_state_fidelity gives a complete API for QEC code-state");
    println!("verification + noise impact studies. Adding pauli_frame_tracking");
    println!("eliminates per-injection tableau overhead — useful for noise-heavy");
    println!("shot sweeps.");
}
