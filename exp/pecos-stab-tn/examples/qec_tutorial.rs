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

//! # QEC tutorial — the STN workflow end-to-end
//!
//! Run with `cargo run --release --example qec_tutorial`.
//!
//! This example walks through a typical quantum-error-correction
//! simulation using `pecos-stab-tn`. Each section maps to one feature
//! of the API; read top-to-bottom to understand how the pieces fit.
//!
//! Contents:
//!   1. Choosing a builder preset for QEC workloads.
//!   2. Defining a stabilizer code (3-qubit bit-flip).
//!   3. Noiseless syndrome extraction and ancilla reuse.
//!   4. Pauli-noise sampling via the `pauli_frame`.
//!   5. Many-round simulation with mid-circuit resets.
//!   6. Interpreting the results.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::{PauliKind, StabMps};
use std::time::Instant;

fn main() {
    println!("=============================================================");
    println!("  STN QEC tutorial");
    println!("=============================================================\n");

    // ------------------------------------------------------------------
    // 1. Builder + preset
    // ------------------------------------------------------------------
    //
    // `StabMps::builder(n).for_qec().build()` sets:
    //   - max_bond_dim = 128 (enough for syndrome rounds without truncation)
    //   - max_truncation_error = 1e-8 (very tight)
    //   - merge_rz = true (batch same-qubit RZ noise)
    //
    // For ion-trap-memory-noise or T-heavy workloads this is the right
    // default. You can layer `pauli_frame_tracking(true)` on top for
    // fast Pauli-noise injection.
    //
    // 3 data + 2 ancillas = 5 qubits total.

    let num_data = 3;
    let num_ancillas = 2; // one per stabilizer generator
    let n = num_data + num_ancillas;
    let ancilla_base = num_data;

    let mut stn = StabMps::builder(n)
        .seed(42)
        .for_qec()
        .pauli_frame_tracking(true)
        .build();

    println!("Step 1: built StabMps with for_qec() preset + pauli_frame_tracking");
    println!("  data qubits  : 0..{num_data}");
    println!("  ancillas     : {num_data}..{n}");
    println!();

    // ------------------------------------------------------------------
    // 2. Define the code
    // ------------------------------------------------------------------
    //
    // Each stabilizer generator is a Vec<(qubit_index, PauliKind)>.
    // For the 3-qubit bit-flip code: Z_0 Z_1 and Z_1 Z_2.
    //
    // `extract_syndromes` handles any Pauli generator — mix X/Y/Z freely.

    let stabilizers: Vec<Vec<(usize, PauliKind)>> = vec![
        vec![(0, PauliKind::Z), (1, PauliKind::Z)],
        vec![(1, PauliKind::Z), (2, PauliKind::Z)],
    ];
    let ancilla_qubits: Vec<QubitId> = (ancilla_base..ancilla_base + num_ancillas)
        .map(QubitId)
        .collect();

    println!("Step 2: defined 3-qubit bit-flip code stabilizers");
    for (i, s) in stabilizers.iter().enumerate() {
        println!("  g{i}: {s:?}");
    }
    println!();

    // ------------------------------------------------------------------
    // 3. Noiseless syndrome extraction
    // ------------------------------------------------------------------
    //
    // `extract_syndromes(generators, ancilla_qubits)`:
    //   1. For each generator, prep_plus (|+⟩) the ancilla.
    //   2. Apply controlled-P with ancilla as control.
    //   3. H + measure ancilla → syndrome bit.
    //   4. reset_qubit(ancilla) so it's ready for the next round.
    //
    // On the codespace, syndrome must be all-zero.

    // Data state starts at |000⟩ — already in the codespace for bit-flip.
    let syndrome = stn.extract_syndromes(&stabilizers, &ancilla_qubits);
    println!("Step 3: noiseless syndrome extraction");
    println!("  syndrome = {syndrome:?} (expected all-false)");
    assert!(syndrome.iter().all(|&b| !b));
    println!();

    // ------------------------------------------------------------------
    // 4. Inject a single X error via the Pauli frame
    // ------------------------------------------------------------------
    //
    // With pauli_frame_tracking(true), `inject_x_in_frame` is O(1) —
    // it toggles a classical bit rather than applying X to the state.
    // The bit propagates through Cliffords and flips measurement
    // outcomes at read time.
    //
    // For bulk injection, use `inject_paulis_in_frame(&[(q, Pauli), ...])`.

    println!("Step 4: inject X_0 error and re-extract syndrome");
    stn.inject_x_in_frame(QubitId(0));
    let syndrome = stn.extract_syndromes(&stabilizers, &ancilla_qubits);
    println!("  syndrome = {syndrome:?} (expected [true, false]: X_0 triggers Z_0Z_1 only)");
    println!();

    // ------------------------------------------------------------------
    // 5. Many rounds with random noise
    // ------------------------------------------------------------------
    //
    // Apply depolarizing noise to each data qubit each round, then
    // extract. Frame tracking + merge_rz makes this fast. We count
    // how often the syndrome is non-zero as a function of noise rate.

    println!("Step 5: many-round detection rate vs depolarizing rate");
    let num_rounds = 5000;
    for &p in &[0.001_f64, 0.005, 0.01, 0.02, 0.05] {
        let mut non_zero_syndromes = 0u32;
        let mut stn = StabMps::builder(n)
            .seed(100 + (p * 1e6) as u64)
            .for_qec()
            .pauli_frame_tracking(true)
            .build();

        let start = Instant::now();
        for _round in 0..num_rounds {
            // Per-round depolarizing on each data qubit.
            for q in 0..num_data {
                stn.apply_depolarizing(QubitId(q), p);
            }
            let s = stn.extract_syndromes(&stabilizers, &ancilla_qubits);
            if s.iter().any(|&b| b) {
                non_zero_syndromes += 1;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let rate = f64::from(non_zero_syndromes) / f64::from(num_rounds);
        println!(
            "  p={p:.3}: detection rate = {rate:.3} ({num_rounds} rounds in {elapsed:.2}s = {:.0} rounds/s)",
            f64::from(num_rounds) / elapsed
        );
    }
    println!();

    // ------------------------------------------------------------------
    // 6. Ion-trap-style RZ memory noise
    // ------------------------------------------------------------------
    //
    // Every "gate timestep" each idle qubit picks up a small rz(θ).
    // merge_rz accumulates these until some gate or measurement forces
    // a flush. Scales well — 25-42× speedup over the eager path.

    println!("Step 6: ion-trap memory noise scenario");
    let num_rounds_ion = 50;
    let steps_per_round = 20;
    let theta = Angle64::from_radians(0.005);

    let mut stn = StabMps::builder(n)
        .seed(77)
        .for_qec()
        .pauli_frame_tracking(true)
        .build();
    let start = Instant::now();
    for _round in 0..num_rounds_ion {
        for _step in 0..steps_per_round {
            // Pass all data qubits in a single slice call — the rz
            // method accumulates into pending_rz[q] for each q at O(1).
            let data: Vec<QubitId> = (0..num_data).map(QubitId).collect();
            stn.rz(theta, &data);
        }
        stn.extract_syndromes(&stabilizers, &ancilla_qubits);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "  {num_rounds_ion} rounds × {steps_per_round} idle steps each, θ={theta:?}: {elapsed:.3}s"
    );
    println!();

    // ------------------------------------------------------------------
    // 7. Flush the frame before reading exact state
    // ------------------------------------------------------------------
    //
    // If you want exact state_vector/amplitude readouts (including
    // complex phase from Y injections), call `flush_pauli_frame_to_state`
    // first. The decomposition-based flush gives EXACT amplitudes even
    // on Clifford-evolved states — no ±1 residual.

    let mut stn = StabMps::builder(2)
        .seed(1)
        .pauli_frame_tracking(true)
        .build();
    stn.h(&[QubitId(0)]);
    stn.cx(&[(QubitId(0), QubitId(1))]);
    stn.inject_y_in_frame(QubitId(0));
    stn.flush_pauli_frame_to_state();
    let sv = stn.state_vector();
    println!("Step 7: Y_0 on a Bell state, exact amplitudes after flush");
    for (i, a) in sv.iter().enumerate() {
        println!("  sv[{i}] = {a:.4}");
    }

    println!("\nTutorial done. Key API summary:");
    println!("  - StabMps::builder(n).for_qec().pauli_frame_tracking(true).build()");
    println!("  - extract_syndromes(generators, ancillas)");
    println!("  - reset_qubit(q), pz(q), px(q)");
    println!("  - inject_{{x,y,z}}_in_frame(q), inject_paulis_in_frame(&[...])");
    println!("  - apply_depolarizing(q, p), apply_depolarizing_all(&qs, p)");
    println!("  - flush_pauli_frame_to_state(): exact state_vector after Y frames");
}
