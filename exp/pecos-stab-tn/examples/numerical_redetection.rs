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

//! Falsifier benchmark for numerical |0> flag re-detection.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;
use std::f64::consts::TAU;
use std::time::Instant;

#[derive(Clone, Copy)]
enum GateMix {
    Random,
    CliffT,
}

fn next_rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn run_circuit(
    num_qubits: usize,
    num_gates: usize,
    seed: u64,
    mix: GateMix,
    numerical_redetection: bool,
) -> (StabMps, usize) {
    let mut stn = StabMps::builder(num_qubits)
        .seed(seed)
        .numerical_flag_redetection(numerical_redetection)
        .build();
    let mut peak_bond = stn.max_bond_dim();
    let mut rng_state = seed.wrapping_add(1);

    for _ in 0..num_gates {
        let gate_type = next_rng(&mut rng_state)
            % match mix {
                GateMix::Random => 8,
                GateMix::CliffT => 6,
            };
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };

        match gate_type {
            0 => {
                stn.h(&[QubitId(q0)]);
            }
            1 => {
                stn.sz(&[QubitId(q0)]);
            }
            2 => {
                stn.x(&[QubitId(q0)]);
            }
            3 => {
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            5 => {
                stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q0)]);
            }
            6 => {
                let bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians((bits % 1000) as f64 * 0.001 * TAU);
                stn.rz(angle, &[QubitId(q0)]);
            }
            _ => {
                let bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians((bits % 1000) as f64 * 0.001 * TAU);
                stn.rx(angle, &[QubitId(q0)]);
            }
        }
        peak_bond = peak_bond.max(stn.max_bond_dim());
    }

    (stn, peak_bond)
}

fn benchmark(
    label: &str,
    num_qubits: usize,
    num_gates: usize,
    num_seeds: u64,
    mix: GateMix,
    numerical_redetection: bool,
) {
    let start = Instant::now();
    let mut total = 0u64;
    let mut fast = 0u64;
    let mut redetect = 0u64;
    let mut standard = 0u64;
    let mut final_bond_sum = 0u64;
    let mut peak_bond_sum = 0u64;

    for seed in 0..num_seeds {
        let (stn, peak_bond) = run_circuit(num_qubits, num_gates, seed, mix, numerical_redetection);
        total += stn.stats.total_nonclifford;
        fast += stn.stats.multi_disent;
        redetect += stn.stats.numerical_redetect;
        standard += stn.stats.multi_std;
        final_bond_sum += stn.max_bond_dim() as u64;
        peak_bond_sum += peak_bond as u64;
    }

    let mode = if numerical_redetection { "ON" } else { "OFF" };
    let fast_rate = if total == 0 {
        0.0
    } else {
        100.0 * fast as f64 / total as f64
    };
    let average_final = final_bond_sum as f64 / num_seeds as f64;
    let average_peak = peak_bond_sum as f64 / num_seeds as f64;
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{label:<13} | {mode:<3} | {total:>5} | {fast_rate:>7.2}% | {redetect:>8} | \
         {standard:>5} | {average_final:>9.2} | {average_peak:>8.2} | {wall_ms:>8.2}"
    );
}

fn main() {
    println!(
        "scenario      | opt | total | fast rate | redetect |   std | avg final | avg peak |  wall ms"
    );
    println!(
        "--------------|-----|-------|-----------|----------|-------|-----------|----------|---------"
    );
    for (label, n, gates, seeds, mix) in [
        ("2q deep", 2, 50, 50, GateMix::Random),
        ("3q T-heavy", 3, 30, 30, GateMix::CliffT),
        ("10q T", 10, 40, 3, GateMix::CliffT),
        ("15q T", 15, 50, 2, GateMix::CliffT),
    ] {
        benchmark(label, n, gates, seeds, mix, false);
        benchmark(label, n, gates, seeds, mix, true);
    }
}
