// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Issue #586 uncapped probability-readback reproducer.

use std::time::Instant;

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::StabMps;

const NUM_QUBITS: usize = 64;

fn main() {
    let started = Instant::now();
    let mut simulator = StabMps::builder(NUM_QUBITS)
        .seed(26_401)
        .max_bond_dim(1 << 32)
        .svd_cutoff(1e-12)
        .max_truncation_error(0.0)
        .merge_rz(false)
        .pauli_frame_tracking(false)
        .build();
    for q in 0..NUM_QUBITS {
        simulator.h(&[QubitId(q)]);
    }
    for q in 0..NUM_QUBITS - 1 {
        simulator.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    // Exact choices from pecos-perf sparse_t_circuit(n=64, n_t=64, seed=26401).
    let injections = [
        (63, 23, true),
        (16, 40, false),
        (28, 49, false),
        (3, 14, false),
        (13, 55, true),
        (7, 11, false),
        (47, 35, false),
        (10, 28, true),
        (29, 35, true),
        (39, 51, false),
        (6, 28, false),
        (7, 16, true),
        (41, 32, false),
        (0, 22, true),
        (8, 40, false),
        (11, 52, false),
        (48, 61, false),
        (31, 24, false),
        (43, 41, false),
        (7, 29, false),
        (46, 31, true),
        (41, 17, true),
        (2, 61, false),
        (29, 15, false),
        (61, 49, false),
        (39, 58, true),
        (46, 4, true),
        (57, 38, false),
        (5, 44, false),
        (63, 26, false),
        (8, 22, false),
        (32, 13, false),
        (60, 50, false),
        (3, 63, false),
        (6, 28, true),
        (21, 13, true),
        (57, 0, true),
        (8, 48, true),
        (48, 5, true),
        (25, 50, false),
        (25, 47, false),
        (38, 42, false),
        (34, 58, true),
        (0, 29, true),
        (19, 11, false),
        (13, 3, true),
        (33, 19, false),
        (4, 39, true),
        (42, 31, false),
        (36, 42, true),
        (21, 31, true),
        (25, 54, false),
        (44, 61, true),
        (49, 62, false),
        (58, 22, true),
        (41, 35, false),
        (49, 25, true),
        (25, 45, false),
        (47, 13, false),
        (7, 28, true),
        (45, 46, true),
        (20, 58, false),
        (8, 56, false),
        (18, 26, false),
    ];
    let s_targets = [
        62, 37, 52, 55, 49, 3, 62, 15, 28, 32, 32, 33, 40, 29, 18, 47,
    ];
    for (injection, &(target, other, dagger)) in injections.iter().enumerate() {
        let angle = if dagger {
            -std::f64::consts::FRAC_PI_4
        } else {
            std::f64::consts::FRAC_PI_4
        };
        simulator.rz(Angle64::from_radians(angle), &[QubitId(target)]);
        simulator.h(&[QubitId(target)]);
        if injection & 1 == 0 {
            simulator.cx(&[(QubitId(target), QubitId(other))]);
        } else {
            simulator.cz(&[(QubitId(target), QubitId(other))]);
        }
        if injection % 4 == 3 {
            simulator.sz(&[QubitId(s_targets[injection / 4])]);
        }
    }
    simulator.flush();
    eprintln!(
        "simulation complete: final_bond={} lifetime_peak_bond={} infidelity={:.16e} elapsed={:?}",
        simulator.max_bond_dim(),
        simulator.lifetime_peak_bond(),
        simulator.truncation_error(),
        started.elapsed(),
    );

    let masks = [
        0x7adb_0d8f_af54_4738_u64,
        0xea75_e4c4_2e73_22aa,
        0x9209_6016_76be_141a,
        0x6eee_15d2_b596_e2f6,
        0xfe7e_e6bc_ea9a_b769,
        0xd510_161b_eb97_9144,
        0xf6cc_7af8_1700_f285,
        0x0a33_a0f8_ea16_8533,
        0xa179_4380_8d6b_c709,
        0x7187_ba8f_e039_a1c7,
        0xf5c2_3124_58f9_eb3f,
        0xdea3_b676_22f6_30e9,
        0x8ed3_b87a_0ce3_7180,
        0x0939_5389_e38e_90df,
        0x3d43_6ceb_280a_e375,
        0x1669_4814_3969_2e7e,
    ];
    let queries = masks
        .into_iter()
        .map(|mask| {
            (0..NUM_QUBITS)
                .map(|q| mask >> q & 1 != 0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let query_started = Instant::now();
    let probabilities = simulator.prob_bitstrings(&queries);
    let mut max_delta = 0.0_f64;
    for (probability, query) in probabilities.iter().zip(&queries).take(4) {
        max_delta =
            max_delta.max((probability - simulator.amplitude_iterative(query).norm_sqr()).abs());
    }
    eprintln!(
        "readback complete: queries={} subset_max_delta={max_delta:.3e} elapsed={:?}",
        probabilities.len(),
        query_started.elapsed(),
    );
}
