// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Cross-implementation A/B harness: decode upstream-frontier sample shots
//! with `FrontierDecoder` on the identical model and column order.
//!
//! Input JSON (produced by an external extraction script from the upstream
//! `frontier` package): `{num_detectors, num_observables, mechanisms:
//! [[p, [detectors], [observables]], ...], shots: [{syndrome, truth_logical}]}`
//! where mechanism order IS the processing order and `syndrome` packs detector
//! `i` into bit `i`.
//!
//! Usage: `bridge_ab <model.json> <k> <delta> <score_alpha>`
//! Prints one `shot,predicted,truth,status` line per shot plus a summary line.

use pecos_decoder_core::dem::SparseDem;
use pecos_frontier::{FrontierConfig, FrontierDecoder};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct BridgeModel {
    num_detectors: usize,
    num_observables: usize,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    shots: Vec<Shot>,
}

#[derive(Deserialize)]
struct Shot {
    /// Fired detector indices (supports arbitrary detector counts).
    fired: Vec<u32>,
    truth_logical: u128,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: bridge_ab <model.json> <k> <delta> <score_alpha>");
    let k: usize = args.next().expect("missing k").parse().expect("k");
    let delta: f64 = args.next().expect("missing delta").parse().expect("delta");
    let score_alpha: f64 = args
        .next()
        .expect("missing score_alpha")
        .parse()
        .expect("score_alpha");

    let model: BridgeModel =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read model json"))
            .expect("parse model json");
    let dem = SparseDem {
        mechanisms: model.mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors: model.num_detectors,
        num_observables: model.num_observables,
    };
    let config = FrontierConfig {
        k,
        delta,
        score_alpha,
        column_order: None,
    };
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config).expect("build decoder");

    let mut failures = 0_u32;
    let mut no_path = 0_u32;
    let started = std::time::Instant::now();
    assert!(
        model.num_observables <= 128,
        "bridge truth_logical is u128; wider observables need a format change"
    );
    let mut syndrome = vec![0_u8; model.num_detectors];
    for (shot, entry) in model.shots.iter().enumerate() {
        syndrome.fill(0);
        for &fired in &entry.fired {
            syndrome[fired as usize] = 1;
        }
        if let Ok(result) = decoder.decode(&syndrome) {
            let words = result.predicted.words();
            assert!(words.iter().skip(2).all(|&w| w == 0), "label fits u128");
            let predicted = u128::from(words.first().copied().unwrap_or(0))
                | (u128::from(words.get(1).copied().unwrap_or(0)) << 64);
            let status = if predicted == entry.truth_logical {
                "ok"
            } else {
                failures += 1;
                "logical_fail"
            };
            println!("{shot},{predicted},{},{status}", entry.truth_logical);
        } else {
            failures += 1;
            no_path += 1;
            println!("{shot},,{},no_path", entry.truth_logical);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let trials = u32::try_from(model.shots.len()).expect("shot count fits u32");
    println!(
        "SUMMARY trials={trials} fail={failures} no_path={no_path} fer={} k={k} delta={delta} alpha={score_alpha} decode_s_mean={}",
        f64::from(failures) / f64::from(trials),
        elapsed / f64::from(trials),
    );
}
