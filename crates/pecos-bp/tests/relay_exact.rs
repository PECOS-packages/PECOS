// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Independent exact-oracle coverage for check-serial Relay-BP.
//!
//! The oracle enumerates error patterns and minimizes their negative log
//! probability. It does not reproduce any message-passing expression.

use std::collections::BTreeSet;

use pecos_bp::BpGraph;
use pecos_bp::relay::{RelayBp, RelayConfig, Schedule};
use pecos_decoder_core::dem::DemCheckMatrix;

const TREE_DEM: &str = "\
error(0.40662808134905276) D0
error(0.023459592534763554) D0 D1
error(0.021196178837122753) D1
";

fn pattern(mask: usize, width: usize) -> Vec<u8> {
    (0..width)
        .map(|bit| u8::from(mask & (1 << bit) != 0))
        .collect()
}

fn syndrome_of(dcm: &DemCheckMatrix, error: &[u8]) -> Vec<u8> {
    (0..dcm.num_detectors)
        .map(|check| {
            (0..dcm.num_mechanisms).fold(0, |parity, variable| {
                parity ^ (dcm.check_matrix[[check, variable]] & error[variable])
            })
        })
        .collect()
}

fn negative_log_probability(error: &[u8], probabilities: &[f64]) -> f64 {
    -error
        .iter()
        .zip(probabilities)
        .map(|(&bit, &probability)| {
            if bit == 0 {
                (-probability).ln_1p()
            } else {
                probability.ln()
            }
        })
        .sum::<f64>()
}

fn exact_minimum_weight_correction(dcm: &DemCheckMatrix, syndrome: &[u8]) -> Vec<u8> {
    let mut candidates = (0..1 << dcm.num_mechanisms)
        .map(|mask| pattern(mask, dcm.num_mechanisms))
        .filter(|error| syndrome_of(dcm, error) == syndrome)
        .map(|error| {
            let cost = negative_log_probability(&error, &dcm.error_priors);
            (cost, error)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    assert!(candidates.len() >= 2);
    assert_ne!(
        candidates[0].0.total_cmp(&candidates[1].0),
        std::cmp::Ordering::Equal,
        "oracle minimum must be unique"
    );
    candidates.remove(0).1
}

#[test]
fn check_serial_matches_exhaustive_minimum_weight_decoding() {
    let dcm = DemCheckMatrix::from_dem_str(TREE_DEM).unwrap();
    let config = RelayConfig {
        schedule: Schedule::CheckSerial,
        alpha: 1.0,
        gamma0: 0.0,
        pre_iterations: 8,
        num_legs: 0,
        leg_iterations: 1,
        gamma_range: (-0.24, 0.66),
        stop_after_converged: 1,
        explicit_gammas: None,
    };
    let mut decoder = RelayBp::new(BpGraph::from_dcm(&dcm), config).unwrap();
    let mut syndromes = BTreeSet::new();

    for source_mask in 0..1 << dcm.num_mechanisms {
        let source = pattern(source_mask, dcm.num_mechanisms);
        let syndrome = syndrome_of(&dcm, &source);
        syndromes.insert(syndrome.clone());
        let exact = exact_minimum_weight_correction(&dcm, &syndrome);

        let outcome = decoder.decode(&syndrome, 0).unwrap();

        assert!(outcome.converged, "failed for syndrome {syndrome:?}");
        assert_eq!(outcome.correction, exact, "syndrome {syndrome:?}");
    }

    assert_eq!(syndromes.len(), 1 << dcm.num_detectors);
}
