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

use pecos_frontier::{FrontierCommittee, FrontierConfig, FrontierDecoder, SparseDem};
use std::collections::BTreeMap;

fn sparse_dem(
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
) -> SparseDem {
    SparseDem {
        mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors,
        num_observables,
    }
}

fn exact_config() -> FrontierConfig {
    FrontierConfig {
        k: usize::MAX,
        delta: f64::INFINITY,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
        metric_mode: pecos_frontier::MetricMode::default(),
        int_metric_scale: 1024,
    }
}

#[test]
fn committee_and_direct_decoder_share_unexplainable_error_text() {
    let dem = sparse_dem(vec![(0.2, vec![0, 1], vec![])], 2, 0);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let mut committee = FrontierCommittee::from_sparse_dem(&dem, exact_config()).unwrap();

    let direct_error = decoder.decode(&[1, 0]).unwrap_err();
    let committee_error = committee.decode(&[1, 0]).unwrap_err();
    assert_eq!(direct_error.to_string(), committee_error.to_string());
}

/// A malformed call must reach the caller as itself, not as a pruning
/// complaint. The committee used to synthesize an unexplainable-syndrome error
/// whenever both legs failed, which reported a dimension mismatch as something
/// the caller could fix by tuning `k` or `delta`.
#[test]
fn committee_propagates_a_dimension_error_instead_of_reporting_no_path() {
    let dem = sparse_dem(vec![(0.2, vec![0, 1], vec![])], 2, 0);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let mut committee = FrontierCommittee::from_sparse_dem(&dem, exact_config()).unwrap();

    // One detector short of the model.
    let direct_error = decoder.decode(&[1]).unwrap_err();
    let committee_error = committee.decode(&[1]).unwrap_err();

    assert!(
        matches!(
            direct_error,
            pecos_frontier::DecoderError::InvalidDimensions { .. }
        ),
        "expected the engine to reject a short syndrome, got {direct_error:?}"
    );
    assert_eq!(direct_error.to_string(), committee_error.to_string());
    assert!(
        !committee_error.to_string().contains("unexplainable"),
        "committee reported a dimension fault as an unexplainable syndrome: {committee_error}"
    );
}
