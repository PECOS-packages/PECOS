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

use pecos_trellis::{SparseDem, backward_deadline_column_order, deadline_column_order};
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

#[test]
fn deadline_order_closes_earlier_rows_first() {
    // D0 has first/last touches 0/2; D1 has 1/3. The keys therefore place
    // both D0 columns before both D1 columns, preserving original-index ties:
    // [0(D0), 2(D0), 1(D1), 3(D1)].
    let dem = sparse_dem(
        vec![
            (0.1, vec![0], vec![]),
            (0.1, vec![1], vec![]),
            (0.1, vec![0], vec![]),
            (0.1, vec![1], vec![]),
        ],
        2,
        0,
    );
    assert_eq!(deadline_column_order(&dem).unwrap(), vec![0, 2, 1, 3]);
}

#[test]
fn detector_free_mechanisms_sort_last_and_empty_dem_stays_empty() {
    let dem = sparse_dem(vec![(0.1, vec![], vec![0]), (0.1, vec![0], vec![])], 1, 1);
    assert_eq!(deadline_column_order(&dem).unwrap(), vec![1, 0]);

    let empty = sparse_dem(Vec::new(), 0, 0);
    assert!(deadline_column_order(&empty).unwrap().is_empty());
    assert!(backward_deadline_column_order(&empty).unwrap().is_empty());
}

#[test]
fn ordering_rejects_invalid_and_duplicate_detector_indices() {
    let out_of_range = sparse_dem(vec![(0.1, vec![1], vec![])], 1, 0);
    assert!(deadline_column_order(&out_of_range).is_err());

    let duplicate = sparse_dem(vec![(0.1, vec![0, 0], vec![])], 1, 0);
    assert!(backward_deadline_column_order(&duplicate).is_err());
}
