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

use pecos_decoder_core::ObservableDecoder;
use pecos_mwpf::{MwpfConfig, MwpfDecoder};

/// Simple repetition-code-like DEM: two detectors, two error mechanisms.
const SIMPLE_DEM: &str = "\
error(0.1) D0 D1 L0
error(0.1) D1
";

/// A d=3 surface-code-like DEM with a hyperedge (3 detectors).
const HYPEREDGE_DEM: &str = "\
error(0.01) D0 D1 L0
error(0.01) D1 D2
error(0.01) D0 D1 D2 L0
error(0.01) D2
";

#[test]
fn construct_from_simple_dem() {
    let decoder = MwpfDecoder::from_dem(SIMPLE_DEM, MwpfConfig::default());
    assert!(decoder.is_ok());
    let decoder = decoder.unwrap();
    assert_eq!(decoder.num_detectors(), 2);
    assert_eq!(decoder.num_observables(), 1);
}

#[test]
fn construct_from_hyperedge_dem() {
    let decoder = MwpfDecoder::from_dem(HYPEREDGE_DEM, MwpfConfig::default());
    assert!(decoder.is_ok());
    let decoder = decoder.unwrap();
    assert_eq!(decoder.num_detectors(), 3);
    assert_eq!(decoder.num_observables(), 1);
}

#[test]
fn decode_no_errors() {
    let mut decoder = MwpfDecoder::from_dem(SIMPLE_DEM, MwpfConfig::default()).unwrap();
    // No detectors triggered -> no observable flips.
    let syndrome = vec![0u8; 2];
    let result = decoder.decode_syndrome(&syndrome).unwrap();
    assert_eq!(result.observable_mask, 0);
}

#[test]
fn decode_single_error() {
    let mut decoder = MwpfDecoder::from_dem(SIMPLE_DEM, MwpfConfig::default()).unwrap();
    // Both D0 and D1 triggered -> mechanism 0 (D0 D1 L0), observable L0 flips.
    let syndrome = vec![1u8, 1];
    let result = decoder.decode_syndrome(&syndrome).unwrap();
    assert_eq!(result.observable_mask, 1);
}

#[test]
fn decode_boundary_error() {
    let mut decoder = MwpfDecoder::from_dem(SIMPLE_DEM, MwpfConfig::default()).unwrap();
    // Only D1 triggered -> mechanism 1 (D1, boundary), no observable flip.
    let syndrome = vec![0u8, 1];
    let result = decoder.decode_syndrome(&syndrome).unwrap();
    assert_eq!(result.observable_mask, 0);
}

#[test]
fn observable_decoder_trait() {
    let mut decoder = MwpfDecoder::from_dem(SIMPLE_DEM, MwpfConfig::default()).unwrap();
    let mask = decoder.decode_to_observables(&[1, 1]).unwrap();
    assert_eq!(mask, 1);
}

#[test]
fn decode_multiple_shots() {
    // Verify solver reuse works (clear() between shots).
    let mut decoder = MwpfDecoder::from_dem(SIMPLE_DEM, MwpfConfig::default()).unwrap();
    for _ in 0..10 {
        let _ = decoder.decode_syndrome(&[1, 1]).unwrap();
        let _ = decoder.decode_syndrome(&[0, 1]).unwrap();
        let _ = decoder.decode_syndrome(&[0, 0]).unwrap();
    }
}

#[test]
fn custom_config() {
    let config = MwpfConfig {
        cluster_node_limit: 10,
        timeout: Some(5.0),
        ..MwpfConfig::default()
    };
    let mut decoder = MwpfDecoder::from_dem(SIMPLE_DEM, config).unwrap();
    let result = decoder.decode_syndrome(&[1, 1]).unwrap();
    assert_eq!(result.observable_mask, 1);
}

#[test]
fn hyperedge_decode() {
    let mut decoder = MwpfDecoder::from_dem(HYPEREDGE_DEM, MwpfConfig::default()).unwrap();
    // All three detectors triggered: the hyperedge mechanism (D0 D1 D2 L0)
    // is the minimum weight explanation.
    let result = decoder.decode_syndrome(&[1, 1, 1]).unwrap();
    assert_eq!(result.observable_mask, 1);
}
