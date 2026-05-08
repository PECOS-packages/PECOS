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

//! Integration tests for the ensemble decoder.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::ensemble::EnsembleDecoder;
use pecos_decoder_core::errors::DecoderError;

/// A decoder that returns a configurable mask based on syndrome content.
struct ConfigurableDecoder {
    /// Observable mask to return when syndrome has any defects.
    defect_mask: u64,
}

impl ObservableDecoder for ConfigurableDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let has_defects = syndrome.iter().any(|&v| v != 0);
        if has_defects {
            Ok(self.defect_mask)
        } else {
            Ok(0)
        }
    }
}

/// A decoder that always fails.
struct FailingDecoder;

impl ObservableDecoder for FailingDecoder {
    fn decode_to_observables(&mut self, _syndrome: &[u8]) -> Result<u64, DecoderError> {
        Err(DecoderError::DecodingFailed("always fails".into()))
    }
}

#[test]
fn test_ensemble_with_three_agreeing_decoders() {
    let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
        Box::new(ConfigurableDecoder { defect_mask: 0b11 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b11 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b11 }),
    ];
    let mut ens = EnsembleDecoder::new(decoders);
    assert_eq!(ens.decode_to_observables(&[1]).unwrap(), 0b11);
    assert_eq!(ens.decode_to_observables(&[0]).unwrap(), 0);
}

#[test]
fn test_ensemble_majority_per_bit() {
    // Decoder 1: flips obs 0 and 1
    // Decoder 2: flips obs 0 only
    // Decoder 3: flips obs 0 only
    // Majority: obs 0 = 3/3 flip, obs 1 = 1/3 flip
    let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
        Box::new(ConfigurableDecoder { defect_mask: 0b11 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b01 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b01 }),
    ];
    let mut ens = EnsembleDecoder::new(decoders);
    assert_eq!(ens.decode_to_observables(&[1]).unwrap(), 0b01);
}

#[test]
fn test_ensemble_weighted_overrides_majority() {
    // 1 decoder votes flip (weight 10), 2 decoders vote no flip (weight 1 each)
    let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
        Box::new(ConfigurableDecoder { defect_mask: 1 }),
        Box::new(ConfigurableDecoder { defect_mask: 0 }),
        Box::new(ConfigurableDecoder { defect_mask: 0 }),
    ];
    let mut ens = EnsembleDecoder::with_weights(decoders, vec![10.0, 1.0, 1.0]);
    // Weight for flip: 10, weight for no flip: 2. Flip wins despite 1/3 majority.
    assert_eq!(ens.decode_to_observables(&[1]).unwrap(), 1);
}

#[test]
fn test_ensemble_propagates_errors() {
    let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
        Box::new(ConfigurableDecoder { defect_mask: 1 }),
        Box::new(FailingDecoder),
    ];
    let mut ens = EnsembleDecoder::new(decoders);
    let result = ens.decode_to_observables(&[1]);
    assert!(result.is_err(), "Should propagate decoder error");
}

#[test]
fn test_ensemble_repeated_shots_consistent() {
    let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
        Box::new(ConfigurableDecoder { defect_mask: 1 }),
        Box::new(ConfigurableDecoder { defect_mask: 1 }),
        Box::new(ConfigurableDecoder { defect_mask: 0 }),
    ];
    let mut ens = EnsembleDecoder::new(decoders);

    // Run the same syndrome 100 times -- must be deterministic.
    for _ in 0..100 {
        assert_eq!(ens.decode_to_observables(&[1]).unwrap(), 1);
        assert_eq!(ens.decode_to_observables(&[0]).unwrap(), 0);
    }
}

#[test]
fn test_ensemble_five_decoders_complex_vote() {
    // 5 decoders, 3 bits:
    // D0: 0b111, D1: 0b101, D2: 0b100, D3: 0b110, D4: 0b010
    // Bit 0: D0,D1 flip (2/5 < majority) -> 0
    // Bit 1: D0,D3,D4 flip (3/5 majority) -> 1
    // Bit 2: D0,D1,D2,D3 flip (4/5 majority) -> 1
    let decoders: Vec<Box<dyn ObservableDecoder>> = vec![
        Box::new(ConfigurableDecoder { defect_mask: 0b111 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b101 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b100 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b110 }),
        Box::new(ConfigurableDecoder { defect_mask: 0b010 }),
    ];
    let mut ens = EnsembleDecoder::new(decoders);
    assert_eq!(ens.decode_to_observables(&[1]).unwrap(), 0b110);
}
