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

//! Implementation of core decoder traits for MWPF

use crate::decoder::MwpfDecoder;

/// Implement `ObservableDecoder` for `MwpfDecoder`.
///
/// This is the primary trait used by the fast decode path
/// (`SampleBatch.decode_count`, `sample_decode_count`, etc.).
impl pecos_decoder_core::ObservableDecoder for MwpfDecoder {
    fn decode_to_observables(
        &mut self,
        syndrome: &[u8],
    ) -> std::result::Result<u64, pecos_decoder_core::DecoderError> {
        let result = self
            .decode_syndrome(syndrome)
            .map_err(|e| pecos_decoder_core::DecoderError::DecodingFailed(e.to_string()))?;
        Ok(result.observable_mask)
    }
}
