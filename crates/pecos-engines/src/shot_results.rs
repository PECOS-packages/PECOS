// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Shot results and data structures for quantum program execution.
//!
//! The result types live in the neutral [`pecos_results`] crate so that any
//! simulation stack can produce them; this module re-exports them under the
//! historical `pecos_engines::shot_results` paths and adds the
//! `ByteMessage`-protocol conveniences that belong to this crate.

pub use pecos_results::*;

use crate::byte_message::ByteMessage;
use pecos_core::errors::PecosError;
use std::collections::BTreeMap;

/// Create a [`Shot`] directly from a [`ByteMessage`] containing measurement
/// results, mapping result IDs to names via `result_id_to_name` (missing IDs
/// fall back to `result_{id}`).
///
/// # Errors
///
/// Returns an error if the `ByteMessage` cannot be parsed or doesn't contain
/// valid measurement results.
pub fn shot_from_byte_message(
    message: &ByteMessage,
    result_id_to_name: &BTreeMap<usize, String>,
) -> Result<Shot, PecosError> {
    let outcomes = message.outcomes()?;

    let mut result = Shot::default();
    for (result_id, value) in outcomes.into_iter().enumerate() {
        let name = result_id_to_name
            .get(&result_id)
            .cloned()
            .unwrap_or_else(|| format!("result_{result_id}"));
        result.data.insert(name, Data::U32(value));
    }

    Ok(result)
}

/// Create a single-shot [`ShotVec`] directly from a [`ByteMessage`]
/// containing measurement results, naming each outcome `result_{id}`.
///
/// # Errors
///
/// Returns a `PecosError` if the measurements cannot be extracted from the
/// `ByteMessage`.
pub fn shot_vec_from_byte_message(message: &ByteMessage) -> Result<ShotVec, PecosError> {
    let outcomes = message.outcomes()?;

    let mut shot_result = Shot::default();
    for (result_id, value) in outcomes.into_iter().enumerate() {
        shot_result
            .data
            .insert(format!("result_{result_id}"), Data::U32(value));
    }

    Ok(ShotVec {
        shots: vec![shot_result],
    })
}
