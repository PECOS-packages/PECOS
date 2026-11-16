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

//! Explicit per-shot outcomes for decoders with opt-in recovery policies.
use crate::obs_mask::ObsMask;

/// A prediction or an explicitly undecoded shot. Ordinary errors remain errors.
#[derive(Clone, Debug)]
pub struct ObservableDecodeOutcome {
    /// None means no prediction; it must never be scored as an identity mask.
    pub prediction: Option<ObsMask>,
    /// Whether the prediction came from a fallback rather than normal decoding.
    pub fallback_used: bool,
    /// Verified syndrome compatibility of a fallback; None means unchecked.
    pub syndrome_consistent: Option<bool>,
    /// Actual pruning thresholds attempted, when the decoder reports them.
    pub deltas_tried: Vec<f64>,
}
impl ObservableDecodeOutcome {
    /// Wrap an ordinary successful prediction without recovery metadata.
    #[must_use]
    pub fn decoded(prediction: ObsMask) -> Self {
        Self {
            prediction: Some(prediction),
            fallback_used: false,
            syndrome_consistent: None,
            deltas_tried: Vec::new(),
        }
    }
}

/// Compact diagnostics for a shot requiring retries, reporting, or fallback.
#[derive(Clone, Debug)]
pub struct ShotRecovery {
    /// Absolute index in the input batch.
    pub shot_index: usize,
    /// No prediction was produced; this shot was excluded from mismatch scoring.
    pub undecoded: bool,
    /// Whether this prediction disagreed with truth; None for an undecoded shot.
    pub logical_error: Option<bool>,
    /// A fallback supplied the prediction.
    pub fallback_used: bool,
    /// Verified fallback syndrome compatibility, or None when unchecked.
    pub syndrome_consistent: Option<bool>,
    /// Actual pruning thresholds attempted.
    pub deltas_tried: Vec<f64>,
}
