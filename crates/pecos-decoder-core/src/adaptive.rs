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

//! Adaptive decoder wrapper for time-varying noise.
//!
//! Wraps any `ObservableDecoder` with automatic rebuilding when the
//! noise model changes. The decoder factory is called with a new DEM
//! whenever `update_noise` is invoked.
//!
//! Use cases:
//! - Neutral atoms: noise drifts on hour timescales (calibration drift)
//! - Trapped ions: slow parameter drift between recalibrations
//! - Any platform where the DEM becomes stale

use crate::ObservableDecoder;
use crate::errors::DecoderError;

type DecoderFactory = dyn FnMut(&str) -> Result<Box<dyn ObservableDecoder>, DecoderError>;

fn fraction(numerator: usize, denominator: usize) -> f64 {
    let numerator = u32::try_from(numerator).expect("monitoring count fits in u32");
    let denominator = u32::try_from(denominator).expect("monitoring window fits in u32");
    f64::from(numerator) / f64::from(denominator)
}

/// Adaptive decoder that rebuilds when noise changes.
///
/// Holds a decoder factory and the current DEM. When `update_dem` is
/// called with a new DEM string, the decoder is rebuilt transparently.
pub struct AdaptiveDecoder {
    decoder: Box<dyn ObservableDecoder>,
    factory: Box<DecoderFactory>,
    current_dem: String,
    rebuild_count: usize,
    /// Calibration monitoring: recent outcomes (true = logical error).
    recent_outcomes: std::collections::VecDeque<bool>,
    /// Size of the monitoring window.
    monitoring_window: usize,
    /// Error rate threshold above which recalibration is recommended.
    recalibration_threshold: f64,
}

impl AdaptiveDecoder {
    /// Create from a DEM string and factory.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the factory fails.
    pub fn new<F>(dem: &str, mut factory: F) -> Result<Self, DecoderError>
    where
        F: FnMut(&str) -> Result<Box<dyn ObservableDecoder>, DecoderError> + 'static,
    {
        let decoder = factory(dem)?;
        Ok(Self {
            decoder,
            factory: Box::new(factory),
            current_dem: dem.to_string(),
            rebuild_count: 0,
            recent_outcomes: std::collections::VecDeque::new(),
            monitoring_window: 1000,
            recalibration_threshold: 0.1,
        })
    }

    /// Update the DEM and rebuild the decoder.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the factory fails on the new DEM.
    pub fn update_dem(&mut self, new_dem: &str) -> Result<(), DecoderError> {
        self.decoder = (self.factory)(new_dem)?;
        self.current_dem = new_dem.to_string();
        self.rebuild_count += 1;
        Ok(())
    }

    /// Number of times the decoder has been rebuilt.
    #[must_use]
    pub fn rebuild_count(&self) -> usize {
        self.rebuild_count
    }

    /// The current DEM string.
    #[must_use]
    pub fn current_dem(&self) -> &str {
        &self.current_dem
    }

    /// Report a logical outcome for calibration monitoring.
    ///
    /// `was_logical_error`: true if this QEC cycle resulted in a logical error.
    /// The adaptive decoder tracks recent error rate and signals when
    /// recalibration may be needed (noise model drift).
    pub fn report_outcome(&mut self, was_logical_error: bool) {
        self.recent_outcomes.push_back(was_logical_error);
        if self.recent_outcomes.len() > self.monitoring_window {
            self.recent_outcomes.pop_front();
        }
    }

    /// Check if recalibration is recommended.
    ///
    /// Returns true if the recent error rate exceeds `recalibration_threshold`.
    /// This suggests the noise model has drifted and the DEM should be regenerated.
    #[must_use]
    pub fn should_recalibrate(&self) -> bool {
        if self.recent_outcomes.len() < self.monitoring_window / 2 {
            return false; // Not enough data
        }
        let errors = self.recent_outcomes.iter().filter(|&&e| e).count();
        let rate = fraction(errors, self.recent_outcomes.len());
        rate > self.recalibration_threshold
    }

    /// Recent logical error rate from monitoring window.
    #[must_use]
    pub fn recent_error_rate(&self) -> f64 {
        if self.recent_outcomes.is_empty() {
            return 0.0;
        }
        let errors = self.recent_outcomes.iter().filter(|&&e| e).count();
        fraction(errors, self.recent_outcomes.len())
    }

    /// Set the monitoring window size and recalibration threshold.
    pub fn set_monitoring(&mut self, window: usize, threshold: f64) {
        self.monitoring_window = window;
        self.recalibration_threshold = threshold;
    }
}

impl ObservableDecoder for AdaptiveDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decoder.decode_to_observables(syndrome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_decode() {
        let dec = AdaptiveDecoder::new("error(0.1) D0\n", |_dem| {
            struct Zero;
            impl ObservableDecoder for Zero {
                fn decode_to_observables(&mut self, _: &[u8]) -> Result<u64, DecoderError> {
                    Ok(0)
                }
            }
            Ok(Box::new(Zero))
        });
        assert!(dec.is_ok());
        let mut dec = dec.unwrap();
        assert_eq!(dec.decode_to_observables(&[0]).unwrap(), 0);
        assert_eq!(dec.rebuild_count(), 0);
    }

    #[test]
    fn test_adaptive_update() {
        let mut dec = AdaptiveDecoder::new("error(0.1) D0\n", |_dem| {
            struct Zero;
            impl ObservableDecoder for Zero {
                fn decode_to_observables(&mut self, _: &[u8]) -> Result<u64, DecoderError> {
                    Ok(0)
                }
            }
            Ok(Box::new(Zero))
        })
        .unwrap();

        dec.update_dem("error(0.2) D0\n").unwrap();
        assert_eq!(dec.rebuild_count(), 1);
        assert_eq!(dec.current_dem(), "error(0.2) D0\n");
    }
}
