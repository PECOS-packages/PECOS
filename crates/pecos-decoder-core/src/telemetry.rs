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

//! Decoder telemetry wrapper for real-time monitoring.
//!
//! Wraps any `ObservableDecoder` with transparent per-decode latency
//! measurement and statistics tracking. Useful for monitoring decoder
//! health in a real QEC system.

use crate::ObservableDecoder;
use crate::errors::DecoderError;
use std::collections::VecDeque;
use std::time::Instant;

/// Live decoder statistics.
#[derive(Debug, Clone)]
pub struct DecoderTelemetry {
    /// Total decodes performed.
    pub decode_count: u64,
    /// Total decode time in nanoseconds.
    pub total_decode_ns: u64,
    /// Sum of syndrome weights (number of defects per decode).
    pub syndrome_weight_sum: u64,
    /// Number of decodes that produced nonzero observable.
    pub nonzero_observable_count: u64,
    /// Maximum single-decode latency in nanoseconds.
    pub max_decode_ns: u64,
    /// Recent decode latencies for rolling statistics.
    pub recent_latencies_ns: VecDeque<u64>,
    /// Maximum size of the rolling window.
    window_size: usize,
}

impl DecoderTelemetry {
    fn new(window_size: usize) -> Self {
        Self {
            decode_count: 0,
            total_decode_ns: 0,
            syndrome_weight_sum: 0,
            nonzero_observable_count: 0,
            max_decode_ns: 0,
            recent_latencies_ns: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    fn record(&mut self, latency_ns: u64, syndrome_weight: u64, obs_nonzero: bool) {
        self.decode_count += 1;
        self.total_decode_ns += latency_ns;
        self.syndrome_weight_sum += syndrome_weight;
        if obs_nonzero {
            self.nonzero_observable_count += 1;
        }
        if latency_ns > self.max_decode_ns {
            self.max_decode_ns = latency_ns;
        }
        if self.recent_latencies_ns.len() >= self.window_size {
            self.recent_latencies_ns.pop_front();
        }
        self.recent_latencies_ns.push_back(latency_ns);
    }

    /// Average decode latency in nanoseconds.
    #[must_use]
    pub fn avg_decode_ns(&self) -> f64 {
        if self.decode_count == 0 {
            0.0
        } else {
            self.total_decode_ns as f64 / self.decode_count as f64
        }
    }

    /// Average syndrome weight (defects per decode).
    #[must_use]
    pub fn avg_syndrome_weight(&self) -> f64 {
        if self.decode_count == 0 {
            0.0
        } else {
            self.syndrome_weight_sum as f64 / self.decode_count as f64
        }
    }

    /// Fraction of decodes that produced a nonzero observable (logical correction).
    #[must_use]
    pub fn correction_rate(&self) -> f64 {
        if self.decode_count == 0 {
            0.0
        } else {
            self.nonzero_observable_count as f64 / self.decode_count as f64
        }
    }

    /// P99 latency from recent window in nanoseconds.
    #[must_use]
    pub fn p99_latency_ns(&self) -> u64 {
        if self.recent_latencies_ns.is_empty() {
            return 0;
        }
        let mut sorted: Vec<u64> = self.recent_latencies_ns.iter().copied().collect();
        sorted.sort_unstable();
        let idx = (sorted.len() * 99 / 100).min(sorted.len() - 1);
        sorted[idx]
    }
}

/// Telemetry-instrumented decoder.
///
/// Wraps any `ObservableDecoder` with transparent latency and statistics
/// tracking. The inner decoder's behavior is unchanged.
pub struct TelemetryDecoder {
    inner: Box<dyn ObservableDecoder>,
    /// Live telemetry data. Read via `telemetry()`.
    stats: DecoderTelemetry,
}

impl TelemetryDecoder {
    /// Create with a rolling window of `window_size` recent latencies.
    #[must_use]
    pub fn new(inner: Box<dyn ObservableDecoder>, window_size: usize) -> Self {
        Self {
            inner,
            stats: DecoderTelemetry::new(window_size),
        }
    }

    /// Access the telemetry data.
    #[must_use]
    pub fn telemetry(&self) -> &DecoderTelemetry {
        &self.stats
    }

    /// Reset all statistics.
    pub fn reset_telemetry(&mut self) {
        self.stats = DecoderTelemetry::new(self.stats.window_size);
    }
}

impl ObservableDecoder for TelemetryDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let syndrome_weight = syndrome.iter().filter(|&&v| v != 0).count() as u64;
        let start = Instant::now();
        let obs = self.inner.decode_to_observables(syndrome)?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.stats.record(elapsed_ns, syndrome_weight, obs != 0);
        Ok(obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDecoder(u64);
    impl ObservableDecoder for FixedDecoder {
        fn decode_to_observables(&mut self, _: &[u8]) -> Result<u64, DecoderError> {
            Ok(self.0)
        }
    }

    #[test]
    fn test_telemetry_counts() {
        let mut dec = TelemetryDecoder::new(Box::new(FixedDecoder(1)), 100);
        dec.decode_to_observables(&[0, 1, 0]).unwrap();
        dec.decode_to_observables(&[0, 0, 0]).unwrap();

        let t = dec.telemetry();
        assert_eq!(t.decode_count, 2);
        assert_eq!(t.nonzero_observable_count, 2); // FixedDecoder always returns 1
        assert_eq!(t.syndrome_weight_sum, 1); // only first syndrome has weight 1
    }

    #[test]
    fn test_telemetry_latency() {
        let mut dec = TelemetryDecoder::new(Box::new(FixedDecoder(0)), 10);
        for _ in 0..5 {
            dec.decode_to_observables(&[]).unwrap();
        }
        let t = dec.telemetry();
        assert_eq!(t.decode_count, 5);
        assert!(t.avg_decode_ns() >= 0.0);
        assert!(t.p99_latency_ns() > 0);
    }

    #[test]
    fn test_reset() {
        let mut dec = TelemetryDecoder::new(Box::new(FixedDecoder(0)), 10);
        dec.decode_to_observables(&[]).unwrap();
        dec.reset_telemetry();
        assert_eq!(dec.telemetry().decode_count, 0);
    }
}
