// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.

//! Fast batch composite processing with geometric sampling.
//!
//! This module provides optimized batch noise processing using geometric sampling
//! with lazy filter checks. The key insight is that for low probability events,
//! geometric sampling generates only O(n*p) candidates, and checking filters
//! on just those candidates is much faster than any approach that touches all n qubits.
//!
//! # Performance
//!
//! For 1M qubits at p=1e-4 with 1% leaked:
//! - Geometric samples: ~100 candidates (O(n*p))
//! - Filter checks: ~100 lookups (O(affected))
//! - Actions: ~99 applied (O(affected))
//! - **Total: ~4 µs**
//!
//! # Example
//!
//! ```
//! use pecos_neo::noise::composite::batch_composite::*;
//! use pecos_neo::noise::composite::prelude::*;
//! use pecos_neo::noise::NoiseContext;
//! use pecos_rng::PecosRng;
//!
//! // Create optimized batch processor
//! let processor = fast_depolarizing(1e-4, pauli());
//!
//! // Process all qubits
//! let mut state = BatchState::new(1_000);
//! state.mark_all_active();
//! let mut ctx = NoiseContext::new();
//! let mut rng = PecosRng::seed_from_u64(42);
//! let result = processor.process_all(&state, &mut ctx, &mut rng);
//! ```

use super::Primitive;
use super::batch::GeometricSampler;
use super::response::CompositeResponse;
use crate::noise::NoiseContext;
use pecos_core::QubitId;
use pecos_rng::PecosRng;
use smallvec::SmallVec;

// ============================================================================
// Batch State Context
// ============================================================================

/// State context for batch operations.
///
/// Tracks leaked and active state for qubits using efficient bit storage.
#[derive(Debug, Clone)]
pub struct BatchState {
    num_qubits: usize,
    /// Leaked state - stored as packed bits for memory efficiency.
    leaked: Vec<u64>,
    /// Active/prepared state - stored as packed bits.
    active: Vec<u64>,
}

impl BatchState {
    /// Create a new batch state.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let num_words = num_qubits.div_ceil(64);
        Self {
            num_qubits,
            leaked: vec![0; num_words],
            active: vec![0; num_words],
        }
    }

    /// Get the number of qubits.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Check if a qubit is leaked.
    #[inline]
    #[must_use]
    pub fn is_leaked(&self, qubit: QubitId) -> bool {
        let idx = qubit.0;
        if idx >= self.num_qubits {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.leaked[word] >> bit) & 1 != 0
    }

    /// Mark a qubit as leaked.
    #[inline]
    pub fn mark_leaked(&mut self, qubit: QubitId) {
        let idx = qubit.0;
        if idx < self.num_qubits {
            let word = idx / 64;
            let bit = idx % 64;
            self.leaked[word] |= 1 << bit;
        }
    }

    /// Mark a qubit as not leaked.
    #[inline]
    pub fn mark_unleaked(&mut self, qubit: QubitId) {
        let idx = qubit.0;
        if idx < self.num_qubits {
            let word = idx / 64;
            let bit = idx % 64;
            self.leaked[word] &= !(1 << bit);
        }
    }

    /// Check if a qubit is active.
    #[inline]
    #[must_use]
    pub fn is_active(&self, qubit: QubitId) -> bool {
        let idx = qubit.0;
        if idx >= self.num_qubits {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        (self.active[word] >> bit) & 1 != 0
    }

    /// Mark a qubit as active.
    #[inline]
    pub fn mark_active(&mut self, qubit: QubitId) {
        let idx = qubit.0;
        if idx < self.num_qubits {
            let word = idx / 64;
            let bit = idx % 64;
            self.active[word] |= 1 << bit;
        }
    }

    /// Mark all qubits as active.
    pub fn mark_all_active(&mut self) {
        self.active.fill(u64::MAX);
        // Clear excess bits in the last word
        let excess = self.num_qubits % 64;
        if excess > 0 && !self.active.is_empty() {
            let last = self.active.len() - 1;
            self.active[last] &= (1u64 << excess) - 1;
        }
    }

    /// Count leaked qubits.
    #[must_use]
    pub fn count_leaked(&self) -> usize {
        self.leaked.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Count active qubits.
    #[must_use]
    pub fn count_active(&self) -> usize {
        self.active.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.leaked.fill(0);
        self.active.fill(0);
    }
}

// ============================================================================
// Batch Flow Result
// ============================================================================

/// Result of batch processing.
#[derive(Debug, Clone, Default)]
pub struct BatchCompositeResult {
    /// Qubits to skip.
    pub skip: SmallVec<[QubitId; 8]>,
    /// Qubits that leaked.
    pub leaked: SmallVec<[QubitId; 8]>,
    /// Qubits that unleaked.
    pub unleaked: SmallVec<[QubitId; 8]>,
    /// Flow responses per qubit.
    pub responses: SmallVec<[(QubitId, CompositeResponse); 16]>,
}

impl BatchCompositeResult {
    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skip.is_empty()
            && self.leaked.is_empty()
            && self.unleaked.is_empty()
            && self.responses.is_empty()
    }
}

// ============================================================================
// Fast Batch Processor
// ============================================================================

/// Fast batch processor using geometric sampling with lazy filter checks.
///
/// This is the recommended approach for high-scale batch noise processing.
/// It uses geometric sampling to generate O(n*p) candidate indices directly,
/// then checks filters only on those candidates.
///
/// # Performance
///
/// For 1M qubits at p=1e-4:
/// - ~100 geometric samples generated
/// - ~100 filter checks (`is_leaked` lookups)
/// - ~99 actions applied
/// - **Total: ~4 µs**
///
/// Compare to bit-vector approach (~2.3 ms) or linear iteration (~20 s).
pub struct FastBatchProcessor<P: Primitive> {
    sampler: GeometricSampler,
    action: P,
    check_not_leaked: bool,
    check_active: bool,
}

impl<P: Primitive> FastBatchProcessor<P> {
    /// Create a new fast batch processor.
    pub fn new(probability: f64, action: P) -> Self {
        Self {
            sampler: GeometricSampler::new(probability),
            action,
            check_not_leaked: false,
            check_active: false,
        }
    }

    /// Enable not-leaked filter check on candidates.
    #[must_use]
    pub fn filter_not_leaked(mut self) -> Self {
        self.check_not_leaked = true;
        self
    }

    /// Enable active filter check on candidates.
    #[must_use]
    pub fn filter_active(mut self) -> Self {
        self.check_active = true;
        self
    }

    /// Process a range of qubits using geometric sampling + lazy filter.
    pub fn process_range(
        &self,
        start: usize,
        end: usize,
        state: &BatchState,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> BatchCompositeResult {
        let mut result = BatchCompositeResult::default();
        let n = end.saturating_sub(start);
        if n == 0 {
            return result;
        }

        // Phase 1: Geometric sampling to get candidate indices (O(n*p))
        let candidates = self.sampler.sample_range(0, n, rng);

        // Phase 2+3: Filter check + action for each candidate (O(affected))
        for local_idx in candidates {
            let global_idx = start + local_idx;
            let qubit = QubitId(global_idx);

            // Lazy filter checks
            if self.check_not_leaked && state.is_leaked(qubit) {
                continue;
            }
            if self.check_active && !state.is_active(qubit) {
                continue;
            }

            // Apply action
            let response = self.action.apply(qubit, ctx, rng);
            if !response.is_none() {
                if response.causes_leak() {
                    result.leaked.push(qubit);
                }
                if matches!(response, CompositeResponse::Unleak) {
                    result.unleaked.push(qubit);
                }
                if response.skips_gate() {
                    result.skip.push(qubit);
                }
                result.responses.push((qubit, response));
            }
        }

        result
    }

    /// Process all qubits in the state.
    pub fn process_all(
        &self,
        state: &BatchState,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> BatchCompositeResult {
        self.process_range(0, state.num_qubits(), state, ctx, rng)
    }
}

/// Create a fast batch processor for depolarizing noise.
///
/// This is the recommended way to create a batch processor for common
/// depolarizing noise scenarios. It automatically filters out leaked qubits.
pub fn fast_depolarizing<P: Primitive>(probability: f64, action: P) -> FastBatchProcessor<P> {
    FastBatchProcessor::new(probability, action).filter_not_leaked()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::composite::action::actions::*;

    #[test]
    fn test_batch_state() {
        let mut state = BatchState::new(1000);

        state.mark_leaked(QubitId(5));
        state.mark_leaked(QubitId(100));
        state.mark_active(QubitId(0));
        state.mark_active(QubitId(5));

        assert!(state.is_leaked(QubitId(5)));
        assert!(state.is_leaked(QubitId(100)));
        assert!(!state.is_leaked(QubitId(0)));

        assert!(state.is_active(QubitId(0)));
        assert!(state.is_active(QubitId(5)));
        assert!(!state.is_active(QubitId(100)));

        assert_eq!(state.count_leaked(), 2);
        assert_eq!(state.count_active(), 2);
    }

    #[test]
    fn test_batch_state_mark_all_active() {
        let mut state = BatchState::new(1000);
        state.mark_all_active();

        assert!(state.is_active(QubitId(0)));
        assert!(state.is_active(QubitId(500)));
        assert!(state.is_active(QubitId(999)));
        assert_eq!(state.count_active(), 1000);
    }

    #[test]
    fn test_fast_batch_processor_basic() {
        let mut state = BatchState::new(100_000);
        state.mark_all_active();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let processor = fast_depolarizing(1e-4, inject_x());
        let result = processor.process_all(&state, &mut ctx, &mut rng);

        // ~10 affected (0.01% of 100K)
        assert!(
            result.responses.len() < 30,
            "Expected ~10, got {}",
            result.responses.len()
        );
    }

    #[test]
    fn test_fast_batch_processor_filters_leaked() {
        let mut state = BatchState::new(10_000);
        state.mark_all_active();

        // Mark 50% as leaked
        for i in (0..10_000).step_by(2) {
            state.mark_leaked(QubitId(i));
        }

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let processor = fast_depolarizing(0.01, inject_x());
        let result = processor.process_all(&state, &mut ctx, &mut rng);

        // Check that no leaked qubits were affected
        for (q, _) in &result.responses {
            assert!(
                !state.is_leaked(*q),
                "Leaked qubit {q:?} should not be affected"
            );
        }

        // ~50 affected (1% of 5K non-leaked)
        assert!(
            result.responses.len() > 25 && result.responses.len() < 80,
            "Expected ~50, got {}",
            result.responses.len()
        );
    }

    #[test]
    fn test_fast_batch_processor_high_leaked_ratio() {
        let mut state = BatchState::new(10_000);
        state.mark_all_active();

        // Mark 99% as leaked
        for i in (0..10_000).filter(|i| i % 100 != 0) {
            state.mark_leaked(QubitId(i));
        }
        assert_eq!(state.count_leaked(), 9900);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let processor = fast_depolarizing(0.1, inject_x());
        let result = processor.process_all(&state, &mut ctx, &mut rng);

        // Only ~10 non-leaked qubits, 10% = ~1
        for (q, _) in &result.responses {
            assert!(
                !state.is_leaked(*q),
                "Leaked qubit {q:?} should not be affected"
            );
        }
    }

    #[test]
    fn test_fast_batch_processor_statistical() {
        let mut state = BatchState::new(100_000);
        state.mark_all_active();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(12345);

        // 1% probability
        let processor = FastBatchProcessor::new(0.01, inject_x());
        let result = processor.process_all(&state, &mut ctx, &mut rng);

        // Should be ~1000 affected (1% of 100K)
        let count = result.responses.len();
        assert!(count > 800 && count < 1200, "Expected ~1000, got {count}");
    }
}
