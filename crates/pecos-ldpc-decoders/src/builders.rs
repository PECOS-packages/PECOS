//! Builder patterns for LDPC decoders
//!
//! This module provides ergonomic builder patterns for constructing LDPC decoders.
//! Instead of passing 10+ parameters to a constructor, you can use a fluent API:
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use pecos_ldpc_decoders::{BpOsdDecoder, SparseMatrix, OsdMethod};
//! use ndarray::arr2;
//!
//! let dense = arr2(&[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]);
//! let pcm = SparseMatrix::from_dense(&dense.view());
//! let decoder = BpOsdDecoder::builder(&pcm)
//!     .error_rate(0.01)
//!     .max_iter(100)
//!     .osd_method(OsdMethod::Osd0)
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use crate::{BpMethod, BpSchedule, InputVectorType, OsdMethod, Result, SparseMatrix, UfMethod};

// ============================================================================
// BP+OSD Decoder Builder
// ============================================================================

/// Builder for `BpOsdDecoder`
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use pecos_ldpc_decoders::{BpOsdDecoder, SparseMatrix, BpMethod, OsdMethod};
/// use ndarray::arr2;
///
/// let dense = arr2(&[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]);
/// let pcm = SparseMatrix::from_dense(&dense.view());
///
/// let decoder = BpOsdDecoder::builder(&pcm)
///     .error_rate(0.01)
///     .max_iter(100)
///     .bp_method(BpMethod::ProductSum)
///     .osd_method(OsdMethod::Osd0)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct BpOsdBuilder<'a> {
    pcm: &'a SparseMatrix,
    error_rate: Option<f64>,
    error_channel: Option<Vec<f64>>,
    max_iter: usize,
    bp_method: BpMethod,
    bp_schedule: BpSchedule,
    ms_scaling_factor: f64,
    osd_method: OsdMethod,
    osd_order: usize,
    input_vector_type: InputVectorType,
    omp_thread_count: Option<usize>,
    serial_schedule_order: Option<Vec<i32>>,
    random_schedule_seed: Option<i32>,
}

impl<'a> BpOsdBuilder<'a> {
    /// Create a new builder with the given parity check matrix
    pub fn new(pcm: &'a SparseMatrix) -> Self {
        Self {
            pcm,
            error_rate: None,
            error_channel: None,
            max_iter: 0, // 0 means adaptive (use n)
            bp_method: BpMethod::ProductSum,
            bp_schedule: BpSchedule::Parallel,
            ms_scaling_factor: 1.0,
            osd_method: OsdMethod::Off,
            osd_order: 0,
            input_vector_type: InputVectorType::Syndrome,
            omp_thread_count: None,
            serial_schedule_order: None,
            random_schedule_seed: None,
        }
    }

    /// Set uniform error rate for all bits
    pub fn error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self.error_channel = None;
        self
    }

    /// Set per-bit error probabilities
    pub fn error_channel(mut self, probs: Vec<f64>) -> Self {
        self.error_channel = Some(probs);
        self.error_rate = None;
        self
    }

    /// Set maximum iterations (0 = adaptive, uses number of columns)
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the BP method (`ProductSum` or `MinimumSum`)
    pub fn bp_method(mut self, method: BpMethod) -> Self {
        self.bp_method = method;
        self
    }

    /// Set the BP schedule (Serial, Parallel, or `SerialRelative`)
    pub fn bp_schedule(mut self, schedule: BpSchedule) -> Self {
        self.bp_schedule = schedule;
        self
    }

    /// Set the minimum-sum scaling factor (only used with `MinimumSum`)
    pub fn ms_scaling_factor(mut self, factor: f64) -> Self {
        self.ms_scaling_factor = factor;
        self
    }

    /// Set the OSD method (Off, Osd0, `OsdE`, or `OsdCs`)
    pub fn osd_method(mut self, method: OsdMethod) -> Self {
        self.osd_method = method;
        self
    }

    /// Set the OSD order (only used with `OsdE` or `OsdCs`)
    pub fn osd_order(mut self, order: usize) -> Self {
        self.osd_order = order;
        self
    }

    /// Set the input vector type (Syndrome, `ReceivedVector`, or Auto)
    pub fn input_vector_type(mut self, input_type: InputVectorType) -> Self {
        self.input_vector_type = input_type;
        self
    }

    /// Set the number of OpenMP threads
    pub fn omp_threads(mut self, count: usize) -> Self {
        self.omp_thread_count = Some(count);
        self
    }

    /// Set a custom serial schedule order
    pub fn serial_schedule_order(mut self, order: Vec<i32>) -> Self {
        self.serial_schedule_order = Some(order);
        self
    }

    /// Set random schedule seed (-1 = disabled)
    pub fn random_schedule_seed(mut self, seed: i32) -> Self {
        self.random_schedule_seed = Some(seed);
        self
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if:
    /// - Neither `error_rate` nor `error_channel` was set
    /// - `error_channel` length doesn't match matrix columns
    /// - OSD is enabled but input type is not Syndrome
    pub fn build(self) -> Result<crate::BpOsdDecoder> {
        crate::BpOsdDecoder::new(
            self.pcm,
            self.error_rate,
            self.error_channel.as_deref(),
            self.max_iter,
            self.bp_method,
            self.bp_schedule,
            self.ms_scaling_factor,
            self.osd_method,
            self.osd_order,
            self.input_vector_type,
            self.omp_thread_count,
            self.serial_schedule_order.as_deref(),
            self.random_schedule_seed,
        )
    }
}

// ============================================================================
// BP+LSD Decoder Builder
// ============================================================================

/// Builder for `BpLsdDecoder`
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use pecos_ldpc_decoders::{BpLsdDecoder, SparseMatrix, OsdMethod};
/// use ndarray::arr2;
///
/// let dense = arr2(&[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]);
/// let pcm = SparseMatrix::from_dense(&dense.view());
///
/// let decoder = BpLsdDecoder::builder(&pcm)
///     .error_rate(0.01)
///     .max_iter(100)
///     .lsd_method(OsdMethod::Osd0)
///     .lsd_order(0)
///     .build()?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct BpLsdBuilder<'a> {
    pcm: &'a SparseMatrix,
    error_rate: Option<f64>,
    error_channel: Option<Vec<f64>>,
    max_iter: usize,
    bp_method: BpMethod,
    bp_schedule: BpSchedule,
    ms_scaling_factor: f64,
    lsd_method: OsdMethod,
    lsd_order: usize,
    bits_per_step: usize,
    input_vector_type: InputVectorType,
    omp_thread_count: Option<usize>,
    serial_schedule_order: Option<Vec<i32>>,
    random_schedule_seed: Option<i32>,
}

impl<'a> BpLsdBuilder<'a> {
    /// Create a new builder with the given parity check matrix
    pub fn new(pcm: &'a SparseMatrix) -> Self {
        Self {
            pcm,
            error_rate: None,
            error_channel: None,
            max_iter: 0,
            bp_method: BpMethod::ProductSum,
            bp_schedule: BpSchedule::Parallel,
            ms_scaling_factor: 1.0,
            lsd_method: OsdMethod::Off,
            lsd_order: 0,
            bits_per_step: 0,
            input_vector_type: InputVectorType::Syndrome,
            omp_thread_count: None,
            serial_schedule_order: None,
            random_schedule_seed: None,
        }
    }

    /// Set uniform error rate for all bits
    pub fn error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self.error_channel = None;
        self
    }

    /// Set per-bit error probabilities
    pub fn error_channel(mut self, probs: Vec<f64>) -> Self {
        self.error_channel = Some(probs);
        self.error_rate = None;
        self
    }

    /// Set maximum iterations (0 = adaptive)
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the BP method
    pub fn bp_method(mut self, method: BpMethod) -> Self {
        self.bp_method = method;
        self
    }

    /// Set the BP schedule
    pub fn bp_schedule(mut self, schedule: BpSchedule) -> Self {
        self.bp_schedule = schedule;
        self
    }

    /// Set the minimum-sum scaling factor
    pub fn ms_scaling_factor(mut self, factor: f64) -> Self {
        self.ms_scaling_factor = factor;
        self
    }

    /// Set the LSD method
    pub fn lsd_method(mut self, method: OsdMethod) -> Self {
        self.lsd_method = method;
        self
    }

    /// Set the LSD order
    pub fn lsd_order(mut self, order: usize) -> Self {
        self.lsd_order = order;
        self
    }

    /// Set bits per step for LSD
    pub fn bits_per_step(mut self, bits: usize) -> Self {
        self.bits_per_step = bits;
        self
    }

    /// Set the input vector type
    pub fn input_vector_type(mut self, input_type: InputVectorType) -> Self {
        self.input_vector_type = input_type;
        self
    }

    /// Set the number of OpenMP threads
    pub fn omp_threads(mut self, count: usize) -> Self {
        self.omp_thread_count = Some(count);
        self
    }

    /// Set a custom serial schedule order
    pub fn serial_schedule_order(mut self, order: Vec<i32>) -> Self {
        self.serial_schedule_order = Some(order);
        self
    }

    /// Set random schedule seed
    pub fn random_schedule_seed(mut self, seed: i32) -> Self {
        self.random_schedule_seed = Some(seed);
        self
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if configuration is invalid.
    pub fn build(self) -> Result<crate::BpLsdDecoder> {
        crate::BpLsdDecoder::new(
            self.pcm,
            self.error_rate,
            self.error_channel.as_deref(),
            self.max_iter,
            self.bp_method,
            self.bp_schedule,
            self.ms_scaling_factor,
            self.lsd_method,
            self.lsd_order,
            self.bits_per_step,
            self.input_vector_type,
            self.omp_thread_count,
            self.serial_schedule_order.as_deref(),
            self.random_schedule_seed,
        )
    }
}

// ============================================================================
// Soft Information BP Decoder Builder
// ============================================================================

/// Builder for `SoftInfoBpDecoder`
#[must_use]
pub struct SoftInfoBpBuilder<'a> {
    pcm: &'a SparseMatrix,
    error_rate: Option<f64>,
    error_channel: Option<Vec<f64>>,
    max_iter: usize,
    bp_method: BpMethod,
    ms_scaling_factor: f64,
    omp_thread_count: Option<usize>,
    serial_schedule_order: Option<Vec<i32>>,
    random_schedule_seed: Option<i32>,
}

impl<'a> SoftInfoBpBuilder<'a> {
    /// Create a new builder with the given parity check matrix
    pub fn new(pcm: &'a SparseMatrix) -> Self {
        Self {
            pcm,
            error_rate: None,
            error_channel: None,
            max_iter: 0,
            bp_method: BpMethod::ProductSum,
            ms_scaling_factor: 1.0,
            omp_thread_count: None,
            serial_schedule_order: None,
            random_schedule_seed: None,
        }
    }

    /// Set uniform error rate for all bits
    pub fn error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self.error_channel = None;
        self
    }

    /// Set per-bit error probabilities
    pub fn error_channel(mut self, probs: Vec<f64>) -> Self {
        self.error_channel = Some(probs);
        self.error_rate = None;
        self
    }

    /// Set maximum iterations (0 = adaptive)
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the BP method
    pub fn bp_method(mut self, method: BpMethod) -> Self {
        self.bp_method = method;
        self
    }

    /// Set the minimum-sum scaling factor
    pub fn ms_scaling_factor(mut self, factor: f64) -> Self {
        self.ms_scaling_factor = factor;
        self
    }

    /// Set the number of OpenMP threads
    pub fn omp_threads(mut self, count: usize) -> Self {
        self.omp_thread_count = Some(count);
        self
    }

    /// Set a custom serial schedule order
    pub fn serial_schedule_order(mut self, order: Vec<i32>) -> Self {
        self.serial_schedule_order = Some(order);
        self
    }

    /// Set random schedule seed
    pub fn random_schedule_seed(mut self, seed: i32) -> Self {
        self.random_schedule_seed = Some(seed);
        self
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if configuration is invalid.
    pub fn build(self) -> Result<crate::SoftInfoBpDecoder> {
        crate::SoftInfoBpDecoder::new(
            self.pcm,
            self.error_rate,
            self.error_channel.as_deref(),
            self.max_iter,
            self.bp_method,
            self.ms_scaling_factor,
            self.omp_thread_count,
            self.serial_schedule_order.as_deref(),
            self.random_schedule_seed,
        )
    }
}

// ============================================================================
// Flip Decoder Builder
// ============================================================================

/// Builder for `FlipDecoder`
#[must_use]
pub struct FlipBuilder<'a> {
    pcm: &'a SparseMatrix,
    max_iter: usize,
    pfreq: usize,
    seed: i32,
}

impl<'a> FlipBuilder<'a> {
    /// Create a new builder with the given parity check matrix
    pub fn new(pcm: &'a SparseMatrix) -> Self {
        Self {
            pcm,
            max_iter: 0,
            pfreq: 0,
            seed: 0,
        }
    }

    /// Set maximum iterations (0 = adaptive)
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set perturbation frequency (0 = never)
    pub fn pfreq(mut self, pfreq: usize) -> Self {
        self.pfreq = pfreq;
        self
    }

    /// Set random seed (0 = random)
    pub fn seed(mut self, seed: i32) -> Self {
        self.seed = seed;
        self
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if configuration is invalid.
    pub fn build(self) -> Result<crate::FlipDecoder> {
        crate::FlipDecoder::new(self.pcm, self.max_iter, self.pfreq, self.seed)
    }
}

// ============================================================================
// Union-Find Decoder Builder
// ============================================================================

/// Builder for `UnionFindDecoder`
#[must_use]
pub struct UnionFindBuilder<'a> {
    pcm: &'a SparseMatrix,
    method: UfMethod,
}

impl<'a> UnionFindBuilder<'a> {
    /// Create a new builder with the given parity check matrix
    pub fn new(pcm: &'a SparseMatrix) -> Self {
        Self {
            pcm,
            method: UfMethod::Inversion,
        }
    }

    /// Set the Union-Find method (Inversion or Peeling)
    pub fn method(mut self, method: UfMethod) -> Self {
        self.method = method;
        self
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if configuration is invalid.
    pub fn build(self) -> Result<crate::UnionFindDecoder> {
        crate::UnionFindDecoder::new(self.pcm, self.method)
    }
}

// ============================================================================
// BeliefFind Decoder Builder
// ============================================================================

/// Builder for `BeliefFindDecoder`
///
/// `BeliefFind` combines BP with Union-Find: it first tries BP, and if that fails
/// to converge, it falls back to Union-Find using the soft information from BP.
#[must_use]
pub struct BeliefFindBuilder<'a> {
    pcm: &'a SparseMatrix,
    error_rate: Option<f64>,
    error_channel: Option<Vec<f64>>,
    max_iter: usize,
    bp_method: BpMethod,
    ms_scaling_factor: f64,
    bp_schedule: BpSchedule,
    omp_thread_count: Option<usize>,
    serial_schedule_order: Option<Vec<i32>>,
    random_schedule_seed: Option<i32>,
    uf_method: UfMethod,
    bits_per_step: usize,
}

impl<'a> BeliefFindBuilder<'a> {
    /// Create a new builder with the given parity check matrix
    pub fn new(pcm: &'a SparseMatrix) -> Self {
        Self {
            pcm,
            error_rate: None,
            error_channel: None,
            max_iter: 0,
            bp_method: BpMethod::ProductSum,
            ms_scaling_factor: 1.0,
            bp_schedule: BpSchedule::Parallel,
            omp_thread_count: None,
            serial_schedule_order: None,
            random_schedule_seed: None,
            uf_method: UfMethod::Inversion,
            bits_per_step: 0,
        }
    }

    /// Set uniform error rate for all bits
    pub fn error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self.error_channel = None;
        self
    }

    /// Set per-bit error probabilities
    pub fn error_channel(mut self, probs: Vec<f64>) -> Self {
        self.error_channel = Some(probs);
        self.error_rate = None;
        self
    }

    /// Set maximum BP iterations (0 = adaptive)
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the BP method
    pub fn bp_method(mut self, method: BpMethod) -> Self {
        self.bp_method = method;
        self
    }

    /// Set the minimum-sum scaling factor
    pub fn ms_scaling_factor(mut self, factor: f64) -> Self {
        self.ms_scaling_factor = factor;
        self
    }

    /// Set the BP schedule
    pub fn bp_schedule(mut self, schedule: BpSchedule) -> Self {
        self.bp_schedule = schedule;
        self
    }

    /// Set the number of OpenMP threads
    pub fn omp_threads(mut self, count: usize) -> Self {
        self.omp_thread_count = Some(count);
        self
    }

    /// Set a custom serial schedule order
    pub fn serial_schedule_order(mut self, order: Vec<i32>) -> Self {
        self.serial_schedule_order = Some(order);
        self
    }

    /// Set random schedule seed
    pub fn random_schedule_seed(mut self, seed: i32) -> Self {
        self.random_schedule_seed = Some(seed);
        self
    }

    /// Set the Union-Find method
    pub fn uf_method(mut self, method: UfMethod) -> Self {
        self.uf_method = method;
        self
    }

    /// Set bits per step for Union-Find (0 = all)
    pub fn bits_per_step(mut self, bits: usize) -> Self {
        self.bits_per_step = bits;
        self
    }

    /// Build the decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if configuration is invalid.
    pub fn build(self) -> Result<crate::BeliefFindDecoder> {
        crate::BeliefFindDecoder::new(
            self.pcm,
            self.error_rate,
            self.error_channel.as_deref(),
            self.max_iter,
            self.bp_method,
            self.ms_scaling_factor,
            self.bp_schedule,
            self.omp_thread_count,
            self.serial_schedule_order.as_deref(),
            self.random_schedule_seed,
            self.uf_method,
            self.bits_per_step,
        )
    }
}

// ============================================================================
// Helper trait for adding builder() methods to decoders
// ============================================================================

/// Extension trait to add `builder()` method to decoder types
pub trait DecoderBuilder<'a> {
    /// The builder type for this decoder
    type Builder;

    /// Create a new builder for this decoder type
    fn builder(pcm: &'a SparseMatrix) -> Self::Builder;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn test_pcm() -> SparseMatrix {
        let dense = arr2(&[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]);
        SparseMatrix::from_dense(&dense.view())
    }

    #[test]
    fn test_bp_osd_builder() {
        let pcm = test_pcm();
        let decoder = BpOsdBuilder::new(&pcm)
            .error_rate(0.01)
            .max_iter(100)
            .bp_method(BpMethod::ProductSum)
            .osd_method(OsdMethod::Osd0)
            .build();

        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.check_count(), 3);
        assert_eq!(decoder.bit_count(), 4);
    }

    #[test]
    fn test_bp_lsd_builder() {
        let pcm = test_pcm();
        let decoder = BpLsdBuilder::new(&pcm)
            .error_rate(0.01)
            .max_iter(100)
            .lsd_method(OsdMethod::Osd0)
            .build();

        assert!(decoder.is_ok());
    }

    #[test]
    fn test_flip_builder() {
        let pcm = test_pcm();
        let decoder = FlipBuilder::new(&pcm).max_iter(100).seed(42).build();

        assert!(decoder.is_ok());
    }

    #[test]
    fn test_union_find_builder() {
        let pcm = test_pcm();
        let decoder = UnionFindBuilder::new(&pcm)
            .method(UfMethod::Inversion)
            .build();

        assert!(decoder.is_ok());
    }
}
