//! High-level decoder interfaces
//!
//! This module provides the main decoder types for LDPC codes:
//! - `BpOsdDecoder`: Belief Propagation with Ordered Statistics Decoding
//! - `BpLsdDecoder`: Belief Propagation with Localised Statistics Decoding
//! - `SoftInfoBpDecoder`: Soft information BP with virtual check nodes
//! - `FlipDecoder`: Simple bit-flipping decoder
//! - `UnionFindDecoder`: Cluster-based decoder using Union-Find
//! - `BeliefFindDecoder`: Hybrid BP + Union-Find decoder

use super::{bridge::ffi, sparse::SparseMatrix};
use crate::{BpMethod, BpSchedule, DecodingResult, InputVectorType, LdpcError, OsdMethod};
use cxx::UniquePtr;
use ndarray::{Array1, ArrayView1};
use std::collections::BTreeMap;

/// Helper function to prepare channel probabilities
fn prepare_channel_probs(
    pcm_cols: usize,
    error_rate: Option<f64>,
    error_channel: Option<&[f64]>,
) -> Result<Vec<f64>, LdpcError> {
    match (error_rate, error_channel) {
        (Some(rate), None) => Ok(vec![rate; pcm_cols]),
        (None, Some(probs)) => {
            if probs.len() != pcm_cols {
                return Err(LdpcError::InvalidInput(
                    "Error channel length must match number of columns".to_string(),
                ));
            }
            Ok(probs.to_vec())
        }
        (None, None) => Err(LdpcError::InvalidInput(
            "Either error_rate or error_channel must be provided".to_string(),
        )),
        (Some(_), Some(_)) => Err(LdpcError::InvalidInput(
            "Cannot specify both error_rate and error_channel".to_string(),
        )),
    }
}

/// BP+OSD Decoder
pub struct BpOsdDecoder {
    inner: UniquePtr<ffi::BpOsdDecoder>,
}

impl BpOsdDecoder {
    /// Create a builder for configuring a new BP+OSD decoder
    ///
    /// This is the recommended way to construct a decoder:
    ///
    /// ```rust,ignore
    /// let decoder = BpOsdDecoder::builder(&pcm)
    ///     .error_rate(0.01)
    ///     .max_iter(100)
    ///     .osd_method(OsdMethod::Osd0)
    ///     .build()?;
    /// ```
    pub fn builder(pcm: &SparseMatrix) -> crate::builders::BpOsdBuilder<'_> {
        crate::builders::BpOsdBuilder::new(pcm)
    }

    /// Create a new BP+OSD decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the input parameters are invalid or
    /// `LdpcError::Ldpc` if the C++ decoder construction fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pcm: &SparseMatrix,
        error_rate: Option<f64>,
        error_channel: Option<&[f64]>,
        max_iter: usize,
        bp_method: BpMethod,
        bp_schedule: BpSchedule,
        ms_scaling_factor: f64,
        osd_method: OsdMethod,
        osd_order: usize,
        input_vector_type: InputVectorType,
        omp_thread_count: Option<usize>,
        serial_schedule_order: Option<&[i32]>,
        random_schedule_seed: Option<i32>,
    ) -> Result<Self, LdpcError> {
        // Validate input type if OSD is enabled
        if osd_method != OsdMethod::Off && input_vector_type != InputVectorType::Syndrome {
            return Err(LdpcError::InvalidInput(
                "OSD decoding requires syndrome input. Please use InputVectorType::Syndrome when OSD is enabled.".to_string()
            ));
        }

        // Prepare channel probabilities
        let channel_probs = prepare_channel_probs(pcm.cols, error_rate, error_channel)?;

        // Create sparse matrix representation for FFI
        let sparse_repr = pcm.to_ffi_repr();

        // Handle adaptive iterations (0 means use n as max_iter)
        let actual_max_iter = if max_iter == 0 { pcm.cols } else { max_iter };

        // Default thread count to 1 if not specified
        let threads = omp_thread_count.unwrap_or(1);

        // Default serial schedule order to empty
        let schedule_order = serial_schedule_order.unwrap_or(&[]);

        // Default random schedule seed to -1 (disabled)
        let seed = random_schedule_seed.unwrap_or(-1);

        let inner = ffi::create_bp_osd_decoder(
            &sparse_repr,
            &channel_probs,
            i32::try_from(actual_max_iter).unwrap_or(i32::MAX),
            bp_method.to_ffi(),
            bp_schedule.to_ffi(),
            ms_scaling_factor,
            osd_method.to_ffi(),
            i32::try_from(osd_order).unwrap_or(0),
            input_vector_type.to_ffi(),
            i32::try_from(threads).unwrap_or(1),
            schedule_order,
            seed,
        )
        .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Self { inner })
    }

    /// Decode an input vector (syndrome or received vector based on `input_vector_type`)
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::Ldpc` if the C++ decoder encounters an error during decoding.
    ///
    /// # Panics
    ///
    /// Panics if the input array is not contiguous in memory.
    pub fn decode(&mut self, input: &ArrayView1<u8>) -> Result<DecodingResult, LdpcError> {
        // Input validation is done in the C++ code based on input_vector_type
        let input_slice = input.as_slice().unwrap();
        let result = ffi::decode_bp_osd(self.inner.pin_mut(), input_slice)
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(DecodingResult {
            decoding: Array1::from_vec(result.decoding),
            converged: result.converged,
            iterations: usize::try_from(result.iterations).unwrap_or(0),
        })
    }

    /// Get log probability ratios from the last decoding
    ///
    /// # Errors
    ///
    /// This method currently does not return errors but the signature is maintained for consistency.
    pub fn log_prob_ratios(&self) -> Result<Array1<f64>, LdpcError> {
        let llrs = ffi::get_log_prob_ratios_osd(&self.inner);
        Ok(Array1::from_vec(llrs))
    }

    /// Get the number of checks (rows in PCM)
    #[must_use]
    pub fn check_count(&self) -> usize {
        usize::try_from(ffi::get_check_count_osd(&self.inner)).unwrap_or(0)
    }

    /// Get the number of bits (columns in PCM)
    #[must_use]
    pub fn bit_count(&self) -> usize {
        usize::try_from(ffi::get_bit_count_osd(&self.inner)).unwrap_or(0)
    }

    /// Get the channel probabilities
    #[must_use]
    pub fn channel_probs(&self) -> Array1<f64> {
        Array1::from_vec(ffi::get_channel_probs_osd(&self.inner))
    }

    /// Get the maximum iterations
    #[must_use]
    pub fn max_iter(&self) -> usize {
        usize::try_from(ffi::get_max_iter_osd(&self.inner)).unwrap_or(0)
    }

    /// Get the BP method
    #[must_use]
    pub fn bp_method(&self) -> BpMethod {
        match ffi::get_bp_method_osd(&self.inner) {
            0 => BpMethod::ProductSum,
            1 => BpMethod::MinimumSum,
            _ => unreachable!(),
        }
    }

    /// Get the BP schedule
    #[must_use]
    pub fn bp_schedule(&self) -> BpSchedule {
        match ffi::get_bp_schedule_osd(&self.inner) {
            0 => BpSchedule::Serial,
            1 => BpSchedule::Parallel,
            2 => BpSchedule::SerialRelative,
            _ => unreachable!(),
        }
    }

    /// Get the minimum-sum scaling factor
    #[must_use]
    pub fn ms_scaling_factor(&self) -> f64 {
        ffi::get_ms_scaling_factor_osd(&self.inner)
    }

    /// Get the OSD method
    #[must_use]
    pub fn osd_method(&self) -> OsdMethod {
        match ffi::get_osd_method_osd(&self.inner) {
            0 => OsdMethod::Off,
            1 => OsdMethod::Osd0,
            2 => OsdMethod::OsdE,
            3 => OsdMethod::OsdCs,
            _ => unreachable!(),
        }
    }

    /// Get the OSD order
    #[must_use]
    pub fn osd_order(&self) -> usize {
        usize::try_from(ffi::get_osd_order_osd(&self.inner)).unwrap_or(0)
    }

    /// Check if the last decoding converged
    #[must_use]
    pub fn converged(&self) -> bool {
        ffi::get_converged_osd(&self.inner)
    }

    /// Get the number of iterations from the last decoding
    #[must_use]
    pub fn iterations(&self) -> usize {
        usize::try_from(ffi::get_iterations_osd(&self.inner)).unwrap_or(0)
    }

    /// Get the BP decoding result (before OSD)
    #[must_use]
    pub fn bp_decoding(&self) -> Array1<u8> {
        Array1::from_vec(ffi::get_bp_decoding_osd(&self.inner))
    }

    /// Get the input vector type
    #[must_use]
    pub fn input_vector_type(&self) -> InputVectorType {
        match ffi::get_input_vector_type_osd(&self.inner) {
            0 => InputVectorType::Syndrome,
            1 => InputVectorType::ReceivedVector,
            2 => InputVectorType::Auto,
            _ => unreachable!(),
        }
    }

    /// Get the OpenMP thread count
    #[must_use]
    pub fn omp_thread_count(&self) -> usize {
        usize::try_from(ffi::get_omp_thread_count_osd(&self.inner)).unwrap_or(1)
    }

    /// Get the random schedule seed
    #[must_use]
    pub fn random_schedule_seed(&self) -> i32 {
        ffi::get_random_schedule_seed_osd(&self.inner)
    }
}

/// BP+LSD Decoder
pub struct BpLsdDecoder {
    inner: UniquePtr<ffi::BpLsdDecoder>,
}

impl BpLsdDecoder {
    /// Create a builder for configuring a new BP+LSD decoder
    ///
    /// This is the recommended way to construct a decoder:
    ///
    /// ```rust,ignore
    /// let decoder = BpLsdDecoder::builder(&pcm)
    ///     .error_rate(0.01)
    ///     .max_iter(100)
    ///     .lsd_method(OsdMethod::Osd0)
    ///     .build()?;
    /// ```
    pub fn builder(pcm: &SparseMatrix) -> crate::builders::BpLsdBuilder<'_> {
        crate::builders::BpLsdBuilder::new(pcm)
    }

    /// Create a new BP+LSD decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the input parameters are invalid or
    /// `LdpcError::Ldpc` if the C++ decoder construction fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pcm: &SparseMatrix,
        error_rate: Option<f64>,
        error_channel: Option<&[f64]>,
        max_iter: usize,
        bp_method: BpMethod,
        bp_schedule: BpSchedule,
        ms_scaling_factor: f64,
        lsd_method: OsdMethod,
        lsd_order: usize,
        bits_per_step: usize,
        input_vector_type: InputVectorType,
        omp_thread_count: Option<usize>,
        serial_schedule_order: Option<&[i32]>,
        random_schedule_seed: Option<i32>,
    ) -> Result<Self, LdpcError> {
        // Validate input type - LSD requires syndrome input
        if input_vector_type != InputVectorType::Syndrome {
            return Err(LdpcError::InvalidInput(
                "LSD decoding requires syndrome input. Please use InputVectorType::Syndrome."
                    .to_string(),
            ));
        }

        // Prepare channel probabilities
        let channel_probs = prepare_channel_probs(pcm.cols, error_rate, error_channel)?;

        // Create sparse matrix representation for FFI
        let sparse_repr = pcm.to_ffi_repr();

        // Handle adaptive iterations (0 means use n as max_iter)
        let actual_max_iter = if max_iter == 0 { pcm.cols } else { max_iter };

        // Default thread count to 1 if not specified
        let threads = omp_thread_count.unwrap_or(1);

        // Default serial schedule order to empty
        let schedule_order = serial_schedule_order.unwrap_or(&[]);

        // Default random schedule seed to -1 (disabled)
        let seed = random_schedule_seed.unwrap_or(-1);

        let inner = ffi::create_bp_lsd_decoder(
            &sparse_repr,
            &channel_probs,
            i32::try_from(actual_max_iter).unwrap_or(i32::MAX),
            bp_method.to_ffi(),
            bp_schedule.to_ffi(),
            ms_scaling_factor,
            lsd_method.to_ffi(),
            i32::try_from(lsd_order).unwrap_or(0),
            i32::try_from(bits_per_step).unwrap_or(0),
            input_vector_type.to_ffi(),
            i32::try_from(threads).unwrap_or(1),
            schedule_order,
            seed,
        )
        .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Self { inner })
    }

    /// Decode an input vector (syndrome or received vector based on `input_vector_type`)
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::Ldpc` if the C++ decoder encounters an error during decoding.
    ///
    /// # Panics
    ///
    /// Panics if the input array is not contiguous in memory.
    pub fn decode(&mut self, input: &ArrayView1<u8>) -> Result<DecodingResult, LdpcError> {
        // Input validation is done in the C++ code based on input_vector_type
        let input_slice = input.as_slice().unwrap();
        let result = ffi::decode_bp_lsd(self.inner.pin_mut(), input_slice)
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(DecodingResult {
            decoding: Array1::from_vec(result.decoding),
            converged: result.converged,
            iterations: usize::try_from(result.iterations).unwrap_or(0),
        })
    }

    /// Get log probability ratios from the last decoding
    ///
    /// # Errors
    ///
    /// This method currently does not return errors but the signature is maintained for consistency.
    pub fn log_prob_ratios(&self) -> Result<Array1<f64>, LdpcError> {
        let llrs = ffi::get_log_prob_ratios_lsd(&self.inner);
        Ok(Array1::from_vec(llrs))
    }

    /// Get the number of checks (rows in PCM)
    #[must_use]
    pub fn check_count(&self) -> usize {
        usize::try_from(ffi::get_check_count_lsd(&self.inner)).unwrap_or(0)
    }

    /// Get the number of bits (columns in PCM)
    #[must_use]
    pub fn bit_count(&self) -> usize {
        usize::try_from(ffi::get_bit_count_lsd(&self.inner)).unwrap_or(0)
    }

    /// Get the channel probabilities
    #[must_use]
    pub fn channel_probs(&self) -> Array1<f64> {
        Array1::from_vec(ffi::get_channel_probs_lsd(&self.inner))
    }

    /// Get the maximum iterations
    #[must_use]
    pub fn max_iter(&self) -> usize {
        usize::try_from(ffi::get_max_iter_lsd(&self.inner)).unwrap_or(0)
    }

    /// Get the BP method
    #[must_use]
    pub fn bp_method(&self) -> BpMethod {
        match ffi::get_bp_method_lsd(&self.inner) {
            0 => BpMethod::ProductSum,
            1 => BpMethod::MinimumSum,
            _ => unreachable!(),
        }
    }

    /// Get the BP schedule
    #[must_use]
    pub fn bp_schedule(&self) -> BpSchedule {
        match ffi::get_bp_schedule_lsd(&self.inner) {
            0 => BpSchedule::Serial,
            1 => BpSchedule::Parallel,
            2 => BpSchedule::SerialRelative,
            _ => unreachable!(),
        }
    }

    /// Get the minimum-sum scaling factor
    #[must_use]
    pub fn ms_scaling_factor(&self) -> f64 {
        ffi::get_ms_scaling_factor_lsd(&self.inner)
    }

    /// Get the LSD method
    #[must_use]
    pub fn lsd_method(&self) -> OsdMethod {
        match ffi::get_lsd_method_lsd(&self.inner) {
            0 => OsdMethod::Off,
            1 => OsdMethod::Osd0,
            2 => OsdMethod::OsdE,
            3 => OsdMethod::OsdCs,
            _ => unreachable!(),
        }
    }

    /// Get the LSD order
    #[must_use]
    pub fn lsd_order(&self) -> usize {
        usize::try_from(ffi::get_lsd_order_lsd(&self.inner)).unwrap_or(0)
    }

    /// Get the bits per step
    #[must_use]
    pub fn bits_per_step(&self) -> usize {
        usize::try_from(ffi::get_bits_per_step_lsd(&self.inner)).unwrap_or(0)
    }

    /// Check if the last decoding converged
    #[must_use]
    pub fn converged(&self) -> bool {
        ffi::get_converged_lsd(&self.inner)
    }

    /// Get the number of iterations from the last decoding
    #[must_use]
    pub fn iterations(&self) -> usize {
        usize::try_from(ffi::get_iterations_lsd(&self.inner)).unwrap_or(0)
    }

    /// Get the input vector type
    #[must_use]
    pub fn input_vector_type(&self) -> InputVectorType {
        match ffi::get_input_vector_type_lsd(&self.inner) {
            0 => InputVectorType::Syndrome,
            1 => InputVectorType::ReceivedVector,
            2 => InputVectorType::Auto,
            _ => unreachable!(),
        }
    }

    /// Get the OpenMP thread count
    #[must_use]
    pub fn omp_thread_count(&self) -> usize {
        usize::try_from(ffi::get_omp_thread_count_lsd(&self.inner)).unwrap_or(1)
    }

    /// Get the random schedule seed
    #[must_use]
    pub fn random_schedule_seed(&self) -> i32 {
        ffi::get_random_schedule_seed_lsd(&self.inner)
    }

    /// Enable or disable statistics collection
    pub fn set_do_stats(&mut self, enable: bool) {
        ffi::set_do_stats_lsd(self.inner.pin_mut(), enable);
    }

    /// Get statistics collection status
    #[must_use]
    pub fn do_stats(&self) -> bool {
        ffi::get_do_stats_lsd(&self.inner)
    }

    /// Get statistics from the last decoding as JSON string
    ///
    /// # Errors
    ///
    /// This method currently does not return errors but the signature is maintained for consistency.
    pub fn get_statistics_json(&self) -> Result<String, LdpcError> {
        Ok(ffi::get_statistics_json_lsd(&self.inner))
    }
}

/// Statistics for a single cluster in LSD decoding
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    /// Number of bits in the final cluster
    pub final_bit_count: usize,
    /// Number of growth steps undergone
    pub undergone_growth_steps: usize,
    /// Number of merges with other clusters
    pub nr_merges: usize,
    /// Cluster size history at each step
    pub size_history: Vec<usize>,
    /// Whether the cluster is still active
    pub active: bool,
    /// Timestep when cluster became valid
    pub got_valid_in_timestep: Option<usize>,
    /// Timestep when cluster became inactive
    pub got_inactive_in_timestep: Option<usize>,
    /// ID of cluster that absorbed this one
    pub absorbed_by_cluster: Option<usize>,
}

/// Statistics from LSD decoding
#[derive(Debug, Clone)]
pub struct LsdStatistics {
    /// Individual cluster statistics
    pub individual_cluster_stats: BTreeMap<usize, ClusterStatistics>,
    /// Elapsed time in microseconds
    pub elapsed_time: u64,
    /// LSD method used
    pub lsd_method: OsdMethod,
    /// LSD order parameter
    pub lsd_order: usize,
}

/// Soft Information BP Decoder
pub struct SoftInfoBpDecoder {
    inner: UniquePtr<ffi::SoftInfoBpDecoder>,
}

impl SoftInfoBpDecoder {
    /// Create a builder for configuring a new Soft Information BP decoder
    pub fn builder(pcm: &SparseMatrix) -> crate::builders::SoftInfoBpBuilder<'_> {
        crate::builders::SoftInfoBpBuilder::new(pcm)
    }

    /// Create a new Soft Information BP decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the input parameters are invalid or
    /// `LdpcError::Ldpc` if the C++ decoder construction fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pcm: &SparseMatrix,
        error_rate: Option<f64>,
        error_channel: Option<&[f64]>,
        max_iter: usize,
        bp_method: BpMethod,
        ms_scaling_factor: f64,
        omp_thread_count: Option<usize>,
        serial_schedule_order: Option<&[i32]>,
        random_schedule_seed: Option<i32>,
    ) -> Result<Self, LdpcError> {
        // Create sparse matrix representation for FFI
        let pcm_repr = pcm.to_ffi_repr();

        let channel_probs = prepare_channel_probs(pcm.cols, error_rate, error_channel)?;

        // Handle adaptive iterations (0 means use n as max_iter)
        let actual_max_iter = if max_iter == 0 { pcm.cols } else { max_iter };

        // Handle optional parameters
        let omp_threads = i32::try_from(omp_thread_count.unwrap_or(1)).unwrap_or(1);
        let schedule_seed = random_schedule_seed.unwrap_or(-1);

        // Prepare serial schedule order
        let schedule_order: Vec<i32> = match serial_schedule_order {
            Some(order) => {
                if order.len() != pcm.cols {
                    return Err(LdpcError::InvalidInput(
                        "Serial schedule order must have length equal to number of bits"
                            .to_string(),
                    ));
                }
                order.to_vec()
            }
            None => (0..i32::try_from(pcm.cols).unwrap_or(0)).collect(),
        };

        let decoder = ffi::create_soft_info_bp_decoder(
            &pcm_repr,
            &channel_probs,
            i32::try_from(actual_max_iter).unwrap_or(i32::MAX),
            bp_method.to_ffi(),
            ms_scaling_factor,
            omp_threads,
            &schedule_order,
            schedule_seed,
        )
        .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Self { inner: decoder })
    }

    /// Decode a soft syndrome
    ///
    /// # Arguments
    /// * `soft_syndrome` - Vector of log-likelihood ratios for the syndrome
    /// * `cutoff` - Cutoff parameter for virtual check nodes
    /// * `sigma` - Standard deviation parameter
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the soft syndrome length doesn't match the check count,
    /// or `LdpcError::Ldpc` if the C++ decoder encounters an error.
    pub fn decode(
        &mut self,
        soft_syndrome: &[f64],
        cutoff: f64,
        sigma: f64,
    ) -> Result<DecodingResult, LdpcError> {
        if soft_syndrome.len() != self.check_count() {
            return Err(LdpcError::InvalidInput(format!(
                "Soft syndrome length {} does not match check count {}",
                soft_syndrome.len(),
                self.check_count()
            )));
        }

        let result = ffi::decode_soft_info_bp(self.inner.pin_mut(), soft_syndrome, cutoff, sigma)
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(DecodingResult {
            decoding: Array1::from_vec(result.decoding),
            converged: result.converged,
            iterations: usize::try_from(result.iterations).unwrap_or(0),
        })
    }

    /// Get the log-probability ratios from the last decoding
    #[must_use]
    pub fn log_prob_ratios(&self) -> Array1<f64> {
        Array1::from_vec(ffi::get_log_prob_ratios_soft(&self.inner))
    }

    // Getter methods
    /// Get the number of checks (rows) in the parity check matrix
    #[must_use]
    pub fn check_count(&self) -> usize {
        usize::try_from(ffi::get_check_count_soft(&self.inner)).unwrap_or(0)
    }

    /// Get the number of bits (columns) in the parity check matrix
    #[must_use]
    pub fn bit_count(&self) -> usize {
        usize::try_from(ffi::get_bit_count_soft(&self.inner)).unwrap_or(0)
    }

    /// Get the channel error probabilities
    #[must_use]
    pub fn channel_probs(&self) -> Vec<f64> {
        ffi::get_channel_probs_soft(&self.inner)
    }

    /// Get the maximum number of iterations
    #[must_use]
    pub fn max_iter(&self) -> usize {
        usize::try_from(ffi::get_max_iter_soft(&self.inner)).unwrap_or(0)
    }

    /// Get the BP method
    #[must_use]
    pub fn bp_method(&self) -> BpMethod {
        match ffi::get_bp_method_soft(&self.inner) {
            1 => BpMethod::MinimumSum,
            _ => BpMethod::ProductSum, // default for 0 and any other value
        }
    }

    /// Get the minimum-sum scaling factor
    #[must_use]
    pub fn ms_scaling_factor(&self) -> f64 {
        ffi::get_ms_scaling_factor_soft(&self.inner)
    }

    /// Check if the decoder converged in the last run
    #[must_use]
    pub fn converged(&self) -> bool {
        ffi::get_converged_soft(&self.inner)
    }

    /// Get the number of iterations from the last decoding
    #[must_use]
    pub fn iterations(&self) -> usize {
        usize::try_from(ffi::get_iterations_soft(&self.inner)).unwrap_or(0)
    }

    /// Get the OpenMP thread count
    #[must_use]
    pub fn omp_thread_count(&self) -> usize {
        usize::try_from(ffi::get_omp_thread_count_soft(&self.inner)).unwrap_or(1)
    }

    /// Get the random schedule seed
    #[must_use]
    pub fn random_schedule_seed(&self) -> i32 {
        ffi::get_random_schedule_seed_soft(&self.inner)
    }
}

/// Flip Decoder (Bit-flipping algorithm)
pub struct FlipDecoder {
    inner: UniquePtr<ffi::FlipDecoder>,
}

impl FlipDecoder {
    /// Create a builder for configuring a new Flip decoder
    pub fn builder(pcm: &SparseMatrix) -> crate::builders::FlipBuilder<'_> {
        crate::builders::FlipBuilder::new(pcm)
    }

    /// Create a new Flip decoder
    ///
    /// # Arguments
    /// * `pcm` - The parity check matrix
    /// * `max_iter` - Maximum iterations (0 = n)
    /// * `pfreq` - Perturbation frequency for tie-breaking (0 = never)
    /// * `seed` - Random seed for perturbations (0 = random)
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::Ldpc` if the C++ decoder construction fails.
    pub fn new(
        pcm: &SparseMatrix,
        max_iter: usize,
        pfreq: usize,
        seed: i32,
    ) -> Result<Self, LdpcError> {
        let pcm_repr = pcm.to_ffi_repr();

        // Handle adaptive iterations
        let actual_max_iter = if max_iter == 0 { pcm.cols } else { max_iter };
        let pfreq_val = if pfreq == 0 {
            i32::MAX
        } else {
            i32::try_from(pfreq).unwrap_or(i32::MAX)
        };

        let decoder = ffi::create_flip_decoder(
            &pcm_repr,
            i32::try_from(actual_max_iter).unwrap_or(i32::MAX),
            pfreq_val,
            seed,
        )
        .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Self { inner: decoder })
    }

    /// Decode a syndrome using bit-flipping
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the syndrome length doesn't match the check count,
    /// or `LdpcError::Ldpc` if the C++ decoder encounters an error.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult, LdpcError> {
        if syndrome.len() != self.check_count() {
            return Err(LdpcError::InvalidInput(format!(
                "Syndrome length {} does not match check count {}",
                syndrome.len(),
                self.check_count()
            )));
        }

        let syndrome_vec: Vec<u8> = syndrome.to_vec();
        let result = ffi::decode_flip(self.inner.pin_mut(), &syndrome_vec)
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(DecodingResult {
            decoding: Array1::from_vec(result.decoding),
            converged: result.converged,
            iterations: usize::try_from(result.iterations).unwrap_or(0),
        })
    }

    // Getter methods
    #[must_use]
    pub fn check_count(&self) -> usize {
        usize::try_from(ffi::get_check_count_flip(&self.inner)).unwrap_or(0)
    }

    #[must_use]
    pub fn bit_count(&self) -> usize {
        usize::try_from(ffi::get_bit_count_flip(&self.inner)).unwrap_or(0)
    }

    #[must_use]
    pub fn max_iter(&self) -> usize {
        usize::try_from(ffi::get_max_iter_flip(&self.inner)).unwrap_or(0)
    }

    #[must_use]
    pub fn converged(&self) -> bool {
        ffi::get_converged_flip(&self.inner)
    }

    #[must_use]
    pub fn iterations(&self) -> usize {
        usize::try_from(ffi::get_iterations_flip(&self.inner)).unwrap_or(0)
    }
}

/// Union Find Decoder
pub struct UnionFindDecoder {
    inner: UniquePtr<ffi::UnionFindDecoder>,
}

/// Union Find method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UfMethod {
    /// Matrix inversion method (general)
    Inversion,
    /// Peeling method (LDPC codes only)
    Peeling,
}

impl UfMethod {
    pub(crate) fn to_ffi(self) -> i32 {
        match self {
            UfMethod::Inversion => 0,
            UfMethod::Peeling => 1,
        }
    }
}

impl UnionFindDecoder {
    /// Create a builder for configuring a new Union-Find decoder
    pub fn builder(pcm: &SparseMatrix) -> crate::builders::UnionFindBuilder<'_> {
        crate::builders::UnionFindBuilder::new(pcm)
    }

    /// Create a new Union Find decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::Ldpc` if the C++ decoder construction fails.
    pub fn new(pcm: &SparseMatrix, uf_method: UfMethod) -> Result<Self, LdpcError> {
        let pcm_repr = pcm.to_ffi_repr();

        let decoder = ffi::create_union_find_decoder(&pcm_repr, uf_method.to_ffi())
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Self { inner: decoder })
    }

    /// Decode a syndrome using Union Find
    ///
    /// # Arguments
    /// * `syndrome` - The syndrome to decode
    /// * `llrs` - Log-likelihood ratios (optional, use empty slice if not available)
    /// * `bits_per_step` - Number of bits to add per growth step (0 = all)
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the syndrome or LLR lengths don't match expected sizes,
    /// or `LdpcError::Ldpc` if the C++ decoder encounters an error.
    pub fn decode(
        &mut self,
        syndrome: &ArrayView1<u8>,
        llrs: &[f64],
        bits_per_step: usize,
    ) -> Result<DecodingResult, LdpcError> {
        if syndrome.len() != self.check_count() {
            return Err(LdpcError::InvalidInput(format!(
                "Syndrome length {} does not match check count {}",
                syndrome.len(),
                self.check_count()
            )));
        }

        if !llrs.is_empty() && llrs.len() != self.bit_count() {
            return Err(LdpcError::InvalidInput(format!(
                "LLR length {} does not match bit count {}",
                llrs.len(),
                self.bit_count()
            )));
        }

        let syndrome_vec: Vec<u8> = syndrome.to_vec();
        let result = ffi::decode_union_find(
            self.inner.pin_mut(),
            &syndrome_vec,
            llrs,
            i32::try_from(bits_per_step).unwrap_or(0),
        )
        .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(DecodingResult {
            decoding: Array1::from_vec(result.decoding),
            converged: result.converged,
            iterations: usize::try_from(result.iterations).unwrap_or(0),
        })
    }

    #[must_use]
    pub fn check_count(&self) -> usize {
        usize::try_from(ffi::get_check_count_uf(&self.inner)).unwrap_or(0)
    }

    #[must_use]
    pub fn bit_count(&self) -> usize {
        usize::try_from(ffi::get_bit_count_uf(&self.inner)).unwrap_or(0)
    }
}

/// `BeliefFind` Decoder - Combines BP with Union Find
///
/// This decoder first attempts BP decoding, and if that fails,
/// falls back to Union Find using the soft information from BP.
pub struct BeliefFindDecoder {
    pcm: SparseMatrix,
    bp_decoder: BpOsdDecoder, // We use BpOsdDecoder with OSD disabled
    uf_decoder: UnionFindDecoder,
    uf_method: UfMethod,
    bits_per_step: usize,
}

impl BeliefFindDecoder {
    /// Create a builder for configuring a new `BeliefFind` decoder
    ///
    /// `BeliefFind` combines BP with Union-Find: it first tries BP, and if that
    /// fails to converge, it falls back to Union-Find using soft information from BP.
    pub fn builder(pcm: &SparseMatrix) -> crate::builders::BeliefFindBuilder<'_> {
        crate::builders::BeliefFindBuilder::new(pcm)
    }

    /// Create a new `BeliefFind` decoder
    ///
    /// # Errors
    ///
    /// Returns `LdpcError` if either the BP or Union Find decoder construction fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pcm: &SparseMatrix,
        error_rate: Option<f64>,
        error_channel: Option<&[f64]>,
        max_iter: usize,
        bp_method: BpMethod,
        ms_scaling_factor: f64,
        bp_schedule: BpSchedule,
        omp_thread_count: Option<usize>,
        serial_schedule_order: Option<&[i32]>,
        random_schedule_seed: Option<i32>,
        uf_method: UfMethod,
        bits_per_step: usize,
    ) -> Result<Self, LdpcError> {
        // Create BP decoder (with OSD disabled)
        let bp_decoder = BpOsdDecoder::new(
            pcm,
            error_rate,
            error_channel,
            max_iter,
            bp_method,
            bp_schedule,
            ms_scaling_factor,
            OsdMethod::Off, // Disable OSD
            0,              // OSD order doesn't matter when disabled
            InputVectorType::Syndrome,
            omp_thread_count,
            serial_schedule_order,
            random_schedule_seed,
        )?;

        // Create Union Find decoder
        let uf_decoder = UnionFindDecoder::new(pcm, uf_method)?;

        // Default bits_per_step to n if 0
        let actual_bits_per_step = if bits_per_step == 0 {
            pcm.cols
        } else {
            bits_per_step
        };

        Ok(Self {
            pcm: pcm.clone(),
            bp_decoder,
            uf_decoder,
            uf_method,
            bits_per_step: actual_bits_per_step,
        })
    }

    /// Decode a syndrome using `BeliefFind` algorithm
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the syndrome length doesn't match the check count,
    /// or `LdpcError` if either the BP or Union Find decoding fails.
    ///
    /// # Panics
    ///
    /// Panics if the log probability ratios array is not contiguous in memory.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult, LdpcError> {
        if syndrome.len() != self.check_count() {
            return Err(LdpcError::InvalidInput(format!(
                "Syndrome length {} does not match check count {}",
                syndrome.len(),
                self.check_count()
            )));
        }

        // First try BP decoding
        let bp_result = self.bp_decoder.decode(syndrome)?;

        // If BP converged, return its result
        if bp_result.converged {
            return Ok(bp_result);
        }

        // BP didn't converge, use Union Find with soft information from BP
        let llrs = self.bp_decoder.log_prob_ratios()?;
        let llrs_slice = llrs.as_slice().unwrap();

        // Convert LLRs to bit weights for Union Find
        // Union Find expects weights where lower values = more likely to be in error
        // LLR: positive = likely 0, negative = likely 1
        // So we convert: weight = 1 / (1 + exp(llr))
        let bit_weights: Vec<f64> = llrs_slice
            .iter()
            .map(|&llr| 1.0 / (1.0 + llr.exp()))
            .collect();

        // Run Union Find decoder with the bit weights
        let uf_result = self
            .uf_decoder
            .decode(syndrome, &bit_weights, self.bits_per_step)?;

        // Return the Union Find result but keep BP iteration count for diagnostics
        Ok(DecodingResult {
            decoding: uf_result.decoding,
            converged: uf_result.converged,
            iterations: bp_result.iterations, // Report BP iterations since UF doesn't iterate
        })
    }

    // Getter methods
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.pcm.rows
    }

    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.pcm.cols
    }

    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.bp_decoder.max_iter()
    }

    #[must_use]
    pub fn bp_method(&self) -> BpMethod {
        self.bp_decoder.bp_method()
    }

    #[must_use]
    pub fn ms_scaling_factor(&self) -> f64 {
        self.bp_decoder.ms_scaling_factor()
    }

    #[must_use]
    pub fn bp_schedule(&self) -> BpSchedule {
        self.bp_decoder.bp_schedule()
    }

    #[must_use]
    pub fn uf_method(&self) -> UfMethod {
        self.uf_method
    }

    #[must_use]
    pub fn bits_per_step(&self) -> usize {
        self.bits_per_step
    }

    #[must_use]
    pub fn omp_thread_count(&self) -> usize {
        self.bp_decoder.omp_thread_count()
    }

    #[must_use]
    pub fn channel_probs(&self) -> Array1<f64> {
        self.bp_decoder.channel_probs()
    }
}
