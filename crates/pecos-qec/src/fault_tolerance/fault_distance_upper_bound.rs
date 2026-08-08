// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Randomized decoder-based upper bounds for detector-error-model fault distance.
//!
//! Each sample augments the detector incidence matrix with an enforced nonempty observable
//! parity, then applies BP-OSD ([arXiv:1904.02703](https://arxiv.org/abs/1904.02703)). Sampling
//! observable combinations follows the randomized upper-bound idea of
//! [arXiv:2308.15140](https://arxiv.org/abs/2308.15140). Every returned decoder vector is checked
//! natively before it can tighten the upper bound. The result is never an exactness claim.

use super::dem_builder::DetectorErrorModel;
use crate::DistanceProblem;
use ndarray::Array1;
use pecos_ldpc_decoders::{BpOsdDecoder, InputVectorType, SparseMatrix};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

pub use pecos_ldpc_decoders::{
    BpMethod as FaultDistanceBpMethod, BpSchedule as FaultDistanceBpSchedule,
    OsdMethod as FaultDistanceOsdMethod,
};

/// Identifies the mathematical status of a randomized fault-distance result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultDistanceBoundKind {
    /// A witnessed upper bound, with no exactness claim.
    UpperBound,
}

/// Selects observable subsets for randomized fault-distance upper-bound samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultDistanceObservableSubsetStrategy {
    /// Try every singleton in index order, then seeded random nonempty subsets.
    EachSingleThenRandom,
    /// Use seeded random nonempty subsets for every sample.
    RandomNonempty,
}

/// Fully explicit configuration for randomized fault-distance upper-bound sampling.
///
/// No clock-derived randomness or implicit decoder parameter is used. Repeating a call with the
/// same detector error model and configuration produces the same sample sequence and result.
#[derive(Clone, Debug, PartialEq)]
pub struct FaultDistanceUpperBoundConfig {
    /// Maximum number of observable-subset samples to run.
    pub samples: usize,
    /// Seed for observable-subset sampling.
    pub seed: u64,
    /// Observable-subset sampling strategy.
    pub observable_subset_strategy: FaultDistanceObservableSubsetStrategy,
    /// Uniform independent mechanism prior passed to BP-OSD.
    pub error_rate: f64,
    /// Maximum BP iterations; zero is rejected instead of selecting an implicit adaptive value.
    pub max_iterations: usize,
    /// BP update method.
    pub bp_method: FaultDistanceBpMethod,
    /// BP update schedule.
    pub bp_schedule: FaultDistanceBpSchedule,
    /// Minimum-sum scaling factor.
    pub min_sum_scaling_factor: f64,
    /// Ordered-statistics postprocessing method.
    pub osd_method: FaultDistanceOsdMethod,
    /// Ordered-statistics postprocessing order.
    pub osd_order: usize,
    /// OpenMP thread count passed to the decoder; zero is rejected.
    pub omp_threads: usize,
}

/// A natively verified randomized fault-distance upper bound and its witness.
///
/// This result is only an upper bound. It does not certify that a lighter undetectable logical
/// fault is absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultDistanceUpperBoundResult {
    /// Hamming weight of the witnessed undetectable logical fault.
    pub weight: usize,
    /// Witnessing mechanism indices in [`DetectorErrorModel::to_mechanisms`] order.
    pub mechanism_indices: Vec<usize>,
    /// Number of observable-subset samples attempted.
    pub samples_run: usize,
    /// Always [`FaultDistanceBoundKind::UpperBound`]; this is not an exactness claim.
    pub bound_kind: FaultDistanceBoundKind,
}

/// Error from configuring or running randomized fault-distance upper-bound sampling.
#[derive(Debug, thiserror::Error)]
pub enum FaultDistanceUpperBoundError {
    /// The uniform decoder prior must be finite and strictly between zero and one.
    #[error("error_rate must be finite and strictly between 0 and 1, got {0}")]
    InvalidErrorRate(f64),
    /// The BP iteration limit must be explicit and nonzero.
    #[error("max_iterations must be greater than zero")]
    ZeroMaxIterations,
    /// The decoder thread count must be explicit and nonzero.
    #[error("omp_threads must be greater than zero")]
    ZeroOmpThreads,
    /// The minimum-sum scale must be finite and positive.
    #[error("min_sum_scaling_factor must be finite and positive, got {0}")]
    InvalidMinSumScalingFactor(f64),
    /// The mechanism count cannot be represented by the decoder's sparse matrix.
    #[error("detector error model has too many mechanisms for the decoder: {0}")]
    TooManyMechanisms(usize),
    /// Sparse augmented-system construction failed.
    #[error("invalid augmented parity system: {0}")]
    SparseMatrix(String),
    /// BP-OSD construction or decoding failed.
    #[error(transparent)]
    Decoder(#[from] pecos_ldpc_decoders::LdpcError),
}

fn validate_config(
    config: &FaultDistanceUpperBoundConfig,
) -> Result<(), FaultDistanceUpperBoundError> {
    if !config.error_rate.is_finite() || !(0.0..1.0).contains(&config.error_rate) {
        return Err(FaultDistanceUpperBoundError::InvalidErrorRate(
            config.error_rate,
        ));
    }
    if config.max_iterations == 0 {
        return Err(FaultDistanceUpperBoundError::ZeroMaxIterations);
    }
    if config.omp_threads == 0 {
        return Err(FaultDistanceUpperBoundError::ZeroOmpThreads);
    }
    if !config.min_sum_scaling_factor.is_finite() || config.min_sum_scaling_factor <= 0.0 {
        return Err(FaultDistanceUpperBoundError::InvalidMinSumScalingFactor(
            config.min_sum_scaling_factor,
        ));
    }
    Ok(())
}

fn random_nonempty_subset(rng: &mut SmallRng, observable_count: usize) -> Vec<bool> {
    let mut subset: Vec<_> = (0..observable_count)
        .map(|_| rng.random_bool(0.5))
        .collect();
    if !subset.iter().any(|&selected| selected) {
        subset[rng.random_range(0..observable_count)] = true;
    }
    subset
}

fn sampled_subset(
    sample: usize,
    strategy: FaultDistanceObservableSubsetStrategy,
    observable_count: usize,
    rng: &mut SmallRng,
) -> Vec<bool> {
    if strategy == FaultDistanceObservableSubsetStrategy::EachSingleThenRandom
        && sample < observable_count
    {
        let mut subset = vec![false; observable_count];
        subset[sample] = true;
        subset
    } else {
        random_nonempty_subset(rng, observable_count)
    }
}

fn verified_mechanism_indices(problem: &DistanceProblem, candidate: &[u8]) -> Option<Vec<usize>> {
    if candidate.iter().any(|&bit| bit > 1) {
        return None;
    }
    let assignment: Vec<_> = candidate.iter().map(|&bit| bit == 1).collect();
    problem.verify_witness(&assignment).ok()?;
    Some(
        assignment
            .iter()
            .enumerate()
            .filter_map(|(index, &selected)| selected.then_some(index))
            .collect(),
    )
}

fn update_verified_upper_bound(
    problem: &DistanceProblem,
    candidate: &[u8],
    best: &mut Option<Vec<usize>>,
) {
    let Some(indices) = verified_mechanism_indices(problem, candidate) else {
        return;
    };
    if best
        .as_ref()
        .is_none_or(|current| (indices.len(), &indices) < (current.len(), current))
    {
        *best = Some(indices);
    }
}

fn detector_entries(
    dem: &DetectorErrorModel,
    mechanisms: &[(f64, Vec<u32>, Vec<u32>)],
) -> Result<(usize, Vec<u32>, Vec<u32>), FaultDistanceUpperBoundError> {
    let detector_rows = mechanisms
        .iter()
        .flat_map(|(_, detectors, _)| detectors)
        .map(|&detector| detector as usize + 1)
        .max()
        .unwrap_or(0)
        .max(dem.num_detectors());
    let mut row_indices = Vec::new();
    let mut column_indices = Vec::new();
    for (column, (_, detectors, _)) in mechanisms.iter().enumerate() {
        let column = u32::try_from(column)
            .map_err(|_| FaultDistanceUpperBoundError::TooManyMechanisms(mechanisms.len()))?;
        for &detector in detectors {
            row_indices.push(detector);
            column_indices.push(column);
        }
    }
    Ok((detector_rows, row_indices, column_indices))
}

fn augmented_matrix(
    mechanisms: &[(f64, Vec<u32>, Vec<u32>)],
    detector_rows: usize,
    detector_row_indices: &[u32],
    detector_column_indices: &[u32],
    observable_subset: &[bool],
) -> Result<Option<SparseMatrix>, FaultDistanceUpperBoundError> {
    let logical_row = u32::try_from(detector_rows).map_err(|_| {
        FaultDistanceUpperBoundError::SparseMatrix(
            "detector row count exceeds the u32 index space".to_string(),
        )
    })?;
    let mut row_indices = detector_row_indices.to_vec();
    let mut column_indices = detector_column_indices.to_vec();
    let mut logical_row_nonempty = false;
    for (column, (_, _, observables)) in mechanisms.iter().enumerate() {
        let odd = observables
            .iter()
            .filter(|&&observable| observable_subset[observable as usize])
            .count()
            % 2
            == 1;
        if odd {
            logical_row_nonempty = true;
            row_indices.push(logical_row);
            column_indices.push(
                u32::try_from(column).map_err(|_| {
                    FaultDistanceUpperBoundError::TooManyMechanisms(mechanisms.len())
                })?,
            );
        }
    }
    if !logical_row_nonempty {
        return Ok(None);
    }
    SparseMatrix::from_coo(
        detector_rows + 1,
        mechanisms.len(),
        row_indices,
        column_indices,
    )
    .map(Some)
    .map_err(FaultDistanceUpperBoundError::SparseMatrix)
}

/// Samples natively verified decoder witnesses to obtain a fault-distance upper bound.
///
/// For a selected nonempty observable subset, this solves `[H; l_S] e = [0; 1]`, where `l_S` is
/// the XOR of the selected observable rows. A decoded vector can tighten the result only after
/// [`DistanceProblem::verify_witness`] independently checks every detector parity and the full
/// nonzero-observable predicate. Consequently, an invalid decoder output is ignored.
///
/// `Ok(None)` means that no verified witness was found; it is not a lower bound or an exactness
/// claim. In particular, a zero-sample configuration returns `Ok(None)`.
///
/// # Errors
///
/// Returns an error for invalid explicit decoder parameters, an unrepresentable augmented sparse
/// system, or a decoder construction/decoding failure.
pub fn randomized_fault_distance_upper_bound(
    dem: &DetectorErrorModel,
    config: &FaultDistanceUpperBoundConfig,
) -> Result<Option<FaultDistanceUpperBoundResult>, FaultDistanceUpperBoundError> {
    if config.samples == 0 {
        return Ok(None);
    }
    validate_config(config)?;

    let (mechanisms, _coordinates) = dem.to_mechanisms();
    let observable_count = mechanisms
        .iter()
        .flat_map(|(_, _, observables)| observables)
        .map(|&observable| observable as usize + 1)
        .max()
        .unwrap_or(0)
        .max(dem.num_observables());
    if observable_count == 0 || mechanisms.is_empty() {
        return Ok(None);
    }
    let (detector_rows, detector_row_indices, detector_column_indices) =
        detector_entries(dem, &mechanisms)?;
    let problem = DistanceProblem::from_dem(dem);
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut best = None;

    for sample in 0..config.samples {
        let subset = sampled_subset(
            sample,
            config.observable_subset_strategy,
            observable_count,
            &mut rng,
        );
        let Some(pcm) = augmented_matrix(
            &mechanisms,
            detector_rows,
            &detector_row_indices,
            &detector_column_indices,
            &subset,
        )?
        else {
            continue;
        };
        let mut decoder = BpOsdDecoder::builder(&pcm)
            .error_rate(config.error_rate)
            .max_iter(config.max_iterations)
            .bp_method(config.bp_method)
            .bp_schedule(config.bp_schedule)
            .ms_scaling_factor(config.min_sum_scaling_factor)
            .osd_method(config.osd_method)
            .osd_order(config.osd_order)
            .input_vector_type(InputVectorType::Syndrome)
            .omp_threads(config.omp_threads)
            .serial_schedule_order(Vec::new())
            .random_schedule_seed(-1)
            .build()?;
        let mut syndrome = Array1::zeros(detector_rows + 1);
        syndrome[detector_rows] = 1;
        let decoded = decoder.decode(&syndrome.view())?;
        update_verified_upper_bound(
            &problem,
            decoded.decoding.as_slice().unwrap_or(&[]),
            &mut best,
        );
    }

    Ok(best.map(|mechanism_indices| FaultDistanceUpperBoundResult {
        weight: mechanism_indices.len(),
        mechanism_indices,
        samples_run: config.samples,
        bound_kind: FaultDistanceBoundKind::UpperBound,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DemOutput, FaultMechanism};

    fn dem_from_effects(effects: &[(Vec<u32>, Vec<u32>)]) -> DetectorErrorModel {
        let mut dem = DetectorErrorModel::new();
        let observable_count = effects
            .iter()
            .flat_map(|(_, observables)| observables)
            .map(|&observable| observable as usize + 1)
            .max()
            .unwrap_or(0);
        for observable in 0..observable_count {
            dem.add_observable(DemOutput::new(
                u32::try_from(observable).expect("test observable id fits in u32"),
            ));
        }
        for (detectors, observables) in effects {
            dem.add_direct_contribution(
                FaultMechanism::from_unsorted(
                    detectors.iter().copied(),
                    observables.iter().copied(),
                ),
                0.01,
            );
        }
        dem
    }

    fn config(samples: usize, seed: u64) -> FaultDistanceUpperBoundConfig {
        FaultDistanceUpperBoundConfig {
            samples,
            seed,
            observable_subset_strategy: FaultDistanceObservableSubsetStrategy::EachSingleThenRandom,
            error_rate: 0.1,
            max_iterations: 100,
            bp_method: FaultDistanceBpMethod::ProductSum,
            bp_schedule: FaultDistanceBpSchedule::Parallel,
            min_sum_scaling_factor: 1.0,
            osd_method: FaultDistanceOsdMethod::Osd0,
            osd_order: 0,
            omp_threads: 1,
        }
    }

    #[test]
    fn repetition_triad_upper_bound_reaches_three() {
        let dem = dem_from_effects(&[(vec![0, 1], vec![0]), (vec![0], vec![]), (vec![1], vec![])]);
        let result = randomized_fault_distance_upper_bound(&dem, &config(8, 7))
            .expect("valid decoder configuration")
            .expect("triad has an undetectable logical fault");
        assert!(result.weight >= 3);
        assert_eq!(result.weight, 3, "sampled upper bound reached exact value");
        assert_eq!(result.bound_kind, FaultDistanceBoundKind::UpperBound);
    }

    #[test]
    fn same_seed_gives_identical_upper_bound_and_witness() {
        let dem = dem_from_effects(&[
            (vec![0, 1], vec![0]),
            (vec![0], vec![]),
            (vec![1], vec![]),
            (vec![2], vec![1]),
            (vec![2], vec![]),
        ]);
        let first = randomized_fault_distance_upper_bound(&dem, &config(16, 91)).unwrap();
        let second = randomized_fault_distance_upper_bound(&dem, &config(16, 91)).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn native_verifier_rejects_invalid_decoder_vector() {
        let dem = dem_from_effects(&[(vec![0, 1], vec![0]), (vec![0], vec![]), (vec![1], vec![])]);
        let problem = DistanceProblem::from_dem(&dem);
        let mut best = Some(vec![0, 1, 2]);

        update_verified_upper_bound(&problem, &[1, 0, 0], &mut best);

        assert_eq!(best, Some(vec![0, 1, 2]));
    }

    #[test]
    fn zero_samples_returns_none() {
        let dem = dem_from_effects(&[(vec![], vec![0])]);
        assert_eq!(
            randomized_fault_distance_upper_bound(&dem, &config(0, 5)).unwrap(),
            None
        );
    }

    #[test]
    fn no_undetectable_logical_fault_returns_none() {
        let dem = dem_from_effects(&[(vec![0], vec![0]), (vec![1], vec![])]);
        assert_eq!(
            randomized_fault_distance_upper_bound(&dem, &config(32, 11)).unwrap(),
            None
        );

        let mut empty = DetectorErrorModel::new();
        empty.add_observable(DemOutput::new(0));
        assert_eq!(
            randomized_fault_distance_upper_bound(&empty, &config(32, 11)).unwrap(),
            None
        );
    }
}
