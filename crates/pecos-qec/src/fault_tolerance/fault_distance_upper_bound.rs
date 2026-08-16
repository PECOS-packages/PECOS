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

//! Randomized decoder-based upper bounds for fault and code distance.
//!
//! Each sample augments the detector incidence matrix with an enforced nonempty observable
//! parity, then applies BP-OSD ([arXiv:1904.02703](https://arxiv.org/abs/1904.02703)). Sampling
//! observable combinations follows the randomized upper-bound idea of
//! [arXiv:2308.15140](https://arxiv.org/abs/2308.15140). Every returned decoder vector is checked
//! natively before it can tighten the upper bound. The result is never an exactness claim.

use super::dem_builder::DetectorErrorModel;
use crate::{DistanceProblem, DistanceProblemError, ParityCheckMatrix, StabilizerCodeSpec};
use ndarray::Array1;
use pecos_ldpc_decoders::{BpOsdDecoder, InputVectorType, SparseMatrix};
use pecos_quantum::F2Matrix;
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

/// A natively verified randomized fault- or code-distance upper bound and its witness.
///
/// This result is only an upper bound. It does not certify that a lighter undetectable logical
/// fault is absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaultDistanceUpperBoundResult {
    /// Hamming weight of the witnessed undetectable logical fault.
    pub weight: usize,
    /// Witnessing indices: DEM mechanism indices for fault bounds, qubit indices for code bounds.
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
    /// The code matrices or stabilizer specification do not define a valid distance problem.
    #[error(transparent)]
    DistanceProblem(#[from] DistanceProblemError),
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

fn verified_witness_indices(problem: &DistanceProblem, candidate: &[u8]) -> Option<Vec<usize>> {
    if candidate.iter().any(|&bit| bit > 1) {
        return None;
    }
    let assignment: Vec<_> = candidate.iter().map(|&bit| bit == 1).collect();
    problem.verified_witness_indices(&assignment).ok()
}

fn update_verified_upper_bound(
    problem: &DistanceProblem,
    candidate: &[u8],
    best: &mut Option<Vec<usize>>,
) {
    let Some(indices) = verified_witness_indices(problem, candidate) else {
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

fn parity_check_entries(
    matrix: &F2Matrix,
) -> Result<(Vec<u32>, Vec<u32>), FaultDistanceUpperBoundError> {
    let mut row_indices = Vec::new();
    let mut column_indices = Vec::new();
    for (row, entries) in matrix.rows().iter().enumerate() {
        let row = u32::try_from(row).map_err(|_| {
            FaultDistanceUpperBoundError::SparseMatrix(
                "check row count exceeds the u32 index space".to_string(),
            )
        })?;
        for (column, &entry) in entries.iter().enumerate() {
            if entry == 1 {
                row_indices.push(row);
                column_indices.push(u32::try_from(column).map_err(|_| {
                    FaultDistanceUpperBoundError::TooManyMechanisms(matrix.num_cols())
                })?);
            }
        }
    }
    Ok((row_indices, column_indices))
}

fn augmented_matrix(
    num_columns: usize,
    detector_rows: usize,
    detector_row_indices: &[u32],
    detector_column_indices: &[u32],
    logical_matrix: &F2Matrix,
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
    for column in 0..num_columns {
        let odd = observable_subset
            .iter()
            .enumerate()
            .filter(|&(row, selected)| *selected && logical_matrix.get(row, column) == 1)
            .count()
            % 2
            == 1;
        if odd {
            logical_row_nonempty = true;
            row_indices.push(logical_row);
            column_indices.push(
                u32::try_from(column)
                    .map_err(|_| FaultDistanceUpperBoundError::TooManyMechanisms(num_columns))?,
            );
        }
    }
    if !logical_row_nonempty {
        return Ok(None);
    }
    SparseMatrix::from_coo(detector_rows + 1, num_columns, row_indices, column_indices)
        .map(Some)
        .map_err(FaultDistanceUpperBoundError::SparseMatrix)
}

fn randomized_distance_upper_bound(
    detector_rows: usize,
    detector_row_indices: &[u32],
    detector_column_indices: &[u32],
    observable_count: usize,
    problem: &DistanceProblem,
    config: &FaultDistanceUpperBoundConfig,
) -> Result<Option<FaultDistanceUpperBoundResult>, FaultDistanceUpperBoundError> {
    if config.samples == 0 {
        return Ok(None);
    }
    validate_config(config)?;

    let (h, l) = problem.matrices();
    debug_assert_eq!(h.num_rows(), detector_rows);
    debug_assert_eq!(l.num_rows(), observable_count);
    if observable_count == 0 || problem.num_vars() == 0 {
        return Ok(None);
    }
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
            problem.num_vars(),
            detector_rows,
            detector_row_indices,
            detector_column_indices,
            l,
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
            problem,
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
    let (detector_rows, detector_row_indices, detector_column_indices) =
        detector_entries(dem, &mechanisms)?;
    let problem = DistanceProblem::from_dem(dem);
    randomized_distance_upper_bound(
        detector_rows,
        &detector_row_indices,
        &detector_column_indices,
        observable_count,
        &problem,
        config,
    )
}

/// Samples natively verified qubit witnesses for a binary code-distance upper bound.
///
/// For each sampled nonempty logical-row subset, this applies BP-OSD to
/// `[H; l_S] e = [0; 1]`. Returned `mechanism_indices` are qubit indices. The result is only an
/// upper bound and never certifies exactness.
///
/// `Ok(None)` means no verified witness was found, including when there are zero samples, zero
/// logical rows, or zero qubits.
///
/// # Errors
///
/// Returns an error for mismatched matrix widths, invalid explicit decoder parameters, an
/// unrepresentable augmented sparse system, or a decoder construction/decoding failure.
pub fn randomized_code_distance_upper_bound(
    h: &ParityCheckMatrix,
    l: &ParityCheckMatrix,
    config: &FaultDistanceUpperBoundConfig,
) -> Result<Option<FaultDistanceUpperBoundResult>, FaultDistanceUpperBoundError> {
    let problem = DistanceProblem::from_css_checks(h, l)?;
    let (row_indices, column_indices) = parity_check_entries(h.matrix())?;
    randomized_distance_upper_bound(
        h.num_checks(),
        &row_indices,
        &column_indices,
        l.num_checks(),
        &problem,
        config,
    )
}

/// Samples natively verified qubit witnesses for a stabilizer-code distance upper bound.
///
/// The symplectic decoder variables use `[X|Z]` order, but returned `mechanism_indices` are
/// physical-qubit indices and `weight` is physical-qubit support. The result is only an upper
/// bound and never certifies exactness. The specification must be a complete ordinary code.
///
/// # Errors
///
/// Returns an error for an incomplete or ill-formed stabilizer specification, invalid explicit
/// decoder parameters, an unrepresentable augmented sparse system, or a decoder
/// construction/decoding failure.
pub fn randomized_stabilizer_code_distance_upper_bound(
    spec: &StabilizerCodeSpec,
    config: &FaultDistanceUpperBoundConfig,
) -> Result<Option<FaultDistanceUpperBoundResult>, FaultDistanceUpperBoundError> {
    spec.verify_as_complete_code()
        .map_err(DistanceProblemError::from)?;
    let problem = DistanceProblem::from_stabilizer_spec(spec)?;
    let (h, l) = problem.matrices();
    let (row_indices, column_indices) = parity_check_entries(h)?;
    randomized_distance_upper_bound(
        h.num_rows(),
        &row_indices,
        &column_indices,
        l.num_rows(),
        &problem,
        config,
    )
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

#[cfg(test)]
mod code_tests {
    use super::*;
    use crate::{StabilizerCode, StabilizerCodeSpecError};
    use std::time::Instant;

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

    fn repetition_pair(n: usize) -> (ParityCheckMatrix, ParityCheckMatrix) {
        let h = ParityCheckMatrix::from_dense(
            (0..n - 1)
                .map(|row| {
                    let mut check = vec![0; n];
                    check[row] = 1;
                    check[row + 1] = 1;
                    check
                })
                .collect(),
        )
        .unwrap();
        let l = ParityCheckMatrix::from_dense(vec![vec![1; n]]).unwrap();
        (h, l)
    }

    fn stabilizer_spec(code: &StabilizerCode) -> StabilizerCodeSpec {
        StabilizerCodeSpec::from_stabilizer_code(code).unwrap()
    }

    #[test]
    fn repetition_code_upper_bound_is_sound_and_tight() {
        let (h, l) = repetition_pair(9);
        let result = randomized_code_distance_upper_bound(&h, &l, &config(1, 7))
            .unwrap()
            .expect("the repetition code has a nonzero codeword");

        assert!(result.weight >= 9);
        assert_eq!(result.weight, 9, "one sample reaches the exact value");
        assert_eq!(result.mechanism_indices, (0..9).collect::<Vec<_>>());
        assert_eq!(result.bound_kind, FaultDistanceBoundKind::UpperBound);
    }

    #[test]
    fn steane_and_five_qubit_upper_bounds_are_sound_and_tight() {
        for (label, spec, samples) in [
            ("Steane", stabilizer_spec(&StabilizerCode::steane()), 32),
            (
                "five-qubit",
                stabilizer_spec(&StabilizerCode::five_qubit()),
                32,
            ),
        ] {
            let result =
                randomized_stabilizer_code_distance_upper_bound(&spec, &config(samples, 17))
                    .unwrap()
                    .unwrap_or_else(|| panic!("{label} sampling found no witness"));
            assert!(result.weight >= 3, "{label} upper bound must be sound");
            assert_eq!(result.weight, 3, "{label} sampling reaches exact value");
            assert_eq!(result.mechanism_indices.len(), result.weight);
            assert!(
                result
                    .mechanism_indices
                    .iter()
                    .all(|&qubit| qubit < spec.num_qubits())
            );
            assert_eq!(result.bound_kind, FaultDistanceBoundKind::UpperBound);
        }
    }

    #[test]
    fn code_upper_bound_is_deterministic_for_same_seed() {
        let spec = stabilizer_spec(&StabilizerCode::five_qubit());
        let config = config(32, 991);
        let first = randomized_stabilizer_code_distance_upper_bound(&spec, &config).unwrap();
        let second = randomized_stabilizer_code_distance_upper_bound(&spec, &config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn code_native_verifier_rejects_invalid_decoder_vector() {
        let (h, l) = repetition_pair(3);
        let problem = DistanceProblem::from_css_checks(&h, &l).unwrap();
        let mut best = Some(vec![0, 1, 2]);

        update_verified_upper_bound(&problem, &[1, 0, 0], &mut best);

        assert_eq!(best, Some(vec![0, 1, 2]));
    }

    #[test]
    fn zero_samples_and_empty_code_dimensions_return_none() {
        let (h, l) = repetition_pair(3);
        assert_eq!(
            randomized_code_distance_upper_bound(&h, &l, &config(0, 5)).unwrap(),
            None
        );

        let no_logicals = ParityCheckMatrix::zeros(0, 3);
        assert_eq!(
            randomized_code_distance_upper_bound(&h, &no_logicals, &config(8, 5)).unwrap(),
            None
        );

        let no_qubits = ParityCheckMatrix::zeros(0, 0);
        assert_eq!(
            randomized_code_distance_upper_bound(&no_qubits, &no_qubits, &config(8, 5)).unwrap(),
            None
        );
    }

    #[test]
    fn full_rank_checks_without_a_logical_witness_return_none() {
        let h = ParityCheckMatrix::from_dense(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]])
            .unwrap();
        let l = ParityCheckMatrix::from_dense(vec![vec![1, 1, 1]]).unwrap();
        assert_eq!(
            randomized_code_distance_upper_bound(&h, &l, &config(8, 11)).unwrap(),
            None
        );
    }

    #[test]
    fn stabilizer_entry_rejects_a_spec_with_no_logical_qubits() {
        let spec = StabilizerCodeSpec::from_stabilizers(1, vec![pecos_core::Z(0)]).unwrap();
        assert!(matches!(
            randomized_stabilizer_code_distance_upper_bound(&spec, &config(8, 3)),
            Err(FaultDistanceUpperBoundError::DistanceProblem(
                DistanceProblemError::StabilizerSpec(StabilizerCodeSpecError::NoLogicalQubits)
            ))
        ));
    }

    #[test]
    #[ignore = "timing probe for randomized gross-code upper bounds"]
    fn randomized_gross_144_12_12_timing_probe() {
        let code = crate::BivariateBicycleCode::new(
            12,
            6,
            &[(3, 0), (0, 1), (0, 2)],
            &[(0, 3), (1, 0), (2, 0)],
        )
        .unwrap();
        assert_eq!(code.num_qubits(), 144);
        assert_eq!(code.num_logical_qubits(), 12);

        for samples in [12, 64, 256] {
            let started = Instant::now();
            let result = randomized_code_distance_upper_bound(
                code.hx(),
                code.logical_x(),
                &config(samples, 17),
            )
            .unwrap();
            println!(
                "gross [[144,12,12]] samples={samples}: result={result:?}, elapsed={:?}",
                started.elapsed()
            );
            assert!(result.is_none_or(|bound| bound.weight >= 12));
        }
    }
}
