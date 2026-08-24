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

//! Small-system qudit state-vector and density-matrix simulators.
//!
//! These simulators use a uniform local dimension and are intended primarily for
//! physical leakage studies and independent noise-model verification. Site zero is
//! the least-significant radix digit. For a local operator, `targets[0]` is the
//! least-significant digit of the operator's row and column indices.

use core::fmt::{Debug, Display, Formatter};
use core::ops::{Deref, DerefMut};
use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_random::{PecosRng, Rng, RngExt, SeedableRng};
use std::error::Error;

const PROBABILITY_TOLERANCE: f64 = 1e-12;
const OPERATOR_TOLERANCE: f64 = 1e-10;

/// Errors returned by the generalized local-dimension simulators.
#[derive(Clone, Debug, PartialEq)]
pub enum QuditError {
    /// The local Hilbert-space dimension must be at least two.
    InvalidLocalDimension(usize),
    /// Computing the global Hilbert-space dimension overflowed `usize`.
    DimensionOverflow,
    /// A state or density operator had the wrong number of entries.
    InvalidStateLength { expected: usize, actual: usize },
    /// A target site does not exist.
    TargetOutOfRange { target: usize, num_sites: usize },
    /// A target site appeared more than once in a local operation.
    DuplicateTarget(usize),
    /// A local operator did not have the required square dimensions.
    InvalidOperatorLength { expected: usize, actual: usize },
    /// A basis-state index or local outcome does not exist.
    InvalidBasisState { state: usize, dimension: usize },
    /// The supplied state has zero norm or an operation produced zero probability.
    ZeroNorm,
    /// A state, channel, or probability contained a non-finite value.
    NonFiniteValue,
    /// A channel probability was outside the inclusive unit interval.
    InvalidProbability(f64),
    /// A state or channel was not normalized within numerical tolerance.
    NotNormalized { norm: f64 },
    /// A Kraus channel contained no operators.
    EmptyKrausChannel,
    /// An operator described as unitary did not satisfy `U^dagger U = I`.
    NonUnitary { deviation: f64 },
    /// Kraus operators did not satisfy `sum_i K_i^dagger K_i = I`.
    NotTracePreserving { deviation: f64 },
    /// A qubit-style measurement was requested for a state with leakage support.
    LeakagePopulation { probability: f64 },
    /// A projective measurement did not partition every local basis state exactly once.
    InvalidMeasurementPartition,
    /// A generalized measurement contained no outcomes or an empty outcome.
    InvalidMeasurementInstrument,
    /// A density operator was not Hermitian within the validation tolerance.
    NonHermitian { deviation: f64 },
    /// A density operator had a negative eigenvalue outside the validation tolerance.
    NotPositiveSemidefinite { minimum_eigenvalue: f64 },
    /// Allocating the requested dense state failed.
    AllocationFailed { entries: usize },
    /// A numerical validation tolerance was negative or non-finite.
    InvalidTolerance(f64),
}

impl Display for QuditError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLocalDimension(d) => {
                write!(f, "local dimension must be at least two, received {d}")
            }
            Self::DimensionOverflow => write!(f, "Hilbert-space dimension overflowed usize"),
            Self::InvalidStateLength { expected, actual } => {
                write!(f, "expected {expected} state entries, received {actual}")
            }
            Self::TargetOutOfRange { target, num_sites } => {
                write!(f, "target {target} is out of range for {num_sites} sites")
            }
            Self::DuplicateTarget(target) => write!(f, "target {target} appears more than once"),
            Self::InvalidOperatorLength { expected, actual } => {
                write!(f, "expected {expected} operator entries, received {actual}")
            }
            Self::InvalidBasisState { state, dimension } => {
                write!(
                    f,
                    "basis state {state} is out of range for dimension {dimension}"
                )
            }
            Self::ZeroNorm => write!(f, "state or selected operation has zero norm"),
            Self::NonFiniteValue => write!(f, "state, channel, or probability is not finite"),
            Self::InvalidProbability(probability) => {
                write!(
                    f,
                    "probability must be between zero and one, received {probability}"
                )
            }
            Self::NotNormalized { norm } => write!(f, "state is not normalized; norm is {norm}"),
            Self::EmptyKrausChannel => write!(f, "a Kraus channel must contain an operator"),
            Self::NonUnitary { deviation } => {
                write!(
                    f,
                    "operator is not unitary; maximum deviation is {deviation}"
                )
            }
            Self::NotTracePreserving { deviation } => write!(
                f,
                "Kraus channel is not trace preserving; maximum deviation is {deviation}"
            ),
            Self::LeakagePopulation { probability } => write!(
                f,
                "computational measurement is undefined with leakage probability {probability}"
            ),
            Self::InvalidMeasurementPartition => write!(
                f,
                "measurement groups must partition every local basis state exactly once"
            ),
            Self::InvalidMeasurementInstrument => write!(
                f,
                "a measurement instrument must contain non-empty outcome operator groups"
            ),
            Self::NonHermitian { deviation } => write!(
                f,
                "density operator is not Hermitian; maximum deviation is {deviation}"
            ),
            Self::NotPositiveSemidefinite { minimum_eigenvalue } => write!(
                f,
                "density operator is not positive semidefinite; minimum eigenvalue is {minimum_eigenvalue}"
            ),
            Self::AllocationFailed { entries } => {
                write!(f, "could not allocate {entries} complex state entries")
            }
            Self::InvalidTolerance(tolerance) => {
                write!(
                    f,
                    "tolerance must be finite and non-negative, received {tolerance}"
                )
            }
        }
    }
}

impl Error for QuditError {}

/// Result of sampling a Kraus trajectory branch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KrausSample {
    /// Index of the selected operator in the supplied Kraus list.
    pub operator_index: usize,
    /// Probability of the selected branch before renormalization.
    pub probability: f64,
}

/// Result of sampling a projective measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasurementSample {
    /// Selected joint basis index or partition-group index.
    pub outcome: usize,
    /// Probability of the selected outcome before collapse.
    pub probability: f64,
}

/// Result of sampling a generalized-measurement trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InstrumentSample {
    /// Selected reported outcome.
    pub outcome: usize,
    /// Selected Kraus operator within that outcome's operator group.
    pub operator_index: usize,
    /// Total probability of the reported outcome.
    pub outcome_probability: f64,
    /// Probability of the selected pure-state trajectory branch.
    pub branch_probability: f64,
}

/// Numerical diagnostics for an exact density operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityMatrixDiagnostics {
    /// Real part of `Tr(rho)`.
    pub trace: f64,
    /// Absolute imaginary part of `Tr(rho)`.
    pub trace_imaginary_error: f64,
    /// Largest elementwise deviation from Hermiticity.
    pub hermiticity_error: f64,
    /// Smallest eigenvalue of the Hermitian part of `rho`.
    pub minimum_eigenvalue: f64,
}

impl DensityMatrixDiagnostics {
    /// Whether the matrix is normalized, Hermitian, and positive semidefinite.
    #[must_use]
    pub fn is_physical(self, tolerance: f64) -> bool {
        (self.trace - 1.0).abs() <= tolerance
            && self.trace_imaginary_error <= tolerance
            && self.hermiticity_error <= tolerance
            && self.minimum_eigenvalue >= -tolerance
    }
}

/// A dense state-vector simulator with a uniform local dimension.
#[derive(Clone, Debug)]
pub struct QuditStateVec<R = PecosRng>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    local_dimension: usize,
    num_sites: usize,
    state: Vec<Complex64>,
    rng: R,
}

/// A qutrit state-vector simulator using the basis `|0>, |1>, |L>`.
///
/// Unlike [`QuditStateVec`], this wrapper fixes the local dimension to three.
#[derive(Clone, Debug)]
pub struct QutritStateVec<R = PecosRng>(QuditStateVec<R>)
where
    R: Rng + SeedableRng + Debug + Clone;

impl QutritStateVec<PecosRng> {
    /// Bytes required for the dense qutrit amplitudes, excluding container overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows.
    pub fn required_memory_bytes(num_sites: usize) -> Result<usize, QuditError> {
        QuditStateVec::required_memory_bytes(num_sites, 3)
    }

    /// Create a qutrit simulator in `|0...0>` with entropy-derived randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn new(num_sites: usize) -> Result<Self, QuditError> {
        Ok(Self(QuditStateVec::new(num_sites, 3)?))
    }

    /// Alias for [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn qutrit(num_sites: usize) -> Result<Self, QuditError> {
        Self::new(num_sites)
    }

    /// Create a qutrit simulator in `|0...0>` with deterministic randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn with_seed(num_sites: usize, seed: u64) -> Result<Self, QuditError> {
        Ok(Self(QuditStateVec::with_seed(num_sites, 3, seed)?))
    }

    /// Alias for [`Self::with_seed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn qutrit_with_seed(num_sites: usize, seed: u64) -> Result<Self, QuditError> {
        Self::with_seed(num_sites, seed)
    }
}

impl<R> QutritStateVec<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Create a qutrit simulator in `|0...0>` with a caller-provided RNG.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn with_rng(num_sites: usize, rng: R) -> Result<Self, QuditError> {
        Ok(Self(QuditStateVec::with_rng(num_sites, 3, rng)?))
    }

    /// Construct a qutrit simulator from a normalized state vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the state length, entries, or normalization is invalid.
    pub fn from_state(num_sites: usize, state: Vec<Complex64>, rng: R) -> Result<Self, QuditError> {
        Ok(Self(QuditStateVec::from_state(num_sites, 3, state, rng)?))
    }

    /// Consume the qutrit wrapper and return its generalized simulator.
    pub fn into_inner(self) -> QuditStateVec<R> {
        self.0
    }
}

impl<R> Deref for QutritStateVec<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    type Target = QuditStateVec<R>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<R> DerefMut for QutritStateVec<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl QuditStateVec<PecosRng> {
    /// Bytes required for the dense state amplitudes, excluding container overhead.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn required_memory_bytes(
        num_sites: usize,
        local_dimension: usize,
    ) -> Result<usize, QuditError> {
        global_dimension(num_sites, local_dimension)?
            .checked_mul(core::mem::size_of::<Complex64>())
            .ok_or(QuditError::DimensionOverflow)
    }

    /// Create `|0...0>` with entropy-derived randomness.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn new(num_sites: usize, local_dimension: usize) -> Result<Self, QuditError> {
        Self::with_rng(num_sites, local_dimension, rand::make_rng())
    }

    /// Create a qutrit simulator in `|0...0>` with entropy-derived randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the qutrit Hilbert-space dimension or allocation is invalid.
    pub fn qutrit(num_sites: usize) -> Result<Self, QuditError> {
        Self::new(num_sites, 3)
    }

    /// Create `|0...0>` with deterministic randomness.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn with_seed(
        num_sites: usize,
        local_dimension: usize,
        seed: u64,
    ) -> Result<Self, QuditError> {
        Self::with_rng(num_sites, local_dimension, PecosRng::seed_from_u64(seed))
    }

    /// Create a qutrit simulator in `|0...0>` with deterministic randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the qutrit Hilbert-space dimension overflows.
    pub fn qutrit_with_seed(num_sites: usize, seed: u64) -> Result<Self, QuditError> {
        Self::with_seed(num_sites, 3, seed)
    }
}

impl<R> QuditStateVec<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Create `|0...0>` with a caller-provided random-number generator.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn with_rng(num_sites: usize, local_dimension: usize, rng: R) -> Result<Self, QuditError> {
        let dimension = global_dimension(num_sites, local_dimension)?;
        let mut state = zeroed_complex_entries(dimension)?;
        state[0] = Complex64::new(1.0, 0.0);
        Ok(Self {
            local_dimension,
            num_sites,
            state,
            rng,
        })
    }

    /// Construct a simulator from a normalized state vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension, length, entries, or normalization is invalid.
    pub fn from_state(
        num_sites: usize,
        local_dimension: usize,
        state: Vec<Complex64>,
        rng: R,
    ) -> Result<Self, QuditError> {
        let dimension = global_dimension(num_sites, local_dimension)?;
        validate_state(&state, dimension)?;
        Ok(Self {
            local_dimension,
            num_sites,
            state,
            rng,
        })
    }

    /// Number of simulated sites.
    #[must_use]
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    /// Local Hilbert-space dimension.
    #[must_use]
    pub fn local_dimension(&self) -> usize {
        self.local_dimension
    }

    /// Global Hilbert-space dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.state.len()
    }

    /// State amplitudes in little-endian radix order.
    #[must_use]
    pub fn state(&self) -> &[Complex64] {
        &self.state
    }

    /// Probability of a global basis state.
    ///
    /// # Errors
    ///
    /// Returns an error if `basis_state` is outside the Hilbert space.
    pub fn probability(&self, basis_state: usize) -> Result<f64, QuditError> {
        self.state
            .get(basis_state)
            .map(Complex64::norm_sqr)
            .ok_or(QuditError::InvalidBasisState {
                state: basis_state,
                dimension: self.dimension(),
            })
    }

    /// Outcome distribution for a full local-basis measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` does not identify a simulated site.
    pub fn outcome_probabilities(&self, target: usize) -> Result<Vec<f64>, QuditError> {
        self.joint_outcome_probabilities(&[target])
    }

    /// Joint local-basis outcome distribution for the ordered target sites.
    ///
    /// `targets[0]` is the least-significant digit of the returned outcome index.
    ///
    /// # Errors
    ///
    /// Returns an error if targets are invalid or the outcome-space dimension overflows.
    pub fn joint_outcome_probabilities(&self, targets: &[usize]) -> Result<Vec<f64>, QuditError> {
        validate_targets(targets, self.num_sites)?;
        let local_size = radix_power(self.local_dimension, targets.len())?;
        let mut probabilities = vec![0.0; local_size];
        for (index, amplitude) in self.state.iter().enumerate() {
            probabilities[extract_local_index(index, targets, self.local_dimension)?] +=
                amplitude.norm_sqr();
        }
        Ok(probabilities)
    }

    /// Apply a row-major local operator to one or more sites.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets, dimensions, or non-finite entries.
    pub fn apply_operator(
        &mut self,
        targets: &[usize],
        operator: &[Complex64],
    ) -> Result<&mut Self, QuditError> {
        let local_size =
            validate_operator(targets, operator, self.num_sites, self.local_dimension)?;
        validate_unitary(operator, local_size)?;
        self.state = apply_operator_to_vector(
            &self.state,
            targets,
            operator,
            local_size,
            self.local_dimension,
        )?;
        Ok(self)
    }

    /// Apply a 2x2 qubit unitary embedded as `U + |L><L| + ...`.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` or the local dimension is invalid.
    pub fn apply_embedded_qubit_unitary(
        &mut self,
        target: usize,
        qubit_unitary: &[Complex64; 4],
    ) -> Result<&mut Self, QuditError> {
        let operator = embedded_qubit_unitary(self.local_dimension, qubit_unitary)?;
        self.apply_operator(&[target], &operator)
    }

    /// Sample and apply a Kraus channel, returning its index and branch probability.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets, operators, or channel normalization.
    pub fn apply_kraus(
        &mut self,
        targets: &[usize],
        operators: &[Vec<Complex64>],
    ) -> Result<KrausSample, QuditError> {
        if operators.is_empty() {
            return Err(QuditError::EmptyKrausChannel);
        }
        let local_size =
            validate_kraus_channel(targets, operators, self.num_sites, self.local_dimension)?;
        let mut branches = Vec::with_capacity(operators.len());
        let mut probabilities = Vec::with_capacity(operators.len());
        for operator in operators {
            let branch = apply_operator_to_vector(
                &self.state,
                targets,
                operator,
                local_size,
                self.local_dimension,
            )?;
            let probability = branch.iter().map(Complex64::norm_sqr).sum::<f64>();
            if !probability.is_finite() {
                return Err(QuditError::NonFiniteValue);
            }
            branches.push(branch);
            probabilities.push(probability.max(0.0));
        }
        let total = probabilities.iter().sum::<f64>();
        if total <= PROBABILITY_TOLERANCE {
            return Err(QuditError::ZeroNorm);
        }
        if (total - 1.0).abs() > channel_normalization_tolerance(local_size) {
            return Err(QuditError::NotNormalized { norm: total });
        }
        let selected = sample_distribution(&mut self.rng, &probabilities, total);
        let scale = probabilities[selected].sqrt();
        self.state = branches.swap_remove(selected);
        for amplitude in &mut self.state {
            *amplitude /= scale;
        }
        Ok(KrausSample {
            operator_index: selected,
            probability: probabilities[selected],
        })
    }

    /// Probabilities of generalized-measurement outcomes without changing the state.
    ///
    /// # Errors
    ///
    /// Returns an error for empty outcomes, invalid operators, or incomplete measurements.
    pub fn instrument_probabilities(
        &self,
        targets: &[usize],
        outcomes: &[Vec<Vec<Complex64>>],
    ) -> Result<Vec<f64>, QuditError> {
        validate_instrument_shape(outcomes)?;
        let operator_slices = outcomes
            .iter()
            .flatten()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let local_size = validate_kraus_slices(
            targets,
            &operator_slices,
            self.num_sites,
            self.local_dimension,
        )?;
        let mut probabilities = vec![0.0; outcomes.len()];
        for (outcome, operators) in outcomes.iter().enumerate() {
            for operator in operators {
                let branch = apply_operator_to_vector(
                    &self.state,
                    targets,
                    operator,
                    local_size,
                    self.local_dimension,
                )?;
                probabilities[outcome] += branch.iter().map(Complex64::norm_sqr).sum::<f64>();
            }
        }
        validate_total_probability(
            probabilities.iter().sum(),
            channel_normalization_tolerance(local_size),
        )?;
        Ok(probabilities)
    }

    /// Sample a generalized quantum measurement described by grouped Kraus operators.
    ///
    /// Each outer element is one reported outcome and may contain multiple Kraus
    /// operators. The union of every outcome's operators must be trace preserving.
    /// A pure-state trajectory samples both the reported outcome and its internal
    /// Kraus branch.
    ///
    /// # Errors
    ///
    /// Returns an error for empty outcomes, invalid operators, or incomplete measurements.
    pub fn measure_instrument(
        &mut self,
        targets: &[usize],
        outcomes: &[Vec<Vec<Complex64>>],
    ) -> Result<InstrumentSample, QuditError> {
        validate_instrument_shape(outcomes)?;
        let indexed_operators = outcomes
            .iter()
            .enumerate()
            .flat_map(|(outcome, operators)| {
                operators
                    .iter()
                    .enumerate()
                    .map(move |(operator, values)| (outcome, operator, values.as_slice()))
            })
            .collect::<Vec<_>>();
        let operator_slices = indexed_operators
            .iter()
            .map(|(_, _, operator)| *operator)
            .collect::<Vec<_>>();
        let local_size = validate_kraus_slices(
            targets,
            &operator_slices,
            self.num_sites,
            self.local_dimension,
        )?;
        let mut branches = Vec::with_capacity(operator_slices.len());
        let mut branch_probabilities = Vec::with_capacity(operator_slices.len());
        let mut outcome_probabilities = vec![0.0; outcomes.len()];
        for ((outcome, _, _), operator_slice) in indexed_operators.iter().zip(operator_slices) {
            let branch = apply_operator_to_vector(
                &self.state,
                targets,
                operator_slice,
                local_size,
                self.local_dimension,
            )?;
            let probability = branch.iter().map(Complex64::norm_sqr).sum::<f64>();
            if !probability.is_finite() {
                return Err(QuditError::NonFiniteValue);
            }
            outcome_probabilities[*outcome] += probability;
            branch_probabilities.push(probability);
            branches.push(branch);
        }
        let total = branch_probabilities.iter().sum::<f64>();
        validate_total_probability(total, channel_normalization_tolerance(local_size))?;
        let selected = sample_distribution(&mut self.rng, &branch_probabilities, total);
        let (outcome, operator_index, _) = indexed_operators[selected];
        let branch_probability = branch_probabilities[selected];
        self.state = branches.swap_remove(selected);
        let scale = branch_probability.sqrt();
        for amplitude in &mut self.state {
            *amplitude /= scale;
        }
        Ok(InstrumentSample {
            outcome,
            operator_index,
            outcome_probability: outcome_probabilities[outcome],
            branch_probability,
        })
    }

    /// Measure a site in its complete local basis and collapse the state.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` is invalid or the selected branch has zero norm.
    pub fn measure(&mut self, target: usize) -> Result<usize, QuditError> {
        Ok(self.measure_joint(&[target])?.outcome)
    }

    /// Measure several sites jointly in their complete local basis.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets or an impossible selected outcome.
    pub fn measure_joint(&mut self, targets: &[usize]) -> Result<MeasurementSample, QuditError> {
        let probabilities = self.joint_outcome_probabilities(targets)?;
        let total = probabilities.iter().sum::<f64>();
        let selected = sample_distribution(&mut self.rng, &probabilities, total);
        let probability = probabilities[selected];
        let mut allowed = vec![false; probabilities.len()];
        allowed[selected] = true;
        self.collapse_local_partition(targets, &allowed, probability)?;
        Ok(MeasurementSample {
            outcome: selected,
            probability,
        })
    }

    /// Measure a coarse-grained partition of the targets' joint local basis.
    ///
    /// Each inner vector lists the joint basis indices belonging to one reported
    /// outcome. The groups must contain every joint basis index exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets, an invalid partition, or an impossible outcome.
    pub fn measure_partition(
        &mut self,
        targets: &[usize],
        groups: &[Vec<usize>],
    ) -> Result<MeasurementSample, QuditError> {
        let basis_probabilities = self.joint_outcome_probabilities(targets)?;
        validate_measurement_partition(groups, basis_probabilities.len())?;
        let probabilities = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&outcome| basis_probabilities[outcome])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f64>();
        let selected = sample_distribution(&mut self.rng, &probabilities, total);
        let mut allowed = vec![false; basis_probabilities.len()];
        for &outcome in &groups[selected] {
            allowed[outcome] = true;
        }
        self.collapse_local_partition(targets, &allowed, probabilities[selected])?;
        Ok(MeasurementSample {
            outcome: selected,
            probability: probabilities[selected],
        })
    }

    /// Measure zero versus one when the site has no support outside the qubit subspace.
    ///
    /// Use [`Self::measure`] for a complete local-basis measurement that can report
    /// leakage. This strict method prevents a caller from silently assigning leaked
    /// population to a detector outcome.
    ///
    /// # Errors
    ///
    /// Returns an error if the target is invalid or has population outside `|0>, |1>`.
    pub fn measure_computational(&mut self, target: usize) -> Result<bool, QuditError> {
        let probabilities = self.outcome_probabilities(target)?;
        let leakage_probability = probabilities.iter().skip(2).sum::<f64>();
        if leakage_probability > PROBABILITY_TOLERANCE {
            return Err(QuditError::LeakagePopulation {
                probability: leakage_probability,
            });
        }
        let selected = sample_distribution(&mut self.rng, &probabilities[..2], 1.0);
        let probability = probabilities[selected];
        let mut allowed = vec![false; self.local_dimension];
        allowed[selected] = true;
        self.collapse_local_partition(&[target], &allowed, probability)?;
        Ok(selected == 1)
    }

    /// Reset a site to local basis state zero.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` does not identify a simulated site.
    pub fn reset(&mut self, target: usize) -> Result<&mut Self, QuditError> {
        let outcome = self.measure(target)?;
        if outcome != 0 {
            let operator = basis_swap(self.local_dimension, 0, outcome)?;
            self.apply_operator(&[target], &operator)?;
        }
        Ok(self)
    }

    /// Prepare a site in one of its local basis states.
    ///
    /// # Errors
    ///
    /// Returns an error if the target or requested local basis state is invalid.
    pub fn prepare_basis(
        &mut self,
        target: usize,
        basis_state: usize,
    ) -> Result<&mut Self, QuditError> {
        if basis_state >= self.local_dimension {
            return Err(QuditError::InvalidBasisState {
                state: basis_state,
                dimension: self.local_dimension,
            });
        }
        self.reset(target)?;
        if basis_state != 0 {
            let operator = basis_swap(self.local_dimension, 0, basis_state)?;
            self.apply_operator(&[target], &operator)?;
        }
        Ok(self)
    }

    /// Reduced density matrix for the ordered target sites of this pure state.
    ///
    /// # Errors
    ///
    /// Returns an error if targets are invalid or the reduced allocation fails.
    pub fn reduced_density_matrix(&self, targets: &[usize]) -> Result<Vec<Complex64>, QuditError> {
        validate_targets(targets, self.num_sites)?;
        let local_size = radix_power(self.local_dimension, targets.len())?;
        let mut reduced = zeroed_complex_entries(square(local_size)?)?;
        let target_mask = target_membership(targets, self.num_sites);
        for row in 0..self.dimension() {
            for column in 0..self.dimension() {
                if traced_digits_match(
                    row,
                    column,
                    &target_mask,
                    self.num_sites,
                    self.local_dimension,
                ) {
                    let local_row = extract_local_index(row, targets, self.local_dimension)?;
                    let local_column = extract_local_index(column, targets, self.local_dimension)?;
                    reduced[local_row * local_size + local_column] +=
                        self.state[row] * self.state[column].conj();
                }
            }
        }
        Ok(reduced)
    }

    fn collapse_local_partition(
        &mut self,
        targets: &[usize],
        allowed: &[bool],
        probability: f64,
    ) -> Result<(), QuditError> {
        if probability <= PROBABILITY_TOLERANCE {
            return Err(QuditError::ZeroNorm);
        }
        let scale = probability.sqrt();
        for (index, amplitude) in self.state.iter_mut().enumerate() {
            let local_index = extract_local_index(index, targets, self.local_dimension)?;
            if allowed[local_index] {
                *amplitude /= scale;
            } else {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        Ok(())
    }
}

/// An exact dense density-matrix simulator with a uniform local dimension.
#[derive(Clone, Debug)]
pub struct QuditDensityMatrix<R = PecosRng>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    local_dimension: usize,
    num_sites: usize,
    dimension: usize,
    density_matrix: Vec<Complex64>,
    rng: R,
}

/// A qutrit density-matrix simulator using the basis `|0>, |1>, |L>`.
///
/// Unlike [`QuditDensityMatrix`], this wrapper fixes the local dimension to three.
#[derive(Clone, Debug)]
pub struct QutritDensityMatrix<R = PecosRng>(QuditDensityMatrix<R>)
where
    R: Rng + SeedableRng + Debug + Clone;

impl QutritDensityMatrix<PecosRng> {
    /// Bytes required for the dense qutrit density operator, excluding container overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows.
    pub fn required_memory_bytes(num_sites: usize) -> Result<usize, QuditError> {
        QuditDensityMatrix::required_memory_bytes(num_sites, 3)
    }

    /// Create `|0...0><0...0|` with entropy-derived randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn new(num_sites: usize) -> Result<Self, QuditError> {
        Ok(Self(QuditDensityMatrix::new(num_sites, 3)?))
    }

    /// Alias for [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn qutrit(num_sites: usize) -> Result<Self, QuditError> {
        Self::new(num_sites)
    }

    /// Create `|0...0><0...0|` with deterministic randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn with_seed(num_sites: usize, seed: u64) -> Result<Self, QuditError> {
        Ok(Self(QuditDensityMatrix::with_seed(num_sites, 3, seed)?))
    }

    /// Alias for [`Self::with_seed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn qutrit_with_seed(num_sites: usize, seed: u64) -> Result<Self, QuditError> {
        Self::with_seed(num_sites, seed)
    }
}

impl<R> QutritDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Create `|0...0><0...0|` with a caller-provided RNG.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension or allocation is invalid.
    pub fn with_rng(num_sites: usize, rng: R) -> Result<Self, QuditError> {
        Ok(Self(QuditDensityMatrix::with_rng(num_sites, 3, rng)?))
    }

    /// Construct an exact simulator from a row-major qutrit density operator.
    ///
    /// # Errors
    ///
    /// Returns an error if the entries do not form a normalized, Hermitian,
    /// positive-semidefinite qutrit density operator.
    pub fn from_density_matrix(
        num_sites: usize,
        density_matrix: Vec<Complex64>,
        rng: R,
    ) -> Result<Self, QuditError> {
        Ok(Self(QuditDensityMatrix::from_density_matrix(
            num_sites,
            3,
            density_matrix,
            rng,
        )?))
    }

    /// Construct from a trace-one matrix without checking Hermiticity or positivity.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix length, entries, or trace is invalid.
    pub fn from_density_matrix_unchecked(
        num_sites: usize,
        density_matrix: Vec<Complex64>,
        rng: R,
    ) -> Result<Self, QuditError> {
        Ok(Self(QuditDensityMatrix::from_density_matrix_unchecked(
            num_sites,
            3,
            density_matrix,
            rng,
        )?))
    }

    /// Consume the qutrit wrapper and return its generalized simulator.
    pub fn into_inner(self) -> QuditDensityMatrix<R> {
        self.0
    }
}

impl<R> Deref for QutritDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    type Target = QuditDensityMatrix<R>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<R> DerefMut for QutritDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl QuditDensityMatrix<PecosRng> {
    /// Bytes required for the dense density operator, excluding container overhead.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn required_memory_bytes(
        num_sites: usize,
        local_dimension: usize,
    ) -> Result<usize, QuditError> {
        let dimension = global_dimension(num_sites, local_dimension)?;
        square(dimension)?
            .checked_mul(core::mem::size_of::<Complex64>())
            .ok_or(QuditError::DimensionOverflow)
    }

    /// Create `|0...0><0...0|` with entropy-derived randomness.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn new(num_sites: usize, local_dimension: usize) -> Result<Self, QuditError> {
        Self::with_rng(num_sites, local_dimension, rand::make_rng())
    }

    /// Create a qutrit simulator in `|0...0><0...0|` with entropy-derived randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the qutrit Hilbert-space dimension or allocation is invalid.
    pub fn qutrit(num_sites: usize) -> Result<Self, QuditError> {
        Self::new(num_sites, 3)
    }

    /// Create `|0...0><0...0|` with deterministic randomness.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn with_seed(
        num_sites: usize,
        local_dimension: usize,
        seed: u64,
    ) -> Result<Self, QuditError> {
        Self::with_rng(num_sites, local_dimension, PecosRng::seed_from_u64(seed))
    }

    /// Create a qutrit simulator in `|0...0><0...0|` with deterministic randomness.
    ///
    /// # Errors
    ///
    /// Returns an error if the qutrit Hilbert-space dimension overflows.
    pub fn qutrit_with_seed(num_sites: usize, seed: u64) -> Result<Self, QuditError> {
        Self::with_seed(num_sites, 3, seed)
    }
}

impl<R> QuditDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Create `|0...0><0...0|` with a caller-provided RNG.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or overflowing Hilbert-space dimension.
    pub fn with_rng(num_sites: usize, local_dimension: usize, rng: R) -> Result<Self, QuditError> {
        let dimension = global_dimension(num_sites, local_dimension)?;
        let matrix_size = square(dimension)?;
        let mut density_matrix = zeroed_complex_entries(matrix_size)?;
        density_matrix[0] = Complex64::new(1.0, 0.0);
        Ok(Self {
            local_dimension,
            num_sites,
            dimension,
            density_matrix,
            rng,
        })
    }

    /// Construct an exact simulator from a row-major density operator.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension, entries, trace, Hermiticity, or positivity is invalid.
    pub fn from_density_matrix(
        num_sites: usize,
        local_dimension: usize,
        density_matrix: Vec<Complex64>,
        rng: R,
    ) -> Result<Self, QuditError> {
        let simulator =
            Self::from_density_matrix_unchecked(num_sites, local_dimension, density_matrix, rng)?;
        simulator.validate_physicality(PROBABILITY_TOLERANCE * 10.0)?;
        Ok(simulator)
    }

    /// Construct from a trace-one matrix without checking Hermiticity or positivity.
    ///
    /// This is useful for diagnostics and tests. Normal simulation should use
    /// [`Self::from_density_matrix`]. Length, finiteness, and trace are still checked.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimension, length, entries, or trace is invalid.
    pub fn from_density_matrix_unchecked(
        num_sites: usize,
        local_dimension: usize,
        density_matrix: Vec<Complex64>,
        rng: R,
    ) -> Result<Self, QuditError> {
        let dimension = global_dimension(num_sites, local_dimension)?;
        let expected = square(dimension)?;
        if density_matrix.len() != expected {
            return Err(QuditError::InvalidStateLength {
                expected,
                actual: density_matrix.len(),
            });
        }
        if density_matrix
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QuditError::NonFiniteValue);
        }
        let simulator = Self {
            local_dimension,
            num_sites,
            dimension,
            density_matrix,
            rng,
        };
        let trace = simulator.trace();
        if (trace.re - 1.0).abs() > PROBABILITY_TOLERANCE || trace.im.abs() > PROBABILITY_TOLERANCE
        {
            return Err(QuditError::NotNormalized { norm: trace.re });
        }
        Ok(simulator)
    }

    /// Number of simulated sites.
    #[must_use]
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    /// Local Hilbert-space dimension.
    #[must_use]
    pub fn local_dimension(&self) -> usize {
        self.local_dimension
    }

    /// Global Hilbert-space dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Row-major density-operator entries.
    #[must_use]
    pub fn density_matrix(&self) -> &[Complex64] {
        &self.density_matrix
    }

    /// Probability of a global basis state.
    ///
    /// # Errors
    ///
    /// Returns an error if `basis_state` is outside the Hilbert space.
    pub fn probability(&self, basis_state: usize) -> Result<f64, QuditError> {
        if basis_state >= self.dimension {
            return Err(QuditError::InvalidBasisState {
                state: basis_state,
                dimension: self.dimension,
            });
        }
        Ok(self.density_matrix[basis_state * self.dimension + basis_state].re)
    }

    /// Trace of the density operator.
    #[must_use]
    pub fn trace(&self) -> Complex64 {
        let dimension = self.dimension();
        (0..dimension)
            .map(|index| self.density_matrix[index * dimension + index])
            .sum()
    }

    /// Purity `Tr(rho^2)` of the density operator.
    #[must_use]
    pub fn purity(&self) -> f64 {
        let dimension = self.dimension();
        let mut purity = Complex64::new(0.0, 0.0);
        for row in 0..dimension {
            for column in 0..dimension {
                purity += self.density_matrix[row * dimension + column]
                    * self.density_matrix[column * dimension + row];
            }
        }
        purity.re
    }

    /// Numerical physicality diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> DensityMatrixDiagnostics {
        let dimension = self.dimension();
        let matrix = DMatrix::from_row_slice(dimension, dimension, &self.density_matrix);
        let hermitian = (&matrix + matrix.adjoint()) * Complex64::new(0.5, 0.0);
        let eigenvalues = hermitian.symmetric_eigen().eigenvalues;
        let minimum_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let mut hermiticity_error: f64 = 0.0;
        for row in 0..dimension {
            for column in 0..dimension {
                hermiticity_error = hermiticity_error.max(
                    (self.density_matrix[row * dimension + column]
                        - self.density_matrix[column * dimension + row].conj())
                    .norm(),
                );
            }
        }
        let trace = self.trace();
        DensityMatrixDiagnostics {
            trace: trace.re,
            trace_imaginary_error: trace.im.abs(),
            hermiticity_error,
            minimum_eigenvalue,
        }
    }

    /// Validate trace, Hermiticity, and positive semidefiniteness.
    ///
    /// # Errors
    ///
    /// Returns an error if `tolerance` is invalid or the density operator is unphysical.
    pub fn validate_physicality(&self, tolerance: f64) -> Result<(), QuditError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(QuditError::InvalidTolerance(tolerance));
        }
        let diagnostics = self.diagnostics();
        if (diagnostics.trace - 1.0).abs() > tolerance
            || diagnostics.trace_imaginary_error > tolerance
        {
            return Err(QuditError::NotNormalized {
                norm: diagnostics.trace,
            });
        }
        if diagnostics.hermiticity_error > tolerance {
            return Err(QuditError::NonHermitian {
                deviation: diagnostics.hermiticity_error,
            });
        }
        if diagnostics.minimum_eigenvalue < -tolerance {
            return Err(QuditError::NotPositiveSemidefinite {
                minimum_eigenvalue: diagnostics.minimum_eigenvalue,
            });
        }
        Ok(())
    }

    /// Outcome distribution for a full local-basis measurement.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` does not identify a simulated site.
    pub fn outcome_probabilities(&self, target: usize) -> Result<Vec<f64>, QuditError> {
        self.joint_outcome_probabilities(&[target])
    }

    /// Joint local-basis outcome distribution for the ordered target sites.
    ///
    /// `targets[0]` is the least-significant digit of the returned outcome index.
    ///
    /// # Errors
    ///
    /// Returns an error if targets are invalid or the outcome-space dimension overflows.
    pub fn joint_outcome_probabilities(&self, targets: &[usize]) -> Result<Vec<f64>, QuditError> {
        validate_targets(targets, self.num_sites)?;
        let dimension = self.dimension();
        let local_size = radix_power(self.local_dimension, targets.len())?;
        let mut probabilities = vec![0.0; local_size];
        for index in 0..dimension {
            probabilities[extract_local_index(index, targets, self.local_dimension)?] +=
                self.density_matrix[index * dimension + index].re;
        }
        Ok(probabilities)
    }

    /// Apply a row-major local unitary to one or more sites.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets, dimensions, or non-finite entries.
    pub fn apply_operator(
        &mut self,
        targets: &[usize],
        operator: &[Complex64],
    ) -> Result<&mut Self, QuditError> {
        let local_size =
            validate_operator(targets, operator, self.num_sites, self.local_dimension)?;
        validate_unitary(operator, local_size)?;
        self.density_matrix = apply_operator_to_density_matrix(
            &self.density_matrix,
            self.dimension(),
            targets,
            operator,
            local_size,
            self.local_dimension,
        )?;
        Ok(self)
    }

    /// Apply a 2x2 qubit unitary embedded as `U + |L><L| + ...`.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` or the local dimension is invalid.
    pub fn apply_embedded_qubit_unitary(
        &mut self,
        target: usize,
        qubit_unitary: &[Complex64; 4],
    ) -> Result<&mut Self, QuditError> {
        let operator = embedded_qubit_unitary(self.local_dimension, qubit_unitary)?;
        self.apply_operator(&[target], &operator)
    }

    /// Apply an exact Kraus channel `rho -> sum_i K_i rho K_i^dagger`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets, operators, or channel normalization.
    pub fn apply_kraus(
        &mut self,
        targets: &[usize],
        operators: &[Vec<Complex64>],
    ) -> Result<&mut Self, QuditError> {
        if operators.is_empty() {
            return Err(QuditError::EmptyKrausChannel);
        }
        let local_size =
            validate_kraus_channel(targets, operators, self.num_sites, self.local_dimension)?;
        let dimension = self.dimension();
        let mut result = zeroed_complex_entries(self.density_matrix.len())?;
        for operator in operators {
            let branch = apply_operator_to_density_matrix(
                &self.density_matrix,
                dimension,
                targets,
                operator,
                local_size,
                self.local_dimension,
            )?;
            for (value, contribution) in result.iter_mut().zip(branch) {
                *value += contribution;
            }
        }
        let trace = (0..dimension)
            .map(|index| result[index * dimension + index])
            .sum::<Complex64>();
        if !trace.re.is_finite() || !trace.im.is_finite() {
            return Err(QuditError::NonFiniteValue);
        }
        let normalization_tolerance = channel_normalization_tolerance(local_size);
        if (trace.re - 1.0).abs() > normalization_tolerance
            || trace.im.abs() > normalization_tolerance
        {
            return Err(QuditError::NotNormalized { norm: trace.re });
        }
        self.density_matrix = result;
        Ok(self)
    }

    /// Probabilities of generalized-measurement outcomes without changing the state.
    ///
    /// # Errors
    ///
    /// Returns an error for empty outcomes, invalid operators, or incomplete measurements.
    pub fn instrument_probabilities(
        &self,
        targets: &[usize],
        outcomes: &[Vec<Vec<Complex64>>],
    ) -> Result<Vec<f64>, QuditError> {
        validate_instrument_shape(outcomes)?;
        let operator_slices = outcomes
            .iter()
            .flatten()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let local_size = validate_kraus_slices(
            targets,
            &operator_slices,
            self.num_sites,
            self.local_dimension,
        )?;
        let dimension = self.dimension();
        let mut probabilities = vec![0.0; outcomes.len()];
        for (outcome, operators) in outcomes.iter().enumerate() {
            for operator in operators {
                let branch = apply_operator_to_density_matrix(
                    &self.density_matrix,
                    dimension,
                    targets,
                    operator,
                    local_size,
                    self.local_dimension,
                )?;
                probabilities[outcome] += (0..dimension)
                    .map(|index| branch[index * dimension + index].re)
                    .sum::<f64>();
            }
        }
        validate_total_probability(
            probabilities.iter().sum(),
            channel_normalization_tolerance(local_size),
        )?;
        Ok(probabilities)
    }

    /// Sample a generalized quantum measurement described by grouped Kraus operators.
    ///
    /// Each outer element is one reported outcome and may contain multiple Kraus
    /// operators. All operators together must be trace preserving. The selected
    /// outcome retains the exact mixed post-measurement state of its operator group.
    ///
    /// # Errors
    ///
    /// Returns an error for empty outcomes, invalid operators, or incomplete measurements.
    pub fn measure_instrument(
        &mut self,
        targets: &[usize],
        outcomes: &[Vec<Vec<Complex64>>],
    ) -> Result<MeasurementSample, QuditError> {
        validate_instrument_shape(outcomes)?;
        let operator_slices = outcomes
            .iter()
            .flatten()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let local_size = validate_kraus_slices(
            targets,
            &operator_slices,
            self.num_sites,
            self.local_dimension,
        )?;
        let dimension = self.dimension();
        let mut outcome_states = Vec::with_capacity(outcomes.len());
        let mut probabilities = Vec::with_capacity(outcomes.len());
        for operators in outcomes {
            let mut outcome_state = zeroed_complex_entries(self.density_matrix.len())?;
            for operator in operators {
                let branch = apply_operator_to_density_matrix(
                    &self.density_matrix,
                    dimension,
                    targets,
                    operator,
                    local_size,
                    self.local_dimension,
                )?;
                for (value, contribution) in outcome_state.iter_mut().zip(branch) {
                    *value += contribution;
                }
            }
            let probability = (0..dimension)
                .map(|index| outcome_state[index * dimension + index].re)
                .sum::<f64>();
            if !probability.is_finite() {
                return Err(QuditError::NonFiniteValue);
            }
            probabilities.push(probability);
            outcome_states.push(outcome_state);
        }
        let total = probabilities.iter().sum::<f64>();
        validate_total_probability(total, channel_normalization_tolerance(local_size))?;
        let selected = sample_distribution(&mut self.rng, &probabilities, total);
        let probability = probabilities[selected];
        self.density_matrix = outcome_states.swap_remove(selected);
        for value in &mut self.density_matrix {
            *value /= probability;
        }
        Ok(MeasurementSample {
            outcome: selected,
            probability,
        })
    }

    /// Measure a site in its complete local basis and collapse the density operator.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` is invalid or the selected branch has zero probability.
    pub fn measure(&mut self, target: usize) -> Result<usize, QuditError> {
        Ok(self.measure_joint(&[target])?.outcome)
    }

    /// Measure several sites jointly in their complete local basis.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets or an impossible selected outcome.
    pub fn measure_joint(&mut self, targets: &[usize]) -> Result<MeasurementSample, QuditError> {
        let probabilities = self.joint_outcome_probabilities(targets)?;
        let total = probabilities.iter().sum::<f64>();
        let selected = sample_distribution(&mut self.rng, &probabilities, total);
        let probability = probabilities[selected];
        let mut allowed = vec![false; probabilities.len()];
        allowed[selected] = true;
        self.collapse_local_partition(targets, &allowed, probability)?;
        Ok(MeasurementSample {
            outcome: selected,
            probability,
        })
    }

    /// Measure a coarse-grained partition of the targets' joint local basis.
    ///
    /// Each inner vector lists the joint basis indices belonging to one reported
    /// outcome. The groups must contain every joint basis index exactly once.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid targets, an invalid partition, or an impossible outcome.
    pub fn measure_partition(
        &mut self,
        targets: &[usize],
        groups: &[Vec<usize>],
    ) -> Result<MeasurementSample, QuditError> {
        let basis_probabilities = self.joint_outcome_probabilities(targets)?;
        validate_measurement_partition(groups, basis_probabilities.len())?;
        let probabilities = groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&outcome| basis_probabilities[outcome])
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f64>();
        let selected = sample_distribution(&mut self.rng, &probabilities, total);
        let mut allowed = vec![false; basis_probabilities.len()];
        for &outcome in &groups[selected] {
            allowed[outcome] = true;
        }
        self.collapse_local_partition(targets, &allowed, probabilities[selected])?;
        Ok(MeasurementSample {
            outcome: selected,
            probability: probabilities[selected],
        })
    }

    /// Measure zero versus one when the site has no support outside the qubit subspace.
    ///
    /// Use [`Self::measure`] for a complete local-basis measurement that can report
    /// leakage. This strict method prevents a caller from silently assigning leaked
    /// population to a detector outcome.
    ///
    /// # Errors
    ///
    /// Returns an error if the target is invalid or has population outside `|0>, |1>`.
    pub fn measure_computational(&mut self, target: usize) -> Result<bool, QuditError> {
        let probabilities = self.outcome_probabilities(target)?;
        let leakage_probability = probabilities.iter().skip(2).sum::<f64>();
        if leakage_probability > PROBABILITY_TOLERANCE {
            return Err(QuditError::LeakagePopulation {
                probability: leakage_probability,
            });
        }
        let selected = sample_distribution(&mut self.rng, &probabilities[..2], 1.0);
        let probability = probabilities[selected];
        let mut allowed = vec![false; self.local_dimension];
        allowed[selected] = true;
        self.collapse_local_partition(&[target], &allowed, probability)?;
        Ok(selected == 1)
    }

    /// Reset a site to local basis state zero without sampling.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` does not identify a simulated site.
    pub fn reset(&mut self, target: usize) -> Result<&mut Self, QuditError> {
        validate_targets(&[target], self.num_sites)?;
        let mut operators = Vec::with_capacity(self.local_dimension);
        let operator_size = square(self.local_dimension)?;
        for input in 0..self.local_dimension {
            let mut operator = zeroed_complex_entries(operator_size)?;
            operator[input] = Complex64::new(1.0, 0.0);
            operators.push(operator);
        }
        self.apply_kraus(&[target], &operators)
    }

    /// Prepare a site in one of its local basis states.
    ///
    /// # Errors
    ///
    /// Returns an error if the target or requested local basis state is invalid.
    pub fn prepare_basis(
        &mut self,
        target: usize,
        basis_state: usize,
    ) -> Result<&mut Self, QuditError> {
        if basis_state >= self.local_dimension {
            return Err(QuditError::InvalidBasisState {
                state: basis_state,
                dimension: self.local_dimension,
            });
        }
        self.reset(target)?;
        if basis_state != 0 {
            let operator = basis_swap(self.local_dimension, 0, basis_state)?;
            self.apply_operator(&[target], &operator)?;
        }
        Ok(self)
    }

    /// Trace out all sites not listed in `targets`.
    ///
    /// # Errors
    ///
    /// Returns an error if targets are invalid or the reduced dimension overflows.
    pub fn reduced_density_matrix(&self, targets: &[usize]) -> Result<Vec<Complex64>, QuditError> {
        validate_targets(targets, self.num_sites)?;
        let local_size = radix_power(self.local_dimension, targets.len())?;
        let dimension = self.dimension();
        let mut reduced = zeroed_complex_entries(square(local_size)?)?;
        let target_mask = target_membership(targets, self.num_sites);
        for row in 0..dimension {
            for column in 0..dimension {
                if traced_digits_match(
                    row,
                    column,
                    &target_mask,
                    self.num_sites,
                    self.local_dimension,
                ) {
                    let local_row = extract_local_index(row, targets, self.local_dimension)?;
                    let local_column = extract_local_index(column, targets, self.local_dimension)?;
                    reduced[local_row * local_size + local_column] +=
                        self.density_matrix[row * dimension + column];
                }
            }
        }
        Ok(reduced)
    }

    fn collapse_local_partition(
        &mut self,
        targets: &[usize],
        allowed: &[bool],
        probability: f64,
    ) -> Result<(), QuditError> {
        if probability <= PROBABILITY_TOLERANCE {
            return Err(QuditError::ZeroNorm);
        }
        let dimension = self.dimension();
        for row in 0..dimension {
            let local_row = extract_local_index(row, targets, self.local_dimension)?;
            for column in 0..dimension {
                let local_column = extract_local_index(column, targets, self.local_dimension)?;
                let element = &mut self.density_matrix[row * dimension + column];
                if allowed[local_row] && allowed[local_column] {
                    *element /= probability;
                } else {
                    *element = Complex64::new(0.0, 0.0);
                }
            }
        }
        Ok(())
    }
}

/// Embed a row-major 2x2 unitary in the computational subspace.
///
/// # Errors
///
/// Returns an error if `local_dimension` is less than two or the supplied matrix is not unitary.
pub fn embedded_qubit_unitary(
    local_dimension: usize,
    qubit_unitary: &[Complex64; 4],
) -> Result<Vec<Complex64>, QuditError> {
    if local_dimension < 2 {
        return Err(QuditError::InvalidLocalDimension(local_dimension));
    }
    validate_unitary(qubit_unitary, 2)?;
    let mut operator = zeroed_complex_entries(square(local_dimension)?)?;
    for level in 2..local_dimension {
        operator[level * local_dimension + level] = Complex64::new(1.0, 0.0);
    }
    operator[0] = qubit_unitary[0];
    operator[1] = qubit_unitary[1];
    operator[local_dimension] = qubit_unitary[2];
    operator[local_dimension + 1] = qubit_unitary[3];
    Ok(operator)
}

/// Create a local basis-state swap operator.
///
/// # Errors
///
/// Returns an error if the local dimension or either basis state is invalid.
pub fn basis_swap(
    local_dimension: usize,
    first: usize,
    second: usize,
) -> Result<Vec<Complex64>, QuditError> {
    if local_dimension < 2 {
        return Err(QuditError::InvalidLocalDimension(local_dimension));
    }
    if first >= local_dimension {
        return Err(QuditError::InvalidBasisState {
            state: first,
            dimension: local_dimension,
        });
    }
    if second >= local_dimension {
        return Err(QuditError::InvalidBasisState {
            state: second,
            dimension: local_dimension,
        });
    }
    let mut operator = zeroed_complex_entries(square(local_dimension)?)?;
    for level in 0..local_dimension {
        let output = if level == first {
            second
        } else if level == second {
            first
        } else {
            level
        };
        operator[output * local_dimension + level] = Complex64::new(1.0, 0.0);
    }
    Ok(operator)
}

/// Qutrit leakage channel that transfers either computational basis state to `|L>`.
///
/// The returned row-major Kraus operators preserve an existing leaked population.
/// Separate jump operators for `|0>` and `|1>` ensure that the environment records
/// which computational level leaked rather than creating unphysical interference.
///
/// # Errors
///
/// Returns an error unless `probability` is finite and in the unit interval.
pub fn qutrit_leakage_channel(probability: f64) -> Result<Vec<Vec<Complex64>>, QuditError> {
    validate_probability(probability)?;
    let stay = (1.0 - probability).sqrt();
    let leak = probability.sqrt();
    Ok(vec![
        vec![
            Complex64::new(stay, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(stay, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(leak, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(leak, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ])
}

/// Qutrit seepage channel that returns `|L>` to an incoherent `|0>`/`|1>` mixture.
///
/// `probability` is the probability that a leaked population seeps during this
/// channel. `zero_fraction` is the conditional probability of returning to `|0>`;
/// the remaining returned population enters `|1>`.
///
/// # Errors
///
/// Returns an error unless both probabilities are finite and in the unit interval.
pub fn qutrit_seepage_channel(
    probability: f64,
    zero_fraction: f64,
) -> Result<Vec<Vec<Complex64>>, QuditError> {
    validate_probability(probability)?;
    validate_probability(zero_fraction)?;
    let stay_leaked = (1.0 - probability).sqrt();
    let to_zero = (probability * zero_fraction).sqrt();
    let to_one = (probability * (1.0 - zero_fraction)).sqrt();
    Ok(vec![
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(stay_leaked, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(to_zero, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(to_one, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ])
}

fn global_dimension(num_sites: usize, local_dimension: usize) -> Result<usize, QuditError> {
    if local_dimension < 2 {
        return Err(QuditError::InvalidLocalDimension(local_dimension));
    }
    radix_power(local_dimension, num_sites)
}

fn square(value: usize) -> Result<usize, QuditError> {
    value
        .checked_mul(value)
        .ok_or(QuditError::DimensionOverflow)
}

fn zeroed_complex_entries(entries: usize) -> Result<Vec<Complex64>, QuditError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(entries)
        .map_err(|_| QuditError::AllocationFailed { entries })?;
    values.resize(entries, Complex64::new(0.0, 0.0));
    Ok(values)
}

fn radix_power(radix: usize, exponent: usize) -> Result<usize, QuditError> {
    radix
        .checked_pow(
            exponent
                .try_into()
                .map_err(|_| QuditError::DimensionOverflow)?,
        )
        .ok_or(QuditError::DimensionOverflow)
}

fn validate_state(state: &[Complex64], expected: usize) -> Result<(), QuditError> {
    if state.len() != expected {
        return Err(QuditError::InvalidStateLength {
            expected,
            actual: state.len(),
        });
    }
    if state
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(QuditError::NonFiniteValue);
    }
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>();
    if (norm - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(QuditError::NotNormalized { norm });
    }
    Ok(())
}

fn validate_probability(probability: f64) -> Result<(), QuditError> {
    if !probability.is_finite() {
        Err(QuditError::NonFiniteValue)
    } else if !(0.0..=1.0).contains(&probability) {
        Err(QuditError::InvalidProbability(probability))
    } else {
        Ok(())
    }
}

fn validate_targets(targets: &[usize], num_sites: usize) -> Result<(), QuditError> {
    let mut seen = vec![false; num_sites];
    for &target in targets {
        if target >= num_sites {
            return Err(QuditError::TargetOutOfRange { target, num_sites });
        }
        if seen[target] {
            return Err(QuditError::DuplicateTarget(target));
        }
        seen[target] = true;
    }
    Ok(())
}

fn validate_measurement_partition(
    groups: &[Vec<usize>],
    basis_size: usize,
) -> Result<(), QuditError> {
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Err(QuditError::InvalidMeasurementPartition);
    }
    let mut seen = vec![false; basis_size];
    for &outcome in groups.iter().flatten() {
        if outcome >= basis_size || seen[outcome] {
            return Err(QuditError::InvalidMeasurementPartition);
        }
        seen[outcome] = true;
    }
    if seen.into_iter().all(core::convert::identity) {
        Ok(())
    } else {
        Err(QuditError::InvalidMeasurementPartition)
    }
}

fn validate_instrument_shape(outcomes: &[Vec<Vec<Complex64>>]) -> Result<(), QuditError> {
    if outcomes.is_empty() || outcomes.iter().any(Vec::is_empty) {
        Err(QuditError::InvalidMeasurementInstrument)
    } else {
        Ok(())
    }
}

fn validate_total_probability(total: f64, tolerance: f64) -> Result<(), QuditError> {
    if !total.is_finite() {
        Err(QuditError::NonFiniteValue)
    } else if total <= PROBABILITY_TOLERANCE {
        Err(QuditError::ZeroNorm)
    } else if (total - 1.0).abs() > tolerance {
        Err(QuditError::NotNormalized { norm: total })
    } else {
        Ok(())
    }
}

fn channel_normalization_tolerance(local_size: usize) -> f64 {
    let scale = u32::try_from(local_size).map_or(f64::from(u32::MAX), f64::from);
    OPERATOR_TOLERANCE * scale
}

fn validate_operator(
    targets: &[usize],
    operator: &[Complex64],
    num_sites: usize,
    local_dimension: usize,
) -> Result<usize, QuditError> {
    validate_targets(targets, num_sites)?;
    let local_size = radix_power(local_dimension, targets.len())?;
    let expected = local_size
        .checked_mul(local_size)
        .ok_or(QuditError::DimensionOverflow)?;
    if operator.len() != expected {
        return Err(QuditError::InvalidOperatorLength {
            expected,
            actual: operator.len(),
        });
    }
    if operator
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(QuditError::NonFiniteValue);
    }
    Ok(local_size)
}

fn validate_unitary(operator: &[Complex64], size: usize) -> Result<(), QuditError> {
    let deviation = completeness_deviation(core::slice::from_ref(&operator), size);
    if deviation > OPERATOR_TOLERANCE {
        Err(QuditError::NonUnitary { deviation })
    } else {
        Ok(())
    }
}

fn validate_kraus_channel(
    targets: &[usize],
    operators: &[Vec<Complex64>],
    num_sites: usize,
    local_dimension: usize,
) -> Result<usize, QuditError> {
    let operator_slices = operators.iter().map(Vec::as_slice).collect::<Vec<_>>();
    validate_kraus_slices(targets, &operator_slices, num_sites, local_dimension)
}

fn validate_kraus_slices(
    targets: &[usize],
    operators: &[&[Complex64]],
    num_sites: usize,
    local_dimension: usize,
) -> Result<usize, QuditError> {
    let mut local_size = 0;
    for &operator in operators {
        local_size = validate_operator(targets, operator, num_sites, local_dimension)?;
    }
    let deviation = completeness_deviation(operators, local_size);
    if deviation > OPERATOR_TOLERANCE {
        Err(QuditError::NotTracePreserving { deviation })
    } else {
        Ok(local_size)
    }
}

fn completeness_deviation(operators: &[&[Complex64]], size: usize) -> f64 {
    let mut maximum: f64 = 0.0;
    for row in 0..size {
        for column in 0..size {
            let actual = operators
                .iter()
                .flat_map(|operator| {
                    (0..size).map(|output| {
                        operator[output * size + row].conj() * operator[output * size + column]
                    })
                })
                .sum::<Complex64>();
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            maximum = maximum.max((actual - expected).norm());
        }
    }
    maximum
}

fn extract_local_index(
    global_index: usize,
    targets: &[usize],
    local_dimension: usize,
) -> Result<usize, QuditError> {
    let mut local_index = 0;
    let mut local_stride = 1;
    for &target in targets {
        let global_stride = radix_power(local_dimension, target)?;
        let digit = (global_index / global_stride) % local_dimension;
        local_index += digit * local_stride;
        local_stride *= local_dimension;
    }
    Ok(local_index)
}

fn replace_local_index(
    mut global_index: usize,
    local_index: usize,
    targets: &[usize],
    local_dimension: usize,
) -> Result<usize, QuditError> {
    let mut local_stride = 1;
    for &target in targets {
        let global_stride = radix_power(local_dimension, target)?;
        let old_digit = (global_index / global_stride) % local_dimension;
        let new_digit = (local_index / local_stride) % local_dimension;
        global_index = global_index - old_digit * global_stride + new_digit * global_stride;
        local_stride *= local_dimension;
    }
    Ok(global_index)
}

fn apply_operator_to_vector(
    state: &[Complex64],
    targets: &[usize],
    operator: &[Complex64],
    local_size: usize,
    local_dimension: usize,
) -> Result<Vec<Complex64>, QuditError> {
    let mut result = zeroed_complex_entries(state.len())?;
    for (input, amplitude) in state.iter().copied().enumerate() {
        if amplitude == Complex64::new(0.0, 0.0) {
            continue;
        }
        let local_input = extract_local_index(input, targets, local_dimension)?;
        for local_output in 0..local_size {
            let coefficient = operator[local_output * local_size + local_input];
            if coefficient != Complex64::new(0.0, 0.0) {
                let output = replace_local_index(input, local_output, targets, local_dimension)?;
                result[output] += coefficient * amplitude;
            }
        }
    }
    Ok(result)
}

fn apply_operator_to_density_matrix(
    density_matrix: &[Complex64],
    dimension: usize,
    targets: &[usize],
    operator: &[Complex64],
    local_size: usize,
    local_dimension: usize,
) -> Result<Vec<Complex64>, QuditError> {
    let mut result = zeroed_complex_entries(density_matrix.len())?;
    for input_row in 0..dimension {
        let local_input_row = extract_local_index(input_row, targets, local_dimension)?;
        for input_column in 0..dimension {
            let value = density_matrix[input_row * dimension + input_column];
            if value == Complex64::new(0.0, 0.0) {
                continue;
            }
            let local_input_column = extract_local_index(input_column, targets, local_dimension)?;
            for local_output_row in 0..local_size {
                let left = operator[local_output_row * local_size + local_input_row];
                if left == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let output_row =
                    replace_local_index(input_row, local_output_row, targets, local_dimension)?;
                for local_output_column in 0..local_size {
                    let right =
                        operator[local_output_column * local_size + local_input_column].conj();
                    if right != Complex64::new(0.0, 0.0) {
                        let output_column = replace_local_index(
                            input_column,
                            local_output_column,
                            targets,
                            local_dimension,
                        )?;
                        result[output_row * dimension + output_column] += left * value * right;
                    }
                }
            }
        }
    }
    Ok(result)
}

fn sample_distribution<R>(rng: &mut R, probabilities: &[f64], total: f64) -> usize
where
    R: Rng + ?Sized,
{
    let mut threshold = rng.random::<f64>() * total;
    let mut last_nonzero = 0;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if probability > 0.0 {
            last_nonzero = index;
        }
        if threshold < probability {
            return index;
        }
        threshold -= probability;
    }
    last_nonzero
}

fn target_membership(targets: &[usize], num_sites: usize) -> Vec<bool> {
    let mut membership = vec![false; num_sites];
    for &target in targets {
        membership[target] = true;
    }
    membership
}

fn traced_digits_match(
    mut row: usize,
    mut column: usize,
    target_membership: &[bool],
    num_sites: usize,
    local_dimension: usize,
) -> bool {
    for is_target in target_membership.iter().copied().take(num_sites) {
        if !is_target && row % local_dimension != column % local_dimension {
            return false;
        }
        row /= local_dimension;
        column /= local_dimension;
    }
    true
}
