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

//! Capability traits for inspecting simulator state.
//!
//! These traits deliberately describe what a backend can expose, rather than a
//! single broad "quantum state" interface. A state-vector simulator can provide
//! amplitudes; a density-matrix simulator can provide matrix elements; symbolic
//! propagation backends should not pretend to expose either.

use crate::clifford_gateable::{CliffordGateable, MeasurementResult};
use crate::dense_stab::DenseStab;
use crate::density_matrix::DensityMatrix;
use crate::quantum_simulator::QuantumSimulator;
use crate::sparse_stab::{SparseStabGeneric, SparseStabHybrid};
use crate::stabilizer::Stabilizer;
use crate::state_vec_aos::StateVecAoS;
use crate::state_vec_soa::StateVecSoA;
use crate::state_vec_soa32::StateVecSoA32;
use crate::state_vec_sparse_aos::SparseStateVecAoS;
use crate::state_vec_sparse_soa::SparseStateVecSoA;
use core::fmt;
use num_complex::Complex64;
use pecos_core::{IndexSet, Pauli, PauliString, Phase, QuarterPhase, QubitId};
use pecos_quantum::PauliStabilizerGroup;
use pecos_random::{Rng, RngExt as _, SeedableRng};
use std::error::Error;
use std::f64::consts::TAU;
use std::fmt::Debug;

/// Error returned by state-inspection capability traits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateAccessError {
    /// `2^num_qubits` cannot be represented as a `usize`.
    DimensionOverflow {
        /// Number of qubits requested.
        num_qubits: usize,
    },
    /// A computational-basis index is outside the state Hilbert space.
    BasisIndexOutOfRange {
        /// Number of qubits in the state.
        num_qubits: usize,
        /// Hilbert-space dimension, when representable.
        dim: usize,
        /// Offending basis index.
        index: usize,
    },
    /// A Pauli string acts outside the state qubit range.
    PauliQubitOutOfRange {
        /// Number of qubits in the state.
        num_qubits: usize,
        /// Offending qubit index.
        qubit: usize,
    },
    /// A sampled/measured qubit is outside the state qubit range.
    QubitOutOfRange {
        /// Number of qubits in the state.
        num_qubits: usize,
        /// Offending qubit index.
        qubit: usize,
    },
    /// A state vector had an unexpected length.
    InvalidStateVectorLength {
        /// Expected vector length.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },
    /// A density matrix had an unexpected shape.
    InvalidDensityMatrixShape {
        /// Expected row/column count.
        expected: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// Stabilizer generators do not define a unique pure state.
    NotPureStabilizerState {
        /// Number of qubits in the represented system.
        num_qubits: usize,
        /// Number of supplied stabilizer generators.
        num_generators: usize,
        /// Rank of the stabilizer generator span.
        rank: usize,
    },
}

impl fmt::Display for StateAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow { num_qubits } => {
                write!(
                    f,
                    "Hilbert dimension overflows usize for {num_qubits} qubits"
                )
            }
            Self::BasisIndexOutOfRange {
                num_qubits,
                dim,
                index,
            } => write!(
                f,
                "basis index {index} is outside the {dim}-element Hilbert space for {num_qubits} qubits"
            ),
            Self::PauliQubitOutOfRange { num_qubits, qubit } => write!(
                f,
                "Pauli string touches qubit {qubit}, outside {num_qubits}-qubit state"
            ),
            Self::QubitOutOfRange { num_qubits, qubit } => {
                write!(f, "qubit {qubit} is outside {num_qubits}-qubit state")
            }
            Self::InvalidStateVectorLength { expected, actual } => {
                write!(
                    f,
                    "invalid state-vector length {actual}; expected {expected}"
                )
            }
            Self::InvalidDensityMatrixShape {
                expected,
                rows,
                cols,
            } => write!(
                f,
                "invalid density-matrix shape {rows}x{cols}; expected {expected}x{expected}"
            ),
            Self::NotPureStabilizerState {
                num_qubits,
                num_generators,
                rank,
            } => write!(
                f,
                "stabilizer generators do not define a unique pure state for {num_qubits} qubits: {num_generators} generators, rank {rank}"
            ),
        }
    }
}

impl Error for StateAccessError {}

/// Common metadata for state-inspection capabilities.
pub trait StateInfo {
    /// Returns the number of qubits represented by the state.
    fn num_qubits(&self) -> usize;

    /// Returns the Hilbert-space dimension `2^num_qubits`.
    ///
    /// # Errors
    ///
    /// Returns [`StateAccessError::DimensionOverflow`] if the dimension cannot
    /// be represented as a `usize`.
    fn hilbert_dim(&self) -> Result<usize, StateAccessError> {
        hilbert_dim(self.num_qubits())
    }
}

impl<T> StateInfo for T
where
    T: QuantumSimulator,
{
    fn num_qubits(&self) -> usize {
        QuantumSimulator::num_qubits(self)
    }
}

/// Returns a Haar-random pure state vector on `num_qubits` qubits.
///
/// The vector is sampled by drawing independent complex normal amplitudes and
/// normalizing them. The output is in little-endian computational-basis order,
/// matching PECOS's state-vector backends.
///
/// # Errors
///
/// Returns [`StateAccessError::DimensionOverflow`] if `2^num_qubits` does not
/// fit in `usize`.
pub fn random_statevector<R>(
    rng: &mut R,
    num_qubits: usize,
) -> Result<Vec<Complex64>, StateAccessError>
where
    R: Rng + ?Sized,
{
    let dim = hilbert_dim(num_qubits)?;
    if dim == 1 {
        return Ok(vec![Complex64::new(1.0, 0.0)]);
    }

    loop {
        let mut state: Vec<Complex64> = (0..dim).map(|_| standard_complex_normal(rng)).collect();
        let norm_sqr = state_norm_sqr(&state);
        if norm_sqr > f64::EPSILON {
            normalize_state_vector(&mut state, norm_sqr);
            return Ok(state);
        }
    }
}

/// Capability for backends that can expose computational-basis amplitudes.
pub trait StateVectorAccess: StateInfo {
    /// Returns one computational-basis amplitude.
    ///
    /// # Errors
    ///
    /// Returns an error if `basis_state` is outside the Hilbert space.
    fn amplitude(&mut self, basis_state: usize) -> Result<Complex64, StateAccessError>;

    /// Returns a dense state vector in little-endian computational-basis order.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows.
    fn state_vector(&mut self) -> Result<Vec<Complex64>, StateAccessError>;

    /// Returns the probability of one computational-basis state.
    ///
    /// # Errors
    ///
    /// Returns an error if `basis_state` is outside the Hilbert space.
    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        Ok(self.amplitude(basis_state)?.norm_sqr())
    }
}

/// Capability for backends that can expose a density matrix.
pub trait DensityMatrixAccess: StateInfo {
    /// Returns a dense density matrix in little-endian computational-basis order.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows.
    fn density_matrix(&mut self) -> Result<Vec<Vec<Complex64>>, StateAccessError>;

    /// Returns one density-matrix element.
    ///
    /// The default implementation materializes the full dense density matrix
    /// and then reads one entry. Backends with cheaper direct element access
    /// should override this method.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is outside the Hilbert space.
    fn density_matrix_element(
        &mut self,
        row: usize,
        col: usize,
    ) -> Result<Complex64, StateAccessError> {
        validate_basis_index(self.num_qubits(), row)?;
        validate_basis_index(self.num_qubits(), col)?;
        let rho = self.density_matrix()?;
        Ok(rho[row][col])
    }

    /// Returns the probability of one computational-basis state.
    ///
    /// # Errors
    ///
    /// Returns an error if `basis_state` is outside the Hilbert space.
    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        Ok(self.density_matrix_element(basis_state, basis_state)?.re)
    }
}

/// Capability for backends that can evaluate Pauli expectation values.
pub trait PauliExpectationAccess: StateInfo {
    /// Returns `<P>` for the supplied Pauli string.
    ///
    /// The result is complex so callers can also query phased Pauli strings
    /// such as `i X`. For Hermitian Paulis with real phase, the imaginary part
    /// should be numerical roundoff only.
    ///
    /// # Errors
    ///
    /// Returns an error if the Pauli string acts outside the state.
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError>;
}

/// Capability for backends that can sample projective Pauli-basis measurements.
///
/// Unlike [`StateVectorAccess`], [`DensityMatrixAccess`], and
/// [`PauliExpectationAccess`], these methods are not passive inspection:
/// sampling mutates the backend by collapsing the measured state and may
/// consume RNG state.
pub trait StateSampling: StateInfo {
    /// Samples/measures qubits in the computational (`Z`) basis.
    ///
    /// Results use the existing PECOS [`MeasurementResult`] convention:
    /// `outcome == false` is the `0`/`+Z` outcome and `outcome == true` is
    /// the `1`/`-Z` outcome.
    ///
    /// # Errors
    ///
    /// Returns an error if any qubit is outside the state.
    fn sample_z(&mut self, qubits: &[QubitId]) -> Result<Vec<MeasurementResult>, StateAccessError>;

    /// Samples/measures qubits in the `X` basis.
    ///
    /// # Errors
    ///
    /// Returns an error if any qubit is outside the state.
    fn sample_x(&mut self, qubits: &[QubitId]) -> Result<Vec<MeasurementResult>, StateAccessError>;

    /// Samples/measures qubits in the `Y` basis.
    ///
    /// # Errors
    ///
    /// Returns an error if any qubit is outside the state.
    fn sample_y(&mut self, qubits: &[QubitId]) -> Result<Vec<MeasurementResult>, StateAccessError>;
}

/// Capability for stabilizer backends that can materialize a dense state vector.
///
/// This is an explicit conversion, not passive inspection. The output has
/// length `2^num_qubits`, so callers should treat it as an exponential-cost
/// debugging and interop path rather than the normal way to query stabilizer
/// states.
pub trait StabilizerStateVectorConversion: StateInfo {
    /// Converts the stabilizer state into a dense state vector in little-endian
    /// computational-basis order.
    ///
    /// The returned vector is normalized and has a deterministic global phase:
    /// the first nonzero amplitude is real and positive.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows or if the
    /// stabilizer generators do not define a unique pure state.
    fn to_state_vector(&self) -> Result<Vec<Complex64>, StateAccessError>;
}

macro_rules! impl_state_sampling {
    (impl<$($generic:tt),+> for $ty:ty where $($where_clause:tt)*) => {
        impl<$($generic),+> StateSampling for $ty
        where
            $($where_clause)*
        {
            fn sample_z(
                &mut self,
                qubits: &[QubitId],
            ) -> Result<Vec<MeasurementResult>, StateAccessError> {
                sample_z_via_clifford(self, qubits)
            }

            fn sample_x(
                &mut self,
                qubits: &[QubitId],
            ) -> Result<Vec<MeasurementResult>, StateAccessError> {
                sample_x_via_clifford(self, qubits)
            }

            fn sample_y(
                &mut self,
                qubits: &[QubitId],
            ) -> Result<Vec<MeasurementResult>, StateAccessError> {
                sample_y_via_clifford(self, qubits)
            }
        }
    };
    (for $ty:ty) => {
        impl StateSampling for $ty {
            fn sample_z(
                &mut self,
                qubits: &[QubitId],
            ) -> Result<Vec<MeasurementResult>, StateAccessError> {
                sample_z_via_clifford(self, qubits)
            }

            fn sample_x(
                &mut self,
                qubits: &[QubitId],
            ) -> Result<Vec<MeasurementResult>, StateAccessError> {
                sample_x_via_clifford(self, qubits)
            }

            fn sample_y(
                &mut self,
                qubits: &[QubitId],
            ) -> Result<Vec<MeasurementResult>, StateAccessError> {
                sample_y_via_clifford(self, qubits)
            }
        }
    };
}

impl_state_sampling!(impl<R> for StateVecSoA<R> where R: Rng);
impl_state_sampling!(impl<R> for SparseStateVecSoA<R> where R: Rng + Debug);
impl_state_sampling!(impl<R> for StateVecAoS<R> where R: Rng + SeedableRng + Debug);
impl_state_sampling!(impl<R> for SparseStateVecAoS<R> where R: Rng + Debug);
impl_state_sampling!(impl<R> for StateVecSoA32<R> where R: Rng);
impl_state_sampling!(impl<R> for DensityMatrix<R> where R: Rng + SeedableRng + Debug + Clone);
impl_state_sampling!(impl<S, R> for SparseStabGeneric<S, R> where S: IndexSet, R: Rng + SeedableRng + Debug);
impl_state_sampling!(impl<R> for SparseStabHybrid<R> where R: Rng + SeedableRng + Debug);
impl_state_sampling!(for Stabilizer);

impl<S, R> StabilizerStateVectorConversion for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: Rng + SeedableRng + Debug,
{
    fn to_state_vector(&self) -> Result<Vec<Complex64>, StateAccessError> {
        stabilizer_group_to_state_vector(&self.to_stabilizer_group())
    }
}

impl<R> StabilizerStateVectorConversion for SparseStabHybrid<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn to_state_vector(&self) -> Result<Vec<Complex64>, StateAccessError> {
        stabilizer_group_to_state_vector(&self.to_stabilizer_group())
    }
}

impl<R> StabilizerStateVectorConversion for DenseStab<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn to_state_vector(&self) -> Result<Vec<Complex64>, StateAccessError> {
        stabilizer_group_to_state_vector(&self.to_stabilizer_group())
    }
}

impl StabilizerStateVectorConversion for Stabilizer {
    fn to_state_vector(&self) -> Result<Vec<Complex64>, StateAccessError> {
        stabilizer_group_to_state_vector(&self.to_stabilizer_group())
    }
}

impl<R> StateVectorAccess for StateVecSoA<R>
where
    R: Rng,
{
    fn amplitude(&mut self, basis_state: usize) -> Result<Complex64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.get_amplitude(basis_state))
    }

    fn state_vector(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        let expected = self.hilbert_dim()?;
        let state = self.state();
        validate_state_vector_len(&state, expected)?;
        Ok(state)
    }

    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.probability(basis_state))
    }
}

impl<R> PauliExpectationAccess for StateVecSoA<R>
where
    R: Rng,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        let num_qubits = StateInfo::num_qubits(self);
        pauli_expectation_from_state_vector(&self.state_vector()?, num_qubits, pauli)
    }
}

impl<R> StateVectorAccess for SparseStateVecSoA<R>
where
    R: Rng + Debug,
{
    fn amplitude(&mut self, basis_state: usize) -> Result<Complex64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.get_amplitude(basis_state))
    }

    fn state_vector(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        let expected = self.hilbert_dim()?;
        let state = self.state();
        validate_state_vector_len(&state, expected)?;
        Ok(state)
    }

    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.probability(basis_state))
    }
}

impl<R> PauliExpectationAccess for SparseStateVecSoA<R>
where
    R: Rng + Debug,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        let num_qubits = StateInfo::num_qubits(self);
        pauli_expectation_from_state_vector(&self.state_vector()?, num_qubits, pauli)
    }
}

impl<R> StateVectorAccess for StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn amplitude(&mut self, basis_state: usize) -> Result<Complex64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.state()[basis_state])
    }

    fn state_vector(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        let expected = self.hilbert_dim()?;
        let state = self.state().to_vec();
        validate_state_vector_len(&state, expected)?;
        Ok(state)
    }

    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.probability(basis_state))
    }
}

impl<R> PauliExpectationAccess for StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        let num_qubits = StateInfo::num_qubits(self);
        pauli_expectation_from_state_vector(&self.state_vector()?, num_qubits, pauli)
    }
}

impl<R> StateVectorAccess for SparseStateVecAoS<R>
where
    R: Rng + Debug,
{
    fn amplitude(&mut self, basis_state: usize) -> Result<Complex64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.get_amplitude(basis_state))
    }

    fn state_vector(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        let dim = self.hilbert_dim()?;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        for (basis_state, amp) in state.iter_mut().enumerate() {
            *amp = self.get_amplitude(basis_state);
        }
        Ok(state)
    }
}

impl<R> PauliExpectationAccess for SparseStateVecAoS<R>
where
    R: Rng + Debug,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        let num_qubits = StateInfo::num_qubits(self);
        pauli_expectation_from_state_vector(&self.state_vector()?, num_qubits, pauli)
    }
}

impl<R> StateVectorAccess for StateVecSoA32<R>
where
    R: Rng,
{
    fn amplitude(&mut self, basis_state: usize) -> Result<Complex64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        let amp = self.get_amplitude(basis_state);
        Ok(Complex64::new(f64::from(amp.re), f64::from(amp.im)))
    }

    fn state_vector(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        let expected = self.hilbert_dim()?;
        let state: Vec<Complex64> = self
            .to_complex_vec()
            .into_iter()
            .map(|amp| Complex64::new(f64::from(amp.re), f64::from(amp.im)))
            .collect();
        validate_state_vector_len(&state, expected)?;
        Ok(state)
    }

    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.probability(basis_state))
    }
}

impl<R> PauliExpectationAccess for StateVecSoA32<R>
where
    R: Rng,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        let num_qubits = StateInfo::num_qubits(self);
        pauli_expectation_from_state_vector(&self.state_vector()?, num_qubits, pauli)
    }
}

impl<R> DensityMatrixAccess for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn density_matrix(&mut self) -> Result<Vec<Vec<Complex64>>, StateAccessError> {
        let expected = self.hilbert_dim()?;
        let rho = self.get_density_matrix();
        validate_density_matrix_shape(&rho, expected)?;
        Ok(rho)
    }

    fn basis_probability(&mut self, basis_state: usize) -> Result<f64, StateAccessError> {
        validate_basis_index(StateInfo::num_qubits(self), basis_state)?;
        Ok(self.probability(basis_state))
    }
}

impl<R> PauliExpectationAccess for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        let num_qubits = StateInfo::num_qubits(self);
        pauli_expectation_from_density_matrix(&self.density_matrix()?, num_qubits, pauli)
    }
}

impl<S, R> PauliExpectationAccess for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: Rng + SeedableRng + Debug,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        pauli_expectation_from_stabilizer_group(
            &self.to_stabilizer_group(),
            StateInfo::num_qubits(self),
            pauli,
        )
    }
}

impl<R> PauliExpectationAccess for SparseStabHybrid<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        pauli_expectation_from_stabilizer_group(
            &self.to_stabilizer_group(),
            StateInfo::num_qubits(self),
            pauli,
        )
    }
}

impl<R> PauliExpectationAccess for DenseStab<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        pauli_expectation_from_stabilizer_group(
            &self.to_stabilizer_group(),
            StateInfo::num_qubits(self),
            pauli,
        )
    }
}

impl PauliExpectationAccess for Stabilizer {
    fn pauli_expectation(&mut self, pauli: &PauliString) -> Result<Complex64, StateAccessError> {
        pauli_expectation_from_stabilizer_group(
            &self.to_stabilizer_group(),
            StateInfo::num_qubits(self),
            pauli,
        )
    }
}

fn hilbert_dim(num_qubits: usize) -> Result<usize, StateAccessError> {
    let shift = u32::try_from(num_qubits)
        .map_err(|_| StateAccessError::DimensionOverflow { num_qubits })?;
    if shift >= usize::BITS {
        return Err(StateAccessError::DimensionOverflow { num_qubits });
    }
    Ok(1usize << shift)
}

fn validate_basis_index(num_qubits: usize, index: usize) -> Result<usize, StateAccessError> {
    let dim = hilbert_dim(num_qubits)?;
    if index >= dim {
        return Err(StateAccessError::BasisIndexOutOfRange {
            num_qubits,
            dim,
            index,
        });
    }
    Ok(dim)
}

fn validate_pauli_support(num_qubits: usize, pauli: &PauliString) -> Result<(), StateAccessError> {
    for qubit in pauli.qubits() {
        if qubit >= num_qubits {
            return Err(StateAccessError::PauliQubitOutOfRange { num_qubits, qubit });
        }
    }
    Ok(())
}

fn validate_qubit_support(num_qubits: usize, qubits: &[QubitId]) -> Result<(), StateAccessError> {
    for qubit in qubits {
        let q = qubit.index();
        if q >= num_qubits {
            return Err(StateAccessError::QubitOutOfRange {
                num_qubits,
                qubit: q,
            });
        }
    }
    Ok(())
}

fn validate_state_vector_len(state: &[Complex64], expected: usize) -> Result<(), StateAccessError> {
    if state.len() != expected {
        return Err(StateAccessError::InvalidStateVectorLength {
            expected,
            actual: state.len(),
        });
    }
    Ok(())
}

fn validate_density_matrix_shape(
    rho: &[Vec<Complex64>],
    expected: usize,
) -> Result<(), StateAccessError> {
    if rho.len() != expected {
        return Err(StateAccessError::InvalidDensityMatrixShape {
            expected,
            rows: rho.len(),
            cols: rho.first().map_or(0, Vec::len),
        });
    }
    for row in rho {
        if row.len() != expected {
            return Err(StateAccessError::InvalidDensityMatrixShape {
                expected,
                rows: rho.len(),
                cols: row.len(),
            });
        }
    }
    Ok(())
}

fn sample_z_via_clifford<T>(
    backend: &mut T,
    qubits: &[QubitId],
) -> Result<Vec<MeasurementResult>, StateAccessError>
where
    T: CliffordGateable + StateInfo,
{
    validate_qubit_support(StateInfo::num_qubits(backend), qubits)?;
    Ok(backend.mz(qubits))
}

fn sample_x_via_clifford<T>(
    backend: &mut T,
    qubits: &[QubitId],
) -> Result<Vec<MeasurementResult>, StateAccessError>
where
    T: CliffordGateable + StateInfo,
{
    validate_qubit_support(StateInfo::num_qubits(backend), qubits)?;
    Ok(backend.mx(qubits))
}

fn sample_y_via_clifford<T>(
    backend: &mut T,
    qubits: &[QubitId],
) -> Result<Vec<MeasurementResult>, StateAccessError>
where
    T: CliffordGateable + StateInfo,
{
    validate_qubit_support(StateInfo::num_qubits(backend), qubits)?;
    Ok(backend.my(qubits))
}

fn stabilizer_group_to_state_vector(
    group: &PauliStabilizerGroup,
) -> Result<Vec<Complex64>, StateAccessError> {
    let num_qubits = group.num_qubits();
    let num_generators = group.num_generators();
    let rank = group.rank();
    if rank != num_qubits {
        return Err(StateAccessError::NotPureStabilizerState {
            num_qubits,
            num_generators,
            rank,
        });
    }

    let dim = hilbert_dim(num_qubits)?;
    let generators = group.stabilizers();
    for seed in 0..dim {
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[seed] = Complex64::new(1.0, 0.0);

        for generator in generators {
            validate_pauli_support(num_qubits, generator)?;
            apply_stabilizer_projector(&mut state, generator);
            if state_norm_sqr(&state) <= f64::EPSILON {
                break;
            }
        }

        let norm_sqr = state_norm_sqr(&state);
        if norm_sqr > f64::EPSILON {
            normalize_state_vector(&mut state, norm_sqr);
            canonicalize_global_phase(&mut state);
            return Ok(state);
        }
    }

    Err(StateAccessError::NotPureStabilizerState {
        num_qubits,
        num_generators,
        rank,
    })
}

fn apply_stabilizer_projector(state: &mut [Complex64], generator: &PauliString) {
    let input = state.to_vec();
    state.fill(Complex64::new(0.0, 0.0));
    for (basis_state, amplitude) in input.iter().copied().enumerate() {
        state[basis_state] += amplitude * 0.5;
        if amplitude.norm_sqr() == 0.0 {
            continue;
        }
        let (output_state, coefficient) = pauli_action_on_basis_index(generator, basis_state);
        state[output_state] += coefficient * amplitude * 0.5;
    }
}

fn state_norm_sqr(state: &[Complex64]) -> f64 {
    state.iter().map(Complex64::norm_sqr).sum()
}

fn standard_complex_normal<R>(rng: &mut R) -> Complex64
where
    R: Rng + ?Sized,
{
    Complex64::new(standard_normal(rng), standard_normal(rng))
}

fn standard_normal<R>(rng: &mut R) -> f64
where
    R: Rng + ?Sized,
{
    loop {
        let u1 = rng.random::<f64>();
        if u1 > 0.0 {
            let u2 = rng.random::<f64>();
            return (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
        }
    }
}

fn normalize_state_vector(state: &mut [Complex64], norm_sqr: f64) {
    let scale = norm_sqr.sqrt();
    for amplitude in state {
        *amplitude /= scale;
    }
}

fn canonicalize_global_phase(state: &mut [Complex64]) {
    let Some(first_nonzero) = state
        .iter()
        .copied()
        .find(|amplitude| amplitude.norm_sqr() > f64::EPSILON)
    else {
        return;
    };
    let phase = first_nonzero / Complex64::new(first_nonzero.norm(), 0.0);
    let correction = phase.conj();
    for amplitude in state {
        *amplitude *= correction;
    }
}

fn pauli_action_on_basis_index(pauli: &PauliString, basis_state: usize) -> (usize, Complex64) {
    let mut output = basis_state;
    let mut coefficient = pauli.phase().to_complex();
    for (single, qubit) in pauli.iter_pairs() {
        let q = qubit.index();
        let bit = (basis_state >> q) & 1;
        match single {
            Pauli::I => {}
            Pauli::X => {
                output ^= 1usize << q;
            }
            Pauli::Y => {
                output ^= 1usize << q;
                coefficient *= if bit == 0 {
                    Complex64::new(0.0, 1.0)
                } else {
                    Complex64::new(0.0, -1.0)
                };
            }
            Pauli::Z => {
                if bit == 1 {
                    coefficient = -coefficient;
                }
            }
        }
    }
    (output, coefficient)
}

fn pauli_expectation_from_state_vector(
    state: &[Complex64],
    num_qubits: usize,
    pauli: &PauliString,
) -> Result<Complex64, StateAccessError> {
    validate_pauli_support(num_qubits, pauli)?;
    let dim = hilbert_dim(num_qubits)?;
    validate_state_vector_len(state, dim)?;
    let mut expectation = Complex64::new(0.0, 0.0);
    for basis_state in 0..dim {
        let (output, coefficient) = pauli_action_on_basis_index(pauli, basis_state);
        expectation += coefficient * state[basis_state] * state[output].conj();
    }
    Ok(expectation)
}

fn pauli_expectation_from_density_matrix(
    rho: &[Vec<Complex64>],
    num_qubits: usize,
    pauli: &PauliString,
) -> Result<Complex64, StateAccessError> {
    validate_pauli_support(num_qubits, pauli)?;
    let dim = hilbert_dim(num_qubits)?;
    validate_density_matrix_shape(rho, dim)?;
    let mut expectation = Complex64::new(0.0, 0.0);
    for (basis_state, rho_row) in rho.iter().enumerate().take(dim) {
        let (output, coefficient) = pauli_action_on_basis_index(pauli, basis_state);
        expectation += coefficient * rho_row[output];
    }
    Ok(expectation)
}

fn pauli_expectation_from_stabilizer_group(
    stabilizers: &PauliStabilizerGroup,
    num_qubits: usize,
    pauli: &PauliString,
) -> Result<Complex64, StateAccessError> {
    validate_pauli_support(num_qubits, pauli)?;

    let mut positive_body = pauli.clone();
    positive_body.set_phase(QuarterPhase::PlusOne);
    if !stabilizers.contains(&positive_body) {
        return Ok(Complex64::new(0.0, 0.0));
    }

    let stabilizer_phase = if stabilizers.contains_with_phase(&positive_body) {
        QuarterPhase::PlusOne
    } else {
        QuarterPhase::MinusOne
    };
    Ok(pauli
        .phase()
        .multiply(&stabilizer_phase.conjugate())
        .to_complex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CliffordGateable, DenseStab, SparseStab, SparseStabHybrid, Stabilizer, StateVec, qid,
    };
    use pecos_core::QubitId;
    use pecos_core::pauli::algebra::i;
    use pecos_core::pauli::*;
    use pecos_random::PecosRng;

    fn assert_close(actual: Complex64, expected: Complex64) {
        assert!(
            (actual - expected).norm() < 1e-10,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_state_vectors_close(actual: &[Complex64], expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual_amp, &expected_amp)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual_amp - expected_amp).norm() < 1e-10,
                "state-vector mismatch at basis {index}: actual={actual_amp}, expected={expected_amp}"
            );
        }
    }

    #[test]
    fn state_vector_access_flushes_pending_gates() {
        let mut state = StateVecSoA::new(1);
        state.h(&qid(0));

        let amp0 = StateVectorAccess::amplitude(&mut state, 0).unwrap();
        let amp1 = StateVectorAccess::amplitude(&mut state, 1).unwrap();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert_close(amp0, Complex64::new(expected, 0.0));
        assert_close(amp1, Complex64::new(expected, 0.0));
        assert!((StateVectorAccess::basis_probability(&mut state, 0).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn random_statevector_is_normalized_and_seed_reproducible() {
        let mut rng = PecosRng::seed_from_u64(123);
        let state = random_statevector(&mut rng, 3).unwrap();
        assert_eq!(state.len(), 8);
        assert!((state_norm_sqr(&state) - 1.0).abs() < 1e-12);

        let mut same_seed = PecosRng::seed_from_u64(123);
        let same = random_statevector(&mut same_seed, 3).unwrap();
        assert_eq!(state, same);

        let mut different_seed = PecosRng::seed_from_u64(456);
        let different = random_statevector(&mut different_seed, 3).unwrap();
        assert_ne!(state, different);
    }

    #[test]
    fn random_statevector_zero_qubits_is_scalar_identity_state() {
        let mut rng = PecosRng::seed_from_u64(123);
        let state = random_statevector(&mut rng, 0).unwrap();
        assert_state_vectors_close(&state, &[Complex64::new(1.0, 0.0)]);
    }

    #[test]
    fn random_statevector_haar_marginal_is_reasonable() {
        let mut rng = PecosRng::seed_from_u64(123);
        let samples = 2_000;
        let mut p0_sum = 0.0;
        for _ in 0..samples {
            let state = random_statevector(&mut rng, 2).unwrap();
            p0_sum += state[0].norm_sqr();
        }
        let mean = p0_sum / f64::from(samples);
        assert!(
            (0.22..0.28).contains(&mean),
            "expected E[|psi_0|^2] near 1/4 for a 4D Haar state, got {mean}"
        );
    }

    #[test]
    fn sparse_state_vector_access_returns_dense_vector() {
        let mut state = StateVec::new(2);
        state.x(&qid(1));

        let dense = state.state_vector().unwrap();
        assert_eq!(dense.len(), 4);
        assert_close(dense[0], Complex64::new(0.0, 0.0));
        assert_close(dense[2], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn all_state_vector_backends_expose_amplitudes() {
        let mut dense_soa = StateVecSoA::new(1);
        let mut dense_aos = StateVecAoS::new(1);
        let mut sparse_soa = SparseStateVecSoA::new(1);
        let mut sparse_aos = SparseStateVecAoS::new(1);
        let mut soa32 = StateVecSoA32::new(1);

        dense_soa.x(&qid(0));
        dense_aos.x(&qid(0));
        sparse_soa.x(&qid(0));
        sparse_aos.x(&qid(0));
        soa32.x(&qid(0));

        assert_close(dense_soa.amplitude(1).unwrap(), Complex64::new(1.0, 0.0));
        assert_close(dense_aos.amplitude(1).unwrap(), Complex64::new(1.0, 0.0));
        assert_close(sparse_soa.amplitude(1).unwrap(), Complex64::new(1.0, 0.0));
        assert_close(sparse_aos.amplitude(1).unwrap(), Complex64::new(1.0, 0.0));
        assert_close(soa32.amplitude(1).unwrap(), Complex64::new(1.0, 0.0));
    }

    #[test]
    fn state_vector_pauli_expectations_match_known_states() {
        let mut plus = StateVec::new(1);
        plus.h(&qid(0));

        assert_close(
            plus.pauli_expectation(&X(0)).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            plus.pauli_expectation(&Z(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );
        assert_close(
            plus.pauli_expectation(&(i * X(0))).unwrap(),
            Complex64::new(0.0, 1.0),
        );
    }

    #[test]
    fn density_matrix_access_and_expectation_match_known_state() {
        let mut state = DensityMatrix::new(1);
        state.h(&qid(0));

        let rho = state.density_matrix().unwrap();
        assert_eq!(rho.len(), 2);
        assert_close(rho[0][0], Complex64::new(0.5, 0.0));
        assert_close(rho[0][1], Complex64::new(0.5, 0.0));
        assert!(
            (DensityMatrixAccess::basis_probability(&mut state, 0).unwrap() - 0.5).abs() < 1e-10
        );
        assert_close(
            state.pauli_expectation(&X(0)).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            state.pauli_expectation(&Z(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );
    }

    #[test]
    fn sparse_stab_pauli_expectation_uses_signed_stabilizer_membership() {
        let mut one = SparseStab::new(1);
        one.x(&qid(0));

        assert_close(
            one.pauli_expectation(&Z(0)).unwrap(),
            Complex64::new(-1.0, 0.0),
        );
        assert_close(
            one.pauli_expectation(&(-Z(0))).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            one.pauli_expectation(&(i * Z(0))).unwrap(),
            Complex64::new(0.0, -1.0),
        );
        assert_close(
            one.pauli_expectation(&X(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );
    }

    #[test]
    fn sparse_stab_pauli_expectation_matches_state_vector_for_bell_state() {
        let mut state_vec = StateVec::new(2);
        let mut stab = SparseStab::new(2);
        state_vec.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        for pauli in [
            X(0) & X(1),
            Y(0) & Y(1),
            Z(0) & Z(1),
            X(0) & Z(1),
            Y(0) & Z(1),
            X(0) & Y(1),
            X(0),
            Z(0),
        ] {
            let expected = state_vec.pauli_expectation(&pauli).unwrap();
            let actual = stab.pauli_expectation(&pauli).unwrap();
            assert_close(actual, expected);
        }
    }

    #[test]
    fn dense_stab_pauli_expectation_matches_state_vector_for_bell_state() {
        let mut state_vec = StateVec::new(2);
        let mut dense = DenseStab::new(2);
        state_vec.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        dense.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        for pauli in [
            X(0) & X(1),
            Y(0) & Y(1),
            Z(0) & Z(1),
            X(0) & Z(1),
            Y(0) & Z(1),
            X(0) & Y(1),
            X(0),
            Z(0),
        ] {
            let expected = state_vec.pauli_expectation(&pauli).unwrap();
            let actual = dense.pauli_expectation(&pauli).unwrap();
            assert_close(actual, expected);
        }
    }

    #[test]
    fn stabilizer_pauli_expectations_match_state_vector_for_three_qubit_ghz() {
        let mut state_vec = StateVec::new(3);
        let mut sparse = SparseStab::new(3);
        let mut dense = DenseStab::new(3);
        let pairs = [(QubitId(0), QubitId(1)), (QubitId(0), QubitId(2))];
        state_vec.h(&qid(0)).cx(&pairs);
        sparse.h(&qid(0)).cx(&pairs);
        dense.h(&qid(0)).cx(&pairs);

        for pauli in [
            X(0) & X(1) & X(2),
            Z(0) & Z(1),
            Z(1) & Z(2),
            Z(0) & Z(2),
            Y(0) & Y(1) & X(2),
            X(0),
            Z(0),
        ] {
            let expected = state_vec.pauli_expectation(&pauli).unwrap();
            let sparse_actual = sparse.pauli_expectation(&pauli).unwrap();
            let dense_actual = dense.pauli_expectation(&pauli).unwrap();
            assert_close(sparse_actual, expected);
            assert_close(dense_actual, expected);
        }
    }

    #[test]
    fn sparse_stab_hybrid_supports_pauli_expectation_access() {
        let mut plus = SparseStabHybrid::new(1);
        plus.h(&qid(0));

        assert_close(
            plus.pauli_expectation(&X(0)).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            plus.pauli_expectation(&Z(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );
        assert_close(
            plus.pauli_expectation(&(i * X(0))).unwrap(),
            Complex64::new(0.0, 1.0),
        );
    }

    #[test]
    fn stabilizer_generator_bridge_preserves_y_phase_convention() {
        let mut sparse = SparseStab::new(1);
        sparse.h(&qid(0)).sz(&qid(0));

        assert_close(
            sparse.pauli_expectation(&Y(0)).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            sparse.pauli_expectation(&X(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );

        let mut hybrid = SparseStabHybrid::new(1);
        hybrid.h(&qid(0)).sz(&qid(0));
        assert_close(
            hybrid.pauli_expectation(&Y(0)).unwrap(),
            Complex64::new(1.0, 0.0),
        );

        let mut dense = DenseStab::new(1);
        dense.h(&qid(0)).sz(&qid(0));
        assert_close(
            dense.pauli_expectation(&Y(0)).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            dense.pauli_expectation(&X(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );
    }

    #[test]
    fn dense_stab_pauli_expectation_preserves_signed_basis_state() {
        let mut one = DenseStab::new(1);
        one.x(&qid(0));

        assert_close(
            one.pauli_expectation(&Z(0)).unwrap(),
            Complex64::new(-1.0, 0.0),
        );
        assert_close(
            one.pauli_expectation(&(-Z(0))).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            one.pauli_expectation(&(i * Z(0))).unwrap(),
            Complex64::new(0.0, -1.0),
        );
    }

    #[test]
    fn default_stabilizer_supports_pauli_expectation_access() {
        let mut bell = Stabilizer::new(2);
        bell.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        assert_close(
            bell.pauli_expectation(&(X(0) & X(1))).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            bell.pauli_expectation(&(Z(0) & Z(1))).unwrap(),
            Complex64::new(1.0, 0.0),
        );
        assert_close(
            bell.pauli_expectation(&Z(0)).unwrap(),
            Complex64::new(0.0, 0.0),
        );
    }

    #[test]
    fn state_sampling_z_collapses_state_vector() {
        let mut state = StateVec::new(1);
        state.h(&qid(0));

        let first = state.sample_z(&qid(0)).unwrap();
        assert_eq!(first.len(), 1);
        assert!(!first[0].is_deterministic);

        let second = state.sample_z(&qid(0)).unwrap();
        assert_eq!(second.len(), 1);
        assert!(second[0].is_deterministic);
        assert_eq!(second[0].outcome, first[0].outcome);
    }

    #[test]
    fn state_sampling_x_and_y_bases_use_existing_measurement_semantics() {
        let mut plus = StateVec::new(1);
        plus.h(&qid(0));
        let x_result = plus.sample_x(&qid(0)).unwrap();
        assert!(x_result[0].is_deterministic);
        assert!(!x_result[0].outcome);

        let mut plus_i = StateVec::new(1);
        plus_i.h(&qid(0)).sz(&qid(0));
        let y_result = plus_i.sample_y(&qid(0)).unwrap();
        assert!(y_result[0].is_deterministic);
        assert!(!y_result[0].outcome);
    }

    #[test]
    fn state_sampling_batch_preserves_requested_qubit_order() {
        let mut state = StateVec::new(2);
        state.x(&qid(0));

        let results = state.sample_z(&[QubitId(1), QubitId(0)]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_deterministic);
        assert!(results[1].is_deterministic);
        assert!(!results[0].outcome);
        assert!(results[1].outcome);
    }

    #[test]
    fn state_sampling_is_available_on_density_matrix_and_stabilizers() {
        let mut density = DensityMatrix::new(1);
        density.x(&qid(0));
        let density_result = density.sample_z(&qid(0)).unwrap();
        assert!(density_result[0].is_deterministic);
        assert!(density_result[0].outcome);

        let mut sparse = SparseStab::new(1);
        sparse.x(&qid(0));
        let sparse_result = sparse.sample_z(&qid(0)).unwrap();
        assert!(sparse_result[0].is_deterministic);
        assert!(sparse_result[0].outcome);

        let mut hybrid = SparseStabHybrid::new(1);
        hybrid.x(&qid(0));
        let hybrid_result = hybrid.sample_z(&qid(0)).unwrap();
        assert!(hybrid_result[0].is_deterministic);
        assert!(hybrid_result[0].outcome);

        let mut default_stabilizer = Stabilizer::new(1);
        default_stabilizer.x(&qid(0));
        let default_result = default_stabilizer.sample_z(&qid(0)).unwrap();
        assert!(default_result[0].is_deterministic);
        assert!(default_result[0].outcome);
    }

    #[test]
    fn state_access_rejects_out_of_range_queries() {
        let mut state = StateVec::new(1);
        assert!(matches!(
            state.amplitude(2).unwrap_err(),
            StateAccessError::BasisIndexOutOfRange { .. }
        ));
        assert!(matches!(
            state.pauli_expectation(&X(1)).unwrap_err(),
            StateAccessError::PauliQubitOutOfRange { .. }
        ));
        let sample_result = state.sample_z(&[QubitId(1)]);
        assert!(matches!(
            sample_result,
            Err(StateAccessError::QubitOutOfRange { .. })
        ));

        let mut rho = DensityMatrix::new(1);
        assert!(matches!(
            rho.density_matrix_element(0, 2).unwrap_err(),
            StateAccessError::BasisIndexOutOfRange { .. }
        ));
    }

    #[test]
    fn sparse_stabilizer_to_state_vector_matches_initial_state() {
        let sparse = SparseStab::new(2);
        let dense = sparse.to_state_vector().unwrap();

        assert_state_vectors_close(
            &dense,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
    }

    #[test]
    fn sparse_stabilizer_to_state_vector_preserves_signed_basis_state() {
        let mut sparse = SparseStab::new(1);
        sparse.x(&qid(0));

        let dense = sparse.to_state_vector().unwrap();
        assert_state_vectors_close(
            &dense,
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        );
    }

    #[test]
    fn sparse_stabilizer_to_state_vector_matches_bell_statevec() {
        let mut sparse = SparseStab::new(2);
        sparse.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        let mut state_vec = StateVec::new(2);
        state_vec.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        let dense = sparse.to_state_vector().unwrap();
        assert_state_vectors_close(&dense, &state_vec.state_vector().unwrap());
    }

    #[test]
    fn dense_stabilizer_to_state_vector_matches_ghz_statevec() {
        let mut dense = DenseStab::new(3);
        let mut state_vec = StateVec::new(3);
        let pairs = [(QubitId(0), QubitId(1)), (QubitId(0), QubitId(2))];
        dense.h(&qid(0)).cx(&pairs);
        state_vec.h(&qid(0)).cx(&pairs);

        let dense_state = dense.to_state_vector().unwrap();
        assert_state_vectors_close(&dense_state, &state_vec.state_vector().unwrap());
    }

    #[test]
    fn sparse_stabilizer_to_state_vector_preserves_complex_phase_state() {
        let mut sparse = SparseStab::new(1);
        sparse.h(&qid(0)).sz(&qid(0));

        let mut state_vec = StateVec::new(1);
        state_vec.h(&qid(0)).sz(&qid(0));

        let dense = sparse.to_state_vector().unwrap();
        assert_state_vectors_close(&dense, &state_vec.state_vector().unwrap());
    }

    #[test]
    fn stabilizer_to_state_vector_is_available_on_hybrid_and_default_wrapper() {
        let mut hybrid = SparseStabHybrid::new(2);
        hybrid.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        let mut default_stabilizer = Stabilizer::new(2);
        default_stabilizer
            .h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))]);

        let hybrid_dense = hybrid.to_state_vector().unwrap();
        let default_dense = default_stabilizer.to_state_vector().unwrap();
        assert_state_vectors_close(&hybrid_dense, &default_dense);
    }
}
