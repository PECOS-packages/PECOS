//! `QuEST` quantum simulator wrapper for PECOS
//!
//! # Thread Safety Warning
//!
//! **CRITICAL**: `QuEST` has a fundamental limitation - it uses a single global environment
//! per process. This means ALL `QuestStateVec` instances share the same underlying `QuEST`
//! environment, which can lead to race conditions and segmentation faults when used
//! concurrently from multiple threads.
//!
//! For safe usage:
//! - Run tests with `--test-threads=1`
//! - Use only one `QuestStateVec` instance per process in production
//! - See `THREAD_SAFETY_WARNING.md` for detailed information

use core::fmt::Debug;
use num_complex::Complex64;
use pecos_random::{PecosRng, Rng, SeedableRng, time_seed};
use std::f64::consts::FRAC_PI_4;
use thiserror::Error;

pub mod bridge;
use bridge::ffi;

pub mod cuda_loader;

pub mod quantum_engine;
pub use quantum_engine::{
    QuestCudaStateVecEngine, QuestDensityMatrixEngine, QuestDensityMatrixEngineBuilder,
    QuestStateVecEngine, QuestStateVectorEngineBuilder, quest_density_matrix, quest_state_vec,
};

pub use pecos_core::rng::RngManageable;
use pecos_core::{Angle64, QubitId};
pub use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};

#[derive(Error, Debug)]
pub enum QuestError {
    #[error("QuEST initialization failed: {0}")]
    InitializationError(String),

    #[error("Invalid qubit index: {0}")]
    InvalidQubit(usize),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("FFI error: {0}")]
    FfiError(#[from] cxx::Exception),
}

pub type Result<T> = std::result::Result<T, QuestError>;

/// RAII wrapper for `QuEST` environment pointer
#[derive(Debug)]
struct QuestEnvWrapper {
    ptr: *mut u8,
}

impl QuestEnvWrapper {
    fn new() -> Result<Self> {
        let ptr = ffi::quest_create_env();
        if ptr.is_null() {
            return Err(QuestError::InitializationError(
                "Failed to create QuEST environment".into(),
            ));
        }
        Ok(Self { ptr })
    }
}

impl Drop for QuestEnvWrapper {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::quest_destroy_env(self.ptr);
            }
        }
    }
}

unsafe impl Send for QuestEnvWrapper {}
unsafe impl Sync for QuestEnvWrapper {}

/// RAII wrapper for `QuEST` qureg pointer
#[derive(Debug)]
struct QuregWrapper {
    ptr: *mut u8,
}

impl QuregWrapper {
    fn new(env: &QuestEnvWrapper, num_qubits: i32, is_density: bool) -> Result<Self> {
        let ptr = unsafe {
            if is_density {
                ffi::quest_create_density_qureg(env.ptr, num_qubits)
            } else {
                ffi::quest_create_qureg(env.ptr, num_qubits)
            }
        };

        if ptr.is_null() {
            return Err(QuestError::InitializationError(
                "Failed to create QuEST qureg".into(),
            ));
        }
        Ok(Self { ptr })
    }
}

impl Drop for QuregWrapper {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::quest_destroy_qureg(self.ptr);
            }
        }
    }
}

unsafe impl Send for QuregWrapper {}
unsafe impl Sync for QuregWrapper {}

/// A quantum state simulator using the `QuEST` state vector representation
#[derive(Debug)]
pub struct QuestStateVec<R = PecosRng>
where
    R: Rng + SeedableRng + Debug,
{
    num_qubits: usize,
    // The QuEST environment must be kept alive for the lifetime of the simulator.
    // This field manages the global QuEST environment reference count via RAII.
    env: QuestEnvWrapper,
    qureg: QuregWrapper,
    rng: R,
}

impl QuestStateVec {
    /// Creates a new `QuestStateVec` with the specified number of qubits.
    ///
    /// # Panics
    ///
    /// Panics if the `QuEST` environment cannot be created or if the quantum register
    /// allocation fails.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, time_seed())
    }
}

impl<R> QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Creates a new `QuestStateVec` with the specified number of qubits and seed.
    ///
    /// # Panics
    ///
    /// Panics if the `QuEST` environment cannot be created or if the quantum register
    /// allocation fails.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let env = QuestEnvWrapper::new().expect("Failed to create QuEST environment");
        let qureg = QuregWrapper::new(
            &env,
            i32::try_from(num_qubits).expect("Too many qubits"),
            false,
        )
        .expect("Failed to create QuEST qureg");
        let rng = R::seed_from_u64(seed);

        let state = Self {
            num_qubits,
            env,
            qureg,
            rng,
        };

        unsafe {
            ffi::quest_init_zero_state(state.qureg.ptr);
        }
        state
    }

    /// Returns the probability of measuring the given computational basis state.
    ///
    /// # Panics
    ///
    /// Panics if the index is too large to be converted to `i64`.
    pub fn probability(&self, index: usize) -> f64 {
        let quest_index = self.convert_basis_state(index);
        unsafe {
            ffi::quest_get_prob_amp(
                self.qureg.ptr,
                i64::try_from(quest_index).expect("Index too large"),
            )
        }
    }

    /// Convert PECOS basis state to `QuEST` basis state by reversing bit order
    #[inline]
    fn convert_basis_state(&self, pecos_basis: usize) -> usize {
        let mut quest_basis = 0;
        for i in 0..self.num_qubits {
            if (pecos_basis >> i) & 1 == 1 {
                // Bit i in PECOS maps to bit (n-1-i) in QuEST
                quest_basis |= 1 << (self.num_qubits - 1 - i);
            }
        }
        quest_basis
    }

    /// Prepares the quantum state in the specified computational basis state.
    ///
    /// # Panics
    ///
    /// Panics if the index is too large to be converted to `i64`.
    pub fn prepare_computational_basis(&mut self, index: usize) {
        let quest_index = self.convert_basis_state(index);
        unsafe {
            ffi::quest_init_classical_state(
                self.qureg.ptr,
                i64::try_from(quest_index).expect("Index too large"),
            );
        }
    }

    pub fn prepare_plus_state(&mut self) {
        unsafe {
            ffi::quest_init_plus_state(self.qureg.ptr);
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get information about the quantum register (for debugging/introspection)
    pub fn get_info(&self) -> ffi::QuregInfo {
        unsafe { ffi::quest_get_qureg_info(self.qureg.ptr) }
    }

    /// Get information about the `QuEST` environment (for debugging/introspection)
    pub fn get_env_info(&self) -> ffi::QuESTEnvInfo {
        unsafe { ffi::quest_get_env_info(self.env.ptr) }
    }

    fn check_qubit_index(&self, qubit: usize) -> Result<()> {
        if qubit >= self.num_qubits {
            Err(QuestError::InvalidQubit(qubit))
        } else {
            Ok(())
        }
    }

    /// Converts from PECOS qubit indexing (qubit 0 is MSB) to `QuEST` indexing (qubit 0 is LSB)
    fn convert_qubit_index(&self, pecos_qubit: usize) -> i32 {
        i32::try_from(self.num_qubits - 1 - pecos_qubit).expect("Qubit index out of range")
    }
}

impl<R> Clone for QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn clone(&self) -> Self {
        // Create a new independent instance with same parameters
        let env = QuestEnvWrapper::new().expect("Failed to create QuEST environment");

        // Clone the quantum state - quest_clone_qureg creates a new qureg with cloned state
        let cloned_qureg_ptr = unsafe { ffi::quest_clone_qureg(self.qureg.ptr) };

        let qureg = QuregWrapper {
            ptr: cloned_qureg_ptr,
        };

        Self {
            num_qubits: self.num_qubits,
            env,
            qureg,
            rng: self.rng.clone(),
        }
    }
}

impl<R> QuantumSimulator for QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn reset(&mut self) -> &mut Self {
        unsafe {
            ffi::quest_init_zero_state(self.qureg.ptr);
        }
        self
    }
}

impl<R> CliffordGateable for QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_hadamard(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_s_gate(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let control = q0.index();
            let target = q1.index();
            self.check_qubit_index(control)
                .expect("Invalid control qubit");
            self.check_qubit_index(target)
                .expect("Invalid target qubit");
            let quest_control = self.convert_qubit_index(control);
            let quest_target = self.convert_qubit_index(target);
            unsafe {
                ffi::quest_apply_cnot(self.qureg.ptr, quest_control, quest_target);
            }
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        use rand::RngExt;

        let mut results = Vec::with_capacity(qubits.len());
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());

            // Get probability of measuring |0⟩ (deterministic calculation)
            let prob_0 = unsafe { ffi::quest_calc_prob_of_outcome(self.qureg.ptr, quest_qubit, 0) };

            // Sample outcome using our seeded Rust RNG
            let outcome = i32::from(self.rng.random::<f64>() >= prob_0);

            // Collapse state to the sampled outcome
            let actual_prob = unsafe {
                ffi::quest_apply_forced_measurement(self.qureg.ptr, quest_qubit, outcome)
            };

            results.push(MeasurementResult {
                outcome: outcome != 0,
                is_deterministic: (actual_prob - 1.0).abs() < f64::EPSILON,
            });
        }
        results
    }

    // Override with native QuEST implementations for better performance

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_pauli_x(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_pauli_y(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_pauli_z(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let control = q0.index();
            let target = q1.index();
            self.check_qubit_index(control)
                .expect("Invalid control qubit");
            self.check_qubit_index(target)
                .expect("Invalid target qubit");
            let quest_control = self.convert_qubit_index(control);
            let quest_target = self.convert_qubit_index(target);
            unsafe {
                ffi::quest_apply_cz(self.qureg.ptr, quest_control, quest_target);
            }
        }
        self
    }

    // SWAP gate - using trait default implementation
    // The native QuEST swap has GPU dependencies that cause linking issues
}

impl<R> ArbitraryRotationGateable for QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_rotation_x(self.qureg.ptr, quest_qubit, theta);
            }
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_rotation_z(self.qureg.ptr, quest_qubit, theta);
            }
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let half_angle = theta / 2.0;
        for &(q1, q2) in pairs {
            self.check_qubit_index(q1.index())
                .expect("Invalid qubit1 index");
            self.check_qubit_index(q2.index())
                .expect("Invalid qubit2 index");

            let half_angle_a = Angle64::from_radians(half_angle);
            let neg_half_angle_a = Angle64::from_radians(-half_angle);
            self.rz(half_angle_a, &[q1]).rz(half_angle_a, &[q2]);
            self.cz(&[(q1, q2)]);
            self.rz(neg_half_angle_a, &[q1]).rz(neg_half_angle_a, &[q2]);
        }
        self
    }

    // Override with native QuEST implementations

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_rotation_y(self.qureg.ptr, quest_qubit, theta);
            }
        }
        self
    }

    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_t_gate(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_phase_shift(self.qureg.ptr, quest_qubit, -FRAC_PI_4);
            }
        }
        self
    }
}

impl<R> RngManageable for QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug,
{
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

// Additional methods for QuestStateVec
impl<R> QuestStateVec<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Returns the complex amplitude of the specified computational basis state.
    ///
    /// # Panics
    ///
    /// Panics if the index is too large to be converted to `i64`.
    pub fn get_amplitude(&self, index: usize) -> Complex64 {
        let complex_amp = unsafe {
            ffi::quest_get_complex_amp(
                self.qureg.ptr,
                i64::try_from(index).expect("Index too large"),
            )
        };
        Complex64::new(complex_amp.real, complex_amp.imag)
    }
}

unsafe impl<R> Send for QuestStateVec<R> where R: Rng + SeedableRng + Debug + Send {}

unsafe impl<R> Sync for QuestStateVec<R> where R: Rng + SeedableRng + Debug + Sync {}

/// A quantum density matrix simulator using `QuEST`'s density matrix representation
#[derive(Debug)]
pub struct QuestDensityMatrix<R = PecosRng>
where
    R: Rng + SeedableRng + Debug,
{
    num_qubits: usize,
    // The QuEST environment must be kept alive for the lifetime of the simulator.
    // This field manages the global QuEST environment reference count via RAII.
    env: QuestEnvWrapper,
    qureg: QuregWrapper,
    rng: R,
}

impl QuestDensityMatrix {
    /// Creates a new `QuestDensityMatrix` with the specified number of qubits.
    ///
    /// # Panics
    ///
    /// Panics if the `QuEST` environment cannot be created or if the quantum register
    /// allocation fails.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, time_seed())
    }
}

impl<R> QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Creates a new `QuestDensityMatrix` with the specified number of qubits and seed.
    ///
    /// # Panics
    ///
    /// Panics if the `QuEST` environment cannot be created or if the quantum register
    /// allocation fails.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let env = QuestEnvWrapper::new().expect("Failed to create QuEST environment");
        let qureg = QuregWrapper::new(
            &env,
            i32::try_from(num_qubits).expect("Too many qubits"),
            true,
        )
        .expect("Failed to create QuEST density matrix");
        let rng = R::seed_from_u64(seed);

        let state = Self {
            num_qubits,
            env,
            qureg,
            rng,
        };

        unsafe {
            ffi::quest_init_zero_state(state.qureg.ptr);
        }
        state
    }

    /// Returns the probability of measuring the given computational basis state.
    ///
    /// # Panics
    ///
    /// Panics if the index is too large to be converted to `i64`.
    pub fn probability(&self, index: usize) -> f64 {
        let quest_index = self.convert_basis_state(index);
        unsafe {
            ffi::quest_get_prob_amp(
                self.qureg.ptr,
                i64::try_from(quest_index).expect("Index too large"),
            )
        }
    }

    /// Convert PECOS basis state to `QuEST` basis state by reversing bit order
    #[inline]
    fn convert_basis_state(&self, pecos_basis: usize) -> usize {
        let mut quest_basis = 0;
        for i in 0..self.num_qubits {
            if (pecos_basis >> i) & 1 == 1 {
                // Bit i in PECOS maps to bit (n-1-i) in QuEST
                quest_basis |= 1 << (self.num_qubits - 1 - i);
            }
        }
        quest_basis
    }

    pub fn purity(&self) -> f64 {
        unsafe { ffi::quest_calc_purity(self.qureg.ptr) }
    }

    /// Prepares the density matrix in the specified computational basis state.
    ///
    /// # Panics
    ///
    /// Panics if the index is too large to be converted to `i64`.
    pub fn prepare_computational_basis(&mut self, index: usize) {
        let quest_index = self.convert_basis_state(index);
        unsafe {
            ffi::quest_init_classical_state(
                self.qureg.ptr,
                i64::try_from(quest_index).expect("Index too large"),
            );
        }
    }

    pub fn prepare_plus_state(&mut self) {
        unsafe {
            ffi::quest_init_plus_state(self.qureg.ptr);
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get information about the quantum register (for debugging/introspection)
    pub fn get_info(&self) -> ffi::QuregInfo {
        unsafe { ffi::quest_get_qureg_info(self.qureg.ptr) }
    }

    /// Get information about the `QuEST` environment (for debugging/introspection)
    pub fn get_env_info(&self) -> ffi::QuESTEnvInfo {
        unsafe { ffi::quest_get_env_info(self.env.ptr) }
    }

    fn check_qubit_index(&self, qubit: usize) -> Result<()> {
        if qubit >= self.num_qubits {
            Err(QuestError::InvalidQubit(qubit))
        } else {
            Ok(())
        }
    }

    /// Converts from PECOS qubit indexing (qubit 0 is MSB) to `QuEST` indexing (qubit 0 is LSB)
    fn convert_qubit_index(&self, pecos_qubit: usize) -> i32 {
        i32::try_from(self.num_qubits - 1 - pecos_qubit).expect("Qubit index out of range")
    }
}

impl<R> Clone for QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn clone(&self) -> Self {
        // Create a new independent instance with same parameters
        let env = QuestEnvWrapper::new().expect("Failed to create QuEST environment");
        let _qureg = QuregWrapper::new(
            &env,
            i32::try_from(self.num_qubits).expect("Too many qubits"),
            true,
        )
        .expect("Failed to create density matrix");

        // Clone the quantum state - quest_clone_qureg creates a new qureg with cloned state
        let cloned_qureg_ptr = unsafe { ffi::quest_clone_qureg(self.qureg.ptr) };

        // Replace the qureg pointer
        let qureg = QuregWrapper {
            ptr: cloned_qureg_ptr,
        };

        Self {
            num_qubits: self.num_qubits,
            env,
            qureg,
            rng: self.rng.clone(),
        }
    }
}

// Implement traits for QuestDensityMatrix (same as QuestStateVec for compatibility)
impl<R> QuantumSimulator for QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn reset(&mut self) -> &mut Self {
        unsafe {
            ffi::quest_init_zero_state(self.qureg.ptr);
        }
        self
    }
}

impl<R> CliffordGateable for QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_hadamard(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_s_gate(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let control = q0.index();
            let target = q1.index();
            self.check_qubit_index(control)
                .expect("Invalid control qubit");
            self.check_qubit_index(target)
                .expect("Invalid target qubit");
            let quest_control = self.convert_qubit_index(control);
            let quest_target = self.convert_qubit_index(target);
            unsafe {
                ffi::quest_apply_cnot(self.qureg.ptr, quest_control, quest_target);
            }
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        use rand::RngExt;

        let mut results = Vec::with_capacity(qubits.len());
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());

            // Get probability of measuring |0⟩ (deterministic calculation)
            let prob_0 = unsafe { ffi::quest_calc_prob_of_outcome(self.qureg.ptr, quest_qubit, 0) };

            // Sample outcome using our seeded Rust RNG
            let outcome = i32::from(self.rng.random::<f64>() >= prob_0);

            // Collapse state to the sampled outcome
            let actual_prob = unsafe {
                ffi::quest_apply_forced_measurement(self.qureg.ptr, quest_qubit, outcome)
            };

            results.push(MeasurementResult {
                outcome: outcome != 0,
                is_deterministic: (actual_prob - 1.0).abs() < f64::EPSILON,
            });
        }
        results
    }

    // Override with native QuEST implementations for better performance

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_pauli_x(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_pauli_y(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_pauli_z(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let control = q0.index();
            let target = q1.index();
            self.check_qubit_index(control)
                .expect("Invalid control qubit");
            self.check_qubit_index(target)
                .expect("Invalid target qubit");
            let quest_control = self.convert_qubit_index(control);
            let quest_target = self.convert_qubit_index(target);
            unsafe {
                ffi::quest_apply_cz(self.qureg.ptr, quest_control, quest_target);
            }
        }
        self
    }

    // SWAP gate - using trait default implementation
    // The native QuEST swap has GPU dependencies that cause linking issues
}

impl<R> ArbitraryRotationGateable for QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_rotation_x(self.qureg.ptr, quest_qubit, theta);
            }
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_rotation_z(self.qureg.ptr, quest_qubit, theta);
            }
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let half_angle = theta / 2.0;
        for &(q1, q2) in pairs {
            self.check_qubit_index(q1.index())
                .expect("Invalid qubit1 index");
            self.check_qubit_index(q2.index())
                .expect("Invalid qubit2 index");

            let half_angle_a = Angle64::from_radians(half_angle);
            let neg_half_angle_a = Angle64::from_radians(-half_angle);
            self.rz(half_angle_a, &[q1]).rz(half_angle_a, &[q2]);
            self.cz(&[(q1, q2)]);
            self.rz(neg_half_angle_a, &[q1]).rz(neg_half_angle_a, &[q2]);
        }
        self
    }

    // Override with native QuEST implementations

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_rotation_y(self.qureg.ptr, quest_qubit, theta);
            }
        }
        self
    }

    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_t_gate(self.qureg.ptr, quest_qubit);
            }
        }
        self
    }

    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.check_qubit_index(q.index())
                .expect("Invalid qubit index");
            let quest_qubit = self.convert_qubit_index(q.index());
            unsafe {
                ffi::quest_apply_phase_shift(self.qureg.ptr, quest_qubit, -FRAC_PI_4);
            }
        }
        self
    }
}

impl<R> RngManageable for QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug,
{
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

// Additional methods for QuestDensityMatrix
impl<R> QuestDensityMatrix<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Returns the complex density matrix element at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if the index is too large to be converted to `i64`.
    pub fn get_density_element(&self, index: usize) -> Complex64 {
        let complex_amp = unsafe {
            ffi::quest_get_complex_amp(
                self.qureg.ptr,
                i64::try_from(index).expect("Index too large"),
            )
        };
        Complex64::new(complex_amp.real, complex_amp.imag)
    }
}

unsafe impl<R> Send for QuestDensityMatrix<R> where R: Rng + SeedableRng + Debug + Send {}

unsafe impl<R> Sync for QuestDensityMatrix<R> where R: Rng + SeedableRng + Debug + Sync {}

#[cfg(test)]
mod tests;
