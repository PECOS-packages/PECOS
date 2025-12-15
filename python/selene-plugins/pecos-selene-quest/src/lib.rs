// Copyright 2025 The PECOS Developers
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

//! PECOS Quest simulator plugin for the Selene quantum emulator.
//!
//! This crate provides a Selene-compatible plugin wrapping the PECOS Quest simulator.
//! Quest is a high-performance quantum simulator that supports arbitrary rotation angles and
//! can utilize GPU acceleration when available.
//!
//! The plugin supports two simulation modes:
//! - State vector: Memory scales as 16 bytes * `2^n_qubits`
//! - Density matrix: Memory scales as 16 bytes * `4^n_qubits`
//!
//! # Attribution
//!
//! This plugin wraps `QuEST` (Quantum Exact Simulation Toolkit), developed by the QuEST-Kit team.
//!
//! - **Repository:** <https://github.com/quest-kit/QuEST>
//! - **License:** MIT License

use anyhow::{Result, anyhow, bail};
use num_complex::Complex64;
#[cfg(feature = "cuda")]
use pecos_quest::QuantumSimulator;
use pecos_quest::{ArbitraryRotationGateable, CliffordGateable, QuestDensityMatrix, QuestStateVec};
use rand_chacha::ChaCha8Rng;
use selene_core::export_simulator_plugin;
use selene_core::simulator::SimulatorInterface;
use selene_core::simulator::interface::SimulatorInterfaceFactory;
use selene_core::utils::MetricValue;
use std::io::Write;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use pecos_quest::cuda_loader;

/// Simulation mode for the Quest plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimulatorMode {
    /// State vector simulation (default)
    #[default]
    StateVector,
    /// Density matrix simulation
    DensityMatrix,
}

impl SimulatorMode {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "state_vector" => Ok(Self::StateVector),
            "density_matrix" => Ok(Self::DensityMatrix),
            _ => bail!("Unknown simulator mode: {s}. Expected 'state_vector' or 'density_matrix'"),
        }
    }
}

/// Wrapper enum to hold either state vector or density matrix simulator.
enum QuestSimulatorInner {
    StateVector(QuestStateVec<ChaCha8Rng>),
    DensityMatrix(QuestDensityMatrix<ChaCha8Rng>),
    #[cfg(feature = "cuda")]
    StateVectorGpu(CudaStateVec),
    #[cfg(feature = "cuda")]
    DensityMatrixGpu(CudaDensityMatrix),
}

/// CUDA-backed state vector wrapper
#[cfg(feature = "cuda")]
struct CudaStateVec {
    env_handle: *mut u8,
    qureg_handle: *mut u8,
    backend: &'static cuda_loader::CudaBackend,
}

#[cfg(feature = "cuda")]
impl CudaStateVec {
    fn new(num_qubits: usize) -> Result<Self> {
        let backend = cuda_loader::try_load_cuda().map_err(|e| {
            anyhow!(
                "Failed to load CUDA backend: {e}\n\n{}",
                cuda_loader::cuda_unavailable_error_message()
            )
        })?;

        let env_handle = unsafe { (backend.create_env)() };
        if env_handle.is_null() {
            bail!("Failed to create CUDA QuEST environment");
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let qureg_handle = unsafe { (backend.create_qureg)(env_handle, num_qubits as i32) };
        if qureg_handle.is_null() {
            unsafe { (backend.destroy_env)(env_handle) };
            bail!("Failed to create CUDA QuEST qureg");
        }

        unsafe { (backend.init_zero_state)(qureg_handle) };

        Ok(Self {
            env_handle,
            qureg_handle,
            backend,
        })
    }

    fn is_gpu_accelerated(&self) -> bool {
        let info = unsafe { (self.backend.get_env_info)(self.env_handle) };
        info.is_gpu_accelerated
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaStateVec {
    fn drop(&mut self) {
        unsafe {
            (self.backend.destroy_qureg)(self.qureg_handle);
            (self.backend.destroy_env)(self.env_handle);
        }
    }
}

/// CUDA-backed density matrix wrapper
#[cfg(feature = "cuda")]
struct CudaDensityMatrix {
    env_handle: *mut u8,
    qureg_handle: *mut u8,
    backend: &'static cuda_loader::CudaBackend,
}

#[cfg(feature = "cuda")]
impl CudaDensityMatrix {
    fn new(num_qubits: usize) -> Result<Self> {
        let backend = cuda_loader::try_load_cuda().map_err(|e| {
            anyhow!(
                "Failed to load CUDA backend: {e}\n\n{}",
                cuda_loader::cuda_unavailable_error_message()
            )
        })?;

        let env_handle = unsafe { (backend.create_env)() };
        if env_handle.is_null() {
            bail!("Failed to create CUDA QuEST environment");
        }

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let qureg_handle = unsafe { (backend.create_density_qureg)(env_handle, num_qubits as i32) };
        if qureg_handle.is_null() {
            unsafe { (backend.destroy_env)(env_handle) };
            bail!("Failed to create CUDA QuEST density qureg");
        }

        unsafe { (backend.init_zero_state)(qureg_handle) };

        Ok(Self {
            env_handle,
            qureg_handle,
            backend,
        })
    }

    fn is_gpu_accelerated(&self) -> bool {
        let info = unsafe { (self.backend.get_env_info)(self.env_handle) };
        info.is_gpu_accelerated
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaDensityMatrix {
    fn drop(&mut self) {
        unsafe {
            (self.backend.destroy_qureg)(self.qureg_handle);
            (self.backend.destroy_env)(self.env_handle);
        }
    }
}

impl QuestSimulatorInner {
    fn new_state_vector(n_qubits: usize, seed: u64) -> Self {
        Self::StateVector(QuestStateVec::with_seed(n_qubits, seed))
    }

    fn new_density_matrix(n_qubits: usize, seed: u64) -> Self {
        Self::DensityMatrix(QuestDensityMatrix::with_seed(n_qubits, seed))
    }

    #[cfg(feature = "cuda")]
    fn new_state_vector_gpu(n_qubits: usize) -> Result<Self> {
        Ok(Self::StateVectorGpu(CudaStateVec::new(n_qubits)?))
    }

    #[cfg(feature = "cuda")]
    fn new_density_matrix_gpu(n_qubits: usize) -> Result<Self> {
        Ok(Self::DensityMatrixGpu(CudaDensityMatrix::new(n_qubits)?))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn rz(&mut self, theta: f64, qubit: usize) {
        match self {
            Self::StateVector(sim) => {
                sim.rz(theta, qubit);
            }
            Self::DensityMatrix(sim) => {
                sim.rz(theta, qubit);
            }
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => unsafe {
                (sim.backend.apply_rotation_z)(sim.qureg_handle, qubit as i32, theta);
            },
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(sim) => unsafe {
                (sim.backend.apply_rotation_z)(sim.qureg_handle, qubit as i32, theta);
            },
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn rx(&mut self, theta: f64, qubit: usize) {
        match self {
            Self::StateVector(sim) => {
                sim.rx(theta, qubit);
            }
            Self::DensityMatrix(sim) => {
                sim.rx(theta, qubit);
            }
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => unsafe {
                (sim.backend.apply_rotation_x)(sim.qureg_handle, qubit as i32, theta);
            },
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(sim) => unsafe {
                (sim.backend.apply_rotation_x)(sim.qureg_handle, qubit as i32, theta);
            },
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn cx(&mut self, control: usize, target: usize) {
        match self {
            Self::StateVector(sim) => {
                sim.cx(control, target);
            }
            Self::DensityMatrix(sim) => {
                sim.cx(control, target);
            }
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => unsafe {
                (sim.backend.apply_cnot)(sim.qureg_handle, control as i32, target as i32);
            },
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(sim) => unsafe {
                (sim.backend.apply_cnot)(sim.qureg_handle, control as i32, target as i32);
            },
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn x(&mut self, qubit: usize) {
        match self {
            Self::StateVector(sim) => {
                sim.x(qubit);
            }
            Self::DensityMatrix(sim) => {
                sim.x(qubit);
            }
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => unsafe {
                (sim.backend.apply_pauli_x)(sim.qureg_handle, qubit as i32);
            },
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(sim) => unsafe {
                (sim.backend.apply_pauli_x)(sim.qureg_handle, qubit as i32);
            },
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn mz(&mut self, qubit: usize) -> pecos_quest::MeasurementResult {
        match self {
            Self::StateVector(sim) => sim.mz(qubit),
            Self::DensityMatrix(sim) => sim.mz(qubit),
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => {
                let outcome = unsafe { (sim.backend.measure)(sim.qureg_handle, qubit as i32) };
                pecos_quest::MeasurementResult {
                    outcome: outcome != 0,
                    is_deterministic: false, // CUDA backend doesn't report this
                }
            }
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(sim) => {
                let outcome = unsafe { (sim.backend.measure)(sim.qureg_handle, qubit as i32) };
                pecos_quest::MeasurementResult {
                    outcome: outcome != 0,
                    is_deterministic: false,
                }
            }
        }
    }

    // State indices are bounded by 2^n_qubits which is always small enough
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn probability(&self, state_index: usize) -> f64 {
        match self {
            Self::StateVector(sim) => sim.probability(state_index),
            Self::DensityMatrix(sim) => sim.probability(state_index),
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => unsafe {
                (sim.backend.get_prob_amp)(sim.qureg_handle, state_index as i64)
            },
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(sim) => unsafe {
                (sim.backend.get_prob_amp)(sim.qureg_handle, state_index as i64)
            },
        }
    }

    // State indices are bounded by 2^n_qubits which is always small enough
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn get_amplitude(&self, state_index: usize) -> Complex64 {
        match self {
            Self::StateVector(sim) => sim.get_amplitude(state_index),
            Self::DensityMatrix(_sim) => {
                // For density matrix, we can't directly get amplitudes
                // This is a limitation - dump_state will need special handling
                Complex64::new(0.0, 0.0)
            }
            #[cfg(feature = "cuda")]
            Self::StateVectorGpu(sim) => {
                let re =
                    unsafe { (sim.backend.get_real_amp)(sim.qureg_handle, state_index as i64) };
                let im =
                    unsafe { (sim.backend.get_imag_amp)(sim.qureg_handle, state_index as i64) };
                Complex64::new(re, im)
            }
            #[cfg(feature = "cuda")]
            Self::DensityMatrixGpu(_sim) => {
                // For density matrix, we can't directly get amplitudes
                Complex64::new(0.0, 0.0)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn is_gpu_accelerated(&self) -> bool {
        match self {
            Self::StateVector(sim) => sim.get_env_info().is_gpu_accelerated,
            Self::DensityMatrix(sim) => sim.get_env_info().is_gpu_accelerated,
            Self::StateVectorGpu(sim) => sim.is_gpu_accelerated(),
            Self::DensityMatrixGpu(sim) => sim.is_gpu_accelerated(),
        }
    }

    /// Reinitialize the state to |0...0>
    #[cfg(feature = "cuda")]
    fn reinit_zero_state(&mut self) {
        match self {
            Self::StateVector(sim) => {
                sim.reset();
            }
            Self::DensityMatrix(sim) => {
                sim.reset();
            }
            Self::StateVectorGpu(sim) => unsafe {
                (sim.backend.init_zero_state)(sim.qureg_handle);
            },
            Self::DensityMatrixGpu(sim) => unsafe {
                (sim.backend.init_zero_state)(sim.qureg_handle);
            },
        }
    }
}

/// The PECOS Quest simulator wrapped for Selene compatibility.
pub struct QuestSimulator {
    /// The underlying PECOS Quest simulator
    simulator: QuestSimulatorInner,
    /// Number of qubits in the system
    n_qubits: u64,
    /// Simulation mode
    mode: SimulatorMode,
    /// Whether GPU is requested
    #[allow(dead_code)]
    use_gpu: bool,
    /// Cumulative probability of postselection outcomes
    cumulative_postselect_probability: f64,
}

impl QuestSimulator {
    /// Convert a `u64` to `usize` for use with the simulator.
    ///
    /// # Safety
    ///
    /// This is safe because `check_memory()` validates that `n_qubits <= 60` (state vector)
    /// or `n_qubits <= 30` (density matrix) before any simulator is created, and all qubit
    /// indices are bounds-checked against `n_qubits` before this function is called.
    /// Thus, the value will always fit in a `usize` on any platform.
    #[allow(clippy::cast_possible_truncation)]
    #[inline]
    const fn to_usize(value: u64) -> usize {
        value as usize
    }

    /// Convert Selene qubit index to PECOS qubit index.
    ///
    /// PECOS Quest internally converts qubit indices from PECOS convention (MSB-first,
    /// qubit 0 = most significant) to Quest convention (LSB-first, qubit 0 = least
    /// significant).
    ///
    /// Selene uses LSB-first convention (like Quest), so Selene qubit 0 should
    /// ultimately map to Quest qubit 0. Since PECOS Quest converts PECOS index i
    /// to Quest index (n-1-i), we need:
    ///   Selene qubit i -> PECOS qubit (n-1-i) -> Quest qubit (n-1-(n-1-i)) = i
    ///
    /// This double conversion ensures Selene qubit indices are preserved in Quest.
    #[inline]
    fn convert_qubit(&self, selene_qubit: u64) -> usize {
        Self::to_usize(self.n_qubits - 1 - selene_qubit)
    }

    /// Create a new CPU simulator with the given seed.
    fn new_simulator_cpu(mode: SimulatorMode, n_qubits: usize, seed: u64) -> QuestSimulatorInner {
        match mode {
            SimulatorMode::StateVector => QuestSimulatorInner::new_state_vector(n_qubits, seed),
            SimulatorMode::DensityMatrix => QuestSimulatorInner::new_density_matrix(n_qubits, seed),
        }
    }

    /// Create a new GPU simulator.
    #[cfg(feature = "cuda")]
    fn new_simulator_gpu(mode: SimulatorMode, n_qubits: usize) -> Result<QuestSimulatorInner> {
        match mode {
            SimulatorMode::StateVector => QuestSimulatorInner::new_state_vector_gpu(n_qubits),
            SimulatorMode::DensityMatrix => QuestSimulatorInner::new_density_matrix_gpu(n_qubits),
        }
    }
}

impl SimulatorInterface for QuestSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, seed: u64) -> Result<()> {
        // For CPU mode: create a fresh simulator with the given seed for deterministic behavior
        // For GPU mode: reinitialize the state (GPU backend doesn't support seeded random)
        #[cfg(feature = "cuda")]
        {
            if self.use_gpu {
                // GPU mode: just reinitialize to zero state
                // Note: GPU measurements are not seeded, so results may differ from CPU
                self.simulator.reinit_zero_state();
                self.cumulative_postselect_probability = 1.0;
                return Ok(());
            }
        }

        // CPU mode: recreate simulator with seed
        self.simulator = Self::new_simulator_cpu(self.mode, Self::to_usize(self.n_qubits), seed);
        self.cumulative_postselect_probability = 1.0;
        Ok(())
    }

    fn shot_end(&mut self) -> Result<()> {
        Ok(())
    }

    fn rxy(&mut self, qubit: u64, theta: f64, phi: f64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RXY(qubit={qubit}, theta={theta}, phi={phi}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = self.convert_qubit(qubit);

        // RXY(theta, phi) = Rz(phi) * Rx(theta) * Rz(-phi)
        // Gates are applied left-to-right in code but the matrix multiplication
        // is right-to-left, so we apply Rz(-phi) first
        self.simulator.rz(-phi, q);
        self.simulator.rx(theta, q);
        self.simulator.rz(phi, q);

        Ok(())
    }

    fn rz(&mut self, qubit: u64, theta: f64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "RZ(qubit={qubit}, theta={theta}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        self.simulator.rz(theta, self.convert_qubit(qubit));
        Ok(())
    }

    fn rzz(&mut self, qubit1: u64, qubit2: u64, theta: f64) -> Result<()> {
        if qubit1 >= self.n_qubits || qubit2 >= self.n_qubits {
            return Err(anyhow!(
                "RZZ(qubit1={qubit1}, qubit2={qubit2}, theta={theta}) is out of bounds. \
                 qubits must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q1 = self.convert_qubit(qubit1);
        let q2 = self.convert_qubit(qubit2);

        // Implement RZZ using CX (CNOT) since PECOS Quest's rzz has incorrect behavior.
        // RZZ(θ) = CNOT(q1, q2) * Rz(θ)_q2 * CNOT(q1, q2)
        // This creates the correct diagonal matrix:
        //   |00⟩ → exp(-iθ/2)|00⟩
        //   |01⟩ → exp(+iθ/2)|01⟩
        //   |10⟩ → exp(+iθ/2)|10⟩
        //   |11⟩ → exp(-iθ/2)|11⟩
        self.simulator.cx(q1, q2);
        self.simulator.rz(theta, q2);
        self.simulator.cx(q1, q2);

        Ok(())
    }

    fn measure(&mut self, qubit: u64) -> Result<bool> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Measure(qubit={qubit}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let converted = self.convert_qubit(qubit);
        let result = self.simulator.mz(converted);
        Ok(result.outcome)
    }

    fn postselect(&mut self, qubit: u64, target_value: bool) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target_value={target_value}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = self.convert_qubit(qubit);

        // Calculate the probability of measuring the target value
        let mut prob_target = 0.0;
        let n_states = 1usize << self.n_qubits;
        for i in 0..n_states {
            let bit = (i >> q) & 1;
            if (bit == 1) == target_value {
                prob_target += self.simulator.probability(i);
            }
        }

        self.cumulative_postselect_probability *= prob_target;

        if prob_target < 1e-10 {
            return Err(anyhow!(
                "Postselection of {target_value} on qubit {qubit} is too unlikely to postselect. \
                 The probability of this outcome is {prob_target:.2e}."
            ));
        }

        // Measure and check if we got the expected outcome
        let result = self.simulator.mz(q);

        if result.outcome != target_value {
            return Err(anyhow!(
                "Postselect(qubit={qubit}, target_value={target_value}) failed. \
                 The measurement outcome was {} but postselection to {target_value} was requested.",
                result.outcome
            ));
        }

        Ok(())
    }

    fn reset(&mut self, qubit: u64) -> Result<()> {
        if qubit >= self.n_qubits {
            return Err(anyhow!(
                "Reset(qubit={qubit}) is out of bounds. \
                 qubit must be less than the number of qubits ({}).",
                self.n_qubits
            ));
        }

        let q = self.convert_qubit(qubit);

        // Measure the qubit and flip if needed to get |0>
        let result = self.simulator.mz(q);
        if result.outcome {
            // If we measured 1, apply X to flip to 0
            self.simulator.x(q);
        }

        Ok(())
    }

    fn get_metric(&mut self, nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        match nth_metric {
            0 => Ok(Some((
                "cumulative_postselect_probability".to_string(),
                MetricValue::F64(self.cumulative_postselect_probability),
            ))),
            _ => Ok(None),
        }
    }

    fn dump_state(&mut self, file: &std::path::Path, qubits: &[u64]) -> Result<()> {
        let handle = std::fs::File::create(file)?;
        let mut writer = std::io::BufWriter::new(handle);

        // Write header identifier (same format as Selene's quest plugin)
        writer.write_all(b"selene-quest")?;

        // Write number of qubits and qubit list
        writer.write_all(self.n_qubits.to_le_bytes().as_slice())?;
        writer.write_all((qubits.len() as u64).to_le_bytes().as_slice())?;
        for &q in qubits {
            writer.write_all(q.to_le_bytes().as_slice())?;
        }

        // Write state vector amplitudes
        let n_states = 1usize << self.n_qubits;
        for i in 0..n_states {
            let amp = self.simulator.get_amplitude(i);
            writer.write_all(amp.re.to_le_bytes().as_slice())?;
            writer.write_all(amp.im.to_le_bytes().as_slice())?;
        }

        Ok(())
    }
}

/// Factory for creating `QuestSimulator` instances.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuestSimulatorFactory;

/// Parse command-line style arguments.
fn parse_args(args: &[impl AsRef<str>]) -> Result<(SimulatorMode, bool)> {
    let mut mode = SimulatorMode::StateVector;
    let mut use_gpu = false;

    for arg in args {
        let arg = arg.as_ref();
        if arg.is_empty() {
            continue;
        }

        if let Some(value) = arg.strip_prefix("--mode=") {
            mode = SimulatorMode::from_str(value)?;
        } else if arg == "--use-gpu" {
            use_gpu = true;
        } else if arg.starts_with("--") {
            bail!("Unknown argument: {arg}");
        }
        // Ignore positional args (like the empty string from Selene)
    }

    Ok((mode, use_gpu))
}

/// Check if there is enough memory to allocate a simulator of the given size.
fn check_memory(n_qubits: u64, mode: SimulatorMode) -> Result<()> {
    if n_qubits == 0 {
        bail!("Number of qubits must be greater than 0");
    }

    let max_qubits = match mode {
        SimulatorMode::StateVector => 60,
        SimulatorMode::DensityMatrix => 30, // 4^30 states = 2^60 elements
    };

    if n_qubits > max_qubits {
        bail!(
            "It is impossible to describe more than {max_qubits} qubits in {} mode \
             on a computer with a 64-bit address space.",
            match mode {
                SimulatorMode::StateVector => "state vector",
                SimulatorMode::DensityMatrix => "density matrix",
            }
        );
    }

    // Each amplitude is a Complex64 = 16 bytes (2 * f64)
    // State vector: 2^n states, Density matrix: 4^n = 2^(2n) states
    let num_elements = match mode {
        SimulatorMode::StateVector => 1_u64 << n_qubits,
        SimulatorMode::DensityMatrix => 1_u64 << (2 * n_qubits),
    };
    let bytes_required = 16_u64.checked_mul(num_elements);

    match bytes_required {
        Some(bytes) => {
            // Just log a warning for large allocations, but let the OS handle
            // actual memory allocation
            if bytes > 1024 * 1024 * 1024 {
                // > 1GB
                eprintln!(
                    "Warning: Allocating {} for {n_qubits} qubits requires \
                     approximately {} bytes",
                    match mode {
                        SimulatorMode::StateVector => "state vector",
                        SimulatorMode::DensityMatrix => "density matrix",
                    },
                    bytes
                );
            }
            Ok(())
        }
        None => {
            bail!("Memory requirement overflow for {n_qubits} qubits");
        }
    }
}

impl SimulatorInterfaceFactory for QuestSimulatorFactory {
    type Interface = QuestSimulator;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        let (mode, use_gpu) = parse_args(args)?;

        check_memory(n_qubits, mode)?;

        let n_qubits_usize = QuestSimulator::to_usize(n_qubits);

        // Create simulator based on GPU flag
        #[cfg(feature = "cuda")]
        let simulator = if use_gpu {
            // Try to create GPU simulator
            QuestSimulator::new_simulator_gpu(mode, n_qubits_usize)?
        } else {
            QuestSimulator::new_simulator_cpu(mode, n_qubits_usize, 0)
        };

        #[cfg(not(feature = "cuda"))]
        let simulator = {
            if use_gpu {
                bail!(
                    "GPU acceleration was requested but this library was not compiled with CUDA support.\n\
                     Please install a CUDA-enabled build of pecos-selene-quest."
                );
            }
            QuestSimulator::new_simulator_cpu(mode, n_qubits_usize, 0)
        };

        // Verify GPU is actually being used if requested
        #[cfg(feature = "cuda")]
        if use_gpu && !simulator.is_gpu_accelerated() {
            bail!(
                "GPU acceleration was requested but the simulator is not using GPU.\n\
                 This could mean:\n\
                 - CUDA is not installed or not properly configured\n\
                 - No compatible GPU was found\n\
                 Please check your CUDA installation and GPU availability."
            );
        }

        Ok(Box::new(QuestSimulator {
            simulator,
            n_qubits,
            mode,
            use_gpu,
            cumulative_postselect_probability: 1.0,
        }))
    }
}

// Export the plugin using Selene's macro
export_simulator_plugin!(crate::QuestSimulatorFactory);

#[cfg(test)]
mod tests {
    use super::{QuestSimulatorFactory, SimulatorMode, parse_args};
    use selene_core::simulator::conformance_testing::run_basic_tests;
    use std::sync::Arc;

    #[test]
    fn test_parse_args_default() {
        let args: Vec<&str> = vec![];
        let (mode, use_gpu) = parse_args(&args).unwrap();
        assert_eq!(mode, SimulatorMode::StateVector);
        assert!(!use_gpu);
    }

    #[test]
    fn test_parse_args_state_vector() {
        let args = vec!["--mode=state_vector"];
        let (mode, use_gpu) = parse_args(&args).unwrap();
        assert_eq!(mode, SimulatorMode::StateVector);
        assert!(!use_gpu);
    }

    #[test]
    fn test_parse_args_density_matrix() {
        let args = vec!["--mode=density_matrix"];
        let (mode, use_gpu) = parse_args(&args).unwrap();
        assert_eq!(mode, SimulatorMode::DensityMatrix);
        assert!(!use_gpu);
    }

    #[test]
    fn test_parse_args_with_gpu() {
        let args = vec!["--mode=state_vector", "--use-gpu"];
        let (mode, use_gpu) = parse_args(&args).unwrap();
        assert_eq!(mode, SimulatorMode::StateVector);
        assert!(use_gpu);
    }

    #[test]
    fn test_parse_args_empty_strings() {
        let args = vec!["", "--mode=density_matrix", ""];
        let (mode, use_gpu) = parse_args(&args).unwrap();
        assert_eq!(mode, SimulatorMode::DensityMatrix);
        assert!(!use_gpu);
    }

    /// Test that requesting GPU on a system without GPU fails with a helpful error.
    #[test]
    fn test_gpu_requested_but_unavailable() {
        use selene_core::simulator::interface::SimulatorInterfaceFactory;

        let factory = Arc::new(QuestSimulatorFactory);
        let result = factory
            .clone()
            .init(2, &["--mode=state_vector", "--use-gpu"]);

        // On a system without GPU, this should fail
        // (If running on a system with GPU, this test would pass differently)
        match result {
            Ok(_) => {
                // GPU is available - that's fine, test passes
            }
            Err(err) => {
                let err_msg = err.to_string();
                // Accept either "not available" (runtime) or "not compiled with CUDA" (compile-time)
                assert!(
                    err_msg.contains("GPU acceleration was requested"),
                    "Expected GPU error, got: {err_msg}"
                );
            }
        }
    }

    /// Test that a Bell state through the Selene wrapper produces correlated measurements.
    /// This validates the RZZ implementation fix (using CNOT instead of PECOS Quest's buggy rzz).
    #[test]
    fn test_bell_state_correlation() {
        use selene_core::simulator::SimulatorInterface;
        use selene_core::simulator::interface::SimulatorInterfaceFactory;

        const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;
        const PI: f64 = std::f64::consts::PI;

        let factory = Arc::new(QuestSimulatorFactory);
        let mut outcomes = [0u32; 4];

        for seed in 0..100u64 {
            let mut sim = factory.clone().init(2, &["--mode=state_vector"]).unwrap();
            sim.shot_start(0, seed).unwrap();

            // Selene's H decomposition on qubit 0
            sim.rxy(0, HALF_PI, -HALF_PI).unwrap();
            sim.rz(0, PI).unwrap();

            // Selene's CNOT decomposition (control=0, target=1)
            sim.rxy(1, HALF_PI, HALF_PI).unwrap();
            sim.rzz(0, 1, HALF_PI).unwrap();
            sim.rz(0, HALF_PI).unwrap();
            sim.rxy(1, HALF_PI, 0.0).unwrap();
            sim.rz(1, -HALF_PI).unwrap();

            // Measure both qubits
            let m0 = sim.measure(0).unwrap();
            let m1 = sim.measure(1).unwrap();

            let idx = usize::from(m0) | (if m1 { 2 } else { 0 });
            outcomes[idx] += 1;
        }

        // Bell state should only produce |00⟩ and |11⟩, never |01⟩ or |10⟩
        assert!(
            outcomes[0b01] == 0 && outcomes[0b10] == 0,
            "Bell state should only have |00⟩ and |11⟩, got {outcomes:?}"
        );
    }

    /// Test Bell state with density matrix mode.
    #[test]
    fn test_bell_state_density_matrix() {
        use selene_core::simulator::SimulatorInterface;
        use selene_core::simulator::interface::SimulatorInterfaceFactory;

        const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;
        const PI: f64 = std::f64::consts::PI;

        let factory = Arc::new(QuestSimulatorFactory);
        let mut outcomes = [0u32; 4];

        for seed in 0..100u64 {
            let mut sim = factory.clone().init(2, &["--mode=density_matrix"]).unwrap();
            sim.shot_start(0, seed).unwrap();

            // Selene's H decomposition on qubit 0
            sim.rxy(0, HALF_PI, -HALF_PI).unwrap();
            sim.rz(0, PI).unwrap();

            // Selene's CNOT decomposition (control=0, target=1)
            sim.rxy(1, HALF_PI, HALF_PI).unwrap();
            sim.rzz(0, 1, HALF_PI).unwrap();
            sim.rz(0, HALF_PI).unwrap();
            sim.rxy(1, HALF_PI, 0.0).unwrap();
            sim.rz(1, -HALF_PI).unwrap();

            // Measure both qubits
            let m0 = sim.measure(0).unwrap();
            let m1 = sim.measure(1).unwrap();

            let idx = usize::from(m0) | (if m1 { 2 } else { 0 });
            outcomes[idx] += 1;
        }

        // Bell state should only produce |00⟩ and |11⟩, never |01⟩ or |10⟩
        assert!(
            outcomes[0b01] == 0 && outcomes[0b10] == 0,
            "Bell state (density matrix) should only have |00⟩ and |11⟩, got {outcomes:?}"
        );
    }

    /// Run Selene's basic conformance tests for the Quest plugin (state vector mode).
    #[test]
    fn basic_conformance_test_state_vector() {
        let interface = Arc::new(QuestSimulatorFactory);
        let args: Vec<String> = vec!["--mode=state_vector".to_string()];
        run_basic_tests(interface, args);
    }

    /// Run Selene's basic conformance tests for the Quest plugin (density matrix mode).
    #[test]
    fn basic_conformance_test_density_matrix() {
        let interface = Arc::new(QuestSimulatorFactory);
        let args: Vec<String> = vec!["--mode=density_matrix".to_string()];
        run_basic_tests(interface, args);
    }

    /// Test GPU acceleration if available.
    /// This test checks if GPU is available and runs a basic test if so.
    /// If GPU is not available, it verifies the error message is helpful.
    #[test]
    fn test_gpu_acceleration() {
        use selene_core::simulator::SimulatorInterface;
        use selene_core::simulator::interface::SimulatorInterfaceFactory;

        let factory = Arc::new(QuestSimulatorFactory);
        let result = factory
            .clone()
            .init(2, &["--mode=state_vector", "--use-gpu"]);

        match result {
            Ok(mut sim) => {
                // GPU is available - run a basic test
                const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;
                const PI: f64 = std::f64::consts::PI;

                sim.shot_start(0, 42).unwrap();

                // Create Bell state
                sim.rxy(0, HALF_PI, -HALF_PI).unwrap();
                sim.rz(0, PI).unwrap();
                sim.rxy(1, HALF_PI, HALF_PI).unwrap();
                sim.rzz(0, 1, HALF_PI).unwrap();
                sim.rz(0, HALF_PI).unwrap();
                sim.rxy(1, HALF_PI, 0.0).unwrap();
                sim.rz(1, -HALF_PI).unwrap();

                let m0 = sim.measure(0).unwrap();
                let m1 = sim.measure(1).unwrap();

                // Bell state: both measurements should be the same
                assert_eq!(m0, m1, "GPU Bell state measurements should be correlated");
            }
            Err(err) => {
                // GPU not available - verify error message is helpful
                let err_msg = err.to_string();
                // Accept either "not available" (runtime) or "not compiled with CUDA" (compile-time)
                assert!(
                    err_msg.contains("GPU acceleration was requested"),
                    "Expected helpful GPU error message, got: {err_msg}"
                );
            }
        }
    }

    /// Test GPU with density matrix mode if available.
    #[test]
    fn test_gpu_density_matrix() {
        use selene_core::simulator::SimulatorInterface;
        use selene_core::simulator::interface::SimulatorInterfaceFactory;

        let factory = Arc::new(QuestSimulatorFactory);
        let result = factory
            .clone()
            .init(2, &["--mode=density_matrix", "--use-gpu"]);

        match result {
            Ok(mut sim) => {
                // GPU is available - run a basic test
                sim.shot_start(0, 42).unwrap();
                let m = sim.measure(0).unwrap();
                // Should measure 0 for |0> state
                assert!(!m, "Initial state should measure 0");
            }
            Err(err) => {
                // GPU not available - verify error message
                let err_msg = err.to_string();
                // Accept either "not available" (runtime) or "not compiled with CUDA" (compile-time)
                assert!(
                    err_msg.contains("GPU acceleration was requested"),
                    "Expected helpful GPU error message, got: {err_msg}"
                );
            }
        }
    }
}
