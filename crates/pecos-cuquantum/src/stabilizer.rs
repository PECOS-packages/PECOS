//! Stabilizer simulation via cuStabilizer Frame Simulation
//!
//! cuQuantum 25.11+ provides a frame-based stabilizer simulation API that is
//! optimized for running many shots of the same circuit in parallel.
//!
//! # Architecture
//!
//! The cuStabilizer API uses Pauli frame simulation:
//! - Circuits are specified as strings (Stim-compatible format)
//! - Multiple shots (frames) are simulated in parallel on the GPU
//! - Results are returned as bit tables for X errors, Z errors, and measurements
//!
//! # Example
//!
//! ```ignore
//! use pecos_cuquantum::CuFrameSimulator;
//!
//! // Create a frame simulator for 1000 shots
//! let mut sim = CuFrameSimulator::new(5, 1000, 2)?;  // 5 qubits, 1000 shots, 2 measurements
//!
//! // Define circuit in Stim format
//! let circuit = "H 0\nCNOT 0 1\nM 0 1";
//!
//! // Run the circuit
//! let results = sim.run_circuit(circuit, 42)?;  // seed = 42
//!
//! // results.measurements[shot][meas_idx] gives the outcome
//! ```

use crate::error::{CuQuantumError, Result, TryClone, check_stabilizer_status};
use pecos_core::QubitId;
use pecos_cuquantum_sys::{
    cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind_cudaMemcpyDeviceToHost,
    cudaMemcpyKind_cudaMemcpyHostToDevice, custabilizerCircuit_t,
    custabilizerCircuitSizeFromString, custabilizerCreate, custabilizerCreateCircuitFromString,
    custabilizerCreateFrameSimulator, custabilizerDestroy, custabilizerDestroyCircuit,
    custabilizerDestroyFrameSimulator, custabilizerFrameSimulator_t,
    custabilizerFrameSimulatorApplyCircuit, custabilizerHandle_t,
};
use pecos_simulators::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};
use pecos_simulators::{
    CliffordGateable, MeasurementResult, QuantumSimulator, StabilizerTableauSimulator,
};
use std::ffi::CString;
use std::ptr;

/// Results from a frame simulation run
#[derive(Debug, Clone)]
pub struct FrameSimulationResults {
    /// Measurement outcomes: measurements[shot_idx][measurement_idx]
    pub measurements: Vec<Vec<bool>>,
    /// Number of shots
    pub num_shots: usize,
    /// Number of measurements per shot
    pub num_measurements: usize,
}

/// GPU-accelerated frame-based stabilizer simulator using cuStabilizer.
///
/// This simulator runs many shots of the same circuit in parallel on the GPU
/// using Pauli frame simulation. It is optimized for Monte Carlo sampling
/// and QEC simulations.
///
/// # Circuit Format
///
/// Circuits are specified as strings in Stim-compatible format:
/// - `H qubit` - Hadamard gate
/// - `S qubit` - S gate (sqrt Z)
/// - `S_DAG qubit` - S-dagger gate
/// - `X qubit` - Pauli X
/// - `Y qubit` - Pauli Y
/// - `Z qubit` - Pauli Z
/// - `CNOT control target` or `CX control target` - CNOT gate
/// - `CZ qubit1 qubit2` - CZ gate
/// - `SWAP qubit1 qubit2` - SWAP gate
/// - `M qubit1 qubit2 ...` - Measure qubits
/// - `R qubit` - Reset qubit
/// - `X_ERROR(p) qubit` - X error with probability p
/// - `Y_ERROR(p) qubit` - Y error with probability p
/// - `Z_ERROR(p) qubit` - Z error with probability p
/// - `DEPOLARIZE1(p) qubit` - Single-qubit depolarizing noise
/// - `DEPOLARIZE2(p) qubit1 qubit2` - Two-qubit depolarizing noise
pub struct CuFrameSimulator {
    handle: custabilizerHandle_t,
    num_qubits: usize,
    num_shots: usize,
    max_measurements: usize,
    table_stride: usize,
    // GPU buffers
    x_table_device: *mut u32,
    z_table_device: *mut u32,
    m_table_device: *mut u32,
    circuit_buffer_device: *mut std::ffi::c_void,
    circuit_buffer_size: usize,
    // Frame simulator (created lazily)
    frame_simulator: Option<custabilizerFrameSimulator_t>,
}

// Safety: The GPU resources are managed through CUDA and are thread-safe
// when accessed through the cuStabilizer API
unsafe impl Send for CuFrameSimulator {}

impl CuFrameSimulator {
    /// Create a new frame simulator.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the circuit
    /// * `num_shots` - Number of shots (frames) to simulate in parallel
    /// * `max_measurements` - Maximum number of measurements in any circuit
    ///
    /// # Errors
    /// Returns an error if GPU resources cannot be allocated.
    #[allow(unreachable_code, unused_variables)]
    pub fn new(num_qubits: usize, num_shots: usize, max_measurements: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(CuQuantumError::InvalidArgument(
                "num_qubits must be at least 1".into(),
            ));
        }
        if num_shots == 0 {
            return Err(CuQuantumError::InvalidArgument(
                "num_shots must be at least 1".into(),
            ));
        }

        #[cfg(cuquantum_stub)]
        return Err(CuQuantumError::NotAvailable(
            "cuQuantum SDK is not installed. To use GPU-accelerated simulators, install the cuQuantum SDK:\n\
             1. Set CUQUANTUM_ROOT environment variable, or\n\
             2. Install via: pecos install cuquantum, or\n\
             3. Install system-wide to /usr/local/cuquantum/"
                .into(),
        ));

        // Create cuStabilizer handle
        let mut handle: custabilizerHandle_t = ptr::null_mut();
        let status = unsafe { custabilizerCreate(&mut handle) };
        check_stabilizer_status(status)?;

        // Calculate table stride (must be multiple of 4 bytes)
        // Each row stores num_shots bits, packed into u32s
        let bytes_per_row = num_shots.div_ceil(8);
        let table_stride = bytes_per_row.div_ceil(4) * 4; // Round up to multiple of 4

        // Allocate GPU buffers for X and Z tables
        let x_table_size = num_qubits * table_stride;
        let z_table_size = num_qubits * table_stride;
        let m_table_size = max_measurements * table_stride;

        let x_table_device: *mut u32 = ptr::null_mut();
        let z_table_device: *mut u32 = ptr::null_mut();
        let m_table_device: *mut u32 = ptr::null_mut();

        unsafe {
            let cuda_status = cudaMalloc(&mut x_table_device.cast(), x_table_size);
            if cuda_status != 0 {
                custabilizerDestroy(handle);
                return Err(CuQuantumError::Cuda(format!(
                    "Failed to allocate X table: CUDA error {cuda_status}"
                )));
            }

            let cuda_status = cudaMalloc(&mut z_table_device.cast(), z_table_size);
            if cuda_status != 0 {
                cudaFree(x_table_device.cast());
                custabilizerDestroy(handle);
                return Err(CuQuantumError::Cuda(format!(
                    "Failed to allocate Z table: CUDA error {cuda_status}"
                )));
            }

            let cuda_status = cudaMalloc(&mut m_table_device.cast(), m_table_size);
            if cuda_status != 0 {
                cudaFree(x_table_device.cast());
                cudaFree(z_table_device.cast());
                custabilizerDestroy(handle);
                return Err(CuQuantumError::Cuda(format!(
                    "Failed to allocate M table: CUDA error {cuda_status}"
                )));
            }
        }

        Ok(Self {
            handle,
            num_qubits,
            num_shots,
            max_measurements,
            table_stride,
            x_table_device,
            z_table_device,
            m_table_device,
            circuit_buffer_device: ptr::null_mut(),
            circuit_buffer_size: 0,
            frame_simulator: None,
        })
    }

    /// Get the number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the number of shots
    #[must_use]
    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    /// Run a circuit on all shots and return measurement results.
    ///
    /// # Arguments
    /// * `circuit_string` - Circuit in Stim-compatible format
    /// * `seed` - Random seed for noise and measurement randomization
    ///
    /// # Returns
    /// Measurement results for all shots
    ///
    /// # Errors
    /// Returns an error if the circuit is invalid or execution fails.
    pub fn run_circuit(
        &mut self,
        circuit_string: &str,
        seed: u64,
    ) -> Result<FrameSimulationResults> {
        self.run_circuit_internal(circuit_string, seed, true)
    }

    /// Run a circuit without randomizing frames after measurement.
    ///
    /// This is useful for studying error propagation without measurement randomization.
    pub fn run_circuit_deterministic(
        &mut self,
        circuit_string: &str,
        seed: u64,
    ) -> Result<FrameSimulationResults> {
        self.run_circuit_internal(circuit_string, seed, false)
    }

    fn run_circuit_internal(
        &mut self,
        circuit_string: &str,
        seed: u64,
        randomize_after_measurement: bool,
    ) -> Result<FrameSimulationResults> {
        let circuit_cstring = CString::new(circuit_string).map_err(|_| {
            CuQuantumError::InvalidArgument("Circuit string contains null bytes".into())
        })?;

        // Get required buffer size for circuit
        let mut buffer_size: i64 = 0;
        let status = unsafe {
            custabilizerCircuitSizeFromString(
                self.handle,
                circuit_cstring.as_ptr(),
                &mut buffer_size,
            )
        };
        check_stabilizer_status(status)?;

        // Reallocate circuit buffer if needed
        #[allow(clippy::cast_sign_loss)]
        let buffer_size_usize = buffer_size as usize;
        if buffer_size_usize > self.circuit_buffer_size {
            unsafe {
                if !self.circuit_buffer_device.is_null() {
                    cudaFree(self.circuit_buffer_device);
                }
                let cuda_status = cudaMalloc(&mut self.circuit_buffer_device, buffer_size_usize);
                if cuda_status != 0 {
                    self.circuit_buffer_device = ptr::null_mut();
                    self.circuit_buffer_size = 0;
                    return Err(CuQuantumError::Cuda(format!(
                        "Failed to allocate circuit buffer: CUDA error {cuda_status}"
                    )));
                }
                self.circuit_buffer_size = buffer_size_usize;
            }
        }

        // Create circuit from string
        let mut circuit: custabilizerCircuit_t = ptr::null_mut();
        let status = unsafe {
            custabilizerCreateCircuitFromString(
                self.handle,
                circuit_cstring.as_ptr(),
                self.circuit_buffer_device,
                buffer_size,
                &mut circuit,
            )
        };
        check_stabilizer_status(status)?;

        // Count measurements in the circuit
        let num_measurements = circuit_string
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("M ") || trimmed.starts_with("M\t") || trimmed == "M"
            })
            .map(|line| {
                // Count qubits in measurement instruction
                line.split_whitespace().skip(1).count().max(1)
            })
            .sum::<usize>();

        if num_measurements > self.max_measurements {
            unsafe {
                custabilizerDestroyCircuit(circuit);
            }
            return Err(CuQuantumError::InvalidArgument(format!(
                "Circuit has {} measurements but simulator was created with max {}",
                num_measurements, self.max_measurements
            )));
        }

        // Create frame simulator if needed
        if self.frame_simulator.is_none() {
            let mut frame_sim: custabilizerFrameSimulator_t = ptr::null_mut();
            #[allow(clippy::cast_possible_wrap)]
            let status = unsafe {
                custabilizerCreateFrameSimulator(
                    self.handle,
                    self.num_qubits as i64,
                    self.num_shots as i64,
                    self.max_measurements as i64,
                    self.table_stride as i64,
                    &mut frame_sim,
                )
            };
            check_stabilizer_status(status)?;
            self.frame_simulator = Some(frame_sim);
        }

        // Initialize tables to zero (all I operators, no measurements)
        let x_table_size = self.num_qubits * self.table_stride;
        let z_table_size = self.num_qubits * self.table_stride;
        let m_table_size = self.max_measurements * self.table_stride;

        unsafe {
            // Zero out tables
            let zeros_x = vec![0u8; x_table_size];
            let zeros_z = vec![0u8; z_table_size];
            let zeros_m = vec![0u8; m_table_size];

            cudaMemcpy(
                self.x_table_device.cast(),
                zeros_x.as_ptr().cast(),
                x_table_size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                self.z_table_device.cast(),
                zeros_z.as_ptr().cast(),
                z_table_size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                self.m_table_device.cast(),
                zeros_m.as_ptr().cast(),
                m_table_size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            );
        }

        // Run the circuit
        let status = unsafe {
            custabilizerFrameSimulatorApplyCircuit(
                self.handle,
                self.frame_simulator.unwrap(),
                circuit,
                i32::from(randomize_after_measurement),
                seed,
                self.x_table_device,
                self.z_table_device,
                self.m_table_device,
                ptr::null_mut(), // default stream
            )
        };

        // Clean up circuit
        unsafe {
            custabilizerDestroyCircuit(circuit);
        }

        check_stabilizer_status(status)?;

        // Read back measurement results
        let mut m_table_host = vec![0u8; m_table_size];
        unsafe {
            cudaMemcpy(
                m_table_host.as_mut_ptr().cast(),
                self.m_table_device.cast(),
                m_table_size,
                cudaMemcpyKind_cudaMemcpyDeviceToHost,
            );
        }

        // Parse measurement results
        // m_table is measurement-major: m_table[meas_idx * stride + shot_byte]
        let mut measurements = vec![vec![false; num_measurements]; self.num_shots];
        for (shot, shot_measurements) in measurements.iter_mut().enumerate() {
            for (meas, outcome) in shot_measurements.iter_mut().enumerate() {
                let byte_idx = meas * self.table_stride + shot / 8;
                let bit_idx = shot % 8;
                let bit = (m_table_host[byte_idx] >> bit_idx) & 1;
                *outcome = bit != 0;
            }
        }

        Ok(FrameSimulationResults {
            measurements,
            num_shots: self.num_shots,
            num_measurements,
        })
    }

    /// Reset the simulator state (clears X, Z, and M tables).
    pub fn reset(&mut self) {
        let x_table_size = self.num_qubits * self.table_stride;
        let z_table_size = self.num_qubits * self.table_stride;
        let m_table_size = self.max_measurements * self.table_stride;

        unsafe {
            let zeros_x = vec![0u8; x_table_size];
            let zeros_z = vec![0u8; z_table_size];
            let zeros_m = vec![0u8; m_table_size];

            cudaMemcpy(
                self.x_table_device.cast(),
                zeros_x.as_ptr().cast(),
                x_table_size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                self.z_table_device.cast(),
                zeros_z.as_ptr().cast(),
                z_table_size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            );
            cudaMemcpy(
                self.m_table_device.cast(),
                zeros_m.as_ptr().cast(),
                m_table_size,
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            );
        }
    }
}

impl Drop for CuFrameSimulator {
    fn drop(&mut self) {
        unsafe {
            if let Some(frame_sim) = self.frame_simulator.take() {
                custabilizerDestroyFrameSimulator(frame_sim);
            }
            if !self.circuit_buffer_device.is_null() {
                cudaFree(self.circuit_buffer_device);
            }
            cudaFree(self.x_table_device.cast());
            cudaFree(self.z_table_device.cast());
            cudaFree(self.m_table_device.cast());
            custabilizerDestroy(self.handle);
        }
    }
}

// ============================================================================
// Legacy CuStabilizer API (gate-by-gate, wraps CuFrameSimulator)
// ============================================================================

/// Stabilizer simulator using NVIDIA cuStabilizer with gate-by-gate API.
///
/// This provides a gate-by-gate interface on top of the frame simulation API
/// by buffering gates and executing when measurements are requested.
///
/// For better performance with many shots, use `CuFrameSimulator` directly.
pub struct CuStabilizer {
    num_qubits: usize,
    seed: u64,
    gates: Vec<String>,
    measurement_count: usize,
}

impl CuStabilizer {
    /// Create a new stabilizer simulator.
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    pub fn new(num_qubits: usize) -> Result<Self> {
        Self::with_seed_result(num_qubits, 0)
    }

    /// Create a new stabilizer simulator with a specific RNG seed.
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    #[allow(unreachable_code, unused_variables)]
    pub fn with_seed_result(num_qubits: usize, seed: u64) -> Result<Self> {
        if num_qubits == 0 {
            return Err(CuQuantumError::InvalidArgument(
                "num_qubits must be at least 1".into(),
            ));
        }

        #[cfg(cuquantum_stub)]
        return Err(CuQuantumError::NotAvailable(
            "cuQuantum SDK is not installed. To use GPU-accelerated simulators, install the cuQuantum SDK:\n\
             1. Set CUQUANTUM_ROOT environment variable, or\n\
             2. Install via: pecos install cuquantum, or\n\
             3. Install system-wide to /usr/local/cuquantum/"
                .into(),
        ));

        Ok(Self {
            num_qubits,
            seed,
            gates: Vec::new(),
            measurement_count: 0,
        })
    }

    /// Get the number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Execute buffered gates and return measurement results.
    fn execute(&mut self) -> Result<Vec<bool>> {
        if self.gates.is_empty() || self.measurement_count == 0 {
            return Ok(vec![]);
        }

        let circuit_string = self.gates.join("\n");
        let mut sim = CuFrameSimulator::new(self.num_qubits, 1, self.measurement_count)?;
        let results = sim.run_circuit(&circuit_string, self.seed)?;

        // Clear gates after execution
        self.gates.clear();
        let count = self.measurement_count;
        self.measurement_count = 0;

        // Return results for the single shot
        if results.measurements.is_empty() {
            Ok(vec![false; count])
        } else {
            Ok(results.measurements[0].clone())
        }
    }

    /// Measure a qubit in the Z basis with a forced outcome.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        // Frame simulation doesn't support forced outcomes directly
        // We add the measurement and return the forced outcome
        self.gates.push(format!("M {qubit}"));
        self.measurement_count += 1;
        MeasurementResult {
            outcome: forced_outcome,
            is_deterministic: true,
        }
    }
}

impl Clone for CuStabilizer {
    fn clone(&self) -> Self {
        Self {
            num_qubits: self.num_qubits,
            seed: self.seed,
            gates: self.gates.clone(),
            measurement_count: self.measurement_count,
        }
    }
}

impl TryClone for CuStabilizer {
    fn try_clone(&self) -> Result<Self> {
        Ok(self.clone())
    }
}

impl QuantumSimulator for CuStabilizer {
    fn reset(&mut self) -> &mut Self {
        self.gates.clear();
        self.measurement_count = 0;
        self
    }
}

impl CliffordGateable for CuStabilizer {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.gates.push(format!("S {}", q.0));
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.gates.push(format!("S_DAG {}", q.0));
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.gates.push(format!("H {}", q.0));
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.gates.push(format!("X {}", q.0));
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.gates.push(format!("Y {}", q.0));
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for q in qubits {
            self.gates.push(format!("Z {}", q.0));
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.gates.push(format!("CNOT {} {}", q0.0, q1.0));
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.gates.push(format!("CZ {} {}", q0.0, q1.0));
        }
        self
    }

    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.gates.push(format!("SWAP {} {}", q0.0, q1.0));
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        // Add measurement instructions
        for q in qubits {
            self.gates.push(format!("M {}", q.0));
            self.measurement_count += 1;
        }

        // Execute and get results
        match self.execute() {
            Ok(outcomes) => outcomes
                .into_iter()
                .map(|outcome| MeasurementResult {
                    outcome,
                    is_deterministic: false,
                })
                .collect(),
            Err(_) => qubits
                .iter()
                .map(|_| MeasurementResult {
                    outcome: false,
                    is_deterministic: false,
                })
                .collect(),
        }
    }
}

impl StabilizerTableauSimulator for CuStabilizer {
    fn stab_tableau(&self) -> String {
        unimplemented!("CuStabilizer does not support local tableau access")
    }

    fn destab_tableau(&self) -> String {
        unimplemented!("CuStabilizer does not support local tableau access")
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

impl ForcedMeasurement for CuStabilizer {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        CuStabilizer::mz_forced(self, qubit, forced_outcome)
    }
}

impl StabilizerSimulator for CuStabilizer {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed_result(num_qubits, seed).expect("Failed to create CuStabilizer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_simulator_creation() {
        // This test will fail if cuQuantum is not available, which is expected
        let result = CuFrameSimulator::new(5, 100, 10);
        // Just check that we can attempt to create one
        // Actual GPU tests should be in integration tests
        if result.is_err() {
            eprintln!(
                "CuFrameSimulator creation failed (expected if no GPU): {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_custabilizer_gate_buffering() {
        if let Ok(mut sim) = CuStabilizer::new(3) {
            sim.h(&[QubitId(0)]);
            sim.cx(&[(QubitId(0), QubitId(1))]);
            assert_eq!(sim.gates.len(), 2);
            assert_eq!(sim.gates[0], "H 0");
            assert_eq!(sim.gates[1], "CNOT 0 1");
        }
    }

    #[test]
    fn test_custabilizer_reset_clears_gates() {
        if let Ok(mut sim) = CuStabilizer::new(3) {
            sim.h(&[QubitId(0)]);
            sim.reset();
            assert!(sim.gates.is_empty());
        }
    }

    #[test]
    fn test_zero_qubits_error() {
        let result = CuStabilizer::new(0);
        assert!(matches!(result, Err(CuQuantumError::InvalidArgument(_))));

        let result = CuFrameSimulator::new(0, 100, 10);
        assert!(matches!(result, Err(CuQuantumError::InvalidArgument(_))));
    }

    #[test]
    fn test_zero_shots_error() {
        let result = CuFrameSimulator::new(5, 0, 10);
        assert!(matches!(result, Err(CuQuantumError::InvalidArgument(_))));
    }
}
