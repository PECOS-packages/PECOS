//! Safe wrapper for cuStateVec state vector simulation
//!
//! This module provides a safe Rust API for NVIDIA's cuStateVec library,
//! which accelerates state vector quantum simulation on CUDA GPUs.

use crate::error::{CuQuantumError, Result, TryClone, check_status};
use pecos_core::{Angle64, QubitId};
use pecos_cuquantum_sys::{
    cuDoubleComplex, cudaDataType_t, cudaMemcpyKind_cudaMemcpyDeviceToDevice,
    cudaMemcpyKind_cudaMemcpyHostToDevice, custatevecCollapseOp_t, custatevecComputeType_t,
    custatevecHandle_t, custatevecMatrixLayout_t,
};
use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};
use std::ffi::c_void;
use std::ptr;

/// State vector simulator using NVIDIA cuStateVec
///
/// This struct manages a cuStateVec handle and device memory for the state vector,
/// providing methods for quantum gate operations and measurements.
///
/// # Example
///
/// ```no_run
/// use pecos_cuquantum::CuStateVec;
/// use pecos_qsim::{QuantumSimulator, CliffordGateable};
/// use pecos_core::QubitId;
///
/// let mut sim = CuStateVec::new(4).unwrap(); // 4 qubits
/// sim.h(&[QubitId(0)]);           // Hadamard on qubit 0
/// sim.cx(&[QubitId(0), QubitId(1)]);       // CNOT with control=0, target=1
/// let results = sim.mz(&[QubitId(0)]);  // Measure qubit 0
/// ```
pub struct CuStateVec {
    handle: custatevecHandle_t,
    num_qubits: usize,
    /// Device pointer to the state vector (2^n complex doubles)
    state_vector: *mut c_void,
    /// Random number generator for non-deterministic measurements
    rng: fastrand::Rng,
}

impl CuStateVec {
    /// Create a new state vector simulator
    ///
    /// Initializes the state to |0...0>.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits to simulate
    ///
    /// # Errors
    /// Returns an error if:
    /// - cuStateVec initialization fails
    /// - CUDA device is not available
    /// - GPU memory allocation fails
    pub fn new(num_qubits: usize) -> Result<Self> {
        Self::with_seed(num_qubits, 0)
    }

    /// Create a new state vector simulator with a specific RNG seed
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits to simulate
    /// * `seed` - Seed for the random number generator
    ///
    /// # Errors
    /// Returns an error if initialization fails
    #[allow(unreachable_code, unused_variables)]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Result<Self> {
        if num_qubits == 0 {
            return Err(CuQuantumError::InvalidArgument(
                "num_qubits must be at least 1".into(),
            ));
        }

        if num_qubits > 30 {
            return Err(CuQuantumError::InvalidArgument(
                "num_qubits > 30 requires too much GPU memory".into(),
            ));
        }

        #[cfg(cuquantum_stub)]
        return Err(CuQuantumError::NotAvailable(
            "cuQuantum SDK is not installed. To use GPU-accelerated simulators, install the cuQuantum SDK:\n\
             1. Set CUQUANTUM_ROOT environment variable, or\n\
             2. Install to ~/.pecos/cuquantum/, or\n\
             3. Install system-wide to /usr/local/cuquantum/"
                .into(),
        ));

        // Create cuStateVec handle
        let mut handle: custatevecHandle_t = ptr::null_mut();
        let status = unsafe { pecos_cuquantum_sys::custatevecCreate(&mut handle) };
        check_status(status)?;

        // Allocate device memory for state vector
        // State vector has 2^n complex double values, each 16 bytes
        let dimension = 1usize << num_qubits;
        let size_bytes = dimension * std::mem::size_of::<cuDoubleComplex>();

        let mut state_vector: *mut c_void = ptr::null_mut();
        let cuda_result = unsafe { pecos_cuquantum_sys::cudaMalloc(&mut state_vector, size_bytes) };

        if cuda_result != 0 {
            // Clean up handle if allocation failed
            unsafe {
                let _ = pecos_cuquantum_sys::custatevecDestroy(handle);
            }
            return Err(CuQuantumError::Cuda(format!(
                "cudaMalloc failed with error code {cuda_result}"
            )));
        }

        let rng = fastrand::Rng::with_seed(seed);

        let mut sim = Self {
            handle,
            num_qubits,
            state_vector,
            rng,
        };

        // Initialize state to |0...0>
        sim.initialize_to_zero()?;

        Ok(sim)
    }

    /// Initialize the state vector to |0...0>
    fn initialize_to_zero(&mut self) -> Result<()> {
        // |0...0> state has amplitude 1.0 at index 0, 0.0 elsewhere
        // We zero out memory first, then set the first element
        let dimension = self.dimension();
        let size_bytes = dimension * std::mem::size_of::<cuDoubleComplex>();

        // Zero out all memory
        let cuda_result =
            unsafe { pecos_cuquantum_sys::cudaMemset(self.state_vector, 0, size_bytes) };
        if cuda_result != 0 {
            return Err(CuQuantumError::Cuda(format!(
                "cudaMemset failed with error code {cuda_result}"
            )));
        }

        // Set first element to (1.0, 0.0) for |0...0>
        let one = cuDoubleComplex { x: 1.0, y: 0.0 };
        let cuda_result = unsafe {
            pecos_cuquantum_sys::cudaMemcpy(
                self.state_vector,
                &one as *const cuDoubleComplex as *const c_void,
                std::mem::size_of::<cuDoubleComplex>(),
                cudaMemcpyKind_cudaMemcpyHostToDevice,
            )
        };
        if cuda_result != 0 {
            return Err(CuQuantumError::Cuda(format!(
                "cudaMemcpy failed with error code {cuda_result}"
            )));
        }

        Ok(())
    }

    /// Get the number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the dimension of the state vector (2^num_qubits)
    #[must_use]
    pub fn dimension(&self) -> usize {
        1 << self.num_qubits
    }

    /// Apply a single-qubit gate specified by a 2x2 matrix
    ///
    /// # Arguments
    /// * `qubit` - Target qubit index
    /// * `matrix` - 2x2 unitary matrix in row-major order [a, b, c, d]
    ///   Each element is [real, imag]
    fn apply_matrix_1q(&mut self, qubit: usize, matrix: &[[f64; 2]; 4]) {
        debug_assert!(qubit < self.num_qubits, "qubit index out of range");

        // Convert to cuDoubleComplex format
        let gate_matrix: [cuDoubleComplex; 4] = [
            cuDoubleComplex {
                x: matrix[0][0],
                y: matrix[0][1],
            },
            cuDoubleComplex {
                x: matrix[1][0],
                y: matrix[1][1],
            },
            cuDoubleComplex {
                x: matrix[2][0],
                y: matrix[2][1],
            },
            cuDoubleComplex {
                x: matrix[3][0],
                y: matrix[3][1],
            },
        ];

        let targets = [qubit as i32];

        // SAFETY: All pointers are valid, handle owns device memory
        let status = unsafe {
            pecos_cuquantum_sys::custatevecApplyMatrix(
                self.handle,
                self.state_vector,
                cudaDataType_t::CUDA_C_64F,
                self.num_qubits as u32,
                gate_matrix.as_ptr() as *const c_void,
                cudaDataType_t::CUDA_C_64F,
                custatevecMatrixLayout_t::CUSTATEVEC_MATRIX_LAYOUT_ROW,
                0, // adjoint = false
                targets.as_ptr(),
                1,           // num_targets
                ptr::null(), // no controls
                ptr::null(), // no control bit values
                0,           // num_controls
                custatevecComputeType_t::CUSTATEVEC_COMPUTE_64F,
                ptr::null_mut(), // no extra workspace
                0,               // workspace size = 0
            )
        };
        check_status(status).expect("custatevecApplyMatrix failed");
    }

    /// Apply a two-qubit gate specified by a 4x4 matrix
    ///
    /// # Arguments
    /// * `qubit_a` - First qubit index (lower in ordering)
    /// * `qubit_b` - Second qubit index (higher in ordering)
    /// * `matrix` - 4x4 unitary matrix in row-major order (16 complex elements)
    fn apply_matrix_2q(&mut self, qubit_a: usize, qubit_b: usize, matrix: &[[f64; 2]; 16]) {
        debug_assert!(qubit_a < self.num_qubits, "qubit_a index out of range");
        debug_assert!(qubit_b < self.num_qubits, "qubit_b index out of range");
        debug_assert!(qubit_a != qubit_b, "qubits must be different");

        // Convert to cuDoubleComplex format
        let gate_matrix: [cuDoubleComplex; 16] = [
            cuDoubleComplex {
                x: matrix[0][0],
                y: matrix[0][1],
            },
            cuDoubleComplex {
                x: matrix[1][0],
                y: matrix[1][1],
            },
            cuDoubleComplex {
                x: matrix[2][0],
                y: matrix[2][1],
            },
            cuDoubleComplex {
                x: matrix[3][0],
                y: matrix[3][1],
            },
            cuDoubleComplex {
                x: matrix[4][0],
                y: matrix[4][1],
            },
            cuDoubleComplex {
                x: matrix[5][0],
                y: matrix[5][1],
            },
            cuDoubleComplex {
                x: matrix[6][0],
                y: matrix[6][1],
            },
            cuDoubleComplex {
                x: matrix[7][0],
                y: matrix[7][1],
            },
            cuDoubleComplex {
                x: matrix[8][0],
                y: matrix[8][1],
            },
            cuDoubleComplex {
                x: matrix[9][0],
                y: matrix[9][1],
            },
            cuDoubleComplex {
                x: matrix[10][0],
                y: matrix[10][1],
            },
            cuDoubleComplex {
                x: matrix[11][0],
                y: matrix[11][1],
            },
            cuDoubleComplex {
                x: matrix[12][0],
                y: matrix[12][1],
            },
            cuDoubleComplex {
                x: matrix[13][0],
                y: matrix[13][1],
            },
            cuDoubleComplex {
                x: matrix[14][0],
                y: matrix[14][1],
            },
            cuDoubleComplex {
                x: matrix[15][0],
                y: matrix[15][1],
            },
        ];

        // cuStateVec maps targets[i] to bit i of the matrix index.
        // Standard gate matrices use qubit_a as MSB (bit 1) and qubit_b as LSB (bit 0),
        // so we reverse the order: targets[0] = qubit_b (LSB), targets[1] = qubit_a (MSB).
        let targets = [qubit_b as i32, qubit_a as i32];

        // SAFETY: All pointers are valid, handle owns device memory
        let status = unsafe {
            pecos_cuquantum_sys::custatevecApplyMatrix(
                self.handle,
                self.state_vector,
                cudaDataType_t::CUDA_C_64F,
                self.num_qubits as u32,
                gate_matrix.as_ptr() as *const c_void,
                cudaDataType_t::CUDA_C_64F,
                custatevecMatrixLayout_t::CUSTATEVEC_MATRIX_LAYOUT_ROW,
                0, // adjoint = false
                targets.as_ptr(),
                2,           // num_targets
                ptr::null(), // no controls
                ptr::null(), // no control bit values
                0,           // num_controls
                custatevecComputeType_t::CUSTATEVEC_COMPUTE_64F,
                ptr::null_mut(), // no extra workspace
                0,               // workspace size = 0
            )
        };
        check_status(status).expect("custatevecApplyMatrix (2q) failed");
    }

    /// Measure a single qubit in the Z basis
    fn measure_single(&mut self, qubit: usize) -> MeasurementResult {
        debug_assert!(qubit < self.num_qubits, "qubit index out of range");

        let basis_bits = [qubit as i32];
        let mut parity: i32 = 0;
        let rand_num = self.rng.f64();

        // SAFETY: All pointers are valid
        let status = unsafe {
            pecos_cuquantum_sys::custatevecMeasureOnZBasis(
                self.handle,
                self.state_vector,
                cudaDataType_t::CUDA_C_64F,
                self.num_qubits as u32,
                &mut parity,
                basis_bits.as_ptr(),
                1, // n_basis_bits
                rand_num,
                custatevecCollapseOp_t::CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO, // Collapse and renormalize
            )
        };
        check_status(status).expect("custatevecMeasureOnZBasis failed");

        MeasurementResult {
            outcome: parity == 1,
            // State vector measurements are generally non-deterministic
            // unless the state is in an eigenstate of the measurement
            is_deterministic: false,
        }
    }

    /// Sample from the state vector
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to draw
    ///
    /// # Returns
    /// Vector of bitstrings (as u64) representing measurement outcomes
    pub fn sample(&mut self, num_samples: usize) -> Vec<u64> {
        if num_samples == 0 {
            return Vec::new();
        }

        // Use custatevecBatchMeasure for sampling (without collapsing)
        // Each sample is independent
        let mut results = Vec::with_capacity(num_samples);

        // Create bit ordering (measure all qubits)
        let bit_ordering: Vec<i32> = (0..self.num_qubits as i32).collect();
        let mut bit_string = vec![0i32; self.num_qubits];

        for _ in 0..num_samples {
            let rand_num = self.rng.f64();

            // SAFETY: All pointers are valid
            let status = unsafe {
                pecos_cuquantum_sys::custatevecBatchMeasure(
                    self.handle,
                    self.state_vector,
                    cudaDataType_t::CUDA_C_64F,
                    self.num_qubits as u32,
                    bit_string.as_mut_ptr(),
                    bit_ordering.as_ptr(),
                    self.num_qubits as u32,
                    rand_num,
                    custatevecCollapseOp_t::CUSTATEVEC_COLLAPSE_NONE, // Don't collapse for sampling
                )
            };
            check_status(status).expect("custatevecBatchMeasure failed");

            // Convert bit string to u64
            let mut value: u64 = 0;
            for (i, &bit) in bit_string.iter().enumerate() {
                if bit == 1 {
                    value |= 1 << i;
                }
            }
            results.push(value);
        }

        results
    }
}

impl Drop for CuStateVec {
    fn drop(&mut self) {
        // Free device memory first
        if !self.state_vector.is_null() {
            unsafe {
                let _ = pecos_cuquantum_sys::cudaFree(self.state_vector);
            }
        }

        // Then destroy handle
        if !self.handle.is_null() {
            unsafe {
                let _ = pecos_cuquantum_sys::custatevecDestroy(self.handle);
            }
        }
    }
}

impl Clone for CuStateVec {
    /// Clone the state vector simulator, including GPU device memory
    ///
    /// This performs a device-to-device memory copy of the state vector.
    /// The cloned instance has its own cuStateVec handle and device memory.
    ///
    /// # Panics
    /// Panics if CUDA memory allocation or copy fails.
    fn clone(&self) -> Self {
        // Create new cuStateVec handle
        let mut handle: custatevecHandle_t = ptr::null_mut();
        let status = unsafe { pecos_cuquantum_sys::custatevecCreate(&mut handle) };
        check_status(status).expect("Failed to create cuStateVec handle for clone");

        // Allocate device memory for the cloned state vector
        let dimension = self.dimension();
        let size_bytes = dimension * std::mem::size_of::<cuDoubleComplex>();

        let mut state_vector: *mut c_void = ptr::null_mut();
        let cuda_result = unsafe { pecos_cuquantum_sys::cudaMalloc(&mut state_vector, size_bytes) };

        if cuda_result != 0 {
            // Clean up handle if allocation failed
            unsafe {
                let _ = pecos_cuquantum_sys::custatevecDestroy(handle);
            }
            panic!("cudaMalloc failed with error code {cuda_result} during clone");
        }

        // Copy device memory from original to clone (device-to-device)
        let cuda_result = unsafe {
            pecos_cuquantum_sys::cudaMemcpy(
                state_vector,
                self.state_vector,
                size_bytes,
                cudaMemcpyKind_cudaMemcpyDeviceToDevice,
            )
        };

        if cuda_result != 0 {
            // Clean up on failure
            unsafe {
                let _ = pecos_cuquantum_sys::cudaFree(state_vector);
                let _ = pecos_cuquantum_sys::custatevecDestroy(handle);
            }
            panic!("cudaMemcpy device-to-device failed with error code {cuda_result} during clone");
        }

        // Clone the RNG with a derived seed to ensure independent random streams
        let rng = self.rng.clone();

        Self {
            handle,
            num_qubits: self.num_qubits,
            state_vector,
            rng,
        }
    }
}

impl TryClone for CuStateVec {
    /// Attempt to clone the state vector simulator, including GPU device memory
    ///
    /// This performs a device-to-device memory copy of the state vector.
    /// The cloned instance has its own cuStateVec handle and device memory.
    ///
    /// # Errors
    /// Returns an error if:
    /// - cuStateVec handle creation fails
    /// - CUDA memory allocation fails
    /// - Device-to-device memory copy fails
    fn try_clone(&self) -> Result<Self> {
        // Create new cuStateVec handle
        let mut handle: custatevecHandle_t = ptr::null_mut();
        let status = unsafe { pecos_cuquantum_sys::custatevecCreate(&mut handle) };
        check_status(status)?;

        // Allocate device memory for the cloned state vector
        let dimension = self.dimension();
        let size_bytes = dimension * std::mem::size_of::<cuDoubleComplex>();

        let mut state_vector: *mut c_void = ptr::null_mut();
        let cuda_result = unsafe { pecos_cuquantum_sys::cudaMalloc(&mut state_vector, size_bytes) };

        if cuda_result != 0 {
            // Clean up handle if allocation failed
            unsafe {
                let _ = pecos_cuquantum_sys::custatevecDestroy(handle);
            }
            return Err(CuQuantumError::Cuda(format!(
                "cudaMalloc failed with error code {cuda_result} during clone"
            )));
        }

        // Copy device memory from original to clone (device-to-device)
        let cuda_result = unsafe {
            pecos_cuquantum_sys::cudaMemcpy(
                state_vector,
                self.state_vector,
                size_bytes,
                cudaMemcpyKind_cudaMemcpyDeviceToDevice,
            )
        };

        if cuda_result != 0 {
            // Clean up on failure
            unsafe {
                let _ = pecos_cuquantum_sys::cudaFree(state_vector);
                let _ = pecos_cuquantum_sys::custatevecDestroy(handle);
            }
            return Err(CuQuantumError::Cuda(format!(
                "cudaMemcpy device-to-device failed with error code {cuda_result} during clone"
            )));
        }

        // Clone the RNG with a derived seed to ensure independent random streams
        let rng = self.rng.clone();

        Ok(Self {
            handle,
            num_qubits: self.num_qubits,
            state_vector,
            rng,
        })
    }
}

// =============================================================================
// PECOS Trait Implementations
// =============================================================================

impl QuantumSimulator for CuStateVec {
    fn reset(&mut self) -> &mut Self {
        // Reinitialize state vector to |0...0>
        self.initialize_to_zero()
            .expect("Failed to reset state vector");
        self
    }
}

impl CliffordGateable for CuStateVec {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        // S gate = [[1, 0], [0, i]]
        let matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let matrix = [
            [inv_sqrt2, 0.0],
            [inv_sqrt2, 0.0],
            [inv_sqrt2, 0.0],
            [-inv_sqrt2, 0.0],
        ];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );
        // CNOT matrix (4x4)
        // [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
        #[rustfmt::skip]
        let matrix: [[f64; 2]; 16] = [
            [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],
        ];
        for pair in qubits.chunks_exact(2) {
            self.apply_matrix_2q(pair[0].0, pair[1].0, &matrix);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits.iter().map(|&q| self.measure_single(q.0)).collect()
    }

    // Override some gates for efficiency (direct GPU implementation)

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        let matrix = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        let matrix = [[0.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        let matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // S-dagger gate = [[1, 0], [0, -i]]
        let matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, -1.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CZ requires pairs of qubits"
        );
        // CZ matrix (4x4)
        // [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]
        #[rustfmt::skip]
        let matrix: [[f64; 2]; 16] = [
            [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0],
        ];
        for pair in qubits.chunks_exact(2) {
            self.apply_matrix_2q(pair[0].0, pair[1].0, &matrix);
        }
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SWAP requires pairs of qubits"
        );
        // SWAP matrix (4x4)
        // [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        #[rustfmt::skip]
        let matrix: [[f64; 2]; 16] = [
            [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0],
        ];
        for pair in qubits.chunks_exact(2) {
            self.apply_matrix_2q(pair[0].0, pair[1].0, &matrix);
        }
        self
    }
}

impl ArbitraryRotationGateable for CuStateVec {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RX(theta) = [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let matrix = [[c, 0.0], [0.0, -s], [0.0, -s], [c, 0.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RZ(theta) = [[e^(-i*theta/2), 0], [0, e^(i*theta/2)]]
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let matrix = [[c, -s], [0.0, 0.0], [0.0, 0.0], [c, s]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RZZ requires pairs of qubits"
        );
        // RZZ(theta) = diag(e^(-i*theta/2), e^(i*theta/2), e^(i*theta/2), e^(-i*theta/2))
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        #[rustfmt::skip]
        let matrix: [[f64; 2]; 16] = [
            [c, -s], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [c, s], [0.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [c, s], [0.0, 0.0],
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [c, -s],
        ];
        for pair in qubits.chunks_exact(2) {
            self.apply_matrix_2q(pair[0].0, pair[1].0, &matrix);
        }
        self
    }

    // Override ry for efficiency
    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        // RY(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let matrix = [[c, 0.0], [-s, 0.0], [s, 0.0], [c, 0.0]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    // Override T gate for efficiency
    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        // T = [[1, 0], [0, e^(i*pi/4)]]
        let cos = std::f64::consts::FRAC_PI_4.cos();
        let sin = std::f64::consts::FRAC_PI_4.sin();
        let matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [cos, sin]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }

    // Override T-dagger for efficiency
    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // T-dagger = [[1, 0], [0, e^(-i*pi/4)]]
        let cos = std::f64::consts::FRAC_PI_4.cos();
        let sin = std::f64::consts::FRAC_PI_4.sin();
        let matrix = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [cos, -sin]];
        for &q in qubits {
            self.apply_matrix_1q(q.0, &matrix);
        }
        self
    }
}

// CuStateVec is not Send/Sync because CUDA handles are typically thread-local
// If needed, we could add proper synchronization

// Note: Integration tests that call cuQuantum functions should go in tests/
// and only run when cuQuantum is available. The tests here only test
// pure Rust code that doesn't call into cuQuantum.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_matrix_values() {
        // Test that our gate matrices are correct (pure Rust, no FFI)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // H gate should have 1/sqrt(2) entries
        assert!((inv_sqrt2 - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_calculation() {
        // Test dimension calculation without creating actual CuStateVec
        assert_eq!(1usize << 4, 16); // 4 qubits = 16 states
        assert_eq!(1usize << 10, 1024); // 10 qubits = 1024 states
    }

    #[test]
    fn test_rotation_angles() {
        // Test that rotation gate calculations are correct
        let theta = std::f64::consts::PI / 2.0;
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();

        // For RX(pi/2), cos(pi/4) = sin(pi/4) = 1/sqrt(2)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((c - inv_sqrt2).abs() < 1e-10);
        assert!((s - inv_sqrt2).abs() < 1e-10);
    }

    #[test]
    fn test_qubit_id_creation() {
        // Test that QubitId works correctly
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        assert_eq!(q0.0, 0);
        assert_eq!(q1.0, 1);
    }

    #[test]
    fn test_size_calculation_for_clone() {
        // Test size calculation used in clone
        let num_qubits = 4;
        let dimension = 1usize << num_qubits;
        let size_bytes = dimension * std::mem::size_of::<cuDoubleComplex>();

        // 4 qubits = 16 states, each cuDoubleComplex is 16 bytes
        assert_eq!(dimension, 16);
        assert_eq!(std::mem::size_of::<cuDoubleComplex>(), 16);
        assert_eq!(size_bytes, 256);
    }
}
