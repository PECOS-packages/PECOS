//! Quantum error correction support
//!
//! This module provides quantum-specific functionality. While most decoders in this crate
//! (BP, BP+OSD, BP+LSD, Flip, Union Find, `BeliefFind`) can be applied to quantum codes by
//! decoding X and Z syndromes separately, this module contains truly quantum-native decoders
//! like MBP (Modified Belief Propagation) that consider X, Y, and Z errors simultaneously.

use super::{bridge::ffi, sparse::SparseMatrix};
use crate::{BpMethod, LdpcError};
use cxx::UniquePtr;
use ndarray::{Array1, Array2, ArrayView1};

/// Pauli error types for quantum codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliError {
    /// Identity (no error)
    I = 0,
    /// X (bit flip) error
    X = 1,
    /// Y (bit and phase flip) error
    Y = 2,
    /// Z (phase flip) error
    Z = 3,
}

/// MBP (Modified Belief Propagation) Decoder for quantum codes
///
/// Unlike classical decoders that handle X and Z syndromes separately,
/// MBP considers X, Y, and Z errors simultaneously using:
/// - GF(4) arithmetic for Pauli operators
/// - Three-channel message passing for correlated X, Y, Z errors
/// - Joint decoding that exploits Y = iXZ relationships
/// - Quantum-specific noise models with XYZ bias
pub struct MbpDecoder {
    inner: UniquePtr<ffi::MbpDecoder>,
    n_stabs: usize,
    hx_rows: usize,
    hz_rows: usize,
}

impl MbpDecoder {
    /// Create a new MBP decoder
    ///
    /// # Arguments
    /// * `hx` - X stabilizer matrix for CSS code
    /// * `hz` - Z stabilizer matrix for CSS code
    /// * `error_rate` - Physical error rate
    /// * `xyz_bias` - Relative probabilities of X, Y, Z errors (will be normalized)
    /// * `max_iter` - Maximum BP iterations (0 = n)
    /// * `bp_method` - BP method (product-sum or min-sum)
    /// * `ms_scaling_factor` - Scaling factor for min-sum
    /// * `omp_thread_count` - Number of OpenMP threads (not used currently)
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the matrices have invalid dimensions or if parameters are out of range.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hx: &SparseMatrix,
        hz: &SparseMatrix,
        error_rate: f64,
        xyz_bias: [f64; 3],
        max_iter: usize,
        bp_method: BpMethod,
        ms_scaling_factor: f64,
        omp_thread_count: Option<usize>,
    ) -> Result<Self, LdpcError> {
        // Validate inputs
        if hx.cols != hz.cols {
            return Err(LdpcError::InvalidInput(
                "HX and HZ must have the same number of columns (qubits)".to_string(),
            ));
        }

        // Normalize XYZ bias
        let bias_sum = xyz_bias[0] + xyz_bias[1] + xyz_bias[2];
        if bias_sum <= 0.0 {
            return Err(LdpcError::InvalidInput(
                "XYZ bias values must sum to a positive number".to_string(),
            ));
        }
        let normalized_bias = [
            xyz_bias[0] / bias_sum,
            xyz_bias[1] / bias_sum,
            xyz_bias[2] / bias_sum,
        ];

        // Create sparse matrix representations
        let hx_repr = hx.to_ffi_repr();
        let hz_matrix_repr = hz.to_ffi_repr();

        let inner = ffi::create_mbp_decoder(
            &hx_repr,
            &hz_matrix_repr,
            error_rate,
            &normalized_bias,
            i32::try_from(max_iter).unwrap_or(i32::MAX),
            bp_method.to_ffi(),
            ms_scaling_factor,
            i32::try_from(omp_thread_count.unwrap_or(1)).unwrap_or(1),
        )
        .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Self {
            inner,
            n_stabs: hx.rows + hz.rows,
            hx_rows: hx.rows,
            hz_rows: hz.rows,
        })
    }

    /// Decode a quantum syndrome
    ///
    /// # Arguments
    /// * `syndrome` - Combined syndrome from Z stabilizers (first) and X stabilizers (after)
    ///
    /// # Returns
    /// * Array of Pauli errors for each qubit (I=0, X=1, Y=2, Z=3)
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the syndrome length doesn't match expected size,
    /// or `LdpcError::Ldpc` if the C++ decoder encounters an error.
    ///
    /// # Panics
    ///
    /// Panics if the syndrome array is not contiguous in memory.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<Array1<PauliError>, LdpcError> {
        if syndrome.len() != self.n_stabs {
            return Err(LdpcError::InvalidInput(format!(
                "Syndrome length {} does not match total stabilizers {}",
                syndrome.len(),
                self.n_stabs
            )));
        }

        let syndrome_slice = syndrome
            .as_slice()
            .expect("syndrome array must be contiguous");
        let result = ffi::decode_mbp(self.inner.pin_mut(), syndrome_slice)
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        // Convert GF(4) values to PauliError enum
        let pauli_errors: Array1<PauliError> = result
            .decoding
            .iter()
            .map(|&val| match val {
                1 => PauliError::X,
                2 => PauliError::Y,
                3 => PauliError::Z,
                _ => PauliError::I, // Default for 0 and any invalid value
            })
            .collect();

        Ok(pauli_errors)
    }

    /// Get the raw GF(4) decoding result
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the syndrome length doesn't match expected size,
    /// or `LdpcError::Ldpc` if the C++ decoder encounters an error.
    ///
    /// # Panics
    ///
    /// Panics if the syndrome array is not contiguous in memory.
    pub fn decode_gf4(&mut self, syndrome: &ArrayView1<u8>) -> Result<Array1<u8>, LdpcError> {
        if syndrome.len() != self.n_stabs {
            return Err(LdpcError::InvalidInput(format!(
                "Syndrome length {} does not match total stabilizers {}",
                syndrome.len(),
                self.n_stabs
            )));
        }

        let syndrome_slice = syndrome
            .as_slice()
            .expect("syndrome array must be contiguous");
        let result = ffi::decode_mbp(self.inner.pin_mut(), syndrome_slice)
            .map_err(|e| LdpcError::Ldpc(e.what().to_string()))?;

        Ok(Array1::from_vec(result.decoding))
    }

    // Getter methods
    #[must_use]
    pub fn check_count(&self) -> usize {
        ffi::get_check_count_mbp(&self.inner) as usize
    }

    #[must_use]
    pub fn bit_count(&self) -> usize {
        ffi::get_bit_count_mbp(&self.inner) as usize
    }

    #[must_use]
    pub fn max_iter(&self) -> usize {
        usize::try_from(ffi::get_max_iter_mbp(&self.inner)).unwrap_or(0)
    }

    #[must_use]
    pub fn converged(&self) -> bool {
        ffi::get_converged_mbp(&self.inner)
    }

    #[must_use]
    pub fn iterations(&self) -> usize {
        usize::try_from(ffi::get_iterations_mbp(&self.inner)).unwrap_or(0)
    }

    #[must_use]
    pub fn hx_rows(&self) -> usize {
        self.hx_rows
    }

    #[must_use]
    pub fn hz_rows(&self) -> usize {
        self.hz_rows
    }
}

/// CSS Code representation
pub struct CssCode {
    /// X stabilizer matrix
    pub hx: SparseMatrix,
    /// Z stabilizer matrix
    pub hz: SparseMatrix,
    /// Number of qubits
    pub n: usize,
    /// Number of X stabilizers
    pub mx: usize,
    /// Number of Z stabilizers
    pub mz: usize,
}

impl CssCode {
    /// Create a new CSS code from X and Z stabilizer matrices
    ///
    /// # Errors
    ///
    /// Returns `LdpcError::InvalidInput` if the matrices have different numbers of columns.
    pub fn new(hx: SparseMatrix, hz: SparseMatrix) -> Result<Self, LdpcError> {
        if hx.cols != hz.cols {
            return Err(LdpcError::InvalidInput(
                "HX and HZ must have the same number of columns (qubits)".to_string(),
            ));
        }

        let n = hx.cols;
        let mx = hx.rows;
        let mz = hz.rows;

        Ok(Self { hx, hz, n, mx, mz })
    }

    /// Get the combined GF(4) parity check matrix
    ///
    /// This combines HX and HZ into a single matrix where:
    /// - HZ checks come first with value 3 (Z stabilizers)
    /// - HX checks come after with value 1 (X stabilizers)
    #[must_use]
    pub fn to_gf4_pcm(&self) -> Array2<u8> {
        let mut gf4_pcm = Array2::zeros((self.mx + self.mz, self.n));

        // Add Z stabilizers (value 3)
        let z_check_matrix = self.hz.to_dense();
        for i in 0..self.mz {
            for j in 0..self.n {
                if z_check_matrix[[i, j]] == 1 {
                    gf4_pcm[[i, j]] = 3;
                }
            }
        }

        // Add X stabilizers (value 1)
        let x_check_matrix = self.hx.to_dense();
        for i in 0..self.mx {
            for j in 0..self.n {
                if x_check_matrix[[i, j]] == 1 {
                    gf4_pcm[[self.mz + i, j]] = 1;
                }
            }
        }

        gf4_pcm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_css_code_creation() {
        // Create a simple repetition code
        let hx = SparseMatrix::from_coo(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]).unwrap();

        let hz = SparseMatrix::from_coo(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]).unwrap();

        let css = CssCode::new(hx, hz).unwrap();
        assert_eq!(css.n, 3);
        assert_eq!(css.mx, 2);
        assert_eq!(css.mz, 2);

        let gf4_pcm = css.to_gf4_pcm();
        assert_eq!(gf4_pcm.shape(), &[4, 3]);
    }

    #[test]
    fn test_mbp_decoder() {
        // Create a simple CSS code (repetition code)
        let hx = SparseMatrix::from_coo(
            1,
            3, // 1 X check, 3 qubits
            vec![0, 0, 0],
            vec![0, 1, 2],
        )
        .unwrap();

        let hz = SparseMatrix::from_coo(
            2,
            3, // 2 Z checks, 3 qubits
            vec![0, 0, 1, 1],
            vec![0, 1, 1, 2],
        )
        .unwrap();

        let mut decoder = MbpDecoder::new(
            &hx,
            &hz,
            0.1,             // error rate
            [1.0, 1.0, 1.0], // equal XYZ bias
            10,              // max iterations
            crate::BpMethod::MinimumSum,
            0.625, // MS scaling
            None,  // thread count
        )
        .unwrap();

        assert_eq!(decoder.bit_count(), 3);
        assert_eq!(decoder.check_count(), 3);
        assert_eq!(decoder.hx_rows(), 1);
        assert_eq!(decoder.hz_rows(), 2);

        // Test decoding with a syndrome
        // Syndrome order: [Z checks, X checks] = [hz syndrome, hx syndrome]
        let syndrome = Array1::from_vec(vec![1, 0, 1]); // Z0=1, Z1=0, X0=1
        let result = decoder.decode(&syndrome.view()).unwrap();

        assert_eq!(result.len(), 3);

        // Test GF4 decoding
        decoder.decode_gf4(&syndrome.view()).unwrap();
    }
}
