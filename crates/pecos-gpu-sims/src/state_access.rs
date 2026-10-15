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

//! Explicit host-snapshot state access for GPU-backed simulators.
//!
//! The generic `pecos_simulators::StateVectorAccess` and
//! `DensityMatrixAccess` traits look like ordinary passive inspection. For GPU
//! simulators, inspection requires synchronization and device-to-host copy. The
//! traits in this module make that transfer explicit in the method names while
//! returning the same little-endian data shapes as the CPU state-access traits.

use num_complex::Complex64;
use pecos_simulators::{QuantumSimulator, StateAccessError};

use crate::{GpuDensityMatrix, GpuStateVec32, GpuStateVec64, GpuStateVecBackend};

/// Explicit host readback for GPU state-vector simulators.
pub trait GpuStateVectorHostAccess {
    /// Returns the number of qubits represented by the device state.
    fn num_qubits(&self) -> usize;

    /// Synchronizes pending GPU work and returns a host-owned state vector.
    ///
    /// The vector is in little-endian computational-basis order. Calling this
    /// method performs a device-to-host transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows.
    fn state_vector_host_snapshot(&mut self) -> Result<Vec<Complex64>, StateAccessError>;

    /// Synchronizes pending GPU work and returns one host-copied amplitude.
    ///
    /// # Errors
    ///
    /// Returns an error if `basis_state` is outside the Hilbert space.
    fn amplitude_host_snapshot(
        &mut self,
        basis_state: usize,
    ) -> Result<Complex64, StateAccessError> {
        validate_basis_index(self.num_qubits(), basis_state)?;
        Ok(self.state_vector_host_snapshot()?[basis_state])
    }
}

/// Explicit host readback for GPU density-matrix simulators.
pub trait GpuDensityMatrixHostAccess {
    /// Returns the number of physical qubits represented by the density matrix.
    fn num_qubits(&self) -> usize;

    /// Synchronizes pending GPU work and returns a host-owned density matrix.
    ///
    /// The matrix is in little-endian computational-basis order. Calling this
    /// method performs a device-to-host transfer and, for the current
    /// Choi-state implementation, reconstructs the density matrix on the host.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hilbert-space dimension overflows.
    fn density_matrix_host_snapshot(&mut self) -> Result<Vec<Vec<Complex64>>, StateAccessError>;

    /// Synchronizes pending GPU work and returns one host-copied density-matrix
    /// element.
    ///
    /// # Errors
    ///
    /// Returns an error if either basis index is outside the Hilbert space.
    fn density_matrix_element_host_snapshot(
        &mut self,
        row: usize,
        col: usize,
    ) -> Result<Complex64, StateAccessError> {
        validate_basis_index(self.num_qubits(), row)?;
        validate_basis_index(self.num_qubits(), col)?;
        Ok(self.density_matrix_host_snapshot()?[row][col])
    }
}

impl GpuStateVectorHostAccess for GpuStateVec32 {
    fn num_qubits(&self) -> usize {
        QuantumSimulator::num_qubits(self)
    }

    fn state_vector_host_snapshot(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        hilbert_dim(GpuStateVectorHostAccess::num_qubits(self))?;
        Ok(self
            .state()
            .into_iter()
            .map(|[re, im]| Complex64::new(f64::from(re), f64::from(im)))
            .collect())
    }
}

impl GpuStateVectorHostAccess for GpuStateVec64 {
    fn num_qubits(&self) -> usize {
        QuantumSimulator::num_qubits(self)
    }

    fn state_vector_host_snapshot(&mut self) -> Result<Vec<Complex64>, StateAccessError> {
        hilbert_dim(GpuStateVectorHostAccess::num_qubits(self))?;
        Ok(self
            .state()
            .into_iter()
            .map(|[re, im]| Complex64::new(re, im))
            .collect())
    }
}

impl<SV: GpuStateVecBackend> GpuDensityMatrixHostAccess for GpuDensityMatrix<SV> {
    fn num_qubits(&self) -> usize {
        self.num_qubits()
    }

    fn density_matrix_host_snapshot(&mut self) -> Result<Vec<Vec<Complex64>>, StateAccessError> {
        hilbert_dim(self.num_qubits())?;
        Ok(self.get_density_matrix())
    }
}

fn validate_basis_index(num_qubits: usize, index: usize) -> Result<(), StateAccessError> {
    let dim = hilbert_dim(num_qubits)?;
    if index >= dim {
        return Err(StateAccessError::BasisIndexOutOfRange {
            num_qubits,
            dim,
            index,
        });
    }
    Ok(())
}

fn hilbert_dim(num_qubits: usize) -> Result<usize, StateAccessError> {
    2usize
        .checked_pow(
            num_qubits
                .try_into()
                .map_err(|_| StateAccessError::DimensionOverflow { num_qubits })?,
        )
        .ok_or(StateAccessError::DimensionOverflow { num_qubits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_basis_indices_against_hilbert_dimension() {
        assert_eq!(validate_basis_index(3, 7), Ok(()));
        assert_eq!(
            validate_basis_index(3, 8),
            Err(StateAccessError::BasisIndexOutOfRange {
                num_qubits: 3,
                dim: 8,
                index: 8,
            })
        );
    }

    #[test]
    fn reports_dimension_overflow() {
        assert_eq!(hilbert_dim(0), Ok(1));
        assert_eq!(hilbert_dim(3), Ok(8));
        assert!(matches!(
            hilbert_dim(usize::BITS as usize),
            Err(StateAccessError::DimensionOverflow { .. })
        ));
    }
}
