use pecos_core::errors::PecosError;
use std::collections::BTreeMap;

use crate::parser::errors::{index_out_of_bounds, unknown_classical_register, unknown_register};

/// Manages quantum and classical registers
#[derive(Debug, Clone)]
pub struct RegisterManager {
    quantum_registers: BTreeMap<String, Vec<usize>>,
    classical_registers: BTreeMap<String, usize>,
    qubit_map: BTreeMap<usize, (String, usize)>,
    total_qubits: usize,
}

impl RegisterManager {
    /// Create a new empty register manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            quantum_registers: BTreeMap::new(),
            classical_registers: BTreeMap::new(),
            qubit_map: BTreeMap::new(),
            total_qubits: 0,
        }
    }

    /// Add a quantum register
    pub fn add_quantum_register(&mut self, name: String, size: usize) {
        let mut qubit_ids = Vec::with_capacity(size);
        for i in 0..size {
            let qubit_id = self.total_qubits;
            qubit_ids.push(qubit_id);
            self.qubit_map.insert(qubit_id, (name.clone(), i));
            self.total_qubits += 1;
        }
        self.quantum_registers.insert(name, qubit_ids);
    }

    /// Add a classical register
    pub fn add_classical_register(&mut self, name: String, size: usize) {
        self.classical_registers.insert(name, size);
    }

    /// Get all quantum registers
    #[must_use]
    pub fn quantum_registers(&self) -> &BTreeMap<String, Vec<usize>> {
        &self.quantum_registers
    }

    /// Get all classical registers
    #[must_use]
    pub fn classical_registers(&self) -> &BTreeMap<String, usize> {
        &self.classical_registers
    }

    /// Get the qubit map
    #[must_use]
    pub fn qubit_map(&self) -> &BTreeMap<usize, (String, usize)> {
        &self.qubit_map
    }

    /// Get total number of qubits
    #[must_use]
    pub fn total_qubits(&self) -> usize {
        self.total_qubits
    }

    /// Resolve a single qubit by register name and index
    ///
    /// # Errors
    ///
    /// Returns an error if the register doesn't exist or the index is out of bounds
    pub fn resolve_qubit(&self, reg_name: &str, idx: usize) -> Result<usize, PecosError> {
        let qubit_ids = self
            .quantum_registers
            .get(reg_name)
            .ok_or_else(|| unknown_register("quantum", reg_name))?;

        if idx >= qubit_ids.len() {
            return Err(index_out_of_bounds(reg_name, idx, qubit_ids.len()));
        }

        Ok(qubit_ids[idx])
    }

    /// Get all qubits in a register
    ///
    /// # Errors
    ///
    /// Returns an error if the register doesn't exist
    pub fn get_register_qubits(&self, reg_name: &str) -> Result<&[usize], PecosError> {
        self.quantum_registers
            .get(reg_name)
            .map(std::vec::Vec::as_slice)
            .ok_or_else(|| unknown_register("quantum", reg_name))
    }

    /// Get classical register size
    ///
    /// # Errors
    ///
    /// Returns an error if the register doesn't exist
    pub fn get_classical_register_size(&self, reg_name: &str) -> Result<usize, PecosError> {
        self.classical_registers
            .get(reg_name)
            .copied()
            .ok_or_else(|| unknown_classical_register(reg_name))
    }

    /// Check if classical register index is valid
    ///
    /// # Errors
    ///
    /// Returns an error if the register doesn't exist or the index is out of bounds
    pub fn validate_classical_index(&self, reg_name: &str, idx: usize) -> Result<(), PecosError> {
        let size = self.get_classical_register_size(reg_name)?;
        if idx >= size {
            Err(index_out_of_bounds(reg_name, idx, size))
        } else {
            Ok(())
        }
    }
}

impl Default for RegisterManager {
    fn default() -> Self {
        Self::new()
    }
}
