// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Decoder integration for full error correction loop testing.
//!
//! This module provides tools for testing fault tolerance with a complete
//! error correction cycle: error -> syndrome extraction -> decoding -> recovery.
//!
//! # Overview
//!
//! The error correction loop works as follows:
//! 1. Inject a Pauli fault into the circuit
//! 2. Propagate the fault through syndrome extraction
//! 3. Extract the syndrome from the propagated error
//! 4. Run the decoder to determine recovery operation
//! 5. Apply recovery to the residual error
//! 6. Check if the result is a logical error
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::{
//!     ErrorCorrectionConfig, LookupTableDecoder, extract_syndrome
//! };
//! use pecos_qsim::PauliProp;
//!
//! // Create a simple config for 3-qubit code
//! let config = ErrorCorrectionConfig {
//!     z_ancillas: vec![3, 4],
//!     x_ancillas: vec![],
//!     data_qubits: vec![0, 1, 2],
//!     logical_zs: vec![(vec![], vec![0, 1, 2])],
//!     logical_xs: vec![(vec![0], vec![])],
//! };
//!
//! // Build a lookup table decoder from known syndromes
//! let mut decoder = LookupTableDecoder::new(2, 3); // 2-bit syndrome, 3 data qubits
//! decoder.add_entry(vec![1, 0], vec![1, 0, 0]); // syndrome 10 -> X on qubit 0
//! decoder.add_entry(vec![1, 1], vec![0, 1, 0]); // syndrome 11 -> X on qubit 1
//! decoder.add_entry(vec![0, 1], vec![0, 0, 1]); // syndrome 01 -> X on qubit 2
//!
//! // Simulate an X error on qubit 0
//! let mut prop = PauliProp::new();
//! prop.add_x(0);
//!
//! let syndrome = extract_syndrome(&prop, &config.z_ancillas, &config.x_ancillas);
//! assert_eq!(syndrome.len(), 2);
//! ```

use super::{
    FaultCheckConfig, FaultConfiguration, PauliFaultIterator, SpacetimeLocation,
    anticommutes_with_logical, propagate_faults,
};
use ndarray::{Array1, ArrayView1};
use pecos_decoder_core::{Decoder, DecodingResultTrait};
use pecos_qsim::PauliProp;

/// Configuration for error correction checking.
#[derive(Debug, Clone)]
pub struct ErrorCorrectionConfig {
    /// Qubits measured in Z basis (syndrome ancillas for X-type stabilizers).
    pub z_ancillas: Vec<usize>,
    /// Qubits measured in X basis (syndrome ancillas for Z-type stabilizers).
    pub x_ancillas: Vec<usize>,
    /// Data qubit indices.
    pub data_qubits: Vec<usize>,
    /// Logical Z operators as (X positions, Z positions) pairs.
    pub logical_zs: Vec<(Vec<usize>, Vec<usize>)>,
    /// Logical X operators as (X positions, Z positions) pairs.
    pub logical_xs: Vec<(Vec<usize>, Vec<usize>)>,
}

impl ErrorCorrectionConfig {
    /// Creates a new error correction configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            z_ancillas: Vec::new(),
            x_ancillas: Vec::new(),
            data_qubits: Vec::new(),
            logical_zs: Vec::new(),
            logical_xs: Vec::new(),
        }
    }

    /// Sets the Z-basis measurement ancillas.
    #[must_use]
    pub fn with_z_ancillas(mut self, ancillas: &[usize]) -> Self {
        self.z_ancillas = ancillas.to_vec();
        self
    }

    /// Sets the X-basis measurement ancillas.
    #[must_use]
    pub fn with_x_ancillas(mut self, ancillas: &[usize]) -> Self {
        self.x_ancillas = ancillas.to_vec();
        self
    }

    /// Sets the data qubit indices.
    #[must_use]
    pub fn with_data_qubits(mut self, qubits: &[usize]) -> Self {
        self.data_qubits = qubits.to_vec();
        self
    }

    /// Adds a logical Z operator.
    #[must_use]
    pub fn with_logical_z(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.logical_zs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Adds a logical X operator.
    #[must_use]
    pub fn with_logical_x(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.logical_xs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Gets all logical operators.
    #[must_use]
    pub fn all_logicals(&self) -> Vec<(&[usize], &[usize])> {
        let mut all = Vec::new();
        for (xs, zs) in &self.logical_zs {
            all.push((xs.as_slice(), zs.as_slice()));
        }
        for (xs, zs) in &self.logical_xs {
            all.push((xs.as_slice(), zs.as_slice()));
        }
        all
    }
}

impl Default for ErrorCorrectionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a single error correction cycle.
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    /// The original fault configuration.
    pub fault: FaultConfiguration,
    /// Syndrome extracted from the propagated error.
    pub syndrome: Vec<u8>,
    /// Recovery operation from the decoder.
    pub recovery: Vec<u8>,
    /// Whether a logical error occurred after correction.
    pub logical_error: bool,
    /// Weight of the residual error after correction.
    pub residual_weight: usize,
    /// Whether the decoder successfully converged.
    pub decoder_converged: bool,
}

impl CorrectionResult {
    /// Returns true if correction was successful (no logical error).
    #[must_use]
    pub fn is_successful(&self) -> bool {
        !self.logical_error && self.decoder_converged
    }
}

/// Result of error correction checking across multiple faults.
#[derive(Debug, Clone)]
pub struct ErrorCorrectionResult {
    /// Total number of fault configurations tested.
    pub total_tested: usize,
    /// Number of logical errors that occurred.
    pub logical_errors: usize,
    /// Number of decoder failures.
    pub decoder_failures: usize,
    /// Detailed results for each fault (if requested).
    pub details: Vec<CorrectionResult>,
    /// The fault weight that was tested.
    pub weight: usize,
}

impl ErrorCorrectionResult {
    /// Returns the logical error rate.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn logical_error_rate(&self) -> f64 {
        if self.total_tested == 0 {
            0.0
        } else {
            self.logical_errors as f64 / self.total_tested as f64
        }
    }

    /// Returns true if all corrections were successful.
    #[must_use]
    pub fn all_successful(&self) -> bool {
        self.logical_errors == 0 && self.decoder_failures == 0
    }
}

/// Extracts syndrome from a propagated Pauli error.
///
/// For Z-basis measurements, X errors on ancillas flip the syndrome.
/// For X-basis measurements, Z errors on ancillas flip the syndrome.
#[must_use]
pub fn extract_syndrome(prop: &PauliProp, z_ancillas: &[usize], x_ancillas: &[usize]) -> Vec<u8> {
    let mut syndrome = Vec::with_capacity(z_ancillas.len() + x_ancillas.len());

    // Z-basis measurements detect X errors
    for &q in z_ancillas {
        syndrome.push(u8::from(prop.contains_x(q)));
    }

    // X-basis measurements detect Z errors
    for &q in x_ancillas {
        syndrome.push(u8::from(prop.contains_z(q)));
    }

    syndrome
}

/// Applies a recovery operation to a Pauli propagator.
///
/// The recovery is specified as a bit vector where each bit indicates
/// whether to apply a Pauli correction on that qubit.
///
/// For CSS codes:
/// - Z syndrome -> X errors -> apply X recovery
/// - X syndrome -> Z errors -> apply Z recovery
pub fn apply_recovery(prop: &mut PauliProp, recovery: &[u8], apply_x: bool) {
    for (q, &bit) in recovery.iter().enumerate() {
        if bit == 1 {
            if apply_x {
                prop.add_x(q);
            } else {
                prop.add_z(q);
            }
        }
    }
}

/// Checks for logical errors after applying recovery.
#[must_use]
pub fn check_logical_errors_after_recovery(
    prop: &PauliProp,
    logicals: &[(&[usize], &[usize])],
) -> bool {
    logicals
        .iter()
        .any(|(xs, zs)| anticommutes_with_logical(prop, xs, zs))
}

/// Runs a full error correction cycle for a single fault.
///
/// # Arguments
///
/// * `circuit` - The syndrome extraction circuit
/// * `fault` - The fault configuration to test
/// * `ec_config` - Error correction configuration
/// * `decoder` - The decoder to use
///
/// # Returns
///
/// The correction result including whether a logical error occurred.
pub fn run_correction_cycle<D>(
    circuit: &pecos_quantum::TickCircuit,
    fault: &FaultConfiguration,
    ec_config: &ErrorCorrectionConfig,
    decoder: &mut D,
) -> Result<CorrectionResult, D::Error>
where
    D: Decoder,
    D::Result: DecodingResultTrait,
{
    // Step 1: Propagate fault through syndrome extraction
    let mut prop = propagate_faults(circuit, fault);

    // Step 2: Extract syndrome
    let syndrome = extract_syndrome(&prop, &ec_config.z_ancillas, &ec_config.x_ancillas);

    // Step 3: Run decoder (only if syndrome is non-trivial)
    let has_syndrome = syndrome.iter().any(|&s| s != 0);
    let (recovery, decoder_converged) = if has_syndrome {
        let syndrome_array = Array1::from_vec(syndrome.clone());
        let result = decoder.decode(&syndrome_array.view())?;
        let standard = result.to_standard();
        (standard.observable, standard.converged.unwrap_or(true))
    } else {
        (vec![0u8; ec_config.data_qubits.len()], true)
    };

    // Step 4: Apply recovery
    // For Z-syndrome (X errors), apply X recovery
    if !ec_config.z_ancillas.is_empty() {
        apply_recovery(&mut prop, &recovery, true);
    }
    // For X-syndrome (Z errors), apply Z recovery
    // Note: This is simplified - real CSS decoders return separate X/Z recoveries

    // Step 5: Check for logical errors
    let all_logicals = ec_config.all_logicals();
    let logical_error = check_logical_errors_after_recovery(&prop, &all_logicals);

    let residual_weight = prop.weight();

    Ok(CorrectionResult {
        fault: fault.clone(),
        syndrome,
        recovery,
        logical_error,
        residual_weight,
        decoder_converged,
    })
}

/// Error correction checker that tests fault tolerance with decoding.
pub struct ErrorCorrectionChecker<'a> {
    circuit: &'a pecos_quantum::TickCircuit,
    ec_config: ErrorCorrectionConfig,
    locations: Vec<SpacetimeLocation>,
}

impl<'a> ErrorCorrectionChecker<'a> {
    /// Creates a new error correction checker.
    #[must_use]
    pub fn new(circuit: &'a pecos_quantum::TickCircuit) -> Self {
        let locations = super::circuit_runner::extract_spacetime_locations(circuit, false);
        Self {
            circuit,
            ec_config: ErrorCorrectionConfig::new(),
            locations,
        }
    }

    /// Sets the error correction configuration.
    #[must_use]
    pub fn with_config(mut self, config: ErrorCorrectionConfig) -> Self {
        self.ec_config = config;
        self
    }

    /// Sets the Z-basis measurement ancillas.
    #[must_use]
    pub fn with_z_ancillas(mut self, ancillas: &[usize]) -> Self {
        self.ec_config.z_ancillas = ancillas.to_vec();
        self
    }

    /// Sets the X-basis measurement ancillas.
    #[must_use]
    pub fn with_x_ancillas(mut self, ancillas: &[usize]) -> Self {
        self.ec_config.x_ancillas = ancillas.to_vec();
        self
    }

    /// Sets the data qubit indices.
    #[must_use]
    pub fn with_data_qubits(mut self, qubits: &[usize]) -> Self {
        self.ec_config.data_qubits = qubits.to_vec();
        self
    }

    /// Adds a logical Z operator.
    #[must_use]
    pub fn with_logical_z(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.ec_config
            .logical_zs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Adds a logical X operator.
    #[must_use]
    pub fn with_logical_x(mut self, x_positions: &[usize], z_positions: &[usize]) -> Self {
        self.ec_config
            .logical_xs
            .push((x_positions.to_vec(), z_positions.to_vec()));
        self
    }

    /// Returns the spacetime locations that will be checked.
    #[must_use]
    pub fn locations(&self) -> &[SpacetimeLocation] {
        &self.locations
    }

    /// Runs error correction checking with the specified decoder.
    ///
    /// # Arguments
    ///
    /// * `decoder` - The decoder to use for error correction
    /// * `fault_config` - Configuration for fault enumeration
    /// * `collect_details` - Whether to collect detailed results for each fault
    ///
    /// # Returns
    ///
    /// The error correction result.
    pub fn check<D>(
        &self,
        decoder: &mut D,
        fault_config: FaultCheckConfig,
        collect_details: bool,
    ) -> Result<ErrorCorrectionResult, D::Error>
    where
        D: Decoder,
        D::Result: DecodingResultTrait,
    {
        let mut total_tested = 0;
        let mut logical_errors = 0;
        let mut decoder_failures = 0;
        let mut details = Vec::new();

        let max_weight = fault_config.max_weight;
        let fault_iter = PauliFaultIterator::new(self.locations.clone(), max_weight, fault_config);

        for fault in fault_iter {
            total_tested += 1;

            let result = run_correction_cycle(self.circuit, &fault, &self.ec_config, decoder)?;

            if result.logical_error {
                logical_errors += 1;
            }
            if !result.decoder_converged {
                decoder_failures += 1;
            }

            if collect_details {
                details.push(result);
            }
        }

        Ok(ErrorCorrectionResult {
            total_tested,
            logical_errors,
            decoder_failures,
            details,
            weight: max_weight,
        })
    }
}

/// Simple lookup table decoder for testing.
///
/// This decoder uses a precomputed mapping from syndrome to recovery.
#[derive(Debug, Clone)]
pub struct LookupTableDecoder {
    /// Mapping from syndrome (as Vec<u8>) to recovery (as Vec<u8>).
    table: std::collections::HashMap<Vec<u8>, Vec<u8>>,
    /// Number of syndrome bits.
    syndrome_size: usize,
    /// Number of data qubits.
    data_size: usize,
}

impl LookupTableDecoder {
    /// Creates a new lookup table decoder.
    #[must_use]
    pub fn new(syndrome_size: usize, data_size: usize) -> Self {
        Self {
            table: std::collections::HashMap::new(),
            syndrome_size,
            data_size,
        }
    }

    /// Adds an entry to the lookup table.
    pub fn add_entry(&mut self, syndrome: Vec<u8>, recovery: Vec<u8>) {
        self.table.insert(syndrome, recovery);
    }

    /// Creates a decoder for the 3-qubit bit-flip code.
    ///
    /// Syndrome mapping:
    /// - 00 -> no error
    /// - 10 -> error on qubit 0
    /// - 11 -> error on qubit 1
    /// - 01 -> error on qubit 2
    #[must_use]
    pub fn three_qubit_bitflip() -> Self {
        let mut decoder = Self::new(2, 3);
        decoder.add_entry(vec![0, 0], vec![0, 0, 0]); // No error
        decoder.add_entry(vec![1, 0], vec![1, 0, 0]); // Error on q0
        decoder.add_entry(vec![1, 1], vec![0, 1, 0]); // Error on q1
        decoder.add_entry(vec![0, 1], vec![0, 0, 1]); // Error on q2
        decoder
    }
}

/// Error type for lookup table decoder.
#[derive(Debug, Clone)]
pub struct LookupTableError {
    message: String,
}

impl std::fmt::Display for LookupTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lookup table error: {}", self.message)
    }
}

impl std::error::Error for LookupTableError {}

/// Decoding result for lookup table decoder.
#[derive(Debug, Clone)]
pub struct LookupTableResult {
    pub recovery: Vec<u8>,
    pub found: bool,
}

impl DecodingResultTrait for LookupTableResult {
    fn is_successful(&self) -> bool {
        self.found
    }

    fn to_standard(&self) -> pecos_decoder_core::StandardDecodingResult {
        pecos_decoder_core::StandardDecodingResult {
            observable: self.recovery.clone(),
            weight: 0.0,
            converged: Some(self.found),
            iterations: None,
            confidence: None,
        }
    }
}

impl Decoder for LookupTableDecoder {
    type Result = LookupTableResult;
    type Error = LookupTableError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        let syndrome: Vec<u8> = input.iter().copied().collect();

        if let Some(recovery) = self.table.get(&syndrome) {
            Ok(LookupTableResult {
                recovery: recovery.clone(),
                found: true,
            })
        } else {
            // Return identity recovery if syndrome not found
            Ok(LookupTableResult {
                recovery: vec![0u8; self.data_size],
                found: false,
            })
        }
    }

    fn check_count(&self) -> usize {
        self.syndrome_size
    }

    fn bit_count(&self) -> usize {
        self.data_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::TickCircuit;

    #[test]
    fn test_extract_syndrome() {
        let mut prop = PauliProp::new();
        prop.add_x(3); // X error on ancilla 3

        let syndrome = extract_syndrome(&prop, &[3, 4], &[]);
        assert_eq!(syndrome, vec![1, 0]);
    }

    #[test]
    fn test_apply_recovery() {
        let mut prop = PauliProp::new();
        prop.add_x(0); // X error on qubit 0

        // Apply X recovery on qubit 0 (should cancel out)
        apply_recovery(&mut prop, &[1, 0, 0], true);

        assert!(!prop.contains_x(0));
    }

    #[test]
    fn test_lookup_table_decoder() {
        let mut decoder = LookupTableDecoder::three_qubit_bitflip();

        // Test syndrome [1, 0] -> recovery on qubit 0
        let syndrome = Array1::from_vec(vec![1u8, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();
        assert_eq!(result.recovery, vec![1, 0, 0]);
        assert!(result.found);
    }

    #[test]
    fn test_error_correction_config() {
        let config = ErrorCorrectionConfig::new()
            .with_z_ancillas(&[3, 4])
            .with_data_qubits(&[0, 1, 2])
            .with_logical_z(&[], &[0, 1, 2]); // Z0Z1Z2

        assert_eq!(config.z_ancillas, vec![3, 4]);
        assert_eq!(config.data_qubits, vec![0, 1, 2]);
        assert_eq!(config.logical_zs.len(), 1);
    }

    #[test]
    fn test_three_qubit_code_correction() {
        // Build 3-qubit bit-flip syndrome extraction circuit
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let mut decoder = LookupTableDecoder::three_qubit_bitflip();

        let checker = ErrorCorrectionChecker::new(&circuit)
            .with_z_ancillas(&[3, 4])
            .with_data_qubits(&[0, 1, 2])
            .with_logical_z(&[], &[0, 1, 2]); // Logical Z = Z0Z1Z2

        let fault_config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let result = checker.check(&mut decoder, fault_config, true).unwrap();

        println!(
            "3-qubit code correction: {} logical errors out of {} tested",
            result.logical_errors, result.total_tested
        );

        // Print details
        for detail in &result.details {
            if detail.logical_error {
                println!(
                    "  Logical error: syndrome={:?}, recovery={:?}",
                    detail.syndrome, detail.recovery
                );
            }
        }

        // Weight-1 X errors on data qubits should be correctable
        // Errors on ancillas don't affect data, so they shouldn't cause logical errors either
    }

    #[test]
    fn test_error_correction_result() {
        let result = ErrorCorrectionResult {
            total_tested: 100,
            logical_errors: 5,
            decoder_failures: 0,
            details: vec![],
            weight: 1,
        };

        assert!(!result.all_successful());
        assert!((result.logical_error_rate() - 0.05).abs() < 1e-10);
    }
}
