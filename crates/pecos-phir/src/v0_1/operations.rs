use crate::v0_1::ast::{ArgItem, MEASUREMENT_PREFIX, Operation};
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use std::collections::HashMap;

/// Handles processing of variable definitions, quantum and classical operations
#[derive(Debug)]
pub struct OperationProcessor {
    /// Mapping of quantum variable names to their sizes
    pub quantum_variables: HashMap<String, usize>,
    /// Mapping of classical variable names to their types and sizes
    pub classical_variables: HashMap<String, (String, usize)>,
    /// Measurement results and internal variable values
    pub measurement_results: HashMap<String, u32>,
    /// Values explicitly exported via the Result operator
    pub exported_values: HashMap<String, u32>,
    /// Mappings from source registers to export names for Result operations
    pub export_mappings: Vec<(String, String)>,
}

impl Default for OperationProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl OperationProcessor {
    /// Creates a new operation processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            quantum_variables: HashMap::new(),
            classical_variables: HashMap::new(),
            measurement_results: HashMap::new(),
            exported_values: HashMap::new(),
            export_mappings: Vec::new(),
        }
    }

    /// Resets the operation processor state
    pub fn reset(&mut self) {
        self.measurement_results.clear();
        self.exported_values.clear();
        self.export_mappings.clear();
    }

    /// Handle variable definition operations
    pub fn handle_variable_definition(
        &mut self,
        data: &str,
        data_type: &str,
        variable: &str,
        size: usize,
    ) {
        match data {
            "qvar_define" if data_type == "qubits" => {
                self.quantum_variables.insert(variable.to_string(), size);
                log::debug!("Defined quantum variable {} of size {}", variable, size);
            }
            "cvar_define" => {
                self.classical_variables
                    .insert(variable.to_string(), (data_type.to_string(), size));
                log::debug!(
                    "Defined classical variable {} of type {} and size {}",
                    variable,
                    data_type,
                    size
                );
            }
            _ => log::warn!(
                "Unknown variable definition: {} {} {}",
                data,
                data_type,
                variable
            ),
        }
    }

    /// Validate variable access
    pub fn validate_variable_access(&self, var: &str, idx: usize) -> Result<(), PecosError> {
        // Check quantum variables
        if let Some(&size) = self.quantum_variables.get(var) {
            if idx >= size {
                return Err(PecosError::Input(format!(
                    "Variable access validation failed: Index {idx} out of bounds for quantum variable '{var}' of size {size}"
                )));
            }
            return Ok(());
        }

        // Check classical variables
        if let Some((_, size)) = self.classical_variables.get(var) {
            if idx >= *size {
                return Err(PecosError::Input(format!(
                    "Variable access validation failed: Index {idx} out of bounds for classical variable '{var}' of size {size}"
                )));
            }
            return Ok(());
        }

        Err(PecosError::Input(format!(
            "Variable access validation failed: Variable '{var}' is not defined in the program"
        )))
    }

    /// Handle classical operations
    pub fn handle_classical_op(
        &mut self,
        cop: &str,
        args: &[ArgItem],
        returns: &[ArgItem],
    ) -> Result<bool, PecosError> {
        // Extract variable name and index from each ArgItem
        let extract_var_idx = |arg: &ArgItem| -> (String, usize) {
            match arg {
                ArgItem::Indexed((name, idx)) => (name.clone(), *idx),
                ArgItem::Simple(name) => (name.clone(), 0),
            }
        };

        // For most operations, validate all variable accesses
        if cop == "Result" {
            // For Result operation, only validate the source variables (args)
            // The return variables are outputs and don't need to be defined
            for arg in args {
                let (var, idx) = extract_var_idx(arg);
                self.validate_variable_access(&var, idx)?;
            }
        } else {
            for arg in args.iter().chain(returns) {
                let (var, idx) = extract_var_idx(arg);
                self.validate_variable_access(&var, idx)?;
            }
        }

        if cop == "Result" {
            if args.len() == 1 && returns.len() == 1 {
                // Extract source and export info
                let (source_register, _) = extract_var_idx(&args[0]);
                let (export_name, _) = extract_var_idx(&returns[0]);

                log::debug!(
                    "Storing export mapping: {} -> {}",
                    source_register,
                    export_name
                );

                // Instead of immediately exporting, store the mapping for later
                // This allows us to apply the export after all measurements are collected
                self.export_mappings
                    .push((source_register.clone(), export_name.clone()));

                return Ok(true);
            }
            log::warn!("Result operation requires exactly one source and one export target");
            return Ok(true);
        }

        Ok(false)
    }

    /// Process a quantum operation and return the gate type, qubit arguments, and angle arguments
    pub fn process_quantum_op(
        &self,
        qop: &str,
        angles: Option<&(Vec<f64>, String)>,
        args: &[(String, usize)],
    ) -> Result<(String, Vec<usize>, Vec<f64>), PecosError> {
        // First validate all variables
        for (var, idx) in args {
            self.validate_variable_access(var, *idx)?;
        }

        // Validate that we have at least one qubit argument
        if args.is_empty() {
            return Err(PecosError::Input(format!(
                "Invalid quantum operation: Operation '{qop}' requires at least one qubit argument"
            )));
        }

        // Extract qubit arguments
        let mut qubit_args = Vec::new();
        for (_, idx) in args {
            qubit_args.push(*idx);
        }

        // Process based on gate type
        match qop {
            // Single-qubit rotation gates
            "RZ" => {
                let theta = angles
                    .as_ref()
                    .map(|(angles, _)| angles[0])
                    .ok_or_else(|| {
                        PecosError::Gate(format!(
                            "Invalid gate parameters: Missing rotation angle for '{qop}' gate"
                        ))
                    })?;
                Ok((qop.to_string(), qubit_args, vec![theta]))
            }
            "R1XY" => {
                if angles.as_ref().map_or(0, |(angles, _)| angles.len()) < 2 {
                    return Err(PecosError::Gate(format!(
                        "Invalid gate parameters: '{qop}' gate requires two angles (phi, theta)"
                    )));
                }
                let (phi, theta) = angles
                    .as_ref()
                    .map(|(angles, _)| (angles[0], angles[1]))
                    .ok_or_else(|| {
                        PecosError::Gate(format!(
                            "Invalid gate parameters: Missing rotation angles for '{qop}' gate"
                        ))
                    })?;
                Ok((qop.to_string(), qubit_args, vec![phi, theta]))
            }

            // Two-qubit gates
            "SZZ" | "ZZ" => {
                if args.len() < 2 {
                    return Err(PecosError::Gate(format!(
                        "Invalid gate parameters: '{qop}' gate requires exactly two qubits"
                    )));
                }
                Ok(("SZZ".to_string(), qubit_args, vec![]))
            }
            "CX" | "CNOT" => {
                if args.len() < 2 {
                    return Err(PecosError::Gate(format!(
                        "Invalid gate parameters: '{qop}' gate requires control and target qubits (2 qubits total)"
                    )));
                }
                Ok(("CX".to_string(), qubit_args, vec![]))
            }

            // Single-qubit Clifford gates and Measurement
            "H" | "X" | "Y" | "Z" | "Measure" => Ok((qop.to_string(), qubit_args, vec![])),

            _ => Err(PecosError::Gate(format!(
                "Unsupported quantum gate operation: Gate type '{qop}' is not implemented"
            ))),
        }
    }

    /// Add quantum operation to byte message builder
    pub fn add_quantum_operation_to_builder(
        &self,
        builder: &mut ByteMessageBuilder,
        gate_type: &str,
        qubit_args: &[usize],
        angle_args: &[f64],
    ) -> Result<(), PecosError> {
        match gate_type {
            "RZ" => {
                builder.add_rz(angle_args[0], &[qubit_args[0]]);
            }
            "R1XY" => {
                builder.add_r1xy(angle_args[0], angle_args[1], &[qubit_args[0]]);
            }
            "SZZ" => {
                builder.add_szz(&[qubit_args[0]], &[qubit_args[1]]);
            }
            "CX" => {
                builder.add_cx(&[qubit_args[0]], &[qubit_args[1]]);
            }
            "H" => {
                builder.add_h(&[qubit_args[0]]);
            }
            "X" => {
                builder.add_x(&[qubit_args[0]]);
            }
            "Y" => {
                builder.add_y(&[qubit_args[0]]);
            }
            "Z" => {
                builder.add_z(&[qubit_args[0]]);
            }
            "Measure" => {
                builder.add_measurements(&[qubit_args[0]], &[qubit_args[0]]);
            }
            _ => {
                return Err(PecosError::Gate(format!(
                    "Unsupported quantum gate operation: Gate type '{gate_type}' is not implemented"
                )));
            }
        }
        Ok(())
    }

    /// Handle measurements and update measurement results
    pub fn handle_measurements(
        &mut self,
        measurements: &[(u32, u32)],
        ops: &[Operation],
    ) -> Result<(), PecosError> {
        for (result_id, outcome) in measurements {
            debug!(
                "PHIR: Received measurement result_id={}, outcome={}",
                result_id, outcome
            );

            // Store the measurement with the standard prefix and result_id
            self.measurement_results
                .insert(format!("{MEASUREMENT_PREFIX}{result_id}"), *outcome);

            // Also directly map this to the classical variable bits
            // For example, if Measure returns [["m", 0]], we should set m_0 = outcome
            for op in ops {
                if let Operation::QuantumOp {
                    qop,
                    args: _,
                    returns,
                    ..
                } = op
                {
                    if qop == "Measure" && !returns.is_empty() {
                        // Get the variable name and index from the returns field
                        let (var_name, var_idx) = &returns[0];

                        // Check if this is the right measurement result
                        if *var_idx == *result_id as usize {
                            // Store with the format "variable_index"
                            let var_key = format!("{var_name}_{var_idx}");
                            self.measurement_results.insert(var_key.clone(), *outcome);
                            log::debug!(
                                "Mapped measurement result_id={} to {}",
                                result_id,
                                var_key
                            );

                            // Also update the register value by setting the appropriate bit
                            let entry = self
                                .measurement_results
                                .entry(var_name.clone())
                                .or_insert(0);
                            *entry |= outcome << var_idx;
                            log::debug!("Updated register {} value to {}", var_name, *entry);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Process export mappings and prepare final results
    #[must_use]
    pub fn process_export_mappings(&self) -> HashMap<String, u32> {
        let mut exported_values = HashMap::new();

        // Process all stored export mappings
        for (source_register, export_name) in &self.export_mappings {
            log::debug!(
                "Processing export mapping: {} -> {}",
                source_register,
                export_name
            );

            // Check for direct register value first
            if let Some(&value) = self.measurement_results.get(source_register) {
                log::debug!(
                    "Found direct register value for {}: {}",
                    source_register,
                    value
                );
                exported_values.insert(export_name.clone(), value);
                continue;
            }

            // Check for indexed values (e.g., m_0, m_1, etc.)
            let mut register_value = 0u32;
            let mut found_values = false;

            for i in 0..32 {
                // Assuming max 32 bits for registers
                let index_key = format!("{source_register}_{i}");
                if let Some(&value) = self.measurement_results.get(&index_key) {
                    register_value |= value << i;
                    found_values = true;
                    log::debug!("Found indexed value {}_{} = {}", source_register, i, value);
                }
            }

            if found_values {
                log::debug!(
                    "Exporting {} = {} (assembled from bits)",
                    export_name,
                    register_value
                );
                exported_values.insert(export_name.clone(), register_value);
                continue;
            }

            // Check raw measurement results as last resort
            // This handles the case where we didn't capture the measurements in indexed form
            let mut measurement_values = Vec::new();

            for (key, &value) in &self.measurement_results {
                if key.starts_with(MEASUREMENT_PREFIX) {
                    if let Some(idx_str) = key.strip_prefix(MEASUREMENT_PREFIX) {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            measurement_values.push((idx, value));
                            log::debug!("Found measurement value {} at index {}", value, idx);
                        }
                    }
                }
            }

            if !measurement_values.is_empty() {
                // Sort by index to maintain correct order
                measurement_values.sort_by_key(|(idx, _)| *idx);
                let combined_value_str: String = measurement_values
                    .iter()
                    .map(|(_, value)| value.to_string())
                    .collect();

                // Convert combined value to a number
                if let Ok(combined_value) = combined_value_str.parse::<u32>() {
                    log::debug!(
                        "Exporting {} = {} (from raw measurements)",
                        export_name,
                        combined_value
                    );
                    exported_values.insert(export_name.clone(), combined_value);
                    continue;
                }
            }

            log::debug!("No values found to export for {}", source_register);
        }

        exported_values
    }
}
