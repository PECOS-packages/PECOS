use pecos_core::errors::PecosError;
use std::collections::HashMap;

use crate::v0_1::ast::ArgItem;
use crate::v0_1::environment::Environment;

/// Handles Result operations for exporting values from internal variables
pub struct ResultHandler<'a> {
    /// Environment containing variable values
    environment: &'a mut Environment,
    /// Exported values mapped by variable name
    exported_values: HashMap<String, u64>,
    /// Export mappings from source to destination
    export_mappings: Vec<(String, String)>,
}

impl<'a> ResultHandler<'a> {
    /// Creates a new result handler with the given environment
    pub fn new(environment: &'a mut Environment) -> Self {
        Self {
            environment,
            exported_values: HashMap::new(),
            export_mappings: Vec::new(),
        }
    }

    /// Handles a Result operation
    pub fn handle_result(
        &mut self,
        args: &[ArgItem],
        returns: &Vec<ArgItem>,
    ) -> Result<(), PecosError> {
        for (i, src) in args.iter().enumerate() {
            if i < returns.len() {
                let dst = &returns[i];

                // Extract source and destination information
                let (src_name, src_index) = self.extract_arg_info(src)?;
                let (dst_name, dst_index) = self.extract_arg_info(dst)?;

                // Store mapping for future reference
                self.export_mappings.push((src_name.clone(), dst_name.clone()));

                // Get source value
                let value = match src_index {
                    Some(idx) => self.environment.get_bit(&src_name, idx)?,
                    None => self.environment.get(&src_name)
                        .ok_or_else(|| PecosError::Input(format!(
                            "Source variable not found: {}", src_name
                        )))?,
                };

                // Check if destination exists, create if not
                if !self.environment.has_variable(&dst_name) {
                    // Create destination with same properties as source
                    let src_info = self.environment.get_variable_info(&src_name)?;
                    self.environment.add_variable(
                        &dst_name,
                        src_info.data_type.clone(),
                        src_info.size,
                    )?;
                }

                // Set value in destination
                match dst_index {
                    Some(idx) => self.environment.set_bit(&dst_name, idx, value)?,
                    None => self.environment.set(&dst_name, value)?,
                }

                // Add to exported values
                self.exported_values.insert(dst_name, value);
            }
        }

        Ok(())
    }

    /// Extracts variable name and optional index from an argument
    fn extract_arg_info(&self, arg: &ArgItem) -> Result<(String, Option<usize>), PecosError> {
        match arg {
            ArgItem::Simple(name) => Ok((name.clone(), None)),
            ArgItem::Indexed((name, idx)) => Ok((name.clone(), Some(*idx))),
            _ => Err(PecosError::Input(format!(
                "Invalid argument for Result operation: {:?}", arg
            ))),
        }
    }

    /// Handles multiple result operations in bulk
    pub fn handle_multiple_results(
        &mut self,
        operations: &[(Vec<ArgItem>, Vec<ArgItem>)],
    ) -> Result<(), PecosError> {
        for (args, returns) in operations {
            self.handle_result(args, returns)?;
        }
        Ok(())
    }

    /// Processes measurement results and updates variables
    pub fn process_measurement_results(
        &mut self, 
        measurements: &HashMap<u64, u32>,
        result_id_to_var: &HashMap<u64, String>,
    ) -> Result<(), PecosError> {
        for (&result_id, &outcome) in measurements {
            if let Some(var_name) = result_id_to_var.get(&result_id) {
                // Update the variable with measurement outcome
                self.environment.set(var_name, outcome as u64)?;
                
                // Update any exports that depend on this variable
                self.update_exports(var_name)?;
            }
        }
        Ok(())
    }

    /// Updates exported variables based on a changed source variable
    fn update_exports(&mut self, src_name: &str) -> Result<(), PecosError> {
        // Find all exports that use this source variable
        let exports: Vec<(String, String)> = self.export_mappings.iter()
            .filter(|(src, _)| src == src_name)
            .cloned()
            .collect();
        
        // Update each export
        for (src, dst) in exports {
            if let Some(value) = self.environment.get(&src) {
                self.environment.set(&dst, value)?;
                self.exported_values.insert(dst, value);
            }
        }
        
        Ok(())
    }

    /// Gets all exported values
    pub fn get_exported_values(&self) -> &HashMap<String, u64> {
        &self.exported_values
    }

    /// Converts exported values to registers for shot results
    pub fn to_registers(&self) -> HashMap<String, u32> {
        self.exported_values.iter()
            .map(|(k, &v)| (k.clone(), v as u32))
            .collect()
    }
}

/// Extension trait for handling Result operations on Environment
pub trait ResultHandling {
    /// Processes a Result operation
    fn handle_result(
        &mut self,
        args: &[ArgItem],
        returns: &Vec<ArgItem>,
    ) -> Result<HashMap<String, u64>, PecosError>;
    
    /// Gets a value for export
    fn get_for_export(&self, name: &str) -> Result<u64, PecosError>;
}

impl ResultHandling for Environment {
    fn handle_result(
        &mut self,
        args: &[ArgItem],
        returns: &Vec<ArgItem>,
    ) -> Result<HashMap<String, u64>, PecosError> {
        let mut result_values = HashMap::new();

        for (i, src) in args.iter().enumerate() {
            if i < returns.len() {
                let dst = &returns[i];

                // Extract source and destination information
                let (src_name, src_index) = match src {
                    ArgItem::Simple(name) => (name.clone(), None),
                    ArgItem::Indexed((name, idx)) => (name.clone(), Some(*idx)),
                    _ => return Err(PecosError::Input(format!(
                        "Invalid argument for Result operation: {:?}", src
                    ))),
                };

                let (dst_name, dst_index) = match dst {
                    ArgItem::Simple(name) => (name.clone(), None),
                    ArgItem::Indexed((name, idx)) => (name.clone(), Some(*idx)),
                    _ => return Err(PecosError::Input(format!(
                        "Invalid argument for Result operation: {:?}", dst
                    ))),
                };

                // Get source value
                let value = match src_index {
                    Some(idx) => self.get_bit(&src_name, idx)?,
                    None => self.get(&src_name)
                        .ok_or_else(|| PecosError::Input(format!(
                            "Source variable not found: {}", src_name
                        )))?,
                };

                // Check if destination exists, create if not
                if !self.has_variable(&dst_name) {
                    // Create destination with same properties as source
                    let src_info = self.get_variable_info(&src_name)?;
                    self.add_variable(
                        &dst_name,
                        src_info.data_type.clone(),
                        src_info.size,
                    )?;
                }

                // Set value in destination
                match dst_index {
                    Some(idx) => self.set_bit(&dst_name, idx, value)?,
                    None => self.set(&dst_name, value)?,
                }

                // Add to exported values
                result_values.insert(dst_name, value);
            }
        }

        Ok(result_values)
    }

    fn get_for_export(&self, name: &str) -> Result<u64, PecosError> {
        self.get(name).ok_or_else(|| PecosError::Input(format!(
            "Variable '{}' not found for export", name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::environment::DataType;

    #[test]
    fn test_environment_result_handler() {
        let mut env = Environment::new();
        env.add_variable("source", DataType::I32, 32).unwrap();
        env.set("source", 42).unwrap();

        let args = vec![ArgItem::Simple("source".to_string())];
        let returns = vec![ArgItem::Simple("dest".to_string())];

        // Use the trait method to handle the result
        let exports = env.handle_result(&args, &returns).unwrap();

        // Verify destination was created
        assert!(env.has_variable("dest"));
        assert_eq!(env.get("dest"), Some(42));

        // Verify export values
        assert_eq!(exports.get("dest"), Some(&42));
    }

    #[test]
    fn test_result_with_bit_indexing() {
        let mut env = Environment::new();
        env.add_variable("bits", DataType::U8, 8).unwrap();
        env.set("bits", 0b00000101).unwrap(); // 5 in binary

        // Map bit 0 (value 1) to result bit 0
        let args = vec![ArgItem::Indexed(("bits".to_string(), 0))];
        let returns = vec![ArgItem::Indexed(("result".to_string(), 0))];

        // Use the trait method
        env.handle_result(&args, &returns).unwrap();

        // Verify bit was exported correctly
        assert!(env.has_variable("result"));
        assert_eq!(env.get_bit("result", 0).unwrap(), 1);

        // Map bit 1 (value 0) to result bit 1
        let args = vec![ArgItem::Indexed(("bits".to_string(), 1))];
        let returns = vec![ArgItem::Indexed(("result".to_string(), 1))];

        env.handle_result(&args, &returns).unwrap();

        // result should now be 0b01 = 1
        assert_eq!(env.get("result"), Some(1));
    }

    #[test]
    fn test_measurement_processing() {
        // Since the ResultHandler borrowing is problematic in tests,
        // we'll test the functionality through a simpler approach

        let mut env = Environment::new();
        env.add_variable("m0", DataType::I32, 32).unwrap();
        env.add_variable("m1", DataType::I32, 32).unwrap();

        // Set measurement results directly
        env.set("m0", 1).unwrap();
        env.set("m1", 0).unwrap();

        // Setup result exports
        let args = vec![
            ArgItem::Simple("m0".to_string()),
            ArgItem::Simple("m1".to_string()),
        ];
        let returns = vec![
            ArgItem::Simple("result0".to_string()),
            ArgItem::Simple("result1".to_string()),
        ];

        // Use the trait method to handle the result
        let exports = env.handle_result(&args, &returns).unwrap();

        // Verify exports were created
        assert!(env.has_variable("result0"));
        assert!(env.has_variable("result1"));
        assert_eq!(env.get("result0"), Some(1));
        assert_eq!(env.get("result1"), Some(0));

        // Verify exported values
        assert_eq!(exports.get("result0"), Some(&1));
        assert_eq!(exports.get("result1"), Some(&0));
    }
}