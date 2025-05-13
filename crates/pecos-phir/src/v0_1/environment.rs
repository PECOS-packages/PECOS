use pecos_core::errors::PecosError;
use std::collections::HashMap;

/// Represents the data type of a variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    /// Signed 8-bit integer
    I8,
    /// Signed 16-bit integer
    I16,
    /// Signed 32-bit integer
    I32,
    /// Signed 64-bit integer
    I64,
    /// Unsigned 8-bit integer
    U8,
    /// Unsigned 16-bit integer
    U16,
    /// Unsigned 32-bit integer
    U32,
    /// Unsigned 64-bit integer
    U64,
    /// Boolean value
    Bool,
    /// Quantum bits (qubits)
    Qubits,
}

impl DataType {
    /// Creates a DataType from a string representation
    pub fn from_str(s: &str) -> Result<Self, PecosError> {
        match s {
            "i8" => Ok(DataType::I8),
            "i16" => Ok(DataType::I16),
            "i32" => Ok(DataType::I32),
            "i64" => Ok(DataType::I64),
            "u8" => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),
            "bool" => Ok(DataType::Bool),
            "qubits" => Ok(DataType::Qubits),
            _ => Err(PecosError::Input(format!("Unsupported data type: {}", s))),
        }
    }

    /// Returns the bit width of the data type
    pub fn bit_width(&self) -> usize {
        match self {
            DataType::I8 | DataType::U8 => 8,
            DataType::I16 | DataType::U16 => 16,
            DataType::I32 | DataType::U32 => 32,
            DataType::I64 | DataType::U64 => 64,
            DataType::Bool => 1,
            DataType::Qubits => 0, // Qubits don't have a fixed bit width
        }
    }

    /// Checks if the data type is signed
    pub fn is_signed(&self) -> bool {
        matches!(self, DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64)
    }

    /// Applies type constraints to a value based on the bit width and signedness
    pub fn constrain_value(&self, value: u64) -> u64 {
        match self {
            DataType::I8 => (value as i8) as u64,
            DataType::I16 => (value as i16) as u64,
            DataType::I32 => (value as i32) as u64,
            DataType::I64 => (value as i64) as u64,
            DataType::U8 => value & 0xFF,
            DataType::U16 => value & 0xFFFF,
            DataType::U32 => value & 0xFFFFFFFF,
            DataType::U64 => value,
            DataType::Bool => value & 1,
            DataType::Qubits => value, // Qubits don't have a fixed bit width
        }
    }
}

/// Metadata for a variable
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Name of the variable
    pub name: String,
    /// Data type of the variable
    pub data_type: DataType,
    /// Size of the variable (number of elements)
    pub size: usize,
}

/// Environment for storing variables with efficient access
#[derive(Debug, Clone)]
pub struct Environment {
    /// Values of all variables (stored as u64 for uniformity)
    values: Vec<u64>,
    /// Maps variable names to indices in the values vector
    name_to_index: HashMap<String, usize>,
    /// Metadata for each variable
    metadata: Vec<VariableInfo>,
}

impl Environment {
    /// Creates a new empty environment
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            name_to_index: HashMap::new(),
            metadata: Vec::new(),
        }
    }

    /// Resets all variable values to zero while keeping their definitions
    pub fn reset_values(&mut self) {
        for value in &mut self.values {
            *value = 0;
        }
    }

    /// Adds a new variable to the environment
    pub fn add_variable(&mut self, name: &str, data_type: DataType, size: usize) -> Result<(), PecosError> {
        if self.name_to_index.contains_key(name) {
            return Err(PecosError::Input(format!(
                "Variable '{}' already exists", name
            )));
        }

        let index = self.values.len();
        self.name_to_index.insert(name.to_string(), index);
        self.values.push(0); // Initialize with zero
        self.metadata.push(VariableInfo {
            name: name.to_string(),
            data_type,
            size,
        });

        Ok(())
    }

    /// Checks if a variable exists in the environment
    pub fn has_variable(&self, name: &str) -> bool {
        self.name_to_index.contains_key(name)
    }

    /// Gets the value of a variable
    pub fn get(&self, name: &str) -> Option<u64> {
        self.name_to_index.get(name).map(|&idx| self.values[idx])
    }

    /// Sets the value of a variable, applying type constraints
    pub fn set(&mut self, name: &str, value: u64) -> Result<(), PecosError> {
        if let Some(&idx) = self.name_to_index.get(name) {
            // Apply constraints based on data type
            let data_type = &self.metadata[idx].data_type;
            let constrained_value = data_type.constrain_value(value);
            self.values[idx] = constrained_value;
            Ok(())
        } else {
            Err(PecosError::Input(format!(
                "Variable '{}' not found", name
            )))
        }
    }

    /// Gets metadata for a variable
    pub fn get_variable_info(&self, name: &str) -> Result<&VariableInfo, PecosError> {
        if let Some(&idx) = self.name_to_index.get(name) {
            Ok(&self.metadata[idx])
        } else {
            Err(PecosError::Input(format!(
                "Variable '{}' not found", name
            )))
        }
    }

    /// Gets metadata for a variable as Option
    pub fn get_variable_info_opt(&self, name: &str) -> Option<&VariableInfo> {
        self.name_to_index.get(name).map(|&idx| &self.metadata[idx])
    }

    /// Gets a specific bit from a variable
    pub fn get_bit(&self, var_name: &str, bit_index: usize) -> Result<u64, PecosError> {
        let value = self.get(var_name)
            .ok_or_else(|| PecosError::Input(format!(
                "Variable '{}' not found", var_name
            )))?;
        
        // Check bit index is in range
        let var_index = *self.name_to_index.get(var_name).unwrap();
        let size = self.metadata[var_index].size;
        
        if bit_index >= size {
            return Err(PecosError::Input(format!(
                "Bit index {} out of range for variable '{}' with size {}", 
                bit_index, var_name, size
            )));
        }
        
        // Extract the bit
        Ok((value >> bit_index) & 1)
    }
    
    /// Sets a specific bit in a variable
    pub fn set_bit(&mut self, var_name: &str, bit_index: usize, bit_value: u64) -> Result<(), PecosError> {
        // Get current value
        let var_index = *self.name_to_index.get(var_name)
            .ok_or_else(|| PecosError::Input(format!(
                "Variable '{}' not found", var_name
            )))?;
        
        let value = self.values[var_index];
        
        // Check bit index is in range
        let size = self.metadata[var_index].size;
        if bit_index >= size {
            return Err(PecosError::Input(format!(
                "Bit index {} out of range for variable '{}' with size {}", 
                bit_index, var_name, size
            )));
        }
        
        // Update the bit
        let mask = 1u64 << bit_index;
        let new_value = if bit_value & 1 == 1 {
            value | mask  // Set bit
        } else {
            value & !mask  // Clear bit
        };
        
        // Set the new value with proper type constraints
        let data_type = &self.metadata[var_index].data_type;
        self.values[var_index] = data_type.constrain_value(new_value);
        Ok(())
    }

    /// Gets all variable names in the environment
    pub fn get_variable_names(&self) -> Vec<String> {
        self.metadata.iter().map(|info| info.name.clone()).collect()
    }

    /// Gets all variables of a specific type
    pub fn get_variables_of_type(&self, data_type: DataType) -> Vec<&VariableInfo> {
        self.metadata.iter()
            .filter(|info| info.data_type == data_type)
            .collect()
    }

    /// Gets all variables in the environment
    pub fn get_all_variables(&self) -> &[VariableInfo] {
        &self.metadata
    }

    /// Gets all measurement result variables and their values
    pub fn get_measurement_results(&self) -> HashMap<String, u64> {
        let mut results = HashMap::new();
        for info in &self.metadata {
            if let Some(value) = self.get(&info.name) {
                results.insert(info.name.clone(), value);
            }
        }
        results
    }

    /// Gets the total number of qubits in the environment
    pub fn count_qubits(&self) -> usize {
        self.get_variables_of_type(DataType::Qubits)
            .iter()
            .map(|info| info.size)
            .sum()
    }

    /// Returns the total number of variables in the environment
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Checks if the environment is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_basic_operations() {
        let mut env = Environment::new();
        
        // Add variables
        env.add_variable("x", DataType::I32, 32).unwrap();
        env.add_variable("y", DataType::U8, 8).unwrap();
        
        // Set values
        env.set("x", 42).unwrap();
        env.set("y", 255).unwrap();
        
        // Get values
        assert_eq!(env.get("x"), Some(42));
        assert_eq!(env.get("y"), Some(255));
        
        // Check variable existence
        assert!(env.has_variable("x"));
        assert!(!env.has_variable("z"));
    }

    #[test]
    fn test_environment_type_constraints() {
        let mut env = Environment::new();
        
        // Add variables with different types
        env.add_variable("i8_var", DataType::I8, 8).unwrap();
        env.add_variable("u8_var", DataType::U8, 8).unwrap();
        
        // Test i8 constraints (-128 to 127)
        env.set("i8_var", 127).unwrap();
        assert_eq!(env.get("i8_var"), Some(127));
        
        env.set("i8_var", 128).unwrap(); // Should wrap to -128
        assert_eq!(env.get("i8_var"), Some(0xFFFFFFFFFFFFFF80)); // -128 as u64
        
        // Test u8 constraints (0 to 255)
        env.set("u8_var", 255).unwrap();
        assert_eq!(env.get("u8_var"), Some(255));
        
        env.set("u8_var", 256).unwrap(); // Should be masked to 0
        assert_eq!(env.get("u8_var"), Some(0));
    }

    #[test]
    fn test_environment_bit_operations() {
        let mut env = Environment::new();
        
        // Add variable
        env.add_variable("bits", DataType::U8, 8).unwrap();
        env.set("bits", 0).unwrap();
        
        // Set bits
        env.set_bit("bits", 0, 1).unwrap(); // Set bit 0
        env.set_bit("bits", 2, 1).unwrap(); // Set bit 2
        
        // Should have value 0b101 = 5
        assert_eq!(env.get("bits"), Some(5));
        
        // Get bits
        assert_eq!(env.get_bit("bits", 0).unwrap(), 1);
        assert_eq!(env.get_bit("bits", 1).unwrap(), 0);
        assert_eq!(env.get_bit("bits", 2).unwrap(), 1);
        
        // Clear a bit
        env.set_bit("bits", 0, 0).unwrap();
        
        // Should have value 0b100 = 4
        assert_eq!(env.get("bits"), Some(4));
    }
}