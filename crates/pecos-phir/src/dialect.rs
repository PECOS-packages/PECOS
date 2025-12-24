/*!
Dialect system for PHIR

This module provides MLIR-style dialect registration and management,
allowing for extensible operations and types.
*/

use crate::error::Result;
use crate::ops::CustomOp;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, RwLock};

/// Dialect definition
pub trait Dialect: Send + Sync {
    /// Get the namespace for this dialect (e.g., "qec", "pulse", "chem")
    fn namespace(&self) -> &'static str;

    /// Get description of the dialect
    fn description(&self) -> &'static str;

    /// Initialize the dialect (register operations, types, etc.)
    ///
    /// # Errors
    ///
    /// Returns an error if dialect initialization fails
    fn initialize(&self, registry: &mut DialectRegistry) -> Result<()>;

    /// Verify an operation from this dialect
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is invalid
    fn verify_operation(&self, _op: &CustomOp) -> Result<()> {
        // Default: no additional verification
        Ok(())
    }

    /// Get operation traits for a custom operation
    fn get_operation_traits(&self, _op_name: &str) -> Vec<crate::traits::OpTrait> {
        // Default: no traits
        Vec::new()
    }
}

/// Registry for dialects and their operations
pub struct DialectRegistry {
    /// Registered dialects
    dialects: BTreeMap<String, Arc<dyn Dialect>>,
    /// Operation definitions by dialect
    operations: BTreeMap<String, BTreeMap<String, OperationDef>>,
    /// Type definitions by dialect
    types: BTreeMap<String, BTreeMap<String, TypeDef>>,
}

/// Operation definition
#[derive(Clone)]
pub struct OperationDef {
    /// Operation name
    pub name: String,
    /// Description
    pub description: String,
    /// Number of operands (-1 for variadic)
    pub num_operands: i32,
    /// Number of results (-1 for variadic)
    pub num_results: i32,
    /// Number of regions
    pub num_regions: usize,
    /// Operation traits
    pub traits: Vec<crate::traits::OpTrait>,
}

/// Type definition
#[derive(Clone)]
pub struct TypeDef {
    /// Type name
    pub name: String,
    /// Description
    pub description: String,
    /// Type parameters
    pub parameters: Vec<TypeParameter>,
}

/// Type parameter definition
#[derive(Clone)]
pub struct TypeParameter {
    /// Parameter name
    pub name: String,
    /// Parameter kind
    pub kind: ParameterKind,
}

#[derive(Clone)]
pub enum ParameterKind {
    /// Integer parameter
    Integer,
    /// Type parameter
    Type,
    /// String parameter
    String,
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DialectRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            dialects: BTreeMap::new(),
            operations: BTreeMap::new(),
            types: BTreeMap::new(),
        }
    }

    /// Register a dialect
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dialect is already registered
    /// - Dialect initialization fails
    pub fn register_dialect<D: Dialect + 'static>(&mut self, dialect: D) -> Result<()> {
        let namespace = dialect.namespace().to_string();

        if self.dialects.contains_key(&namespace) {
            return Err(crate::error::PhirError::Internal(format!(
                "Dialect '{namespace}' already registered"
            )));
        }

        let dialect = Arc::new(dialect);
        self.dialects.insert(namespace.clone(), dialect.clone());

        // Initialize the dialect
        dialect.initialize(self)?;

        Ok(())
    }

    /// Register an operation for a dialect
    ///
    /// # Errors
    ///
    /// Currently always succeeds, but returns Result for future extensibility
    pub fn register_operation(&mut self, dialect: &str, op: OperationDef) -> Result<()> {
        self.operations
            .entry(dialect.to_string())
            .or_default()
            .insert(op.name.clone(), op);
        Ok(())
    }

    /// Register a type for a dialect
    ///
    /// # Errors
    ///
    /// Currently always succeeds, but returns Result for future extensibility
    pub fn register_type(&mut self, dialect: &str, ty: TypeDef) -> Result<()> {
        self.types
            .entry(dialect.to_string())
            .or_default()
            .insert(ty.name.clone(), ty);
        Ok(())
    }

    /// Get a registered dialect
    #[must_use]
    pub fn get_dialect(&self, namespace: &str) -> Option<Arc<dyn Dialect>> {
        self.dialects.get(namespace).cloned()
    }

    /// Get operation definition
    #[must_use]
    pub fn get_operation(&self, dialect: &str, name: &str) -> Option<&OperationDef> {
        self.operations.get(dialect).and_then(|ops| ops.get(name))
    }

    /// Verify a custom operation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The dialect is not registered
    /// - The operation is unknown
    /// - Operation parameters are invalid
    pub fn verify_custom_operation(&self, op: &CustomOp) -> Result<()> {
        // Get the dialect
        let dialect = self.get_dialect(&op.dialect).ok_or_else(|| {
            crate::error::PhirError::Validation(Box::new(
                crate::error::ValidationError::UnknownDialect(op.dialect.clone()),
            ))
        })?;

        // Check if operation is registered
        let _op_def = self.get_operation(&op.dialect, &op.name).ok_or_else(|| {
            crate::error::PhirError::Validation(Box::new(
                crate::error::ValidationError::UnknownOperation(format!(
                    "{}.{}",
                    op.dialect, op.name
                )),
            ))
        })?;

        // Let the dialect verify
        dialect.verify_operation(op)?;

        Ok(())
    }
}

// Global dialect registry
static GLOBAL_REGISTRY: LazyLock<Arc<RwLock<DialectRegistry>>> =
    LazyLock::new(|| Arc::new(RwLock::new(DialectRegistry::new())));

/// Register a dialect globally
///
/// # Errors
///
/// Returns an error if the dialect is already registered or if dialect initialization fails
///
/// # Panics
///
/// Panics if the global registry lock is poisoned
pub fn register_dialect<D: Dialect + 'static>(dialect: D) -> Result<()> {
    GLOBAL_REGISTRY.write().unwrap().register_dialect(dialect)
}

/// Get the global dialect registry
#[must_use]
pub fn get_registry() -> Arc<RwLock<DialectRegistry>> {
    GLOBAL_REGISTRY.clone()
}

// Example dialects

/// Quantum Error Correction dialect
pub struct QECDialect;

impl Dialect for QECDialect {
    fn namespace(&self) -> &'static str {
        "qec"
    }

    fn description(&self) -> &'static str {
        "Quantum Error Correction operations and types"
    }

    fn initialize(&self, registry: &mut DialectRegistry) -> Result<()> {
        use crate::traits::OpTrait;

        // Register QEC operations
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "syndrome_extract".to_string(),
                description: "Extract error syndrome".to_string(),
                num_operands: -1, // Variable number of qubits
                num_results: -1,  // Variable number of syndrome bits
                num_regions: 0,
                traits: vec![],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "decode".to_string(),
                description: "Decode syndrome to get corrections".to_string(),
                num_operands: -1,
                num_results: -1,
                num_regions: 0,
                traits: vec![OpTrait::NoSideEffect],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "logical_gate".to_string(),
                description: "Logical gate on encoded qubits".to_string(),
                num_operands: -1,
                num_results: -1,
                num_regions: 0,
                traits: vec![OpTrait::PureQuantum],
            },
        )?;

        // Register QEC types
        registry.register_type(
            self.namespace(),
            TypeDef {
                name: "stabilizer_code".to_string(),
                description: "Stabilizer error correcting code".to_string(),
                parameters: vec![
                    TypeParameter {
                        name: "n".to_string(),
                        kind: ParameterKind::Integer,
                    },
                    TypeParameter {
                        name: "k".to_string(),
                        kind: ParameterKind::Integer,
                    },
                    TypeParameter {
                        name: "d".to_string(),
                        kind: ParameterKind::Integer,
                    },
                ],
            },
        )?;

        Ok(())
    }
}

/// Pulse-level control dialect
pub struct PulseDialect;

impl Dialect for PulseDialect {
    fn namespace(&self) -> &'static str {
        "pulse"
    }

    fn description(&self) -> &'static str {
        "Pulse-level quantum control operations"
    }

    fn initialize(&self, registry: &mut DialectRegistry) -> Result<()> {
        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "play".to_string(),
                description: "Play a pulse waveform".to_string(),
                num_operands: 2, // waveform, channel
                num_results: 0,
                num_regions: 0,
                traits: vec![],
            },
        )?;

        registry.register_operation(
            self.namespace(),
            OperationDef {
                name: "capture".to_string(),
                description: "Capture signal from readout".to_string(),
                num_operands: 1, // channel
                num_results: 1,  // signal
                num_regions: 0,
                traits: vec![],
            },
        )?;

        Ok(())
    }
}

/// Initialize standard dialects
///
/// # Errors
///
/// Returns an error if any dialect registration fails
pub fn init_standard_dialects() -> Result<()> {
    register_dialect(QECDialect)?;
    register_dialect(PulseDialect)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_registration() {
        let mut registry = DialectRegistry::new();

        let qec = QECDialect;
        assert!(registry.register_dialect(qec).is_ok());

        // Should not be able to register twice
        let qec2 = QECDialect;
        assert!(registry.register_dialect(qec2).is_err());

        // Should be able to get the dialect
        assert!(registry.get_dialect("qec").is_some());
        assert!(registry.get_dialect("unknown").is_none());
    }

    #[test]
    fn test_operation_registration() {
        let mut registry = DialectRegistry::new();
        registry.register_dialect(QECDialect).unwrap();

        // Check that operations were registered
        assert!(registry.get_operation("qec", "syndrome_extract").is_some());
        assert!(registry.get_operation("qec", "decode").is_some());
        assert!(registry.get_operation("qec", "unknown").is_none());
    }
}
