use pecos_core::errors::PecosError;
use std::fmt::Debug;

/// Trait for foreign object implementations in QASM
pub trait ForeignObject: Debug + Send + Sync {
    /// Clone the foreign object
    fn clone_box(&self) -> Box<dyn ForeignObject>;

    /// Initialize object before running a series of simulations
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    fn init(&mut self) -> Result<(), PecosError>;

    /// Create new instance/internal state for a new shot
    ///
    /// # Errors
    /// Returns an error if instance creation fails.
    fn new_instance(&mut self) -> Result<(), PecosError>;

    /// Execute a function given a list of arguments
    ///
    /// # Errors
    /// Returns an error if the function does not exist or execution fails.
    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError>;
}

/// Dummy foreign object for when no foreign object is needed
#[derive(Debug, Clone)]
pub struct DummyForeignObject {}

impl DummyForeignObject {
    /// Create a new dummy foreign object
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DummyForeignObject {
    fn default() -> Self {
        Self::new()
    }
}

impl ForeignObject for DummyForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        Box::new(Self::default())
    }

    fn init(&mut self) -> Result<(), PecosError> {
        Ok(())
    }

    fn new_instance(&mut self) -> Result<(), PecosError> {
        Ok(())
    }

    fn exec(&mut self, func_name: &str, _args: &[i64]) -> Result<Vec<i64>, PecosError> {
        Err(PecosError::Input(format!(
            "Dummy foreign object cannot execute function: {func_name}"
        )))
    }
}
