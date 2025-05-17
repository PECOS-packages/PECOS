use pecos_core::errors::PecosError;
use std::any::Any;
use std::fmt::Debug;

/// Trait for foreign object implementations
pub trait ForeignObject: Debug + Send + Sync {
    /// Clone the foreign object
    fn clone_box(&self) -> Box<dyn ForeignObject>;
    /// Initialize object before running a series of simulations
    fn init(&mut self) -> Result<(), PecosError>;

    /// Create new instance/internal state
    fn new_instance(&mut self) -> Result<(), PecosError>;

    /// Get a list of function names available from the object
    fn get_funcs(&self) -> Vec<String>;

    /// Execute a function given a list of arguments
    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError>;

    /// Cleanup resources
    fn teardown(&mut self) {}

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get as Any for downcasting (mutable)
    fn as_any_mut(&mut self) -> &mut dyn Any;
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

    fn get_funcs(&self) -> Vec<String> {
        vec![]
    }

    fn exec(&mut self, func_name: &str, _args: &[i64]) -> Result<Vec<i64>, PecosError> {
        Err(PecosError::Input(format!(
            "Dummy foreign object cannot execute function: {func_name}"
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
