use dyn_clone::DynClone;
use pecos_core::errors::PecosError;

/// Core engine trait for processing inputs to outputs.
pub trait Engine: DynClone + Send + Sync {
    type Input;
    type Output;

    /// Process a single input.
    ///
    /// # Errors
    /// Returns `PecosError` if processing fails.
    fn process(&mut self, input: Self::Input) -> Result<Self::Output, PecosError>;

    /// Reset engine state for reuse between simulation runs.
    ///
    /// # Errors
    /// Returns `PecosError` if the reset fails.
    fn reset(&mut self) -> Result<(), PecosError>;
}
