//! Shot-local host decoding, separate from physical runtime scheduling.
use pecos_core::errors::PecosError;
use std::sync::Arc;

/// A fresh session is created on first use in each shot, on its host worker.
/// Implementations receive only program-supplied observations and public IDs.
/// They must reject unsupported models, malformed inputs and failed decodes.
pub trait RuntimeDecoderSession: Send + Sync {
    /// Return a frame mask whose meaning is defined by the supplied model mapping.
    /// The program applies it through ordinary runtime operations or classical logic.
    ///
    /// # Errors
    /// Return an error for an unknown window, invalid contract or decoding failure.
    fn decode(
        &mut self,
        window_id: u64,
        instance_id: u64,
        syndrome: &[u8],
    ) -> Result<u64, PecosError>;
}

/// Immutable factory configuration may be shared by workers; sessions may not.
pub type RuntimeDecoderFactory =
    Arc<dyn Fn() -> Result<Box<dyn RuntimeDecoderSession>, PecosError> + Send + Sync>;
