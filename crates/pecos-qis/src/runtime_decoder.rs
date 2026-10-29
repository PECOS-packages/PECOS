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

    /// Decode with the v1 JSON contract. Implementers must validate the context
    /// and schema binding; silently falling back to an unbound call is forbidden.
    ///
    /// # Errors
    /// The default rejects versioned calls so existing sessions cannot ignore them.
    fn decode_v1(
        &mut self,
        _window_id: u64,
        _instance_id: u64,
        _syndrome: &[u8],
        _context: &[u8],
    ) -> Result<u64, PecosError> {
        Err(PecosError::Input(
            "decoder session does not support context v1".into(),
        ))
    }
}

pub(crate) struct PendingDecode {
    pub window: u64,
    pub instance: u64,
    pub syndrome: Vec<u8>,
    pub context: Option<Vec<u8>>,
}

/// Immutable factory configuration may be shared by workers; sessions may not.
pub type RuntimeDecoderFactory =
    Arc<dyn Fn() -> Result<Box<dyn RuntimeDecoderSession>, PecosError> + Send + Sync>;
