//! Shot-local host decoding, separate from physical runtime scheduling.
use pecos_core::errors::PecosError;
pub use pecos_qis_ffi_types::RuntimeObservedRead;
use std::sync::Arc;

/// A fresh session is created on first use in each shot, on its host worker.
/// Implementations receive program observations, public identities and optional
/// outcome-free runtime-emission data.
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

    /// Decode observed data with the executed runtime-emitted interval.
    /// # Errors
    /// Opt-in sessions must implement this method; never fall back silently.
    fn decode_observed_with_batches_v1(
        &mut self,
        _window_id: u64,
        _instance_id: u64,
        _syndrome: &[u8],
        _context: &[u8],
        _reads: &[RuntimeObservedRead],
        _prefix: &crate::decoder_batches::DecoderBatchPrefix,
    ) -> Result<u64, PecosError> {
        Err(PecosError::Input(
            "decoder session does not support batch provenance".into(),
        ))
    }

    /// Check observed read sources and parities before decoding. No fallback.
    ///
    /// # Errors
    /// The default rejects sessions without observed-read support.
    fn decode_observed_v1(
        &mut self,
        _window_id: u64,
        _instance_id: u64,
        _syndrome: &[u8],
        _context: &[u8],
        _reads: &[RuntimeObservedRead],
    ) -> Result<u64, PecosError> {
        Err(PecosError::Input(
            "decoder session does not support observed reads".into(),
        ))
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum DecoderInputMode {
    #[default]
    Program,
    RuntimeBatches,
}

pub(crate) struct PendingDecode {
    pub window: u64,
    pub instance: u64,
    pub syndrome: Vec<u8>,
    pub context: Option<Vec<u8>>,
    pub observed_reads: Option<Vec<RuntimeObservedRead>>,
}

/// Immutable factory configuration may be shared by workers; sessions may not.
pub type RuntimeDecoderFactory =
    Arc<dyn Fn() -> Result<Box<dyn RuntimeDecoderSession>, PecosError> + Send + Sync>;
