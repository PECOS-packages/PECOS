//! Bounded, outcome-free runtime emission history for a live decoder provider.
use crate::runtime::{Result, RuntimeError};
use pecos_qis_ffi_types::QuantumOp;

pub const MAX_DECODER_BATCHES: usize = 4096;
pub const MAX_DECODER_BATCH_OPERATIONS: usize = 65536;

/// One native batch after checked normalization, source-result routing and idle
/// insertion. No future handles, outcomes, noise samples or seeds are retained.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecoderBatch {
    pub batch_index: usize,
    pub start_time_nanos: u64,
    pub duration_nanos: u64,
    pub normalization_applied: bool,
    pub operations: Vec<QuantumOp>,
}

/// Delta of emitted batches since shot start or the previous decoder call.
/// `QisEngine` publishes it only after executing the ready prefix and delivering
/// its feedback. This is not a flush or a certificate that future work is known.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecoderBatchPrefix {
    pub version: u8,
    pub runtime_shot_id: u64,
    pub first_batch_index: usize,
    pub next_batch_index: usize,
    pub batches: Vec<DecoderBatch>,
}

#[derive(Default)]
pub(crate) struct DecoderBatchBuffer {
    pub enabled: bool,
    first: usize,
    operations: usize,
    batches: Vec<DecoderBatch>,
}

impl DecoderBatchBuffer {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }
    pub fn reset(&mut self) {
        *self = Self {
            enabled: self.enabled,
            ..Self::default()
        };
    }

    pub fn push(&mut self, batch: DecoderBatch) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if batch.batch_index != self.first + self.batches.len()
            || self.batches.len() >= MAX_DECODER_BATCHES
            || batch.operations.len() > MAX_DECODER_BATCH_OPERATIONS - self.operations
        {
            return Err(RuntimeError::ExecutionError(
                "decoder batch provenance sequence or interval limit exceeded".into(),
            ));
        }
        self.operations += batch.operations.len();
        self.batches.push(batch);
        Ok(())
    }

    pub fn take(&mut self, shot: u64, next: usize) -> Result<DecoderBatchPrefix> {
        if !self.enabled || next != self.first + self.batches.len() {
            return Err(RuntimeError::ExecutionError(
                "decoder batch provenance unavailable".into(),
            ));
        }
        let prefix = DecoderBatchPrefix {
            version: 1,
            runtime_shot_id: shot,
            first_batch_index: self.first,
            next_batch_index: next,
            batches: std::mem::take(&mut self.batches),
        };
        self.first = next;
        self.operations = 0;
        Ok(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(index: usize, count: usize) -> DecoderBatch {
        DecoderBatch {
            batch_index: index,
            start_time_nanos: 0,
            duration_nanos: 0,
            normalization_applied: false,
            operations: vec![QuantumOp::Reset(0); count],
        }
    }

    #[test]
    fn decoder_batch_bounds_and_empty_intervals_reset() {
        let mut buffer = DecoderBatchBuffer::new(true);
        buffer.push(batch(0, MAX_DECODER_BATCH_OPERATIONS)).unwrap();
        assert!(buffer.push(batch(1, 1)).is_err());
        assert_eq!(buffer.take(3, 1).unwrap().batches.len(), 1);
        let empty = buffer.take(3, 1).unwrap();
        assert_eq!((empty.first_batch_index, empty.next_batch_index), (1, 1));
        assert!(empty.batches.is_empty());
        buffer.reset();
        for index in 0..MAX_DECODER_BATCHES {
            buffer.push(batch(index, 0)).unwrap();
        }
        assert!(buffer.push(batch(MAX_DECODER_BATCHES, 0)).is_err());
        buffer.reset();
        assert!(buffer.push(batch(1, 0)).is_err());
        buffer.push(batch(0, 1)).unwrap();
        assert!(buffer.take(4, 2).is_err());
        assert_eq!(buffer.take(4, 1).unwrap().runtime_shot_id, 4);
    }
}
