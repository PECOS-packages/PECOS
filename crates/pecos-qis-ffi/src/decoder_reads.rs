//! Shot-local observations. No owned data or lock crosses a blocking native read.
use pecos_qis_ffi_types::{
    MAX_RUNTIME_READ_NAME_BYTES, MAX_RUNTIME_READ_TOTAL_NAME_BYTES, MAX_RUNTIME_READS,
    RuntimeObservedRead,
};
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct DecoderReads {
    reads: Vec<RuntimeObservedRead>,
    next: BTreeMap<String, u64>,
    name_bytes: usize,
    pending: Option<RuntimeObservedRead>,
}

impl DecoderReads {
    fn prepare(&mut self, name: String, source_id: u64) -> Result<(), String> {
        if self.pending.is_some() {
            return Err("decoder read already pending".into());
        }
        if name.is_empty()
            || name.len() > MAX_RUNTIME_READ_NAME_BYTES
            || name.chars().any(|c| c < ' ')
        {
            return Err("invalid decoder read name".into());
        }
        if self.reads.len() >= MAX_RUNTIME_READS {
            return Err("decoder read count limit exceeded".into());
        }
        if self.name_bytes + name.len() > MAX_RUNTIME_READ_TOTAL_NAME_BYTES {
            return Err("decoder read name byte limit exceeded".into());
        }
        self.name_bytes += name.len();
        let next = self.next.entry(name.clone()).or_default();
        self.pending = Some(RuntimeObservedRead {
            name,
            occurrence: *next,
            source_id,
            value: false,
        });
        *next += 1;
        Ok(())
    }

    fn commit(&mut self, value: u64) -> Result<(), String> {
        if value > 1 {
            return Err("native Boolean getter returned a nonbinary value".into());
        }
        let mut read = self
            .pending
            .take()
            .ok_or("decoder read has no pending observation")?;
        read.value = value != 0;
        self.reads.push(read);
        Ok(())
    }

    fn take(&mut self) -> Result<Vec<RuntimeObservedRead>, String> {
        if self.pending.is_some() {
            return Err("cannot decode with an unfinished observed read".into());
        }
        Ok(std::mem::take(self).reads)
    }
}

fn with_reads<T>(f: impl FnOnce(&mut DecoderReads) -> Result<T, String>) -> Result<T, String> {
    let ctx = crate::get_execution_context().ok_or("no execution context")?;
    let mut reads = unsafe { &*ctx }
        .decoder_reads
        .lock()
        .map_err(|_| "poisoned decoder reads")?;
    f(&mut reads)
}

pub(crate) fn prepare_read(name: String, source: u64) -> Result<(), String> {
    with_reads(|reads| reads.prepare(name, source))
}
pub(crate) fn commit_read(value: u64) -> Result<(), String> {
    with_reads(|reads| reads.commit(value))
}
pub(crate) fn take_interval() -> Result<Vec<RuntimeObservedRead>, String> {
    with_reads(DecoderReads::take)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intervals_snapshot_reused_sources_and_reset_occurrences() {
        let mut reads = DecoderReads::default();
        for source in [0, 0, 7] {
            reads.prepare("read".into(), source).unwrap();
            reads.commit(1).unwrap();
        }
        let captured = reads.take().unwrap();
        assert_eq!(
            captured
                .iter()
                .map(|r| (r.occurrence, r.source_id))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 0), (2, 7)]
        );
        reads.prepare("read".into(), 8).unwrap();
        reads.commit(0).unwrap();
        assert_eq!(reads.take().unwrap()[0].occurrence, 0);
        assert_eq!(reads.take().unwrap(), Vec::<RuntimeObservedRead>::new());
    }

    #[test]
    fn pending_reads_cannot_be_published_and_reset_discards_them() {
        let context = crate::ExecutionContext::new();
        context
            .decoder_reads
            .lock()
            .unwrap()
            .prepare("pending".into(), 0)
            .unwrap();
        assert!(context.decoder_reads.lock().unwrap().take().is_err());
        assert!(context.decoder_reads.lock().unwrap().commit(2).is_err());
        context.reset_outputs();
        let mut reads = context.decoder_reads.lock().unwrap();
        assert_eq!(reads.take().unwrap(), Vec::<RuntimeObservedRead>::new());
        reads.prepare("pending".into(), 0).unwrap();
        reads.commit(1).unwrap();
        assert_eq!(reads.take().unwrap()[0].occurrence, 0);
    }

    #[test]
    fn bounds_fail_before_the_native_getter_is_entered() {
        let mut reads = DecoderReads::default();
        for name in [String::new(), "a".repeat(257), "bad\0name".into()] {
            assert!(reads.prepare(name, 0).is_err());
        }
        for _ in 0..MAX_RUNTIME_READS {
            reads.prepare("r".into(), 0).unwrap();
            reads.commit(0).unwrap();
        }
        assert!(
            reads
                .prepare("r".into(), 0)
                .unwrap_err()
                .contains("count limit")
        );
        assert_eq!(reads.take().unwrap().len(), MAX_RUNTIME_READS);
        for _ in 0..MAX_RUNTIME_READ_TOTAL_NAME_BYTES / MAX_RUNTIME_READ_NAME_BYTES {
            reads
                .prepare("n".repeat(MAX_RUNTIME_READ_NAME_BYTES), 0)
                .unwrap();
            reads.commit(0).unwrap();
        }
        assert!(
            reads
                .prepare("r".into(), 0)
                .unwrap_err()
                .contains("byte limit")
        );
    }
}
