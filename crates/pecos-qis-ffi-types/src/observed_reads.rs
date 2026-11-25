//! Bounded runtime-getter observations, separate from caller context bytes.
use std::collections::BTreeMap;

pub const MAX_RUNTIME_READS: usize = 1 << 16;
pub const MAX_RUNTIME_READ_NAME_BYTES: usize = 256;
pub const MAX_RUNTIME_READ_TOTAL_NAME_BYTES: usize = 1 << 20;

/// A successful named native read. Source IDs are dense shot-local measurement
/// occurrences, never native handles, simulator fault identities or RNG state.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuntimeObservedRead {
    pub name: String,
    pub occurrence: u64,
    pub source_id: u64,
    pub value: bool,
}

/// Validate bounds, names, source identities and per-name occurrence order.
///
/// # Errors
/// Reject malformed observations before invoking a host session.
pub fn validate_observed_reads(reads: &[RuntimeObservedRead]) -> Result<(), String> {
    if reads.len() > MAX_RUNTIME_READS {
        return Err("decoder read count limit exceeded".into());
    }
    let mut bytes = 0usize;
    let mut next = BTreeMap::new();
    for read in reads {
        if read.name.is_empty()
            || read.name.len() > MAX_RUNTIME_READ_NAME_BYTES
            || read.name.chars().any(|c| c < ' ')
            || read.source_id >= crate::RUNTIME_REQUEST_RESULT_ID as u64
        {
            return Err("invalid decoder read name or measurement source".into());
        }
        bytes += read.name.len(); // Each term and the number of terms are bounded.
        if bytes > MAX_RUNTIME_READ_TOTAL_NAME_BYTES {
            return Err("decoder read name byte limit exceeded".into());
        }
        let count = next.entry(&read.name).or_insert(0u64);
        if read.occurrence != *count {
            return Err("decoder read occurrences must be contiguous per name".into());
        }
        *count += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_validation_rejects_bad_order_and_reserved_source() {
        let first = RuntimeObservedRead {
            name: "read".into(),
            occurrence: 0,
            source_id: 0,
            value: true,
        };
        let mut second = first.clone();
        assert!(validate_observed_reads(&[first.clone(), second.clone()]).is_err());
        second.occurrence = 1;
        validate_observed_reads(&[first, second.clone()]).unwrap();
        second.occurrence = 0;
        second.source_id = crate::RUNTIME_REQUEST_RESULT_ID as u64;
        assert!(validate_observed_reads(&[second]).is_err());
        validate_observed_reads(&[]).unwrap();
    }
}
