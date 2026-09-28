//! Owned native scheduler output, before idle insertion or physical interpretation.
//!
//! This is a Rust extraction API, not an executable wire format. An opaque custom
//! event is data, never acknowledgement that its physical effect was simulated.

/// One operation in native emission order. Qubit IDs are runtime physical IDs.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeScheduledOp {
    Rxy {
        qubit_id: u64,
        theta: f64,
        phi: f64,
    },
    Rz {
        qubit_id: u64,
        theta: f64,
    },
    Rzz {
        qubit_id_1: u64,
        qubit_id_2: u64,
        theta: f64,
    },
    Measure {
        qubit_id: u64,
        result_id: u64,
    },
    MeasureLeaked {
        qubit_id: u64,
        result_id: u64,
    },
    Reset {
        qubit_id: u64,
    },
    Rpp {
        qubit_id_1: u64,
        qubit_id_2: u64,
        theta: f64,
        phi: f64,
    },
    Custom {
        tag: usize,
        data: Vec<u8>,
    },
}

/// Measurement identity captured before runtime result mappings can change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledMeasurement {
    /// Index in the containing batch's operation vector.
    pub operation_index: usize,
    /// Future allocated by the native runtime.
    pub runtime_result: u64,
    /// Result expected by the source program.
    pub program_result: usize,
    /// Source measurement requires a leakage-aware result, even if the ABI
    /// emitted an ordinary measurement operation.
    pub leakage_aware: bool,
}

/// One original batch; simultaneous operations are never split or re-timed.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledBatch {
    /// Runtime-local shot identity. Not a host run/worker identity.
    pub runtime_shot_id: u64,
    /// Zero-based batch ordinal across calls within this runtime shot.
    pub batch_index: usize,
    /// Original start time, in integer nanoseconds.
    pub start_time_nanos: u64,
    /// Original duration, in integer nanoseconds; end-time overflow is rejected.
    pub duration_nanos: u64,
    /// Owned operations, including opaque events, in original emission order.
    pub operations: Vec<RuntimeScheduledOp>,
    /// Both result namespaces, indexed back into `operations`.
    pub measurements: Vec<ScheduledMeasurement>,
}

// Fixed extraction budgets per API call. Bound allocations inside callbacks as
// well as retention across batches; callers own returned data after extraction.
#[cfg(feature = "selene")]
pub(crate) const MAX_BATCHES: usize = 64;
#[cfg(feature = "selene")]
pub(crate) const MAX_OPERATIONS: usize = 4096;
#[cfg(feature = "selene")]
pub(crate) const MAX_PAYLOAD_BYTES: usize = 262_144;

#[cfg(feature = "selene")]
#[derive(Default)]
pub(crate) struct ScheduledOutput {
    pub batches: Vec<ScheduledBatch>,
    pub operations: usize,
    pub payload_bytes: usize,
}
