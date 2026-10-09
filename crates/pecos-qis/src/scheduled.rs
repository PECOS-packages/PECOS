//! Owned native scheduler output, before idle insertion or physical interpretation.
//!
//! This is a Rust extraction API, not an executable wire format. An opaque custom
//! event is data, never acknowledgement that its physical effect was simulated.

/// One operation in native emission order. Qubit IDs are runtime physical IDs.
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeScheduledOp {
    /// Rotation about an axis in the XY plane.
    Rxy {
        /// Runtime physical qubit.
        qubit_id: u64,
        /// Rotation angle in radians.
        theta: f64,
        /// XY-plane axis angle in radians.
        phi: f64,
    },
    /// Rotation about the Z axis.
    Rz {
        /// Runtime physical qubit.
        qubit_id: u64,
        /// Rotation angle in radians.
        theta: f64,
    },
    /// Two-qubit ZZ rotation.
    Rzz {
        /// First runtime physical qubit.
        qubit_id_1: u64,
        /// Second runtime physical qubit.
        qubit_id_2: u64,
        /// Rotation angle in radians.
        theta: f64,
    },
    /// Boolean measurement into a native future.
    Measure {
        /// Runtime physical qubit.
        qubit_id: u64,
        /// Native future identifier.
        result_id: u64,
    },
    /// Leakage-aware measurement into a native future.
    MeasureLeaked {
        /// Runtime physical qubit.
        qubit_id: u64,
        /// Native future identifier.
        result_id: u64,
    },
    /// Reset a physical qubit.
    Reset {
        /// Runtime physical qubit.
        qubit_id: u64,
    },
    /// Two-qubit rotation about a shared axis in the XY plane.
    Rpp {
        /// First runtime physical qubit.
        qubit_id_1: u64,
        /// Second runtime physical qubit.
        qubit_id_2: u64,
        /// Rotation angle in radians.
        theta: f64,
        /// XY-plane axis angle in radians.
        phi: f64,
    },
    /// Uninterpreted native event.
    Custom {
        /// Plugin-defined native event tag.
        tag: usize,
        /// Owned bytes copied from the callback.
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
    /// Result expected by the source program; may repeat when a slot is re-measured.
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

// Bounds on each indivisible native callback batch, not the entire drain.
// Forced drains also have aggregate limits, including empty native batches.
#[cfg(feature = "selene")]
pub(crate) const MAX_OPERATIONS: usize = 4096;
#[cfg(feature = "selene")]
pub(crate) const MAX_PAYLOAD_BYTES: usize = 262_144;

#[cfg(feature = "selene")]
#[derive(Default)]
pub(crate) struct ScheduledOutput {
    pub batches: Vec<ScheduledBatch>,
    pub drain_budget: Option<crate::scheduled_transport::EventBudget>,
}
