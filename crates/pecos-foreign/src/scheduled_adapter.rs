//! Experimental native normalizers for the existing scheduled noise controller.
//!
//! The matching C contract is `include/pecos_scheduled_adapter.h`. This interface
//! deliberately exposes neither the simulator nor noise RNG or measurement values.
use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::{
    Gate, GateType,
    runtime_frame::ShotContext,
    scheduled_events::{
        ScheduledBatchAdapter, ScheduledCrosstalk, ScheduledEventBatch, ScheduledEventNoise,
        ScheduledEventOp, ScheduledGateBuffer,
    },
    scheduled_frame::ScheduledNoise,
};
use std::{
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::{Arc, Mutex},
};

const ABI: u32 = 1;
const CAPABILITIES: u64 = 3;
/// Maximum opaque configuration size copied into a factory.
pub const MAX_CONFIG_BYTES: usize = 1024 * 1024;
fn error(message: impl Into<String>) -> PecosError {
    PecosError::Processing(message.into())
}

#[repr(C)]
struct Operation {
    kind: u32,
    reserved: u32,
    words: [u64; 5],
    payload: *const u8,
    payload_len: usize,
}
#[repr(C)]
struct Measurement {
    operation_index: u64,
    runtime_result: u64,
    program_result: u64,
}
#[repr(C)]
struct BatchView {
    runtime_shot_id: u64,
    batch_index: u64,
    start_nanos: u64,
    duration_nanos: u64,
    operation_count: u64,
    measurement_count: u64,
    context: *mut c_void,
    read_operation: unsafe extern "C" fn(*mut c_void, u64, *mut Operation) -> i32,
    read_measurement: unsafe extern "C" fn(*mut c_void, u64, *mut Measurement) -> i32,
}
#[repr(C)]
struct Sink {
    context: *mut c_void,
    emit_gate: unsafe extern "C" fn(*mut c_void, *const u64) -> i32,
    emit_source_gate: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    emit_crosstalk: unsafe extern "C" fn(*mut c_void, u32, *const u64, usize) -> i32,
}
type Create = unsafe extern "C" fn(*const u8, usize, u64, u64, u64, *mut *mut c_void) -> i32;
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Descriptor {
    abi_version: u32,
    struct_size: u32,
    capabilities: u64,
    reserved: u64,
    create: Option<Create>,
    validate: Option<unsafe extern "C" fn(*const c_void, *const BatchView) -> i32>,
    translate: Option<unsafe extern "C" fn(*mut c_void, *const BatchView, *const Sink) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
}
type Init = unsafe extern "C" fn(u32, usize, *mut Descriptor) -> i32;
#[derive(Clone, Copy)]
struct Functions {
    create: Create,
    validate: unsafe extern "C" fn(*const c_void, *const BatchView) -> i32,
    translate: unsafe extern "C" fn(*mut c_void, *const BatchView, *const Sink) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void),
}
struct Loaded {
    functions: Functions,
    _library: libloading::Library,
}
/// Cloneable immutable factory. Each created adapter owns independent native state.
#[derive(Clone)]
pub struct NativeScheduledAdapter {
    loaded: Arc<Loaded>,
    config: Arc<[u8]>,
}
impl NativeScheduledAdapter {
    /// Load a trusted normalizer. It is not a runtime or full error-model plugin.
    ///
    /// # Safety
    /// The library must satisfy the C header's pointer, ownership, threading,
    /// side-effect-free validation and no-unwind contracts, including constructors
    /// run by the OS loader. Native plugins are not sandboxed.
    /// # Errors
    /// Rejects load/symbol failures, incompatible descriptors and oversized config.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self, PecosError> {
        if config.len() > MAX_CONFIG_BYTES {
            return Err(error("native adapter configuration limit"));
        }
        // SAFETY: the caller accepts the trusted library contract.
        let library: libloading::Library = unsafe {
            #[cfg(unix)]
            {
                libloading::os::unix::Library::open(
                    Some(path),
                    libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL,
                )
                .map(Into::into)
            }
            #[cfg(not(unix))]
            {
                libloading::Library::new(path)
            }
        }
        .map_err(|e| error(format!("native adapter load: {e}")))?;
        let init = unsafe { library.get::<Init>(b"pecos_scheduled_adapter_init\0") }
            .map_err(|e| error(format!("native adapter symbol: {e}")))?;
        let mut descriptor = Descriptor::default();
        let status = unsafe { init(ABI, size_of::<Descriptor>(), &raw mut descriptor) };
        if status != 0 {
            return Err(error(format!("native adapter descriptor status {status}")));
        }
        if descriptor.abi_version != ABI
            || usize::try_from(descriptor.struct_size).ok() != Some(size_of::<Descriptor>())
            || descriptor.capabilities != CAPABILITIES
            || descriptor.reserved != 0
            || descriptor.create.is_none()
            || descriptor.validate.is_none()
            || descriptor.translate.is_none()
            || descriptor.destroy.is_none()
        {
            return Err(error("incompatible native adapter descriptor"));
        }
        let missing = || error("missing native adapter callback");
        let functions = Functions {
            create: descriptor.create.ok_or_else(missing)?,
            validate: descriptor.validate.ok_or_else(missing)?,
            translate: descriptor.translate.ok_or_else(missing)?,
            destroy: descriptor.destroy.ok_or_else(missing)?,
        };
        Ok(Self {
            loaded: Arc::new(Loaded {
                functions,
                _library: library,
            }),
            config: config.into(),
        })
    }
    /// Attach to the existing checked scheduled-noise execution path.
    #[must_use]
    pub fn with_noise(self, profile: impl Into<ScheduledNoise>) -> ScheduledEventNoise {
        ScheduledEventNoise::new(profile, move |context| self.create(context))
    }
    /// Create an independent adapter for one host shot.
    /// # Errors
    /// Rejects unrepresentable context, failed creation and null successful handles.
    pub fn create(
        &self,
        context: ShotContext,
    ) -> Result<Box<dyn ScheduledBatchAdapter>, PecosError> {
        let worker = u64::try_from(context.worker).map_err(|_| error("worker ID overflow"))?;
        let shot = u64::try_from(context.shot).map_err(|_| error("shot ID overflow"))?;
        let mut handle = std::ptr::null_mut();
        let d = self.loaded.functions;
        let status = unsafe {
            (d.create)(
                self.config.as_ptr(),
                self.config.len(),
                context.run,
                worker,
                shot,
                &raw mut handle,
            )
        };
        if status != 0 || handle.is_null() {
            if !handle.is_null() {
                unsafe { (d.destroy)(handle) };
            }
            return Err(error(format!(
                "native adapter create status {status}, null handle {}",
                handle.is_null()
            )));
        }
        Ok(Box::new(Session {
            loaded: self.loaded.clone(),
            state: Mutex::new(State {
                handle,
                failed: false,
            }),
        }))
    }
}
struct State {
    handle: *mut c_void,
    failed: bool,
}
// SAFETY: the accepted plugin contract permits moving a session across threads.
// The mutex serializes every access; independent sessions may run concurrently.
unsafe impl Send for State {}
struct Session {
    loaded: Arc<Loaded>,
    state: Mutex<State>,
}
impl Drop for Session {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // The retained library outlives this call, even for poisoned sessions.
        unsafe { (self.loaded.functions.destroy)(state.handle) };
    }
}
struct Call<'a, 'b> {
    batch: &'a ScheduledEventBatch,
    output: Option<&'a mut ScheduledGateBuffer<'b>>,
    failure: Option<PecosError>,
}
// All host callbacks share a synchronous call-scoped state. The trusted plugin
// must not retain it, invoke concurrently, or recursively invoke callbacks.
unsafe fn callback(
    context: *mut c_void,
    f: impl FnOnce(&mut Call<'_, '_>) -> Result<(), PecosError>,
) -> i32 {
    if context.is_null() {
        return 1;
    }
    let state = unsafe { &mut *context.cast::<Call<'_, '_>>() };
    if state.failure.is_some() {
        return 1;
    }
    match catch_unwind(AssertUnwindSafe(|| f(state))) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            state.failure = Some(e);
            1
        }
        Err(_) => {
            state.failure = Some(error("native adapter host callback panicked"));
            1
        }
    }
}
fn index(value: u64) -> Result<usize, PecosError> {
    usize::try_from(value).map_err(|_| error("native adapter index overflow"))
}
fn source(batch: &ScheduledEventBatch, i: u64) -> Result<&ScheduledEventOp, PecosError> {
    batch
        .operations
        .get(index(i)?)
        .ok_or_else(|| error("native adapter operation index out of bounds"))
}
fn gate_words(g: &Gate) -> Result<[u64; 5], PecosError> {
    g.validate().map_err(error)?;
    let (code, n, a) = match g.gate_type {
        GateType::RXY1Q => (1, 1, 2),
        GateType::RZ => (2, 1, 1),
        GateType::RZZ => (3, 2, 1),
        GateType::RXYXY2Q => (4, 2, 2),
        GateType::PZ => (5, 1, 0),
        GateType::MZ => (6, 1, 0),
        GateType::MeasureZLeaked => (7, 1, 0),
        gate if gate.is_detection() => {
            return Err(error(
                pecos_engines::scheduled_frame::unsupported_detection_message(gate),
            ));
        }
        _ => return Err(error("unsupported native adapter source gate")),
    };
    if g.qubits.len() != n
        || g.angles.len() != a
        || !g.params.is_empty()
        || !g.meas_ids.is_empty()
        || g.channel.is_some()
        || (n == 2 && g.qubits[0] == g.qubits[1])
    {
        return Err(error("invalid native adapter source fields"));
    }
    Ok([
        code,
        u64::try_from(g.qubits[0].0).map_err(|_| error("target overflow"))?,
        if n == 2 {
            u64::try_from(g.qubits[1].0).map_err(|_| error("target overflow"))?
        } else {
            0
        },
        g.angles.first().map_or(0, Angle64::fraction),
        g.angles.get(1).map_or(0, Angle64::fraction),
    ])
}
fn words_gate(w: [u64; 5]) -> Result<Gate, PecosError> {
    let [code, q0, q1, a, b] = w;
    if (!matches!(code, 3 | 4) && q1 != 0)
        || (code >= 5 && a != 0)
        || (!matches!(code, 1 | 4) && b != 0)
    {
        return Err(error("nonzero unused native adapter gate fields"));
    }
    let q0 = index(q0)?;
    let q1 = index(q1)?;
    let a = Angle64::new(a);
    let b = Angle64::new(b);
    Ok(match code {
        1 => Gate::rxy1q(a, b, &[q0]),
        2 => Gate::rz(a, &[q0]),
        3 => Gate::rzz(a, &[(q0, q1)]),
        4 => Gate::rxyxy2q(a, b, &[(q0, q1)]),
        5 => Gate::pz(&[q0]),
        6 => Gate::mz(&[q0]),
        7 => Gate::measure_z_leaked(&[q0]),
        _ => return Err(error("unknown native adapter gate code")),
    })
}
unsafe extern "C" fn read_operation(context: *mut c_void, i: u64, out: *mut Operation) -> i32 {
    unsafe {
        callback(context, |s| {
            if out.is_null() {
                return Err(error("null native adapter operation output"));
            }
            let mut op = Operation {
                kind: 0,
                reserved: 0,
                words: [0; 5],
                payload: std::ptr::null(),
                payload_len: 0,
            };
            match source(s.batch, i)? {
                ScheduledEventOp::Gate(g) => op.words = gate_words(g)?,
                ScheduledEventOp::Custom { tag, payload } => {
                    op.kind = 1;
                    op.words[0] = *tag;
                    op.payload = payload.as_ptr();
                    op.payload_len = payload.len();
                }
            }
            out.write(op);
            Ok(())
        })
    }
}
unsafe extern "C" fn read_measurement(context: *mut c_void, i: u64, out: *mut Measurement) -> i32 {
    unsafe {
        callback(context, |s| {
            if out.is_null() {
                return Err(error("null native adapter measurement output"));
            }
            let m = s
                .batch
                .measurements
                .get(index(i)?)
                .ok_or_else(|| error("native adapter measurement index out of bounds"))?;
            out.write(Measurement {
                operation_index: u64::try_from(m.operation_index)
                    .map_err(|_| error("measurement index overflow"))?,
                runtime_result: m.runtime_result,
                program_result: m.program_result,
            });
            Ok(())
        })
    }
}
unsafe extern "C" fn emit_gate(context: *mut c_void, words: *const u64) -> i32 {
    unsafe {
        callback(context, |s| {
            let out = s
                .output
                .as_mut()
                .ok_or_else(|| error("no native adapter output sink"))?;
            if out.remaining_bytes() < 40 {
                return Err(error("native adapter expansion limit"));
            }
            if words.is_null() {
                return Err(error("null native adapter gate"));
            }
            let w = words.cast::<[u64; 5]>().read();
            out.push(words_gate(w)?)
        })
    }
}
unsafe extern "C" fn emit_source_gate(context: *mut c_void, i: u64) -> i32 {
    unsafe {
        callback(context, |s| {
            let out = s
                .output
                .as_mut()
                .ok_or_else(|| error("no native adapter output sink"))?;
            if out.remaining_bytes() < 40 {
                return Err(error("native adapter expansion limit"));
            }
            match source(s.batch, i)? {
                ScheduledEventOp::Gate(g) => out.push(g.as_ref().clone()),
                ScheduledEventOp::Custom { .. } => {
                    Err(error("custom event is not an original gate"))
                }
            }
        })
    }
}
unsafe extern "C" fn emit_crosstalk(
    context: *mut c_void,
    kind: u32,
    ids: *const u64,
    len: usize,
) -> i32 {
    unsafe {
        callback(context, |s| {
            let out = s
                .output
                .as_mut()
                .ok_or_else(|| error("no native adapter output sink"))?;
            if kind > 1
                || len > out.remaining_bytes().saturating_sub(40) / 8
                || out.remaining_bytes() < 40
                || len > isize::MAX as usize / 8
            {
                return Err(error("native adapter exposure kind or size limit"));
            }
            if len > 0 && ids.is_null() {
                return Err(error("null native adapter spectator list"));
            }
            let ids = if len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(ids, len)
            };
            let ids = ids
                .iter()
                .map(|q| index(*q))
                .collect::<Result<Vec<_>, _>>()?;
            out.push_crosstalk(if kind == 0 {
                ScheduledCrosstalk::Local(ids)
            } else {
                ScheduledCrosstalk::GlobalExcluding(ids)
            })
        })
    }
}
impl Session {
    fn call(
        &self,
        batch: &ScheduledEventBatch,
        output: Option<&mut ScheduledGateBuffer<'_>>,
    ) -> Result<(), PecosError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| error("poisoned native adapter lock"))?;
        if state.failed {
            return Err(error("native adapter session failed; reset host"));
        }
        let translating = output.is_some();
        let mut call = Call {
            batch,
            output,
            failure: None,
        };
        let context = (&raw mut call).cast::<c_void>();
        let view = BatchView {
            runtime_shot_id: batch.runtime_shot_id,
            batch_index: batch.batch_index,
            start_nanos: batch.start_nanos,
            duration_nanos: batch.duration_nanos,
            operation_count: u64::try_from(batch.operations.len())
                .map_err(|_| error("operation count overflow"))?,
            measurement_count: u64::try_from(batch.measurements.len())
                .map_err(|_| error("measurement count overflow"))?,
            context,
            read_operation,
            read_measurement,
        };
        let sink = Sink {
            context,
            emit_gate,
            emit_source_gate,
            emit_crosstalk,
        };
        let d = self.loaded.functions;
        let status = unsafe {
            if translating {
                (d.translate)(state.handle, &raw const view, &raw const sink)
            } else {
                (d.validate)(state.handle, &raw const view)
            }
        };
        let phase = if translating { "translate" } else { "validate" };
        if let Some(e) = call.failure {
            state.failed = true;
            return Err(error(format!(
                "native adapter {phase} status {status}; host callback: {e}"
            )));
        }
        if status != 0 {
            state.failed = true;
            return Err(error(format!("native adapter {phase} status {status}")));
        }
        Ok(())
    }
}
impl ScheduledBatchAdapter for Session {
    fn validate(&self, batch: &ScheduledEventBatch) -> Result<(), PecosError> {
        self.call(batch, None)
    }
    fn translate(
        &mut self,
        batch: &ScheduledEventBatch,
        output: &mut ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError> {
        self.call(batch, Some(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_engines::scheduled_frame::{TimedBatch, encode_timed_batches};

    #[test]
    fn native_adapter_rejects_detection() {
        for kind in [
            GateType::DetectLeaked,
            GateType::DetectLost,
            GateType::DetectLeakedLost,
        ] {
            let error = gate_words(&Gate::simple(kind, vec![0.into()]))
                .unwrap_err()
                .to_string();
            assert!(error.contains(&format!("{kind} is a non-collapsing detect gate")));
        }
    }

    #[test]
    fn ffi_words_match_transport_exactly_including_low_angle_bits() {
        let gates = vec![
            Gate::rxy1q(Angle64::new(1), Angle64::new(u64::MAX), &[0]),
            Gate::rz(Angle64::new(3), &[1]),
            Gate::rzz(Angle64::new(5), &[(0, 1)]),
            Gate::rxyxy2q(Angle64::new(7), Angle64::new(11), &[(0, 1)]),
            Gate::pz(&[0]),
            Gate::mz(&[0]),
            Gate::measure_z_leaked(&[1]),
        ];
        let message = encode_timed_batches(&[TimedBatch {
            runtime_shot_id: 0,
            batch_index: 0,
            start_nanos: 0,
            duration_nanos: 0,
            gates: gates.clone(),
        }])
        .unwrap();
        for (i, g) in gates.iter().enumerate() {
            let words = gate_words(g).unwrap();
            let expected: Vec<_> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
            assert_eq!(&message.as_bytes()[56 + i * 40..96 + i * 40], expected);
            assert_eq!(words_gate(words).unwrap(), *g);
        }
    }
    #[test]
    fn host_callback_panic_is_latched_and_not_reentered() {
        let batch = ScheduledEventBatch {
            runtime_shot_id: 0,
            batch_index: 0,
            start_nanos: 0,
            duration_nanos: 0,
            operations: vec![],
            measurements: vec![],
        };
        let mut call = Call {
            batch: &batch,
            output: None,
            failure: None,
        };
        let ctx = (&raw mut call).cast();
        assert_eq!(
            unsafe { callback(ctx, |_| panic!("synthetic callback failure")) },
            1
        );
        assert!(call.failure.is_some());
        assert_eq!(unsafe { callback(ctx, |_| panic!("must not enter")) }, 1);
    }
}
