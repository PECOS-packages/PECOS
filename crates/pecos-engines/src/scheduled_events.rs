//! Mandatory v4 scheduled batches and a restricted, shot-local batch adapter.
//!
//! Adapters normalize complete inputs before quantum execution. They cannot observe
//! measurement outcomes or access simulator/RNG state. This is not an arbitrary
//! execution-time physical-event interface. The admitted noise remains idle-Z.
use crate::noise::{IntoNoiseModel, NoiseModel};
use crate::runtime_frame::{ShotContext, error};
use crate::scheduled_frame::{
    self, MAX_SCHEDULE_BYTES, ScheduledIdleModel, ScheduledIdleZ, TimedBatch,
};
use crate::{ByteMessage, ControlEngine, EngineStage, Gate, GateType};
use pecos_core::{RngManageable, errors::PecosError};
use pecos_random::PecosRng;
use std::{any::Any, collections::BTreeSet, sync::Arc};

/// Maximum source or expanded operations per original batch.
pub const MAX_BATCH_OPERATIONS: usize = 4096;
/// Maximum total opaque payload bytes per original batch.
pub const MAX_BATCH_PAYLOAD: usize = 256 * 1024;

/// Native order, including events which must never silently disappear.
#[derive(Clone, Debug)]
pub enum ScheduledEventOp {
    /// An ordinary gate supported by the checked idle profile.
    Gate(Box<Gate>),
    /// An opaque event, interpreted only by an explicitly installed adapter.
    Custom { tag: u64, payload: Vec<u8> },
}
/// Both measurement namespaces, associated with an original operation position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledResult {
    pub operation_index: usize,
    pub runtime_result: u64,
    pub program_result: u64,
}
/// One original native batch. Adapters may not retime or split this boundary.
#[derive(Clone, Debug)]
pub struct ScheduledEventBatch {
    pub runtime_shot_id: u64,
    pub batch_index: u64,
    pub start_nanos: u64,
    pub duration_nanos: u64,
    pub operations: Vec<ScheduledEventOp>,
    pub measurements: Vec<ScheduledResult>,
}
fn measured(gate: &Gate) -> bool {
    matches!(gate.gate_type, GateType::MZ | GateType::MeasureLeaked)
}
fn validate(batch: &ScheduledEventBatch) -> Result<(), PecosError> {
    if batch.operations.len() > MAX_BATCH_OPERATIONS
        || batch
            .start_nanos
            .checked_add(batch.duration_nanos)
            .is_none()
    {
        return Err(error("scheduled event batch limit or time overflow"));
    }
    let mut payload = 0usize;
    let mut positions = Vec::new();
    for (i, op) in batch.operations.iter().enumerate() {
        match op {
            ScheduledEventOp::Gate(g) => {
                scheduled_frame::fields(g)?;
                if measured(g) {
                    positions.push(i);
                }
            }
            ScheduledEventOp::Custom { payload: bytes, .. } => {
                payload = payload
                    .checked_add(bytes.len())
                    .ok_or_else(|| error("event payload overflow"))?;
                if payload > MAX_BATCH_PAYLOAD {
                    return Err(error("event payload limit"));
                }
            }
        }
    }
    if !positions
        .iter()
        .copied()
        .eq(batch.measurements.iter().map(|m| m.operation_index))
    {
        return Err(error("invalid scheduled event measurement positions"));
    }
    Ok(())
}
fn identities(batches: &[ScheduledEventBatch]) -> Result<(), PecosError> {
    let mut native = BTreeSet::new();
    let mut source = BTreeSet::new();
    for b in batches {
        validate(b)?;
        for m in &b.measurements {
            if !native.insert(m.runtime_result) || !source.insert(m.program_result) {
                return Err(error("duplicate scheduled event measurement identity"));
            }
        }
    }
    Ok(())
}
struct Writer(Vec<u8>);
impl Writer {
    fn bytes(&mut self, bytes: &[u8]) -> Result<(), PecosError> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > MAX_SCHEDULE_BYTES)
        {
            return Err(error("scheduled event transport size limit"));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(|_| error("event transport allocation failed"))?;
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    fn word(&mut self, v: u64) -> Result<(), PecosError> {
        self.bytes(&v.to_le_bytes())
    }
}
/// Encode mandatory v4; ordinary consumers must reject its version.
///
/// # Errors
/// Rejects invalid records, identities, payloads and total size above 64 MiB.
pub fn encode_event_batches(batches: &[ScheduledEventBatch]) -> Result<ByteMessage, PecosError> {
    identities(batches)?;
    let mut w = Writer(Vec::new());
    w.bytes(&[0; 16])?;
    for b in batches {
        let gates = b
            .operations
            .iter()
            .filter_map(|o| match o {
                ScheduledEventOp::Gate(g) => Some(g.as_ref().clone()),
                ScheduledEventOp::Custom { .. } => None,
            })
            .collect();
        let nested = scheduled_frame::encode_timed_batches(&[TimedBatch {
            runtime_shot_id: b.runtime_shot_id,
            batch_index: b.batch_index,
            start_nanos: b.start_nanos,
            duration_nanos: b.duration_nanos,
            gates,
        }])?;
        w.word(nested.as_bytes().len() as u64)?;
        w.bytes(nested.as_bytes())?;
        w.word(
            b.operations
                .iter()
                .filter(|o| matches!(o, ScheduledEventOp::Custom { .. }))
                .count() as u64,
        )?;
        for (i, op) in b.operations.iter().enumerate() {
            if let ScheduledEventOp::Custom { tag, payload } = op {
                w.word(i as u64)?;
                w.word(*tag)?;
                w.word(payload.len() as u64)?;
                w.bytes(payload)?;
            }
        }
        w.word(b.measurements.len() as u64)?;
        for m in &b.measurements {
            w.word(m.operation_index as u64)?;
            w.word(m.runtime_result)?;
            w.word(m.program_result)?;
        }
    }
    w.0[..4].copy_from_slice(&crate::byte_message::protocol::BATCH_MAGIC.to_le_bytes());
    w.0[4] = 4;
    w.0[8..12].copy_from_slice(
        &u32::try_from(batches.len())
            .map_err(|_| error("batch count overflow"))?
            .to_le_bytes(),
    );
    let size = u32::try_from(w.0.len()).map_err(|_| error("transport size overflow"))?;
    w.0[12..16].copy_from_slice(&size.to_le_bytes());
    Ok(ByteMessage::new(&w.0))
}
struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PecosError> {
        if n > self.0.len() {
            return Err(error("truncated scheduled event transport"));
        }
        let (part, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(part)
    }
    fn word(&mut self) -> Result<u64, PecosError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("word")))
    }
    fn size(&mut self) -> Result<usize, PecosError> {
        usize::try_from(self.word()?).map_err(|_| error("event size overflow"))
    }
}
/// Decode an entire v4 input without invoking adapters or touching physics.
///
/// # Errors
/// Rejects malformed, unknown-version, oversized or ambiguous records.
pub fn decode_event_batches(input: &ByteMessage) -> Result<Vec<ScheduledEventBatch>, PecosError> {
    let bytes = input.as_bytes();
    if bytes.len() < 16
        || bytes.len() > MAX_SCHEDULE_BYTES
        || bytes[..4] != crate::byte_message::protocol::BATCH_MAGIC.to_le_bytes()
        || bytes[4..8] != [4, 0, 0, 0]
        || u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize != bytes.len()
    {
        return Err(error("invalid scheduled event header/version/length"));
    }
    let count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if count > (bytes.len() - 16) / 80 {
        return Err(error("invalid scheduled event batch count"));
    }
    let mut r = Reader(&bytes[16..]);
    let mut result = Vec::new();
    for _ in 0..count {
        let n = r.size()?;
        // A nested v3 frame holds exactly one bounded batch, without opaque data.
        if n > 56 + MAX_BATCH_OPERATIONS * 40 {
            return Err(error("event gate count limit"));
        }
        let mut nested = scheduled_frame::decode(&ByteMessage::new(r.take(n)?))?;
        if nested.len() != 1 {
            return Err(error("expected one nested scheduled batch"));
        }
        let b = nested.remove(0);
        let events = r.size()?;
        if events > MAX_BATCH_OPERATIONS || b.gates.len() > MAX_BATCH_OPERATIONS - events {
            return Err(error("event operation count limit"));
        }
        let mut opaque = Vec::new();
        let mut total = 0usize;
        let mut previous = None;
        for _ in 0..events {
            let pos = r.size()?;
            let tag = r.word()?;
            let len = r.size()?;
            total = total
                .checked_add(len)
                .ok_or_else(|| error("event payload overflow"))?;
            if total > MAX_BATCH_PAYLOAD
                || pos >= b.gates.len() + events
                || previous.is_some_and(|p| pos <= p)
            {
                return Err(error("invalid event position or payload limit"));
            }
            previous = Some(pos);
            opaque.push((pos, tag, r.take(len)?.to_vec()));
        }
        let mut gates = b.gates.into_iter();
        let mut opaque = opaque.into_iter().peekable();
        let mut operations = Vec::new();
        for i in 0..gates.len() + events {
            if opaque.peek().is_some_and(|e| e.0 == i) {
                let (_, tag, payload) = opaque.next().ok_or_else(|| error("missing event"))?;
                operations.push(ScheduledEventOp::Custom { tag, payload });
            } else {
                operations.push(ScheduledEventOp::Gate(Box::new(
                    gates.next().ok_or_else(|| error("missing gate"))?,
                )));
            }
        }
        let n = r.size()?;
        if n > operations.len() {
            return Err(error("event measurement count limit"));
        }
        let mut measurements = Vec::new();
        for _ in 0..n {
            measurements.push(ScheduledResult {
                operation_index: r.size()?,
                runtime_result: r.word()?,
                program_result: r.word()?,
            });
        }
        result.push(ScheduledEventBatch {
            runtime_shot_id: b.runtime_shot_id,
            batch_index: b.batch_index,
            start_nanos: b.start_nanos,
            duration_nanos: b.duration_nanos,
            operations,
            measurements,
        });
    }
    if !r.0.is_empty() {
        return Err(error("trailing scheduled event data"));
    }
    identities(&result)?;
    Ok(result)
}

/// Bounded gate output for one original batch; no nested events or measurements
/// beyond those in the source are permitted by the execution owner.
#[derive(Default)]
pub struct ScheduledGateBuffer {
    gates: Vec<Gate>,
    failure: Option<String>,
}
impl ScheduledGateBuffer {
    /// Append an admitted ordinary gate.
    /// # Errors
    /// Rejects unsupported fields/gates and expansion above the per-batch bound.
    pub fn push(&mut self, gate: Gate) -> Result<(), PecosError> {
        if let Some(message) = &self.failure {
            return Err(error(message));
        }
        let admitted = scheduled_frame::fields(&gate).map(|_| ()).and_then(|()| {
            if self.gates.len() == MAX_BATCH_OPERATIONS {
                Err(error("event expansion limit"))
            } else {
                Ok(())
            }
        });
        if let Err(failure) = admitted {
            self.failure = Some(failure.to_string());
            return Err(failure);
        }
        self.gates.push(gate);
        Ok(())
    }
}
/// Trusted deterministic normalizer, freshly constructed for each host shot.
///
/// Validation must inspect every event schema without mutating state. Translation
/// may update shot-local classical state; it runs before any quantum execution of
/// this input. Neither method receives outcomes, RNG or simulator access. Arbitrary
/// execution-time effects, leakage and crosstalk are not supported by this profile.
/// Implementations are responsible for bounding their own retained state.
pub trait ScheduledBatchAdapter: Send + Sync {
    /// Validate a complete batch without side effects.
    /// # Errors
    /// Reject every unsupported schema, tag, field or semantic requirement.
    fn validate(&self, batch: &ScheduledEventBatch) -> Result<(), PecosError>;
    /// Normalize one batch, retaining the order and targets of measurements.
    /// # Errors
    /// Any error poisons the owner until whole-host reset; no rollback is promised.
    fn translate(
        &mut self,
        batch: &ScheduledEventBatch,
        output: &mut ScheduledGateBuffer,
    ) -> Result<(), PecosError>;
}
type Factory =
    Arc<dyn Fn(ShotContext) -> Result<Box<dyn ScheduledBatchAdapter>, PecosError> + Send + Sync>;
/// Explicit opt-in to mandatory v4 with the same bounded idle-Z physics as v3.
#[derive(Clone)]
pub struct ScheduledEventIdleZ {
    profile: ScheduledIdleZ,
    factory: Factory,
}
impl ScheduledEventIdleZ {
    /// The factory must return an independent session; captured configuration may
    /// be shared, but mutable adapter state must not be shared between shots.
    pub fn new<F>(profile: ScheduledIdleZ, factory: F) -> Self
    where
        F: Fn(ShotContext) -> Result<Box<dyn ScheduledBatchAdapter>, PecosError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            profile,
            factory: Arc::new(factory),
        }
    }
}
impl IntoNoiseModel for ScheduledEventIdleZ {
    fn into_noise_model(self) -> Box<dyn NoiseModel> {
        Box::new(ScheduledEventModel {
            inner: self.profile.build_model(),
            factory: self.factory,
            adapter: None,
        })
    }
}
pub(crate) struct ScheduledEventModel {
    pub(crate) inner: ScheduledIdleModel,
    factory: Factory,
    adapter: Option<Box<dyn ScheduledBatchAdapter>>,
}
impl Clone for ScheduledEventModel {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            factory: self.factory.clone(),
            adapter: None,
        }
    }
}
impl ScheduledEventModel {
    pub(crate) fn normalize(
        &mut self,
        batches: &[ScheduledEventBatch],
        context: ShotContext,
    ) -> Result<Vec<TimedBatch>, PecosError> {
        if self.adapter.is_none() {
            self.adapter = Some((self.factory)(context)?);
        }
        let adapter = self.adapter.as_mut().expect("initialized adapter");
        for batch in batches {
            adapter.validate(batch)?;
        }
        let mut translated = Vec::new();
        let mut size = 16usize;
        for batch in batches {
            let mut output = ScheduledGateBuffer::default();
            adapter.translate(batch, &mut output)?;
            if let Some(message) = output.failure {
                return Err(error(&message));
            }
            let original: Vec<_> = batch
                .operations
                .iter()
                .filter_map(|op| match op {
                    ScheduledEventOp::Gate(g) if measured(g) => Some((g.gate_type, &g.qubits)),
                    _ => None,
                })
                .collect();
            if !original.into_iter().eq(output
                .gates
                .iter()
                .filter(|g| measured(g))
                .map(|g| (g.gate_type, &g.qubits)))
            {
                return Err(error("adapter changed measurement order, kind or targets"));
            }
            size = size
                .checked_add(40 + output.gates.len() * 40)
                .ok_or_else(|| error("expanded schedule overflow"))?;
            if size > MAX_SCHEDULE_BYTES {
                return Err(error("expanded schedule limit"));
            }
            translated.push(TimedBatch {
                runtime_shot_id: batch.runtime_shot_id,
                batch_index: batch.batch_index,
                start_nanos: batch.start_nanos,
                duration_nanos: batch.duration_nanos,
                gates: output.gates,
            });
        }
        Ok(translated)
    }
}
impl ControlEngine for ScheduledEventModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(
        &mut self,
        _: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        Err(error("scheduled event capability requires execution owner"))
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.inner.continue_processing(reply)
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.adapter = None;
        self.inner.reset()
    }
}
impl NoiseModel for ScheduledEventModel {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl RngManageable for ScheduledEventModel {
    type Rng = PecosRng;
    fn rng(&self) -> &PecosRng {
        self.inner.rng()
    }
    fn rng_mut(&mut self) -> &mut PecosRng {
        self.inner.rng_mut()
    }
    fn set_rng(&mut self, rng: PecosRng) {
        self.inner.set_rng(rng);
    }
}
