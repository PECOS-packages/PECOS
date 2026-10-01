//! Mandatory v4 scheduled batches and a restricted, shot-local batch adapter.
//!
//! Adapters normalize complete inputs before quantum execution. They cannot observe
//! measurement outcomes or access simulator/RNG state. This is not an arbitrary
//! execution-time physical-event interface. The admitted noise is restricted to checked local profiles.
use crate::noise::{IntoNoiseModel, NoiseModel};
use crate::runtime_frame::{ShotContext, error, processing_error};
use crate::scheduled_frame::{
    self, MAX_SCHEDULE_BYTES, PreparedSchedule, ScheduleTimeline, ScheduledIdleModel,
    ScheduledNoise, TimedBatch,
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
impl ScheduledEventBatch {
    /// Original timing and ordinary gates, excluding opaque events.
    #[must_use]
    pub fn source_gates(&self) -> TimedBatch {
        TimedBatch {
            runtime_shot_id: self.runtime_shot_id,
            batch_index: self.batch_index,
            start_nanos: self.start_nanos,
            duration_nanos: self.duration_nanos,
            gates: self
                .operations
                .iter()
                .filter_map(|op| match op {
                    ScheduledEventOp::Gate(g) => Some(g.as_ref().clone()),
                    ScheduledEventOp::Custom { .. } => None,
                })
                .collect(),
        }
    }
}
fn measured(gate: &Gate) -> bool {
    matches!(gate.gate_type, GateType::MZ | GateType::MeasureLeaked)
}
fn validate_batch(batch: &ScheduledEventBatch) -> Result<(), PecosError> {
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
fn validate_batches(batches: &[ScheduledEventBatch]) -> Result<(), PecosError> {
    let mut native = BTreeSet::new();
    let mut source = BTreeSet::new();
    for b in batches {
        validate_batch(b)?;
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
    validate_batches(batches)?;
    let mut w = Writer(Vec::new());
    w.bytes(&[0; 16])?;
    for b in batches {
        let nested = scheduled_frame::encode_timed_batches(&[b.source_gates()])?;
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
        let nested_bytes = r.size()?;
        // A nested v3 frame holds exactly one bounded batch, without opaque data.
        if nested_bytes > 56 + MAX_BATCH_OPERATIONS * 40 {
            return Err(error("event gate count limit"));
        }
        let mut nested = scheduled_frame::decode(&ByteMessage::new(r.take(nested_bytes)?))?;
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
        let measurement_count = r.size()?;
        if measurement_count > operations.len() {
            return Err(error("event measurement count limit"));
        }
        let mut measurements = Vec::new();
        for _ in 0..measurement_count {
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
    validate_batches(&result)?;
    Ok(result)
}

/// Bounded gate output for one original batch; no nested events or measurements
/// beyond those in the source are permitted by the execution owner.
/// This writer borrows owner-held state; adapters cannot construct replacements.
///
/// ```compile_fail
/// use pecos_engines::{Gate, scheduled_events::ScheduledGateBuffer};
/// fn replace(out: &mut ScheduledGateBuffer<'_>) {
///     assert!(out.push(Gate::x(&[0])).is_err());
///     *out = ScheduledGateBuffer::default();
/// }
/// ```
pub struct ScheduledGateBuffer<'a> {
    state: &'a mut GateOutput,
}
#[derive(Default)]
struct GateOutput {
    gates: Vec<Gate>,
    failure: Option<String>,
}
impl ScheduledGateBuffer<'_> {
    /// Append an admitted ordinary gate.
    /// # Errors
    /// Rejects unsupported fields/gates and expansion above the per-batch bound.
    pub fn push(&mut self, gate: Gate) -> Result<(), PecosError> {
        if let Some(message) = &self.state.failure {
            return Err(error(message));
        }
        let admitted = scheduled_frame::fields(&gate).map(|_| ()).and_then(|()| {
            if self.state.gates.len() == MAX_BATCH_OPERATIONS {
                Err(error("event expansion limit"))
            } else {
                Ok(())
            }
        });
        if let Err(failure) = admitted {
            self.state.failure = Some(failure.to_string());
            return Err(failure);
        }
        self.state.gates.push(gate);
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
        output: &mut ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError>;
}
type Factory =
    Arc<dyn Fn(ShotContext) -> Result<Box<dyn ScheduledBatchAdapter>, PecosError> + Send + Sync>;
/// Compatibility alias retaining the original public event-wrapper name.
/// This wraps a supplied profile; unlike the Z-only `ScheduledIdleZ` constructor,
/// the alias does not restrict that profile to Z/RZ channels.
pub type ScheduledEventIdleZ = ScheduledEventNoise;
/// Compatibility name for [`ScheduledEventNoise`], retaining the supplied profile.
pub type ScheduledEventIdleNoise = ScheduledEventNoise;
/// Explicit opt-in to mandatory v4 with the same checked local physics as v3.
#[derive(Clone)]
pub struct ScheduledEventNoise {
    profile: ScheduledNoise,
    factory: Factory,
}
impl ScheduledEventNoise {
    /// The factory must return an independent session; captured configuration may
    /// be shared, but mutable adapter state must not be shared between shots.
    pub fn new<F>(profile: impl Into<ScheduledNoise>, factory: F) -> Self
    where
        F: Fn(ShotContext) -> Result<Box<dyn ScheduledBatchAdapter>, PecosError>
            + Send
            + Sync
            + 'static,
    {
        Self {
            profile: profile.into(),
            factory: Arc::new(factory),
        }
    }
}
impl IntoNoiseModel for ScheduledEventNoise {
    fn into_noise_model(self) -> Box<dyn NoiseModel> {
        let inner = self.profile.build_model();
        Box::new(ScheduledEventModel {
            source_timeline: ScheduleTimeline::new(inner.qubits()),
            inner,
            factory: self.factory,
            adapter: None,
        })
    }
}
pub(crate) struct ScheduledEventModel {
    pub(crate) inner: ScheduledIdleModel,
    source_timeline: ScheduleTimeline,
    factory: Factory,
    adapter: Option<Box<dyn ScheduledBatchAdapter>>,
}
impl Clone for ScheduledEventModel {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            source_timeline: self.source_timeline.clone(),
            factory: self.factory.clone(),
            adapter: None,
        }
    }
}
pub(crate) struct AdmittedEvents {
    batches: Vec<ScheduledEventBatch>,
    source: ScheduleTimeline,
}
impl ScheduledEventModel {
    pub(crate) fn admit(&self, input: &ByteMessage) -> Result<AdmittedEvents, PecosError> {
        // Component-wise clones retain timelines but cannot clone a live adapter.
        if self.adapter.is_none() && self.source_timeline.has_native_shot() {
            return Err(processing_error("cloned event session requires reset"));
        }
        let batches = decode_event_batches(input)?;
        let source = self
            .source_timeline
            .prepare(
                &self.inner.config,
                batches
                    .iter()
                    .map(ScheduledEventBatch::source_gates)
                    .collect(),
            )?
            .timeline;
        Ok(AdmittedEvents { batches, source })
    }
    pub(crate) fn prepare(
        &mut self,
        admitted: AdmittedEvents,
        context: ShotContext,
    ) -> Result<(PreparedSchedule, ScheduleTimeline), PecosError> {
        let normalized = self.normalize(&admitted.batches, context, MAX_SCHEDULE_BYTES)?;
        Ok((self.inner.prepare_batches(normalized)?, admitted.source))
    }
    pub(crate) fn commit(&mut self, prepared: PreparedSchedule, source: ScheduleTimeline) {
        self.inner.commit(prepared);
        self.source_timeline = source;
    }
    fn normalize(
        &mut self,
        batches: &[ScheduledEventBatch],
        context: ShotContext,
        size_limit: usize,
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
            let mut output = GateOutput::default();
            adapter.translate(batch, &mut ScheduledGateBuffer { state: &mut output })?;
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
            size = expanded_size(size, output.gates.len(), size_limit)?;
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
fn expanded_size(current: usize, gates: usize, limit: usize) -> Result<usize, PecosError> {
    let size = gates
        .checked_mul(40)
        .and_then(|bytes| bytes.checked_add(40))
        .and_then(|bytes| current.checked_add(bytes))
        .ok_or_else(|| error("expanded schedule overflow"))?;
    if size > limit {
        return Err(error("expanded schedule limit"));
    }
    Ok(size)
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
        self.inner.reset()?;
        self.source_timeline = ScheduleTimeline::new(self.inner.qubits());
        Ok(())
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_enforces_aggregate_budget_across_batches() {
        struct Expand;
        impl ScheduledBatchAdapter for Expand {
            fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
                Ok(())
            }
            fn translate(
                &mut self,
                _: &ScheduledEventBatch,
                output: &mut ScheduledGateBuffer<'_>,
            ) -> Result<(), PecosError> {
                output.push(Gate::pz(&[0]))
            }
        }
        let batches = (0..2)
            .map(|batch_index| ScheduledEventBatch {
                runtime_shot_id: 1,
                batch_index,
                start_nanos: 0,
                duration_nanos: 0,
                operations: vec![],
                measurements: vec![],
            })
            .collect::<Vec<_>>();
        // Exercise the production normalization loop with a small budget; the
        // real 64 MiB arithmetic boundary is covered separately without a huge
        // expanded Gate allocation. Public admission always uses the fixed limit.
        for (limit, accepted) in [(176, true), (175, false)] {
            let mut noise = ScheduledEventIdleNoise::new(
                crate::scheduled_frame::ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(),
                |_| Ok(Box::new(Expand)),
            )
            .into_noise_model();
            let model = noise
                .as_any_mut()
                .downcast_mut::<ScheduledEventModel>()
                .unwrap();
            let result = model.normalize(
                &batches,
                ShotContext {
                    run: 0,
                    worker: 0,
                    shot: 0,
                },
                limit,
            );
            assert_eq!(result.is_ok(), accepted);
        }
    }
    #[test]
    fn expanded_schedule_size_boundary_and_overflow() {
        assert_eq!(
            expanded_size(MAX_SCHEDULE_BYTES - 80, 1, MAX_SCHEDULE_BYTES).unwrap(),
            MAX_SCHEDULE_BYTES
        );
        assert!(
            expanded_size(MAX_SCHEDULE_BYTES - 79, 1, MAX_SCHEDULE_BYTES)
                .unwrap_err()
                .to_string()
                .contains("expanded schedule limit")
        );
        assert!(expanded_size(16, usize::MAX, MAX_SCHEDULE_BYTES).is_err());
        assert!(expanded_size(usize::MAX, 1, MAX_SCHEDULE_BYTES).is_err());
    }
}
