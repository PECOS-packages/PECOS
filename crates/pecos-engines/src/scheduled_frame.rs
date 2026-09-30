//! Mandatory v3 batches for the checked idle-Z profile. Not a custom-event format.
use crate::noise::{GeneralNoiseModel, GeneralNoiseModelBuilder, IntoNoiseModel, NoiseModel};
use crate::runtime_frame::{error, processing_error};
use crate::{ByteMessage, ControlEngine, EngineStage, Gate, GateType};
use pecos_core::{Angle64, RngManageable, errors::PecosError};
use pecos_random::PecosRng;
use std::{any::Any, collections::BTreeMap};

/// Maximum complete transport size. Extraction itself remains uncapped.
pub const MAX_SCHEDULE_BYTES: usize = 64 * 1024 * 1024;
/// Original native timing and gates. Opaque events cannot be represented.
#[derive(Clone, Debug)]
pub struct TimedBatch {
    /// Runtime-local identity, separate from host run/worker/shot context.
    pub runtime_shot_id: u64,
    /// Consecutive ordinal within the native shot.
    pub batch_index: u64,
    /// Original batch start in nanoseconds.
    pub start_nanos: u64,
    /// Original batch duration in nanoseconds.
    pub duration_nanos: u64,
    /// Operations in native emission order.
    pub gates: Vec<Gate>,
}
/// Checked, immutable profile; gate/readout/leakage/crosstalk noise is disabled.
#[derive(Clone, Debug)]
pub struct ScheduledIdleZ {
    qubits: usize,
    linear: f64,
    sine: f64,
    coherent: f64,
}
impl ScheduledIdleZ {
    /// Rates are inverse seconds (linear) or radians/second (sine and coherent).
    /// # Errors
    /// Rejects capacity outside 1..=16 or nonfinite/negative rates.
    pub fn new(qubits: usize, linear: f64, sine: f64, coherent: f64) -> Result<Self, PecosError> {
        let config = Self {
            qubits,
            linear,
            sine,
            coherent,
        };
        if !(1..=16).contains(&qubits) {
            return Err(error("scheduled capacity must be 1..=16"));
        }
        config.validate_idle(1.0)?;
        Ok(config)
    }
    fn validate_idle(&self, seconds: f64) -> Result<(), PecosError> {
        if [self.linear, self.sine, self.coherent]
            .into_iter()
            .any(|rate| !rate.is_finite() || rate < 0.0 || !(rate * seconds).is_finite())
        {
            return Err(error("invalid scheduled idle rate or duration product"));
        }
        Ok(())
    }
}
impl IntoNoiseModel for ScheduledIdleZ {
    fn into_noise_model(self) -> Box<dyn NoiseModel> {
        Box::new(self.build_model())
    }
}
impl ScheduledIdleZ {
    pub(crate) fn build_model(self) -> ScheduledIdleModel {
        let z = BTreeMap::from([("Z".to_owned(), 1.0)]);
        let rz = BTreeMap::from([("RZ".to_owned(), 1.0)]);
        let inner = GeneralNoiseModelBuilder::new()
            .with_p_idle_linear(self.linear, &z)
            .with_p_idle_sin_squared(self.sine, &z)
            .with_p_idle_coherent(self.coherent, &rz)
            .build();
        ScheduledIdleModel {
            timeline: ScheduleTimeline::new(self.qubits),
            config: self,
            inner,
        }
    }
}
pub(crate) fn fields(g: &Gate) -> Result<(u64, u64, u64, f64, f64), PecosError> {
    g.validate().map_err(|s| error(&s))?;
    let (code, n, angles) = match g.gate_type {
        GateType::RXY1Q => (1, 1, 2),
        GateType::RZ => (2, 1, 1),
        GateType::RZZ => (3, 2, 1),
        GateType::RXYXY2Q => (4, 2, 2),
        // PZ is the only admitted preparation. If adding another (such as PX),
        // revisit the idle-omission rule in ScheduleTimeline::prepare.
        GateType::PZ => (5, 1, 0),
        GateType::MZ => (6, 1, 0),
        GateType::MeasureLeaked => (7, 1, 0),
        _ => return Err(error("unsupported scheduled gate")),
    };
    if g.qubits.len() != n
        || g.angles.len() != angles
        || !g.params.is_empty()
        || !g.meas_ids.is_empty()
        || g.channel.is_some()
        || (n == 2 && g.qubits[0] == g.qubits[1])
    {
        return Err(error("unsupported scheduled gate fields"));
    }
    Ok((
        code,
        g.qubits[0].0 as u64,
        if n == 2 { g.qubits[1].0 as u64 } else { 0 },
        g.angles.first().map_or(0.0, Angle64::to_radians),
        g.angles.get(1).map_or(0.0, Angle64::to_radians),
    ))
}
/// Encode mandatory v3 transport: 16-byte header, 40-byte batch headers and gates.
/// All fields are little endian. No unknown records, implicit padding or events.
/// # Errors
/// Rejects unsupported gates and messages exceeding the transport limit.
pub fn encode_timed_batches(batches: &[TimedBatch]) -> Result<ByteMessage, PecosError> {
    let size = batches
        .iter()
        .try_fold(16usize, |n, b| {
            b.gates
                .len()
                .checked_mul(40)
                .and_then(|v| v.checked_add(40))
                .and_then(|v| v.checked_add(n))
        })
        .ok_or_else(|| error("schedule size overflow"))?;
    if size > MAX_SCHEDULE_BYTES {
        return Err(error("schedule transport size limit"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| processing_error("schedule allocation failed"))?;
    bytes.extend_from_slice(&crate::byte_message::protocol::BATCH_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&[3, 0, 0, 0]);
    bytes.extend_from_slice(
        &u32::try_from(batches.len())
            .map_err(|_| error("batch count overflow"))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(
        &u32::try_from(size)
            .map_err(|_| error("size overflow"))?
            .to_le_bytes(),
    );
    for batch in batches {
        for v in [
            batch.runtime_shot_id,
            batch.batch_index,
            batch.start_nanos,
            batch.duration_nanos,
            batch.gates.len() as u64,
        ] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for gate in &batch.gates {
            let (op, q0, q1, theta, phi) = fields(gate)?;
            for v in [op, q0, q1, theta.to_bits(), phi.to_bits()] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    Ok(ByteMessage::new(&bytes))
}
fn word(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("bounded record"),
    )
}
pub(crate) fn decode(input: &ByteMessage) -> Result<Vec<TimedBatch>, PecosError> {
    let bytes = input.as_bytes();
    if bytes.len() < 16
        || bytes.len() > MAX_SCHEDULE_BYTES
        || bytes[..4] != crate::byte_message::protocol::BATCH_MAGIC.to_le_bytes()
        || bytes[4..8] != [3, 0, 0, 0]
        || u32::from_le_bytes(bytes[12..16].try_into().expect("header")) as usize != bytes.len()
    {
        return Err(error("invalid scheduled header/version/length"));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().expect("header")) as usize;
    if count > (bytes.len() - 16) / 40 {
        return Err(error("invalid scheduled batch count"));
    }
    let mut result = Vec::new();
    let mut offset = 16;
    for _ in 0..count {
        if bytes.len() - offset < 40 {
            return Err(error("truncated batch"));
        }
        let n =
            usize::try_from(word(bytes, offset + 32)).map_err(|_| error("gate count overflow"))?;
        let mut batch = TimedBatch {
            runtime_shot_id: word(bytes, offset),
            batch_index: word(bytes, offset + 8),
            start_nanos: word(bytes, offset + 16),
            duration_nanos: word(bytes, offset + 24),
            gates: Vec::new(),
        };
        offset += 40;
        if n > (bytes.len() - offset) / 40 {
            return Err(error("truncated gates"));
        }
        for _ in 0..n {
            let op = word(bytes, offset);
            let q0 = word(bytes, offset + 8);
            let q1 = word(bytes, offset + 16);
            let theta = f64::from_bits(word(bytes, offset + 24));
            let phi = f64::from_bits(word(bytes, offset + 32));
            if !theta.is_finite() || !phi.is_finite() {
                return Err(error("nonfinite scheduled angle"));
            }
            let q0 = usize::try_from(q0).map_err(|_| error("target overflow"))?;
            let q1 = usize::try_from(q1).map_err(|_| error("target overflow"))?;
            let a = Angle64::from_radians(theta);
            let p = Angle64::from_radians(phi);
            let gate = match op {
                1 => Gate::rxy1q(a, p, &[q0]),
                2 => Gate::rz(a, &[q0]),
                3 => Gate::rzz(a, &[(q0, q1)]),
                4 => Gate::rxyxy2q(a, p, &[(q0, q1)]),
                5 => Gate::pz(&[q0]),
                6 => Gate::mz(&[q0]),
                7 => Gate::measure_leaked(&[q0]),
                _ => return Err(error("unknown mandatory scheduled operation")),
            };
            if !matches!(op, 3 | 4) && q1 != 0
                || (op >= 5 && theta.to_bits() != 0)
                || (op != 1 && op != 4 && phi.to_bits() != 0)
            {
                return Err(error("nonzero unused scheduled fields"));
            }
            fields(&gate)?;
            batch.gates.push(gate);
            offset += 40;
        }
        result.push(batch);
    }
    if offset != bytes.len() {
        return Err(error("trailing scheduled data"));
    }
    Ok(result)
}
#[derive(Clone)]
pub(crate) struct ScheduledIdleModel {
    pub(crate) config: ScheduledIdleZ,
    inner: GeneralNoiseModel,
    timeline: ScheduleTimeline,
}
/// Admission history is independent of gate output and noise controller state.
#[derive(Clone)]
pub(crate) struct ScheduleTimeline {
    native_shot: Option<u64>,
    next_batch: u64,
    ends: Vec<u64>,
}
pub(crate) struct PreparedSchedule {
    pub(crate) messages: Vec<ByteMessage>,
    pub(crate) timeline: ScheduleTimeline,
}
impl ScheduledIdleModel {
    pub(crate) fn qubits(&self) -> usize {
        self.config.qubits
    }
    pub(crate) fn prepare(&self, input: &ByteMessage) -> Result<PreparedSchedule, PecosError> {
        self.prepare_batches(decode(input)?)
    }
    pub(crate) fn prepare_batches(
        &self,
        batches: Vec<TimedBatch>,
    ) -> Result<PreparedSchedule, PecosError> {
        self.timeline.prepare(&self.config, batches)
    }
    pub(crate) fn commit(&mut self, prepared: PreparedSchedule) {
        self.timeline = prepared.timeline;
    }
    pub(crate) fn start_admitted(
        &mut self,
        message: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.inner.start(message)
    }
}
impl ScheduleTimeline {
    pub(crate) fn has_native_shot(&self) -> bool {
        self.native_shot.is_some()
    }
    pub(crate) fn new(qubits: usize) -> Self {
        Self {
            native_shot: None,
            next_batch: 0,
            ends: vec![0; qubits],
        }
    }
    pub(crate) fn prepare(
        &self,
        config: &ScheduledIdleZ,
        batches: Vec<TimedBatch>,
    ) -> Result<PreparedSchedule, PecosError> {
        let mut prepared = PreparedSchedule {
            messages: Vec::new(),
            timeline: self.clone(),
        };
        for batch in batches {
            if prepared
                .timeline
                .native_shot
                .is_some_and(|id| id != batch.runtime_shot_id)
                || batch.batch_index != prepared.timeline.next_batch
            {
                return Err(error("scheduled identity mismatch"));
            }
            prepared.timeline.native_shot = Some(batch.runtime_shot_id);
            prepared.timeline.next_batch = prepared
                .timeline
                .next_batch
                .checked_add(1)
                .ok_or_else(|| error("batch ordinal overflow"))?;
            let end = batch
                .start_nanos
                .checked_add(batch.duration_nanos)
                .ok_or_else(|| error("schedule time overflow"))?;
            let mut touched = vec![false; config.qubits];
            let mut builder = ByteMessage::quantum_operations_builder();
            for gate in batch.gates {
                for q in &gate.qubits {
                    let q = q.0;
                    if q >= touched.len() || batch.start_nanos < prepared.timeline.ends[q] {
                        return Err(error("scheduled capacity or timing overlap"));
                    }
                    if !touched[q] {
                        let gap = batch.start_nanos - prepared.timeline.ends[q];
                        let seconds = std::time::Duration::from_nanos(gap).as_secs_f64();
                        config.validate_idle(seconds)?;
                        // This profile has only local Z/RZ idle channels. PZ
                        // erases their effect, so do not sample a discarded channel.
                        if gap != 0 && gate.gate_type != GateType::PZ {
                            builder.idle(seconds, &[q]);
                        }
                        touched[q] = true;
                    }
                }
                builder.add_gate_command(&gate);
            }
            for (q, touched) in touched.into_iter().enumerate() {
                if touched {
                    prepared.timeline.ends[q] = end;
                }
            }
            let msg = builder.build();
            if !msg.is_empty()? {
                prepared.messages.push(msg);
            }
        }
        Ok(prepared)
    }
}
impl ControlEngine for ScheduledIdleModel {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(
        &mut self,
        _: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        Err(error(
            "scheduled idle profile requires mandatory v3 transport",
        ))
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.inner.continue_processing(reply)
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.inner.reset()?;
        self.timeline = ScheduleTimeline::new(self.config.qubits);
        Ok(())
    }
}
impl NoiseModel for ScheduledIdleModel {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl RngManageable for ScheduledIdleModel {
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
mod timing_tests {
    use super::*;

    fn batch(index: u64, start: u64, duration: u64, gates: Vec<Gate>) -> TimedBatch {
        TimedBatch {
            runtime_shot_id: 9,
            batch_index: index,
            start_nanos: start * 1_000_000_000,
            duration_nanos: duration * 1_000_000_000,
            gates,
        }
    }
    fn config() -> ScheduledIdleZ {
        ScheduledIdleZ::new(2, 0.2, 0.3, 0.4).unwrap()
    }
    fn ops(prepared: &PreparedSchedule, index: usize) -> Vec<Gate> {
        prepared.messages[index].quantum_ops().unwrap()
    }
    fn idle_qubits_and_seconds(gates: &[Gate]) -> Vec<(usize, f64)> {
        gates
            .iter()
            .filter(|g| g.gate_type == GateType::Idle)
            .map(|g| (g.qubits[0].0, g.params[0]))
            .collect()
    }
    #[test]
    fn preparation_omission_is_per_qubit_and_preserves_busy_duration() {
        let prepared = ScheduleTimeline::new(2)
            .prepare(
                &config(),
                vec![
                    batch(0, 2, 3, vec![Gate::pz(&[0]), Gate::rz(Angle64::ZERO, &[1])]),
                    batch(1, 7, 1, vec![Gate::mz(&[0]), Gate::mz(&[1])]),
                ],
            )
            .unwrap();
        assert_eq!(idle_qubits_and_seconds(&ops(&prepared, 0)), vec![(1, 2.0)]);
        assert_eq!(
            idle_qubits_and_seconds(&ops(&prepared, 1)),
            vec![(0, 2.0), (1, 2.0)]
        );
    }
    #[test]
    fn only_first_operation_on_each_qubit_controls_preparation_omission() {
        for prep_first in [false, true] {
            let mut gates = vec![Gate::pz(&[0]), Gate::rz(Angle64::ZERO, &[0])];
            if !prep_first {
                gates.reverse();
            }
            let prepared = ScheduleTimeline::new(2)
                .prepare(&config(), vec![batch(0, 2, 1, gates)])
                .unwrap();
            assert_eq!(
                idle_qubits_and_seconds(&ops(&prepared, 0)),
                if prep_first { vec![] } else { vec![(0, 2.0)] }
            );
        }
    }
}
