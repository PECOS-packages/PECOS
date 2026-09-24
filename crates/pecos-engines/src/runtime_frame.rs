//! Opt-in, bounded synchronous frames for the legacy general-noise controller.
//!
//! Version 2 is a sequential single-qubit wire subset, not a simultaneous runtime
//! batch format. Events observe neither raw nor processed measurement results.
use crate::noise::{GeneralNoiseModel, GeneralNoiseModelBuilder, IntoNoiseModel, NoiseModel};
use crate::{ByteMessage, ControlEngine, EngineStage, Gate, GateType, PecosError};
use pecos_core::{QubitId, RngManageable};
use pecos_random::PecosRng;
use std::any::Any;

/// Independently invented metadata event; no physical or RNG effect.
pub const METADATA: u32 = 4101;
/// Independently invented unconditional X effect, without program-gate noise.
pub const FLIP: u32 = 4102;
/// Hard wire bound, checked before decoding or copying records.
pub const MAX_FRAME_BYTES: usize = 16 + 128 * 24;

/// Host-assigned identity, unchanged across inputs of one shot. Not a seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShotContext {
    pub run: u64,
    pub worker: usize,
    pub shot: usize,
}

/// Finite admission budgets. Expansion includes generated continuation commands.
#[derive(Clone, Copy, Debug)]
pub struct FrameLimits {
    pub records: usize,
    pub expanded_operations: usize,
}
impl Default for FrameLimits {
    fn default() -> Self {
        Self {
            records: 128,
            expanded_operations: 65536,
        }
    }
}

/// Sequential records. Measurement identities and simultaneous batches are absent.
#[derive(Clone, Debug)]
pub enum FrameRecord {
    Gate(Box<Gate>),
    Event { id: u32, target: u32 },
}

impl FrameRecord {
    /// Construct a gate record; validation occurs during encoding/admission.
    #[must_use]
    pub fn gate(gate: Gate) -> Self {
        Self::Gate(Box::new(gate))
    }
}

pub(crate) fn error(s: &str) -> PecosError {
    PecosError::Input(s.into())
}
pub(crate) fn processing_error(s: &str) -> PecosError {
    PecosError::Processing(s.into())
}

// Public admission bounds make overflow unreachable; keep every factor checked
// so this calculation remains defensive if those limits change.
fn expansion_bound(count: usize, qubits: usize) -> Result<usize, PecosError> {
    qubits
        .checked_add(1)
        .and_then(|n| n.checked_mul(16))
        .and_then(|factor| count.checked_mul(factor))
        .ok_or_else(|| error("expansion overflow"))
}

fn opcode(g: &Gate) -> Result<u32, PecosError> {
    g.validate().map_err(|e| error(&e))?;
    if g.qubits.len() != 1
        || !g.angles.is_empty()
        || !g.meas_ids.is_empty()
        || g.channel.is_some()
        || g.params.len() != usize::from(g.gate_type == GateType::Idle)
        || g.params.iter().any(|p| !p.is_finite() || *p < 0.0)
    {
        return Err(error("unsupported frame gate fields or measurement IDs"));
    }
    match g.gate_type {
        GateType::PZ => Ok(1),
        GateType::X => Ok(2),
        GateType::Z => Ok(3),
        GateType::H => Ok(4),
        GateType::MZ => Ok(5),
        GateType::MeasureLeaked => Ok(6),
        GateType::Idle => Ok(7),
        GateType::MeasCrosstalkLocalPayload => Ok(8),
        _ => Err(error("unsupported frame gate")),
    }
}

/// Encode the complete mandatory v2 subset. Unknown event IDs are rejected.
///
/// Header: PECS numeric magic, version=2, zero flags/reserved, count and total
/// bytes. Each record: kind u8, three zero bytes, payload length u32. Gate payload:
/// opcode u32, target u32, idle seconds f64 (zero otherwise). Event payload:
/// ID u32 and target u32. All integers/floats are little-endian; no padding.
///
/// # Errors
/// Rejects unsupported gates, event IDs, or more than 128 records.
pub fn encode_frame(records: &[FrameRecord]) -> Result<ByteMessage, PecosError> {
    if records.len() > 128 {
        return Err(error("frame record limit"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(MAX_FRAME_BYTES)
        .map_err(|_| processing_error("frame allocation failed"))?;
    bytes.extend_from_slice(&crate::byte_message::protocol::BATCH_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&[2, 0, 0, 0]);
    bytes.extend_from_slice(
        &u32::try_from(records.len())
            .map_err(|_| error("record count overflow"))?
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&[0; 4]);
    for record in records {
        match record {
            FrameRecord::Gate(g) => {
                let op = opcode(g)?;
                let q = u32::try_from(g.qubits[0].0).map_err(|_| error("target overflow"))?;
                bytes.extend_from_slice(&[10, 0, 0, 0]);
                bytes.extend_from_slice(&16u32.to_le_bytes());
                bytes.extend_from_slice(&op.to_le_bytes());
                bytes.extend_from_slice(&q.to_le_bytes());
                bytes.extend_from_slice(&g.params.first().copied().unwrap_or(0.0).to_le_bytes());
            }
            FrameRecord::Event { id, target } => {
                if !matches!(*id, METADATA | FLIP) {
                    return Err(error("unknown mandatory event"));
                }
                bytes.extend_from_slice(&[30, 0, 0, 0]);
                bytes.extend_from_slice(&8u32.to_le_bytes());
                bytes.extend_from_slice(&id.to_le_bytes());
                bytes.extend_from_slice(&target.to_le_bytes());
            }
        }
    }
    let len = u32::try_from(bytes.len()).map_err(|_| error("byte length overflow"))?;
    bytes[12..16].copy_from_slice(&len.to_le_bytes());
    Ok(ByteMessage::new(&bytes))
}

fn word(bytes: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(bytes[pos..pos + 4].try_into().expect("validated field"))
}

pub(crate) fn decode(
    input: &ByteMessage,
    config: &RuntimeGeneralNoise,
) -> Result<Vec<FrameRecord>, PecosError> {
    let bytes = input.as_bytes();
    if bytes.len() < 16
        || bytes.len() > MAX_FRAME_BYTES
        || word(bytes, 0) != crate::byte_message::protocol::BATCH_MAGIC
        || bytes[4..8] != [2, 0, 0, 0]
        || word(bytes, 12) as usize != bytes.len()
    {
        return Err(error("invalid frame header/version/length"));
    }
    let count = word(bytes, 8) as usize;
    if count > config.limits.records {
        return Err(error("frame record limit"));
    }
    // For admitted singleton gates: <=8 idle commands; <=4 SQ commands;
    // <=3 preparation commands plus <=qubits raw crosstalk results; each
    // crosstalk payload adds <=1 measurement and <=1 continuation command.
    // 16*(qubits+1) per record conservatively covers all commands and outcomes.
    let bound = expansion_bound(count, config.qubits)?;
    if bound > config.limits.expanded_operations {
        return Err(error("frame expansion budget"));
    }
    let mut records = Vec::new();
    records
        .try_reserve_exact(count)
        .map_err(|_| processing_error("frame allocation failed"))?;
    let mut pos = 16;
    for _ in 0..count {
        if bytes.len() - pos < 8 {
            return Err(error("truncated mandatory record"));
        }
        let kind = bytes[pos];
        let len = word(bytes, pos + 4) as usize;
        if bytes[pos + 1..pos + 4] != [0, 0, 0] || len > bytes.len() - pos - 8 {
            return Err(error("invalid mandatory record length/flags"));
        }
        pos += 8;
        let record = match (kind, len) {
            (10, 16) => {
                let op = word(bytes, pos);
                let q = word(bytes, pos + 4) as usize;
                let duration = f64::from_le_bytes(
                    bytes[pos + 8..pos + 16]
                        .try_into()
                        .expect("validated length"),
                );
                if q >= config.qubits
                    || !duration.is_finite()
                    || duration < 0.0
                    || (op != 7 && duration != 0.0)
                {
                    return Err(error("invalid gate target or parameter"));
                }
                if op == 7 {
                    config.inner.validate_frame_idle(duration)?;
                }
                let gate = match op {
                    1 => Gate::pz(&[q]),
                    2 => Gate::x(&[q]),
                    3 => Gate::z(&[q]),
                    4 => Gate::h(&[q]),
                    5 => Gate::mz(&[q]),
                    6 => Gate::measure_leaked(&[q]),
                    7 => Gate::idle(duration, vec![QubitId(q)]),
                    8 => Gate::new(
                        GateType::MeasCrosstalkLocalPayload,
                        vec![],
                        vec![],
                        vec![QubitId(q)],
                    ),
                    _ => return Err(error("unknown mandatory gate opcode")),
                };
                FrameRecord::gate(gate)
            }
            (30, 8) => {
                let id = word(bytes, pos);
                let target = word(bytes, pos + 4);
                if target as usize >= config.qubits
                    || !matches!(id, METADATA | FLIP)
                    || (id == FLIP && !config.physical)
                {
                    return Err(error(
                        "unsupported mandatory event, target or physical profile",
                    ));
                }
                FrameRecord::Event { id, target }
            }
            _ => return Err(error("unknown mandatory record")),
        };
        records.push(record);
        pos += len;
    }
    if pos != bytes.len() {
        return Err(error("trailing frame bytes"));
    }
    Ok(records)
}

/// Checked opt-in configuration, accepted by the existing `SimBuilder::noise`.
/// Inner model access is deliberately not exposed; capabilities cannot go stale.
#[derive(Clone)]
pub struct RuntimeNoise {
    inner: GeneralNoiseModel,
    limits: FrameLimits,
    qubits: usize,
    physical: bool,
}
impl RuntimeNoise {
    /// Compile supported general-noise configuration and finite storage limits.
    ///
    /// # Errors
    /// Rejects invalid combined probabilities, qubit counts and budgets.
    pub fn new(
        inner: GeneralNoiseModelBuilder,
        qubits: usize,
        limits: FrameLimits,
    ) -> Result<Self, PecosError> {
        inner.validate_configuration().map_err(error)?;
        if !(1..=16).contains(&qubits)
            || !(1..=128).contains(&limits.records)
            || !(1..=65536).contains(&limits.expanded_operations)
        {
            return Err(error("unsupported runtime frame limits"));
        }
        let physical = inner.simple_probabilities().is_some();
        let inner = inner.build();
        inner.validate_runtime_configuration()?;
        Ok(Self {
            inner,
            limits,
            qubits,
            physical,
        })
    }
}
impl IntoNoiseModel for RuntimeNoise {
    fn into_noise_model(self) -> Box<dyn NoiseModel> {
        Box::new(RuntimeGeneralNoise {
            inner: self.inner,
            limits: self.limits,
            qubits: self.qubits,
            physical: self.physical,
        })
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeGeneralNoise {
    inner: GeneralNoiseModel,
    limits: FrameLimits,
    pub(crate) qubits: usize,
    physical: bool,
}
impl RuntimeGeneralNoise {
    // QuantumSystem has already admitted this complete input before latching.
    pub(crate) fn start_validated(
        &mut self,
        input: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.inner.start(input)
    }

    pub(crate) fn preflight_legacy(&self, input: &ByteMessage) -> Result<(), PecosError> {
        // Restrict opt-in v1 inputs too: the retained model must never acquire
        // out-of-range bookkeeping that invalidates a later expansion bound.
        let bytes = input.as_bytes();
        if !(16..=MAX_FRAME_BYTES).contains(&bytes.len()) {
            return Err(error("legacy frame byte limit"));
        }
        // The shared parser reserves from the declared count. Bound it before
        // parsing, even when the wire contains fewer records than it claims.
        let count = word(bytes, 8) as usize;
        if count > self.limits.records
            || expansion_bound(count, self.qubits)? > self.limits.expanded_operations
        {
            return Err(error("frame record or expansion limit"));
        }
        let gates = input.quantum_ops()?;
        let mut canonical = ByteMessage::quantum_operations_builder();
        for gate in &gates {
            opcode(gate)?;
            if gate.gate_type == GateType::Idle {
                self.inner.validate_frame_idle(gate.params[0])?;
            }
            canonical.add_gate_command(gate);
            if gate.qubits[0].0 >= self.qubits {
                return Err(error("gate target outside runtime profile"));
            }
        }
        if canonical.build().as_bytes() != input.as_bytes() {
            return Err(error("noncanonical or unknown legacy record"));
        }
        Ok(())
    }
}
impl ControlEngine for RuntimeGeneralNoise {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(
        &mut self,
        input: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.preflight_legacy(&input)?;
        self.start_validated(input)
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.inner.continue_processing(reply)
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.inner.reset()
    }
}
impl NoiseModel for RuntimeGeneralNoise {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl RngManageable for RuntimeGeneralNoise {
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

// No Clone, public token or reference escapes the synchronous process call.
pub(crate) struct FrameExecutor<'a> {
    pub(crate) model: &'a mut RuntimeGeneralNoise,
    pub(crate) simulator: &'a mut dyn crate::QuantumEngine,
}
impl FrameExecutor<'_> {
    pub(crate) fn execute(&mut self, records: Vec<FrameRecord>) -> Result<ByteMessage, PecosError> {
        let mut original = ByteMessage::quantum_operations_builder();
        for record in &records {
            if let FrameRecord::Gate(g) = record {
                original.add_gate_command(g);
            }
        }
        // Exactly one original start; boundary offsets do not resample or complete.
        let (expanded, ends) = self
            .model
            .inner
            .start_runtime_frame(&original.build(), records.len())?;
        let bound = expansion_bound(records.len(), self.model.qubits)
            .map_err(|_| processing_error("expansion overflow"))?;
        if ends.last().copied().unwrap_or(0) as usize > bound {
            return Err(processing_error("expansion invariant exceeded"));
        }
        // Metadata does not introduce simulator dispatch boundaries.
        let mut reply = if records
            .iter()
            .any(|r| matches!(r, FrameRecord::Event { id: FLIP, .. }))
        {
            let gates = expanded.quantum_ops()?;
            if gates.len() > bound {
                return Err(processing_error("expansion invariant exceeded"));
            }
            let mut raw = Vec::new();
            raw.try_reserve_exact(bound)
                .map_err(|_| processing_error("outcome allocation failed"))?;
            let mut ends = ends.into_iter();
            let mut start = 0;
            let mut end = 0;
            for record in records {
                match record {
                    FrameRecord::Gate(_) => {
                        end = ends
                            .next()
                            .ok_or_else(|| processing_error("missing expansion boundary"))?
                            as usize;
                    }
                    FrameRecord::Event { id: FLIP, target } => {
                        let slice = gates
                            .get(start..end)
                            .ok_or_else(|| processing_error("invalid expansion boundary"))?;
                        self.execute_segment(slice, &mut raw, bound)?;
                        self.simulator.process(
                            ByteMessage::quantum_operations_builder()
                                .x(&[target as usize])
                                .build(),
                        )?;
                        start = end;
                    }
                    FrameRecord::Event { .. } => {}
                }
            }
            let slice = gates
                .get(start..end)
                .ok_or_else(|| processing_error("invalid expansion boundary"))?;
            self.execute_segment(slice, &mut raw, bound)?;
            ByteMessage::outcomes_builder().add_outcomes(&raw).build()
        } else {
            self.simulator.process(expanded)?
        };
        // The admitted singleton profile can generate one crosstalk continuation.
        for _ in 0..2 {
            match self.model.inner.continue_processing(reply)? {
                EngineStage::Complete(outcomes) => return Ok(outcomes),
                EngineStage::NeedsProcessing(commands) => {
                    reply = self.simulator.process(commands)?;
                }
            }
        }
        Err(processing_error("continuation budget exceeded"))
    }

    fn execute_segment(
        &mut self,
        gates: &[Gate],
        raw: &mut Vec<usize>,
        bound: usize,
    ) -> Result<(), PecosError> {
        if gates.is_empty() {
            return Ok(());
        }
        let reply = self.simulator.process(
            ByteMessage::quantum_operations_builder()
                .add_gate_commands(gates)
                .build(),
        )?;
        let values = reply.outcomes()?;
        if raw.len() + values.len() > bound {
            return Err(processing_error("outcome budget exceeded"));
        }
        raw.extend(values.into_iter().map(|v| v as usize));
        Ok(())
    }
}

static NEXT_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub(crate) fn next_run() -> Result<u64, PecosError> {
    NEXT_RUN
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |v| v.checked_add(1),
        )
        .map_err(|_| processing_error("run identity exhausted"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_bound_checks_each_arithmetic_step() {
        assert_eq!(expansion_bound(128, 16).unwrap(), 34816);
        assert_eq!(expansion_bound(0, 16).unwrap(), 0);
        for (count, qubits) in [(1, usize::MAX), (1, usize::MAX / 16), (usize::MAX, 1)] {
            assert!(
                matches!(expansion_bound(count, qubits), Err(PecosError::Input(message)) if message == "expansion overflow")
            );
        }
    }
}
