//! Mandatory v6 batches for checked local noise profiles. Not a custom-event format.
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
/// Checked Z-only convenience profile; other noise channels are disabled.
#[derive(Clone, Debug)]
pub struct ScheduledIdleZ(ScheduledNoise);
impl ScheduledIdleZ {
    /// Rates are inverse seconds (linear) or radians/second (sine and coherent).
    /// # Errors
    /// Rejects capacity outside 1..=16 or nonfinite/negative rates.
    pub fn new(qubits: usize, linear: f64, sine: f64, coherent: f64) -> Result<Self, PecosError> {
        Ok(Self(
            ScheduledNoise::new(qubits)?
                .with_linear(linear, None)?
                .with_sine(sine, None)?
                .with_coherent(coherent, None)?,
        ))
    }
}
impl From<ScheduledIdleZ> for ScheduledNoise {
    fn from(profile: ScheduledIdleZ) -> Self {
        profile.0
    }
}
impl IntoNoiseModel for ScheduledIdleZ {
    fn into_noise_model(self) -> Result<Box<dyn NoiseModel>, PecosError> {
        self.0.into_noise_model()
    }
}
/// Compatibility name for [`ScheduledNoise`]. It preserves the entire checked
/// profile, including local gate, preparation and readout faults.
pub type ScheduledIdleNoise = ScheduledNoise;
/// Checked scheduled noise configuration. `new` leaves all channels disabled;
/// idle setters configure local memory noise, and [`ScheduledLocalNoise`] adds
/// gate, preparation and readout faults by lossless conversion.
/// Uses the existing general-noise sampling and leakage semantics. Capacity is
/// fixed at 1..=16 and the scheduled host requires a state-vector engine.
#[derive(Clone, Debug)]
pub struct ScheduledNoise {
    qubits: usize,
    local: Option<LocalFaults>,
    linear: f64,
    sine: f64,
    coherent: f64,
    linear_model: BTreeMap<String, f64>,
    sine_model: BTreeMap<String, f64>,
    coherent_model: BTreeMap<String, f64>,
}
/// Checked local Pauli gate faults, preparation faults and asymmetric readout.
/// Idle channels use the supplied profile. Single-qubit faults are uniform over
/// X/Y/Z; two-qubit faults default to uniform nonidentity Pauli pairs and accept
/// a checked distribution. Two-qubit rotation faults can use checked angle scaling.
/// Emitted RZ gates receive single-qubit faults too; virtual frame updates must
/// be resolved by the runtime/adapter rather than represented as physical RZ.
/// Preparation faults can leak or flip the qubit. Gate emission/leakage, seepage
/// and crosstalk are disabled.
/// Capacity remains 1..=16 with a state-vector engine. Measurements capture
/// leakage before subsequent operations, allowing same-batch reset and reuse.
#[derive(Clone, Debug)]
pub struct ScheduledLocalNoise {
    idle: ScheduledNoise,
    local: LocalFaults,
}
#[derive(Clone, Debug)]
struct LocalFaults {
    p1: f64,
    p2: f64,
    prep: f64,
    prep_leak_ratio: f64,
    p2_pauli_model: Option<BTreeMap<String, f64>>,
    p2_angle_params: Option<(f64, f64, f64, f64)>,
    p2_angle_power: Option<f64>,
    meas0: f64,
    meas1: f64,
}
impl ScheduledLocalNoise {
    /// Set local fault probabilities while preserving the supplied idle channels.
    /// Replaces all existing local fault probabilities if the profile already has them,
    /// including resetting preparation leakage to zero, two-qubit faults to uniform,
    /// and two-qubit angle scaling to its constant-rate defaults.
    /// # Errors
    /// Every probability must be finite and in [0, 1].
    pub fn new(
        idle: ScheduledNoise,
        p1: f64,
        p2: f64,
        prep: f64,
        meas0: f64,
        meas1: f64,
    ) -> Result<Self, PecosError> {
        if [p1, p2, prep, meas0, meas1]
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(error(
                "scheduled local probabilities must be finite and in [0, 1]",
            ));
        }
        Ok(Self {
            idle,
            local: LocalFaults {
                p1,
                p2,
                prep,
                prep_leak_ratio: 0.0,
                p2_pauli_model: None,
                p2_angle_params: None,
                p2_angle_power: None,
                meas0,
                meas1,
            },
        })
    }
    /// Set signed principal-angle coefficients using the existing general model.
    /// For `x = |theta|/pi`, the multiplier is `a*x^power+b` for negative angles,
    /// `c*x^power+d` for positive angles, and `(b+d)/2` at zero. Angles are reduced
    /// to `(-pi, pi]`, with both half-turn endpoints on the positive branch.
    /// This changes only the event probability, preserving the Pauli distribution
    /// and all other faults. It does not rewrite gates or runtime timing.
    /// # Errors
    /// Rejects nonfinite coefficients or effective probabilities outside [0, 1]
    /// anywhere in the principal-angle range, including when p2 is zero.
    pub fn with_p2_angle_params(
        mut self,
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    ) -> Result<Self, PecosError> {
        self.local.p2_angle_params = Some((a, b, c, d));
        self.local.validate_angles()?;
        Ok(self)
    }
    /// Set a finite positive angle exponent, preserving all other settings.
    /// The default exponent is one; with constant coefficients it has no effect.
    /// # Errors
    /// Rejects nonfinite/nonpositive powers and invalid combined probabilities.
    pub fn with_p2_angle_power(mut self, power: f64) -> Result<Self, PecosError> {
        self.local.p2_angle_power = Some(power);
        self.local.validate_angles()?;
        Ok(self)
    }
    /// Set the conditional distribution of a two-qubit Pauli fault.
    /// Keys are ordered pairs of I/X/Y/Z excluding II. The first character acts
    /// on the emitted gate's first operand, not the lower-numbered qubit. Missing
    /// pairs have zero weight. This replaces the previous distribution without
    /// changing the fault probability, idle channels or preparation leakage.
    /// # Errors
    /// Rejects invalid keys or nonfinite/negative weights, even when p2 is zero.
    /// Weights must sum to one within 1e-10; empty and all-zero maps are rejected.
    pub fn with_p2_pauli_model(mut self, model: BTreeMap<String, f64>) -> Result<Self, PecosError> {
        for (pair, weight) in &model {
            if pair.len() != 2
                || pair == "II"
                || !pair
                    .bytes()
                    .all(|axis| matches!(axis, b'I' | b'X' | b'Y' | b'Z'))
            {
                return Err(error(
                    "scheduled two-qubit Pauli keys must be I/X/Y/Z pairs excluding II",
                ));
            }
            if !weight.is_finite() || *weight < 0.0 {
                return Err(error(
                    "scheduled two-qubit Pauli weights must be finite and nonnegative",
                ));
            }
        }
        let total: f64 = model.values().sum();
        if !total.is_finite() || (total - 1.0).abs() > 1e-10 {
            return Err(error("scheduled two-qubit Pauli weights must sum to one"));
        }
        self.local.p2_pauli_model = Some(model);
        Ok(self)
    }
    /// Set the conditional probability of leakage when a preparation fault occurs.
    /// The remaining preparation faults apply X. Preparation always resets the
    /// qubit and clears prior leakage before sampling this new fault. The default
    /// ratio is zero; a zero preparation fault probability leaves this inactive.
    /// # Errors
    /// Rejects nonfinite ratios or values outside [0, 1], even when prep is zero.
    pub fn with_prep_leak_ratio(mut self, ratio: f64) -> Result<Self, PecosError> {
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err(error(
                "scheduled preparation leakage ratio must be finite and in [0, 1]",
            ));
        }
        self.local.prep_leak_ratio = ratio;
        Ok(self)
    }
}
impl LocalFaults {
    fn configure_angles(&self, mut builder: GeneralNoiseModelBuilder) -> GeneralNoiseModelBuilder {
        if let Some((a, b, c, d)) = self.p2_angle_params {
            builder = builder.with_p2_angle_params(a, b, c, d);
        }
        if let Some(power) = self.p2_angle_power {
            builder = builder.with_p2_angle_power(power);
        }
        builder
    }
    fn validate_angles(&self) -> Result<(), PecosError> {
        self.configure_angles(GeneralNoiseModelBuilder::new().with_p2(self.p2))
            .validate_configuration()
            .map_err(PecosError::Input)
    }
}
/// Lossless conversion for scheduled transport composition; all configured faults remain enabled.
impl From<ScheduledLocalNoise> for ScheduledNoise {
    fn from(profile: ScheduledLocalNoise) -> Self {
        let mut noise = profile.idle;
        noise.local = Some(profile.local);
        noise
    }
}
impl IntoNoiseModel for ScheduledLocalNoise {
    fn into_noise_model(self) -> Result<Box<dyn NoiseModel>, PecosError> {
        ScheduledNoise::from(self).into_noise_model()
    }
}

impl ScheduledNoise {
    /// Construct a profile with all idle families disabled.
    /// # Errors
    /// Rejects capacity outside 1..=16.
    pub fn new(qubits: usize) -> Result<Self, PecosError> {
        if !(1..=16).contains(&qubits) {
            return Err(error("scheduled capacity must be 1..=16"));
        }
        Ok(Self {
            qubits,
            local: None,
            linear: 0.0,
            sine: 0.0,
            coherent: 0.0,
            linear_model: BTreeMap::from([("Z".to_owned(), 1.0)]),
            sine_model: BTreeMap::from([("Z".to_owned(), 1.0)]),
            coherent_model: BTreeMap::from([("RZ".to_owned(), 1.0)]),
        })
    }
    fn validate_model(model: &BTreeMap<String, f64>, axes: &[&str]) -> Result<(), PecosError> {
        for (axis, value) in model {
            if !axes.contains(&axis.as_str()) {
                return Err(error(&format!(
                    "invalid scheduled idle axis '{axis}'; expected one of {}",
                    axes.join(", ")
                )));
            }
            if !value.is_finite() || *value < 0.0 {
                return Err(error(&format!(
                    "invalid scheduled idle weight for '{axis}': must be finite and nonnegative"
                )));
            }
        }
        Ok(())
    }
    /// Set the linear event rate in inverse seconds and its X/Y/Z/L distribution.
    /// Passing `None` retains the current distribution (initially Z only).
    /// A single event is drawn with probability min(rate * seconds, 1).
    /// # Errors
    /// Rejects invalid rates, axes or weights; weights must sum to one within 1e-10.
    pub fn with_linear(
        mut self,
        rate: f64,
        model: impl Into<Option<BTreeMap<String, f64>>>,
    ) -> Result<Self, PecosError> {
        if let Some(model) = model.into() {
            Self::validate_model(&model, &["X", "Y", "Z", "L"])?;
            let total: f64 = model.values().sum();
            if !total.is_finite() || (total - 1.0).abs() > 1e-10 {
                return Err(error("scheduled linear weights must sum to one"));
            }
            self.linear_model = model;
        }
        self.linear = rate;
        self.validate_idle(1.0)?;
        Ok(self)
    }
    /// Set the sine-squared rate in radians/second and independent X/Y/Z/L multipliers.
    /// Passing `None` retains the current multipliers (initially Z only).
    /// Multipliers are not normalized. Each axis has probability sin(rate * multiplier * seconds)^2.
    /// # Errors
    /// Rejects invalid rates, axes, multipliers or overflowing products.
    pub fn with_sine(
        mut self,
        rate: f64,
        model: impl Into<Option<BTreeMap<String, f64>>>,
    ) -> Result<Self, PecosError> {
        if let Some(model) = model.into() {
            Self::validate_model(&model, &["X", "Y", "Z", "L"])?;
            self.sine_model = model;
        }
        self.sine = rate;
        self.validate_idle(1.0)?;
        Ok(self)
    }
    /// Set the coherent rate in radians/second and unnormalized RX/RY/RZ multipliers.
    /// Passing `None` retains the current multipliers (initially RZ only).
    /// # Errors
    /// Rejects invalid rates, axes, multipliers or overflowing products.
    pub fn with_coherent(
        mut self,
        rate: f64,
        model: impl Into<Option<BTreeMap<String, f64>>>,
    ) -> Result<Self, PecosError> {
        if let Some(model) = model.into() {
            Self::validate_model(&model, &["RX", "RY", "RZ"])?;
            self.coherent_model = model;
        }
        self.coherent = rate;
        self.validate_idle(1.0)?;
        Ok(self)
    }
    fn validate_idle(&self, seconds: f64) -> Result<(), PecosError> {
        if [self.linear, self.sine, self.coherent]
            .into_iter()
            .any(|rate| !rate.is_finite() || rate < 0.0 || !(rate * seconds).is_finite())
            || !self
                .sine_model
                .values()
                .all(|m| (self.sine * m * seconds).is_finite())
            || !self
                .coherent_model
                .values()
                .all(|m| (self.coherent * m * seconds).is_finite())
        {
            return Err(error("invalid scheduled idle rate or duration product"));
        }
        Ok(())
    }
    pub(crate) fn build_model(self) -> ScheduledIdleModel {
        let mut builder = GeneralNoiseModelBuilder::new()
            .with_p_idle_linear(self.linear, &self.linear_model)
            .with_p_idle_sin_squared(self.sine, &self.sine_model)
            .with_p_idle_coherent(self.coherent, &self.coherent_model);
        if let Some(local) = &self.local {
            builder = builder
                .with_p1(local.p1)
                .with_p2(local.p2)
                .with_p_prep(local.prep)
                .with_prep_leak_ratio(local.prep_leak_ratio)
                .with_p_meas_0(local.meas0)
                .with_p_meas_1(local.meas1);
            if let Some(model) = &local.p2_pauli_model {
                builder = builder.with_p2_pauli_model(model);
            }
            builder = local.configure_angles(builder);
        }
        let inner = builder.build();
        ScheduledIdleModel {
            timeline: ScheduleTimeline::new(self.qubits),
            config: self,
            inner,
        }
    }
}
impl IntoNoiseModel for ScheduledNoise {
    fn into_noise_model(self) -> Result<Box<dyn NoiseModel>, PecosError> {
        Ok(Box::new(self.build_model()))
    }
}
pub(crate) fn fields(g: &Gate) -> Result<(u64, u64, u64, u64, u64), PecosError> {
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
        g.angles.first().map_or(0, Angle64::fraction),
        g.angles.get(1).map_or(0, Angle64::fraction),
    ))
}
/// Encode mandatory v6 transport: 16-byte header, 40-byte batch headers and gates.
/// Angle words are exact u64 fractions. All fields are little endian. No unknown records, implicit padding or events.
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
    bytes.extend_from_slice(&[
        crate::byte_message::protocol::SCHEDULED_FRAME_VERSION,
        0,
        0,
        0,
    ]);
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
            for v in [op, q0, q1, theta, phi] {
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
/// Validate the common scheduled header before interpreting version-specific records.
pub(crate) fn validate_scheduled_header(
    bytes: &[u8],
    expected_version: u32,
) -> Result<(), PecosError> {
    if bytes.len() < 16 {
        return Err(error(&format!(
            "scheduled message length {} is below the 16-byte header minimum; supply a complete scheduled message",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != crate::byte_message::protocol::BATCH_MAGIC {
        return Err(error(&format!(
            "scheduled magic {magic:#x} is invalid; encode the input with the scheduled message builder"
        )));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != expected_version {
        return Err(error(&format!(
            "scheduled version {version} is unsupported; rebuild the message with version {expected_version}, which stores angles as exact u64 fractions"
        )));
    }
    let declared = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    if bytes.len() > MAX_SCHEDULE_BYTES || declared != bytes.len() {
        return Err(error(&format!(
            "scheduled message length {} disagrees with declared length {declared} or exceeds limit {MAX_SCHEDULE_BYTES}; encode a complete message within the limit",
            bytes.len()
        )));
    }
    Ok(())
}

pub(crate) fn decode(input: &ByteMessage) -> Result<Vec<TimedBatch>, PecosError> {
    let bytes = input.as_bytes();
    validate_scheduled_header(
        bytes,
        u32::from(crate::byte_message::protocol::SCHEDULED_FRAME_VERSION),
    )?;
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
            let theta = word(bytes, offset + 24);
            let phi = word(bytes, offset + 32);
            let q0 = usize::try_from(q0).map_err(|_| error("target overflow"))?;
            let q1 = usize::try_from(q1).map_err(|_| error("target overflow"))?;
            let a = Angle64::new(theta);
            let p = Angle64::new(phi);
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
                || (op >= 5 && theta != 0)
                || (op != 1 && op != 4 && phi != 0)
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
    pub(crate) config: ScheduledNoise,
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
        message: &ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.inner.start_scheduled(message)
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
        config: &ScheduledNoise,
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
            if batch.gates.len() > crate::scheduled_events::MAX_BATCH_OPERATIONS {
                return Err(error("scheduled batch operation limit"));
            }
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
                        // All admitted idle channels are local. PZ erases
                        // their effect and clears leakage before any local prep fault,
                        // so do not sample a discarded channel.
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
            "scheduled idle profile requires mandatory v6 transport",
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
    #[test]
    fn scheduled_angles_keep_every_fraction_bit() {
        for fraction in [0, 1, u64::MAX, pecos_core::radians!(-0.73).fraction()] {
            let a = Angle64::new(fraction);
            let gates = vec![
                Gate::rz(a, &[0]),
                Gate::rxy1q(a, -a, &[1]),
                Gate::rxyxy2q(a, -a, &[(0, 1)]),
            ];
            let message = encode_timed_batches(&[batch(0, 0, 0, gates.clone())]).unwrap();
            assert_eq!(decode(&message).unwrap()[0].gates, gates);
            let mut old = message.as_bytes().to_vec();
            old[4] = 3;
            assert!(decode(&ByteMessage::new(&old)).is_err());
        }
    }

    fn config() -> ScheduledNoise {
        ScheduledIdleZ::new(2, 0.2, 0.3, 0.4).unwrap().into()
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

#[cfg(test)]
mod local_gate_contract {
    use super::*;
    #[test]
    fn emitted_rz_gets_a_pauli_fault_after_the_gate() {
        let profile: ScheduledNoise =
            ScheduledLocalNoise::new(ScheduledNoise::new(1).unwrap(), 1.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap()
                .into();
        let mut model = profile.build_model();
        let stage = model
            .start_admitted(
                &ByteMessage::quantum_operations_builder()
                    .rz(pecos_core::radians!(0.4), &[0])
                    .build(),
            )
            .unwrap();
        let EngineStage::NeedsProcessing(commands) = stage else {
            panic!("expected gates");
        };
        let gates = commands.quantum_ops().unwrap();
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].gate_type, GateType::RZ);
        assert!(matches!(
            gates[1].gate_type,
            GateType::X | GateType::Y | GateType::Z
        ));
        assert_eq!(gates[1].qubits, gates[0].qubits);
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;

    #[test]
    fn scheduled_headers_check_length_and_magic_before_version() {
        for version in [6_u32, 7] {
            let decode = |bytes: &[u8]| -> Result<(), PecosError> {
                let input = ByteMessage::new(bytes);
                if version == 6 {
                    super::decode(&input).map(|_| ())
                } else {
                    crate::scheduled_events::decode_event_batches(&input).map(|_| ())
                }
            };
            for bytes in [vec![], vec![0xff; 8]] {
                let error = decode(&bytes).unwrap_err().to_string();
                assert!(error.contains("length") && error.contains("16"), "{error}");
            }
            let error = decode(&[0xff; 16]).unwrap_err().to_string();
            assert!(error.contains("magic"), "{error}");
            let mut bytes = vec![0; 16];
            bytes[..4].copy_from_slice(&crate::byte_message::protocol::BATCH_MAGIC.to_le_bytes());
            bytes[4..8].copy_from_slice(&(version - 3).to_le_bytes());
            bytes[12..16].copy_from_slice(&16_u32.to_le_bytes());
            let error = decode(&bytes).unwrap_err().to_string();
            assert!(
                error.contains("version") && error.contains(&format!("version {version}")),
                "{error}"
            );
            bytes[4..8].copy_from_slice(&version.to_le_bytes());
            decode(&bytes).unwrap();
        }
    }
}
