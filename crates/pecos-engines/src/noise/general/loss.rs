//! Atom loss, independently configured from leakage.
use super::{GeneralNoiseModel, HookChannel, TransitionSite};
use crate::noise::noise_rng::NoiseRng;
use crate::noise::weighted_sampler::WeightedSampler;
use crate::{ByteMessageBuilder, Gate, GateType};
use pecos_core::errors::PecosError;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Gate site at which a loss channel runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossSite {
    /// Before a generic single-qubit operation.
    BeforeSingleQubit,
    /// After a generic single-qubit operation.
    AfterSingleQubit,
    /// Before a two-qubit operation.
    BeforeTwoQubit,
    /// After a two-qubit operation.
    AfterTwoQubit,
}

impl LossSite {
    fn arity(self) -> usize {
        match self {
            Self::BeforeSingleQubit | Self::AfterSingleQubit => 1,
            Self::BeforeTwoQubit | Self::AfterTwoQubit => 2,
        }
    }

    fn before(self) -> bool {
        matches!(self, Self::BeforeSingleQubit | Self::BeforeTwoQubit)
    }
}

#[derive(Debug, Clone)]
enum ChannelKind {
    Events {
        probability: f64,
        events: WeightedSampler<String>,
    },
    Recovery {
        false_negative: f64,
        false_positive: f64,
    },
}

#[derive(Debug, Clone)]
struct LossChannel {
    site: LossSite,
    kind: ChannelKind,
}

/// Products of the overall scale and each source's section scale.
#[derive(Clone, Copy)]
pub(super) struct LossScales {
    pub measurement: f64,
    pub prep: f64,
    pub idle: f64,
    pub single_qubit: f64,
    pub two_qubit: f64,
}

impl LossScales {
    fn at_site(self, site: LossSite) -> f64 {
        match site {
            LossSite::BeforeSingleQubit | LossSite::AfterSingleQubit => self.single_qubit,
            LossSite::BeforeTwoQubit | LossSite::AfterTwoQubit => self.two_qubit,
        }
    }
}

/// Validated loss configuration and shot-local atom state.
#[derive(Debug, Clone)]
pub struct QubitLoss {
    pub(super) measurement: f64,
    pub(super) prep: f64,
    pub(super) idle_rate: f64,
    reload_on_prep: bool,
    lost_partner_probability: f64,
    lost_partner_pauli: Option<WeightedSampler<String>>,
    channels: Vec<LossChannel>,
    lost: BTreeSet<usize>,
    own_event: BTreeSet<usize>,
    suppressed: BTreeSet<usize>,
}

#[derive(Debug, Clone)]
enum RawChannel {
    Events(LossSite, f64, BTreeMap<String, f64>),
    Recovery(LossSite, f64, f64),
}

/// Builder for independent atom-loss sources and recovery channels.
#[derive(Debug, Clone)]
pub struct QubitLossBuilder {
    measurement: f64,
    prep: f64,
    idle_rate: f64,
    reload_on_prep: bool,
    default_recovery: bool,
    lost_partner_probability: f64,
    lost_partner_pauli: Option<BTreeMap<String, f64>>,
    channels: Vec<RawChannel>,
}

impl Default for QubitLossBuilder {
    fn default() -> Self {
        Self {
            measurement: 0.0,
            prep: 0.0,
            idle_rate: 0.0,
            reload_on_prep: false,
            default_recovery: true,
            lost_partner_probability: 1.0,
            lost_partner_pauli: None,
            channels: Vec::new(),
        }
    }
}

fn invalid(field: &str, value: impl std::fmt::Display, fix: &str) -> PecosError {
    PecosError::Input(format!("loss.{field} = {value}: {fix}"))
}

fn probability(field: &str, value: f64) -> Result<(), PecosError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid(field, value, "use a finite probability in [0, 1]"))
    }
}

fn rate(value: f64) -> Result<(), PecosError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid(
            "idle_rate",
            value,
            "use a finite non-negative rate",
        ))
    }
}

fn distribution(
    field: &str,
    weights: &BTreeMap<String, f64>,
    arity: usize,
    pauli: bool,
) -> Result<WeightedSampler<String>, PecosError> {
    for (label, &weight) in weights {
        let valid = if pauli {
            label.len() == 1 && label.chars().all(|c| "IXYZ".contains(c))
        } else if arity == 1 {
            label.len() == 1 && label.chars().all(|c| "XYZL".contains(c))
        } else {
            label.len() == 2 && label != "II" && label.chars().all(|c| "IXYZL".contains(c))
        };
        if !valid {
            return Err(invalid(
                field,
                format!("{label:?}: {weight}"),
                if pauli {
                    "use I/X/Y/Z labels"
                } else if arity == 1 {
                    "use X/Y/Z/L labels at a single-qubit site"
                } else {
                    "use ordered pairs of I/X/Y/Z/L except II at a two-qubit site"
                },
            ));
        }
        probability(&format!("{field}.{label}"), weight)?;
    }
    WeightedSampler::try_new(weights).map_err(|error| {
        invalid(
            field,
            format!("{weights:?}"),
            &format!("provide a nonempty distribution summing to 1; {error}"),
        )
    })
}

impl QubitLossBuilder {
    /// Set measurement loss probability.
    #[must_use]
    pub fn with_measurement(mut self, value: f64) -> Self {
        self.measurement = value;
        self
    }

    /// Set preparation loss probability.
    #[must_use]
    pub fn with_prep(mut self, value: f64) -> Self {
        self.prep = value;
        self
    }

    /// Set loss rate per unit idle time.
    #[must_use]
    pub fn with_idle_rate(mut self, value: f64) -> Self {
        self.idle_rate = value;
        self
    }

    /// Reload missing atoms during PZ and MPZ preparation.
    #[must_use]
    pub fn with_reload_on_prep(mut self, value: bool) -> Self {
        self.reload_on_prep = value;
        self
    }

    /// Add default before-two-qubit recovery if no recovery was specified.
    #[must_use]
    pub fn with_default_recovery(mut self, value: bool) -> Self {
        self.default_recovery = value;
        self
    }

    /// Set the conditional loss probability next to an already-lost atom.
    #[must_use]
    pub fn with_lost_partner_loss_probability(mut self, value: f64) -> Self {
        self.lost_partner_probability = value;
        self
    }

    /// Set the survivor Pauli distribution when spreading does not lose it.
    #[must_use]
    pub fn with_lost_partner_pauli_model(mut self, weights: &BTreeMap<String, f64>) -> Self {
        self.lost_partner_pauli = Some(weights.clone());
        self
    }

    /// Append a joint loss/Pauli events channel.
    #[must_use]
    pub fn add_events_channel(
        mut self,
        site: LossSite,
        probability: f64,
        events: &BTreeMap<String, f64>,
    ) -> Self {
        self.channels
            .push(RawChannel::Events(site, probability, events.clone()));
        self
    }

    /// Append recovery with false-negative and false-positive detection rates.
    #[must_use]
    pub fn add_recovery_channel(
        mut self,
        site: LossSite,
        false_negative: f64,
        false_positive: f64,
    ) -> Self {
        self.channels
            .push(RawChannel::Recovery(site, false_negative, false_positive));
        self
    }

    /// Validate raw parameters and compile distributions.
    ///
    /// # Errors
    /// Names the invalid field, value and correction, including channel indices.
    pub fn build(self) -> Result<QubitLoss, PecosError> {
        probability("measurement", self.measurement)?;
        probability("prep", self.prep)?;
        rate(self.idle_rate)?;
        probability(
            "lost_partner.loss_probability",
            self.lost_partner_probability,
        )?;
        let lost_partner_pauli = self
            .lost_partner_pauli
            .as_ref()
            .map(|weights| distribution("lost_partner.pauli_model", weights, 1, true))
            .transpose()?;
        let mut channels = Vec::new();
        for (i, raw) in self.channels.into_iter().enumerate() {
            let (site, kind) = match raw {
                RawChannel::Events(site, p, weights) => {
                    probability(&format!("channels[{i}].probability"), p)?;
                    (
                        site,
                        ChannelKind::Events {
                            probability: p,
                            events: distribution(
                                &format!("channels[{i}].events"),
                                &weights,
                                site.arity(),
                                false,
                            )?,
                        },
                    )
                }
                RawChannel::Recovery(site, false_negative, false_positive) => {
                    probability(&format!("channels[{i}].false_negative"), false_negative)?;
                    probability(&format!("channels[{i}].false_positive"), false_positive)?;
                    (
                        site,
                        ChannelKind::Recovery {
                            false_negative,
                            false_positive,
                        },
                    )
                }
            };
            channels.push(LossChannel { site, kind });
        }
        if self.default_recovery
            && !channels
                .iter()
                .any(|c| matches!(c.kind, ChannelKind::Recovery { .. }))
        {
            channels.push(LossChannel {
                site: LossSite::BeforeTwoQubit,
                kind: ChannelKind::Recovery {
                    false_negative: 1e-3,
                    false_positive: 1e-3,
                },
            });
        }
        Ok(QubitLoss {
            measurement: self.measurement,
            prep: self.prep,
            idle_rate: self.idle_rate,
            reload_on_prep: self.reload_on_prep,
            lost_partner_probability: self.lost_partner_probability,
            lost_partner_pauli,
            channels,
            lost: BTreeSet::new(),
            own_event: BTreeSet::new(),
            suppressed: BTreeSet::new(),
        })
    }
}

// An empty key would otherwise render as a bare trailing separator.
fn key_name(key: &str) -> &str {
    if key.is_empty() { "\"\"" } else { key }
}

// Object prefixes end in a separator; literal unknown keys must retain theirs.
fn invalid_object(field: &str, value: &Value) -> PecosError {
    let path = format!("loss.{field}");
    PecosError::Input(format!(
        "{} = {value}: use a JSON object",
        path.trim_end_matches('.')
    ))
}

fn object<'a>(
    value: &'a Value,
    field: &str,
    keys: &[&str],
) -> Result<&'a Map<String, Value>, PecosError> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid_object(field, value))?;
    for (key, value) in obj {
        if !keys.contains(&key.as_str()) {
            return Err(invalid(
                &format!("{field}{}", key_name(key)),
                value,
                "remove this unknown key",
            ));
        }
    }
    Ok(obj)
}

fn boolean(field: &str, value: &Value) -> Result<bool, PecosError> {
    value
        .as_bool()
        .ok_or_else(|| invalid(field, value, "use a JSON boolean"))
}

fn number(field: &str, value: &Value) -> Result<f64, PecosError> {
    value
        .as_f64()
        .ok_or_else(|| invalid(field, value, "use a JSON number"))
}

fn weights(field: &str, value: &Value) -> Result<BTreeMap<String, f64>, PecosError> {
    value
        .as_object()
        .ok_or_else(|| {
            invalid(
                field,
                value,
                "use an object mapping labels to probabilities",
            )
        })?
        .iter()
        .map(|(k, v)| Ok((k.clone(), number(&format!("{field}.{}", key_name(k)), v)?)))
        .collect()
}

fn required<'a>(
    obj: &'a Map<String, Value>,
    name: &str,
    prefix: &str,
) -> Result<&'a Value, PecosError> {
    obj.get(name)
        .ok_or_else(|| PecosError::Input(format!("loss.{prefix}{name} is required")))
}

impl QubitLoss {
    /// Start with zero sources, full partner spreading, and default recovery.
    #[must_use]
    pub fn builder() -> QubitLossBuilder {
        QubitLossBuilder::default()
    }

    /// Parse the complete JSON loss section, shared with Python.
    ///
    /// # Errors
    /// Rejects unknown fields, wrong JSON types and invalid parameters.
    pub fn from_json_value(value: &Value) -> Result<Option<Self>, PecosError> {
        let obj = value.as_object().ok_or_else(|| invalid_object("", value))?;
        if !boolean("enabled", required(obj, "enabled", "")?)? {
            if let Some(key) = obj.keys().find(|k| *k != "enabled") {
                return Err(PecosError::Input(format!(
                    "loss.{} is set but loss.enabled is false",
                    key_name(key)
                )));
            }
            return Ok(None);
        }
        let obj = object(
            value,
            "",
            &[
                "enabled",
                "measurement",
                "prep",
                "idle_rate",
                "lost_partner",
                "reload_on_prep",
                "default_recovery",
                "channels",
            ],
        )?;
        let mut builder = Self::builder();
        if let Some(v) = obj.get("measurement") {
            builder = builder.with_measurement(number("measurement", v)?);
        }
        if let Some(v) = obj.get("prep") {
            builder = builder.with_prep(number("prep", v)?);
        }
        if let Some(v) = obj.get("idle_rate") {
            builder = builder.with_idle_rate(number("idle_rate", v)?);
        }
        if let Some(v) = obj.get("reload_on_prep") {
            builder = builder.with_reload_on_prep(boolean("reload_on_prep", v)?);
        }
        if let Some(v) = obj.get("default_recovery") {
            builder = builder.with_default_recovery(boolean("default_recovery", v)?);
        }
        if let Some(v) = obj.get("lost_partner") {
            let partner = object(v, "lost_partner.", &["loss_probability", "pauli_model"])?;
            if let Some(v) = partner.get("loss_probability") {
                builder = builder.with_lost_partner_loss_probability(number(
                    "lost_partner.loss_probability",
                    v,
                )?);
            }
            if let Some(v) = partner.get("pauli_model") {
                builder =
                    builder.with_lost_partner_pauli_model(&weights("lost_partner.pauli_model", v)?);
            }
        }
        if let Some(v) = obj.get("channels") {
            let channels = v
                .as_array()
                .ok_or_else(|| invalid("channels", v, "use a JSON array"))?;
            for (i, v) in channels.iter().enumerate() {
                let prefix = format!("channels[{i}].");
                let channel = v
                    .as_object()
                    .ok_or_else(|| invalid(&format!("channels[{i}]"), v, "use a channel object"))?;
                let kind = required(channel, "kind", &prefix)?;
                let keys = match kind.as_str() {
                    Some("events") => &["site", "kind", "probability", "events"][..],
                    Some("recovery") => &["site", "kind", "false_negative", "false_positive"][..],
                    _ => {
                        return Err(invalid(
                            &format!("{prefix}kind"),
                            kind,
                            "use events or recovery",
                        ));
                    }
                };
                object(v, &prefix, keys)?;
                let site_value = required(channel, "site", &prefix)?;
                let site = match site_value.as_str() {
                    Some("before_single_qubit") => LossSite::BeforeSingleQubit,
                    Some("after_single_qubit") => LossSite::AfterSingleQubit,
                    Some("before_two_qubit") => LossSite::BeforeTwoQubit,
                    Some("after_two_qubit") => LossSite::AfterTwoQubit,
                    _ => {
                        return Err(invalid(
                            &format!("{prefix}site"),
                            site_value,
                            "use before_single_qubit, after_single_qubit, before_two_qubit or after_two_qubit",
                        ));
                    }
                };
                let numeric = |name| {
                    number(
                        &format!("{prefix}{name}"),
                        required(channel, name, &prefix)?,
                    )
                };
                builder = if kind == "events" {
                    builder.add_events_channel(
                        site,
                        numeric("probability")?,
                        &weights(
                            &format!("{prefix}events"),
                            required(channel, "events", &prefix)?,
                        )?,
                    )
                } else {
                    builder.add_recovery_channel(
                        site,
                        numeric("false_negative")?,
                        numeric("false_positive")?,
                    )
                };
            }
        }
        builder.build().map(Some)
    }

    pub(super) fn scale(&mut self, scales: LossScales) {
        self.measurement *= scales.measurement;
        self.prep *= scales.prep;
        self.idle_rate *= scales.idle;
        for channel in &mut self.channels {
            if let ChannelKind::Events { probability, .. } = &mut channel.kind {
                *probability *= scales.at_site(channel.site);
            }
        }
    }

    // Validate from raw values so diagnostics can identify the scale product.
    pub(super) fn validate_scaled(&self, scales: LossScales) -> Result<(), String> {
        let check = |field: &str, raw: f64, scale: f64, is_rate: bool| {
            let scaled = raw * scale;
            if scaled.is_finite() && scaled >= 0.0 && (is_rate || scaled <= 1.0) {
                return Ok(());
            }
            let fix = if is_rate {
                "reduce the rate or scale factors to keep the scaled rate finite and non-negative"
            } else {
                "reduce the probability or scale factors to keep the scaled probability finite in [0, 1]"
            };
            Err(format!(
                "loss.{field} after scaling = {scaled} ({raw} x scale {scale}): {fix}"
            ))
        };
        check("measurement", self.measurement, scales.measurement, false)?;
        check("prep", self.prep, scales.prep, false)?;
        check("idle_rate", self.idle_rate, scales.idle, true)?;
        for (i, channel) in self.channels.iter().enumerate() {
            if let ChannelKind::Events { probability, .. } = channel.kind {
                check(
                    &format!("channels[{i}].probability"),
                    probability,
                    scales.at_site(channel.site),
                    false,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn compose(&self, site: LossSite, order: &mut Vec<HookChannel>) {
        let mut first = Vec::new();
        let mut last = Vec::new();
        for (i, c) in self
            .channels
            .iter()
            .enumerate()
            .filter(|(_, c)| c.site == site)
        {
            if site.before() && matches!(c.kind, ChannelKind::Events { .. }) {
                last.push(HookChannel::Loss(i));
            } else {
                first.push(HookChannel::Loss(i));
            }
        }
        first.append(order);
        first.extend(last);
        *order = first;
    }

    fn event(
        lost: &mut BTreeSet<usize>,
        qubit: usize,
        leaked: &mut BTreeSet<usize>,
        output: &mut Vec<Gate>,
    ) {
        output.push(Gate::pz(&[qubit]));
        lost.insert(qubit);
        leaked.remove(&qubit);
    }

    fn recover(
        lost: &mut BTreeSet<usize>,
        qubit: usize,
        leaked: &mut BTreeSet<usize>,
        rng: &mut NoiseRng,
        output: &mut Vec<Gate>,
    ) {
        output.push(Gate::pz(&[qubit]));
        if rng.occurs(0.5) {
            output.push(Gate::x(&[qubit]));
        }
        lost.remove(&qubit);
        leaked.remove(&qubit);
    }

    fn channel(
        &mut self,
        index: usize,
        gate: &Gate,
        leaked: &mut BTreeSet<usize>,
        rng: &mut NoiseRng,
    ) -> Vec<Gate> {
        let Self {
            channels,
            lost,
            own_event,
            ..
        } = self;
        let channel = &channels[index];
        let mut output = Vec::new();
        for group in gate.qubits.chunks(channel.site.arity()) {
            match &channel.kind {
                ChannelKind::Recovery {
                    false_negative,
                    false_positive,
                } => {
                    for q in group {
                        let q = usize::from(*q);
                        let reset = if lost.contains(&q) {
                            !occurs(rng, *false_negative)
                        } else {
                            occurs(rng, *false_positive)
                        };
                        if reset {
                            Self::recover(lost, q, leaked, rng, &mut output);
                        }
                    }
                }
                ChannelKind::Events {
                    probability,
                    events,
                } => {
                    if group.iter().any(|q| lost.contains(&usize::from(*q)))
                        || !occurs(rng, *probability)
                    {
                        continue;
                    }
                    let label = events.sample(rng);
                    for (q, event) in group.iter().zip(label.chars()) {
                        let q = usize::from(*q);
                        if event == 'L' {
                            Self::event(lost, q, leaked, &mut output);
                            if channel.site == LossSite::BeforeTwoQubit {
                                own_event.extend(group.iter().map(|q| usize::from(*q)));
                            }
                        } else if !leaked.contains(&q) {
                            pauli(event, q, &mut output);
                        }
                    }
                }
            }
        }
        output
    }

    fn spread(
        &mut self,
        qubits: &[pecos_core::QubitId],
        leaked: &mut BTreeSet<usize>,
        rng: &mut NoiseRng,
        output: &mut Vec<Gate>,
    ) {
        self.suppressed
            .extend(qubits.iter().map(|q| usize::from(*q)));
        if qubits
            .iter()
            .any(|q| self.own_event.contains(&usize::from(*q)))
        {
            return;
        }
        for q in qubits {
            let q = usize::from(*q);
            if self.lost.contains(&q) {
                continue;
            }
            if occurs(rng, self.lost_partner_probability) {
                Self::event(&mut self.lost, q, leaked, output);
            } else if let Some(model) = &self.lost_partner_pauli {
                let label = model.sample(rng);
                if !leaked.contains(&q) {
                    pauli(label.as_bytes()[0] as char, q, output);
                }
            }
        }
    }

    pub(super) fn clear_command(&mut self) {
        self.own_event.clear();
        self.suppressed.clear();
    }

    pub(super) fn reset(&mut self) {
        self.lost.clear();
        self.clear_command();
    }
}

fn occurs(rng: &mut NoiseRng, p: f64) -> bool {
    p > 0.0 && rng.occurs(p)
}

fn pauli(event: char, q: usize, output: &mut Vec<Gate>) {
    match event {
        'X' => output.push(Gate::x(&[q])),
        'Y' => output.push(Gate::y(&[q])),
        'Z' => output.push(Gate::z(&[q])),
        'I' => (),
        _ => unreachable!("validated loss Pauli"),
    }
}

/// Physical state captured when a tracked measurement is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MeasurementState {
    Present,
    Leaked,
    Lost,
}

impl GeneralNoiseModel {
    /// Whether independent atom loss is configured.
    #[must_use]
    pub fn has_loss(&self) -> bool {
        self.loss.is_some()
    }

    pub(super) fn is_lost(&self, q: usize) -> bool {
        self.loss
            .as_ref()
            .is_some_and(|loss| loss.lost.contains(&q))
    }

    pub(super) fn measurement_state(&self, q: usize) -> MeasurementState {
        if self.is_lost(q) {
            MeasurementState::Lost
        } else if self.is_leaked(q) {
            MeasurementState::Leaked
        } else {
            MeasurementState::Present
        }
    }

    pub(super) fn loss_source(
        &mut self,
        qubits: &[pecos_core::QubitId],
        probability: f64,
        builder: &mut ByteMessageBuilder,
    ) {
        if let Some(loss) = &mut self.loss {
            let mut output = Vec::new();
            for q in qubits {
                let q = usize::from(*q);
                if !loss.lost.contains(&q) && occurs(&mut self.rng, probability) {
                    QubitLoss::event(&mut loss.lost, q, &mut self.leaked_qubits, &mut output);
                }
            }
            builder.add_gate_commands(&output);
        }
    }

    pub(super) fn reload(&mut self, q: usize) {
        if let Some(loss) = &mut self.loss
            && loss.reload_on_prep
        {
            loss.lost.remove(&q);
            self.leaked_qubits.remove(&q);
        }
    }

    pub(super) fn clear_loss_command(&mut self) {
        if let Some(loss) = &mut self.loss {
            loss.clear_command();
        }
    }

    pub(super) fn suppress_for_loss(&mut self, qubits: &[pecos_core::QubitId]) {
        if let Some(loss) = &mut self.loss {
            loss.suppressed
                .extend(qubits.iter().map(|q| usize::from(*q)));
        }
    }

    pub(super) fn spread_loss(&mut self, qubits: &[pecos_core::QubitId], output: &mut Vec<Gate>) {
        if let Some(loss) = &mut self.loss {
            loss.spread(qubits, &mut self.leaked_qubits, &mut self.rng, output);
        }
    }

    pub(super) fn loss_hook(
        &mut self,
        index: usize,
        gate: &Gate,
        builder: &mut ByteMessageBuilder,
    ) {
        if let Some(loss) = &mut self.loss {
            builder.add_gate_commands(&loss.channel(
                index,
                gate,
                &mut self.leaked_qubits,
                &mut self.rng,
            ));
        }
    }

    pub(super) fn skip_loss_hook(
        &self,
        gate: &Gate,
        qubits: &[usize],
        site: TransitionSite,
    ) -> bool {
        let Some(loss) = &self.loss else {
            return false;
        };
        let after = matches!(site, TransitionSite::AfterP1 | TransitionSite::AfterP2);
        let skip = |q: &usize| loss.lost.contains(q) || (after && loss.suppressed.contains(q));
        if gate.gate_type == GateType::Custom {
            gate.qubits
                .iter()
                .map(|q| usize::from(*q))
                .any(|q| skip(&q))
        } else {
            qubits.iter().any(skip)
        }
    }

    pub(super) fn lost_readout(&mut self, flavor: GateType, outcome: u32) -> usize {
        if !self.noiseless_gates.contains(&flavor)
            && let Some(value) = self.configured_readout(flavor, outcome, MeasurementState::Lost)
        {
            return value;
        }
        if matches!(
            flavor,
            GateType::MeasureZLost | GateType::MeasureZLeakedLost
        ) {
            3
        } else {
            usize::from(!self.noiseless_gates.contains(&flavor) && self.rng.occurs(self.p_meas_0))
        }
    }
}

impl GeneralNoiseModel {
    // Measurements always retain all operands; Custom is one operation.
    pub(super) fn filter_lost_gate<'a>(
        &self,
        gate: &'a Gate,
    ) -> Option<std::borrow::Cow<'a, Gate>> {
        if self.loss.is_none() || super::is_tracked_measurement(gate.gate_type) {
            return Some(std::borrow::Cow::Borrowed(gate));
        }
        let arity = if gate.gate_type == GateType::Custom {
            gate.qubits.len().max(1)
        } else {
            gate.gate_type.quantum_arity().max(1)
        };
        let mut filtered = gate.clone();
        filtered.qubits = gate
            .qubits
            .chunks(arity)
            .filter(|group| !group.iter().any(|q| self.is_lost(usize::from(*q))))
            .flatten()
            .copied()
            .collect();
        if !gate.qubits.is_empty() && filtered.qubits.is_empty() {
            None
        } else {
            Some(std::borrow::Cow::Owned(filtered))
        }
    }

    pub(super) fn loss_preparation<'a>(
        &mut self,
        gate: &'a Gate,
    ) -> Option<std::borrow::Cow<'a, Gate>> {
        for q in &gate.qubits {
            self.reload(usize::from(*q));
        }
        self.filter_lost_gate(gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::GeneralNoiseConfig;
    use serde_json::json;

    fn recovery_count(loss: &QubitLoss) -> usize {
        loss.channels
            .iter()
            .filter(|c| matches!(c.kind, ChannelKind::Recovery { .. }))
            .count()
    }

    #[test]
    fn t08_default_recovery() {
        let default = QubitLoss::from_json_value(&json!({"enabled":true}))
            .unwrap()
            .unwrap();
        assert_eq!(default.channels.len(), 1);
        assert_eq!(default.channels[0].site, LossSite::BeforeTwoQubit);
        assert!(matches!(
            default.channels[0].kind,
            ChannelKind::Recovery {
                false_negative: 1e-3,
                false_positive: 1e-3
            }
        ));
        let explicit = QubitLoss::builder()
            .add_recovery_channel(LossSite::AfterSingleQubit, 0.2, 0.3)
            .build()
            .unwrap();
        assert_eq!(recovery_count(&explicit), 1);
        assert_eq!(explicit.channels[0].site, LossSite::AfterSingleQubit);
        assert_eq!(
            recovery_count(
                &QubitLoss::builder()
                    .with_default_recovery(false)
                    .build()
                    .unwrap()
            ),
            0
        );
    }

    #[test]
    fn t08_published_numeric_instances() {
        let pl = 0.01;
        let pc = 0.2;
        let pd = 0.003;
        let r = 4.0 * pd / 15.0;
        let w = (1.0 - pl) / (2.0 - pl);
        let p_readout = 0.002;
        let q = p_readout / 2.0;
        let p_pauli = 0.001;
        let p_cz: f64 = 0.01;
        let prime = 1.0 - (1.0 - p_cz).sqrt();
        let bluvstein_q: f64 = 0.00183;
        let mut instances = vec![
            json!({"loss":{"enabled":true,"default_recovery":false,"reload_on_prep":true,"lost_partner":{"loss_probability":1},"channels":[{"site":"before_two_qubit","kind":"events","probability":pl,"events":{"LL":pc,"LI":(1.0-pc)*3.0/16.0,"IL":(1.0-pc)*3.0/16.0,"LZ":(1.0-pc)*3.0/16.0,"ZL":(1.0-pc)*3.0/16.0,"LX":(1.0-pc)/16.0,"XL":(1.0-pc)/16.0,"LY":(1.0-pc)/16.0,"YL":(1.0-pc)/16.0}}]}}),
            json!({"loss":{"enabled":true,"default_recovery":false,"reload_on_prep":true,"lost_partner":{"loss_probability":0,"pauli_model":{"I":1.0-p_pauli,"X":p_pauli/3.0,"Y":p_pauli/3.0,"Z":p_pauli/3.0}},"channels":[{"site":"after_two_qubit","kind":"events","probability":1.0-(1.0-q)*(1.0-q),"events":{"LI":(1.0-q)/(2.0-q),"IL":(1.0-q)/(2.0-q),"LL":q/(2.0-q)}}]},"measurement":{"readout":{"MeasureZLost":{"0":{"0":1.0-p_readout/2.0,"3":p_readout/2.0},"1":{"1":1.0-p_readout/2.0,"3":p_readout/2.0},"lost":{"3":1.0-p_readout/2.0,"0":p_readout/4.0,"1":p_readout/4.0}}}}}),
            json!({"loss":{"enabled":true,"default_recovery":false,"reload_on_prep":true,"lost_partner":{"loss_probability":0},"channels":[{"site":"before_two_qubit","kind":"events","probability":2.0*prime,"events":{"LI":0.25,"IL":0.25,"LZ":0.25,"ZL":0.25}}]}}),
            json!({"loss":{"enabled":true,"default_recovery":false,"reload_on_prep":true,"measurement":0.006,"prep":0.006,"idle_rate":0.09e-6,"lost_partner":{"loss_probability":bluvstein_q,"pauli_model":{"I":0.9984,"X":5e-5,"Y":5e-5,"Z":0.0015}},"channels":[{"site":"after_two_qubit","kind":"events","probability":1.0-(1.0-bluvstein_q).powi(2),"events":{"LI":(1.0-bluvstein_q)/(2.0-bluvstein_q),"IL":(1.0-bluvstein_q)/(2.0-bluvstein_q),"LL":bluvstein_q/(2.0-bluvstein_q)}}]}}),
            json!({"loss":{"enabled":true,"default_recovery":false,"reload_on_prep":false,"idle_rate":pl,"lost_partner":{"loss_probability":0}}}),
        ];
        for refined in [false, true] {
            let identity = if refined {
                (1.0 - 2.0 * r) / 2.0
            } else {
                1.0 - 3.0 * r
            };
            let z = if refined { (1.0 - 2.0 * r) / 2.0 } else { r };
            instances.push(json!({"loss":{"enabled":true,"default_recovery":false,"reload_on_prep":true,"lost_partner":{"loss_probability":pl,"pauli_model":{"I":1.0-3.0*r,"X":r,"Y":r,"Z":r}},"channels":[{"site":"before_two_qubit","kind":"events","probability":pl*(2.0-pl),"events":{"LL":pl/(2.0-pl),"LI":w*identity,"IL":w*identity,"LX":w*r,"XL":w*r,"LY":w*r,"YL":w*r,"LZ":w*z,"ZL":w*z}}]}}));
        }
        for value in instances {
            let model = GeneralNoiseConfig::from_json(&value.to_string())
                .unwrap()
                .build()
                .unwrap();
            assert_eq!(recovery_count(model.loss.as_ref().unwrap()), 0, "{value}");
        }
    }

    #[test]
    fn t16_loss_clears_leak_and_prevents_releaking() {
        let mut model = GeneralNoiseModel::builder()
            .with_loss(
                QubitLoss::builder()
                    .with_default_recovery(false)
                    .with_measurement(1.0)
                    .build()
                    .unwrap(),
            )
            .with_leakage_scale(0.5)
            .build();
        model.mark_as_leaked(0);
        let mut output = ByteMessageBuilder::new();
        model.loss_source(&[0.into()], 1.0, &mut output);
        assert!(!model.leaked_qubits.contains(&0));
        assert!(model.is_lost(0));
        let rng = model.rng.clone();
        model.mark_as_leaked(0);
        model.apply_channel_leakage(0, &mut output);
        assert!(model.leak(0).is_none());
        assert!(!model.leaked_qubits.contains(&0));
        assert_eq!(model.rng.inner(), rng.inner());
        assert_eq!(output.build().quantum_ops().unwrap(), [Gate::pz(&[0])]);
        assert_eq!(model.prepared_qubits, BTreeSet::new());
        model.reset_noise_model();
        model.mark_as_leaked(0);
        let mut input = crate::ByteMessage::quantum_operations_builder();
        input.mz(&[0]);
        model.apply_noise_on_start(&input.build()).unwrap();
        assert!(model.is_lost(0));
        assert!(!model.leaked_qubits.contains(&0));
        assert_eq!(model.measured_qubits[0].2, MeasurementState::Leaked);
    }

    #[test]
    fn t21_scaling_exact_products_and_conditionals() {
        let events = BTreeMap::from([("L".into(), 0.4), ("X".into(), 0.6)]);
        let pair_events = BTreeMap::from([("LI".into(), 1.0)]);
        let pauli = BTreeMap::from([("Z".into(), 1.0)]);
        let raw = QubitLoss::builder()
            .with_measurement(0.8)
            .with_prep(0.7)
            .with_idle_rate(4.0)
            .with_lost_partner_loss_probability(0.9)
            .with_lost_partner_pauli_model(&pauli)
            .add_events_channel(LossSite::BeforeSingleQubit, 0.6, &events)
            .add_recovery_channel(LossSite::BeforeTwoQubit, 0.1, 0.2)
            .add_events_channel(LossSite::AfterTwoQubit, 0.5, &pair_events)
            .build()
            .unwrap();
        let model = GeneralNoiseModel::builder()
            .with_loss(raw.clone())
            .with_scale(0.5)
            .with_meas_scale(0.5)
            .with_prep_scale(0.4)
            .with_idle_scale(0.25)
            .with_p1_scale(0.2)
            .with_p2_scale(0.1)
            .build();
        let scaled = model.loss.unwrap();
        assert_eq!(scaled.measurement, 0.8 * (0.5 * 0.5));
        assert_eq!(scaled.prep, 0.7 * (0.4 * 0.5));
        assert_eq!(scaled.idle_rate, 4.0 * (0.25 * 0.5));
        assert_eq!(
            scaled.lost_partner_probability,
            raw.lost_partner_probability
        );
        assert_eq!(
            scaled.lost_partner_pauli.unwrap().get_weighted_map(),
            &pauli
        );
        for (i, scale, weights) in [(0, 0.2 * 0.5, &events), (2, 0.1 * 0.5, &pair_events)] {
            let ChannelKind::Events {
                probability: p,
                events: ref sampler,
            } = scaled.channels[i].kind
            else {
                panic!()
            };
            let ChannelKind::Events {
                probability: original,
                ..
            } = raw.channels[i].kind
            else {
                panic!()
            };
            assert_eq!(p, original * scale);
            assert_eq!(sampler.get_weighted_map(), weights);
        }
        assert!(matches!(
            scaled.channels[1].kind,
            ChannelKind::Recovery {
                false_negative: 0.1,
                false_positive: 0.2
            }
        ));
    }

    fn commands(gates: &[Gate]) -> crate::ByteMessage {
        let mut input = crate::ByteMessage::quantum_operations_builder();
        input.add_gate_commands(gates);
        input.build()
    }

    #[test]
    fn prep_lost_operand_remains_prepared() {
        let mut model = GeneralNoiseModel::builder()
            .with_loss(
                QubitLoss::builder()
                    .with_default_recovery(false)
                    .with_prep(1.0)
                    .build()
                    .unwrap(),
            )
            .build();
        model
            .apply_noise_on_start(&commands(&[Gate::pz(&[0])]))
            .unwrap();
        assert!(model.is_lost(0));
        assert!(model.prepared_qubits.contains(&0));
    }

    #[test]
    fn prep_faults_then_loss_then_crosstalk_rng_order() {
        for seed in 0..16 {
            let mut model = GeneralNoiseModel::builder()
                .with_seed(seed)
                .with_p_prep(0.5)
                .with_prep_leak_ratio(0.0)
                .with_p_prep_crosstalk(0.5)
                .with_loss(
                    QubitLoss::builder()
                        .with_default_recovery(false)
                        .with_prep(0.5)
                        .build()
                        .unwrap(),
                )
                .build();
            model.prepared_qubits.insert(2);
            let mut expected_rng = model.rng.clone();
            let mut expected = vec![Gate::pz(&[0, 1])];
            for q in [0, 1] {
                if expected_rng.occurs(0.5) {
                    expected_rng.occurs(0.0); // Existing prep leak selection consumes a draw.
                    expected.push(Gate::x(&[q]));
                }
            }
            for q in [0, 1] {
                if expected_rng.occurs(0.5) {
                    expected.push(Gate::pz(&[q]));
                }
            }
            if expected_rng.occurs(0.5) {
                expected.push(Gate::mz(&[2]));
            }
            let actual = model
                .apply_noise_on_start(&commands(&[Gate::pz(&[0, 1])]))
                .unwrap();
            assert_eq!(actual.quantum_ops().unwrap(), expected, "seed {seed}");
            let mut actual_rng = model.rng.clone();
            let actual_tail: [u64; 4] = std::array::from_fn(|_| actual_rng.inner_mut().next_u64());
            let expected_tail: [u64; 4] =
                std::array::from_fn(|_| expected_rng.inner_mut().next_u64());
            assert_eq!(
                actual_tail, expected_tail,
                "baseline clone: prep faults, prep loss, then crosstalk"
            );
        }
    }

    #[test]
    fn reset_abandoned_after_hook_clears_command_flags() {
        use crate::noise::{P2TransitionStep, PauliLeakageChannel, StateTransitionChannel};
        use crate::{ControlEngine, EngineStage};
        let transition = StateTransitionChannel::from_transitions(
            1.0,
            &BTreeMap::from([("0->0".into(), 1.0), ("1->1".into(), 1.0)]),
        )
        .unwrap();
        let z = PauliLeakageChannel::new(1.0, &BTreeMap::from([("Z".into(), 1.0)])).unwrap();
        // Find a reproducible shot with one own-event pair suppressed and another
        // pair awaiting an after-site transition measurement.
        let mut model = (0..32)
            .find_map(|seed| {
                let mut model = GeneralNoiseModel::builder()
                    .with_seed(seed)
                    .with_loss(
                        QubitLoss::builder()
                            .with_default_recovery(false)
                            .add_events_channel(
                                LossSite::BeforeTwoQubit,
                                0.5,
                                &BTreeMap::from([("LI".into(), 1.0)]),
                            )
                            .build()
                            .unwrap(),
                    )
                    .add_p1_pauli_leakage_channel_after_gate(&z)
                    .add_p2_transition_step_after_gate(&P2TransitionStep::independent(
                        transition.clone(),
                        transition.clone(),
                    ))
                    .build();
                let stage = model
                    .start(commands(&[Gate::cx(&[(0, 1), (2, 3)])]))
                    .unwrap();
                let loss = model.loss.as_ref().unwrap();
                (matches!(stage, EngineStage::NeedsProcessing(_))
                    && model.pending_hook.is_some()
                    && !loss.own_event.is_empty()
                    && !loss.suppressed.is_empty())
                .then_some(model)
            })
            .expect("a seed with both an own event and a deferred after hook");
        let q = *model.loss.as_ref().unwrap().own_event.first().unwrap();
        ControlEngine::reset(&mut model).unwrap();
        let mut spreading = model.clone();
        let output = model
            .apply_noise_on_start(&commands(&[Gate::h(&[q])]))
            .unwrap();
        assert!(
            output.quantum_ops().unwrap().contains(&Gate::z(&[q])),
            "next 1q after-hook must not inherit suppression"
        );
        let mut loss_output = ByteMessageBuilder::new();
        spreading.loss_source(&[q.into()], 1.0, &mut loss_output);
        spreading
            .apply_noise_on_start(&commands(&[Gate::cx(&[(q, q + 1)])]))
            .unwrap();
        assert!(
            spreading.is_lost(q + 1),
            "next pair must spread: its loss predates this command"
        );
    }
}
