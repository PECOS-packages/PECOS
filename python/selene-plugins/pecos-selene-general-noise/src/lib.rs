//! Device-neutral Selene adapter for PECOS's general noise model.

mod simulator;

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use pecos_core::Angle64;
use pecos_core::gate_type::GateType;
use pecos_engines::prelude::*;
use selene_core::error_model::BatchResult;
use selene_core::error_model::interface::{ErrorModelInterface, ErrorModelInterfaceFactory};
use selene_core::export_error_model_plugin;
use selene_core::runtime::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;
use selene_core::utils::MetricValue;
use serde::Deserialize;

use crate::simulator::SeleneSimulator;

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PreparationConfig {
    probability: Option<f64>,
    leakage_ratio: Option<f64>,
    crosstalk_probability: Option<f64>,
    average_crosstalk_probability: Option<f64>,
    scale: Option<f64>,
    crosstalk_scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct GateConfig {
    probability: Option<f64>,
    average_infidelity: Option<f64>,
    pauli_model: Option<BTreeMap<String, f64>>,
    emission_ratio: Option<f64>,
    emission_model: Option<BTreeMap<String, f64>>,
    seepage_probability: Option<f64>,
    scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TwoQubitConfig {
    probability: Option<f64>,
    average_infidelity: Option<f64>,
    pauli_model: Option<BTreeMap<String, f64>>,
    emission_ratio: Option<f64>,
    emission_model: Option<BTreeMap<String, f64>>,
    seepage_probability: Option<f64>,
    scale: Option<f64>,
    angle_coefficients: Option<[f64; 4]>,
    angle_power: Option<f64>,
    idle_after_gate: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct IdleConfig {
    linear_rate: Option<f64>,
    linear_model: Option<BTreeMap<String, f64>>,
    sin_squared_rate: Option<f64>,
    sin_squared_model: Option<BTreeMap<String, f64>>,
    coherent_rate: Option<f64>,
    coherent_model: Option<BTreeMap<String, f64>>,
    scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MeasurementConfig {
    p0_to_1: Option<f64>,
    p1_to_0: Option<f64>,
    global_crosstalk_probability: Option<f64>,
    local_crosstalk_probability: Option<f64>,
    crosstalk_model: Option<BTreeMap<String, f64>>,
    local_groups: Vec<Vec<usize>>,
    scale: Option<f64>,
    crosstalk_scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ScalingConfig {
    overall: Option<f64>,
    leakage: Option<f64>,
    emission: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Config {
    preparation: PreparationConfig,
    measurement: MeasurementConfig,
    single_qubit: GateConfig,
    two_qubit: TwoQubitConfig,
    idle: IdleConfig,
    scaling: ScalingConfig,
    noiseless_gates: Vec<String>,
}

impl Config {
    fn build_model(&self) -> Result<Box<dyn NoiseModel>> {
        catch_unwind(AssertUnwindSafe(|| {
            let mut builder = GeneralNoiseModel::builder();

            if let Some(value) = self.scaling.overall {
                builder = builder.with_scale(value);
            }
            if let Some(value) = self.scaling.leakage {
                builder = builder.with_leakage_scale(value);
            }
            if let Some(value) = self.scaling.emission {
                builder = builder.with_emission_scale(value);
            }
            for name in &self.noiseless_gates {
                let gate = GateType::from_str(name)
                    .unwrap_or_else(|message| panic!("invalid noiseless gate {name:?}: {message}"));
                builder = builder.with_noiseless_gate(gate);
            }

            let prep = &self.preparation;
            assert!(
                prep.crosstalk_probability.is_none()
                    || prep.average_crosstalk_probability.is_none(),
                "preparation accepts crosstalk_probability or average_crosstalk_probability, not both"
            );
            if let Some(value) = prep.probability {
                builder = builder.with_p_prep(value);
            }
            if let Some(value) = prep.leakage_ratio {
                builder = builder.with_prep_leak_ratio(value);
            }
            if let Some(value) = prep.crosstalk_probability {
                builder = builder.with_p_prep_crosstalk(value);
            }
            if let Some(value) = prep.average_crosstalk_probability {
                assert!(
                    value.is_finite() && (0.0..=5.0 / 18.0).contains(&value),
                    "average preparation crosstalk probability must be between 0 and 5/18"
                );
                builder = builder.with_average_p_prep_crosstalk(value);
            }
            if let Some(value) = prep.scale {
                builder = builder.with_prep_scale(value);
            }
            if let Some(value) = prep.crosstalk_scale {
                builder = builder.with_p_prep_crosstalk_scale(value);
            }

            let one = &self.single_qubit;
            assert!(
                one.probability.is_none() || one.average_infidelity.is_none(),
                "single_qubit accepts probability or average_infidelity, not both"
            );
            if let Some(value) = one.probability {
                builder = builder.with_p1(value);
            }
            if let Some(value) = one.average_infidelity {
                builder = builder.with_average_p1(value);
            }
            if let Some(value) = &one.pauli_model {
                builder = builder.with_p1_pauli_model(value);
            }
            if let Some(value) = one.emission_ratio {
                builder = builder.with_p1_emission_ratio(value);
            }
            if let Some(value) = &one.emission_model {
                builder = builder.with_p1_emission_model(value);
            }
            if let Some(value) = one.seepage_probability {
                builder = builder.with_p1_seepage_prob(value);
            }
            if let Some(value) = one.scale {
                builder = builder.with_p1_scale(value);
            }

            let two = &self.two_qubit;
            assert!(
                two.probability.is_none() || two.average_infidelity.is_none(),
                "two_qubit accepts probability or average_infidelity, not both"
            );
            if let Some(value) = two.probability {
                builder = builder.with_p2(value);
            }
            if let Some(value) = two.average_infidelity {
                builder = builder.with_average_p2(value);
            }
            if let Some(value) = &two.pauli_model {
                builder = builder.with_p2_pauli_model(value);
            }
            if let Some(value) = two.emission_ratio {
                builder = builder.with_p2_emission_ratio(value);
            }
            if let Some(value) = &two.emission_model {
                builder = builder.with_p2_emission_model(value);
            }
            if let Some(value) = two.seepage_probability {
                builder = builder.with_p2_seepage_prob(value);
            }
            if let Some(value) = two.scale {
                builder = builder.with_p2_scale(value);
            }
            if let Some([a, b, c, d]) = two.angle_coefficients {
                builder = builder.with_p2_angle_params(a, b, c, d);
            }
            if let Some(value) = two.angle_power {
                builder = builder.with_p2_angle_power(value);
            }
            if let Some(value) = two.idle_after_gate {
                builder = builder.with_idle_after_2q(value);
            }

            let idle = &self.idle;
            if let Some(rate) = idle.linear_rate {
                let model = idle.linear_model.clone().unwrap_or_else(|| {
                    BTreeMap::from([
                        ("X".to_string(), 1.0 / 3.0),
                        ("Y".to_string(), 1.0 / 3.0),
                        ("Z".to_string(), 1.0 / 3.0),
                    ])
                });
                builder = builder.with_p_idle_linear(rate, &model);
            } else if idle.linear_model.is_some() {
                panic!("idle.linear_model requires idle.linear_rate");
            }
            if let Some(rate) = idle.sin_squared_rate {
                let model = idle
                    .sin_squared_model
                    .clone()
                    .unwrap_or_else(|| BTreeMap::from([("Z".to_string(), 1.0)]));
                builder = builder.with_p_idle_sin_squared(rate, &model);
            } else if idle.sin_squared_model.is_some() {
                panic!("idle.sin_squared_model requires idle.sin_squared_rate");
            }
            if let Some(rate) = idle.coherent_rate {
                let model = idle
                    .coherent_model
                    .clone()
                    .unwrap_or_else(|| BTreeMap::from([("RZ".to_string(), 1.0)]));
                builder = builder.with_p_idle_coherent(rate, &model);
            } else if idle.coherent_model.is_some() {
                panic!("idle.coherent_model requires idle.coherent_rate");
            }
            if let Some(value) = idle.scale {
                builder = builder.with_idle_scale(value);
            }

            let measurement = &self.measurement;
            if let Some(value) = measurement.p0_to_1 {
                builder = builder.with_p_meas_0(value);
            }
            if let Some(value) = measurement.p1_to_0 {
                builder = builder.with_p_meas_1(value);
            }
            if let Some(value) = measurement.global_crosstalk_probability {
                builder = builder.with_p_meas_crosstalk_global(value);
            }
            if let Some(value) = measurement.local_crosstalk_probability {
                builder = builder.with_p_meas_crosstalk_local(value);
            }
            if let Some(value) = &measurement.crosstalk_model {
                builder = builder.with_p_meas_crosstalk_model(value);
            }
            if let Some(value) = measurement.scale {
                builder = builder.with_meas_scale(value);
            }
            if let Some(value) = measurement.crosstalk_scale {
                builder = builder.with_p_meas_crosstalk_scale(value);
            }

            Box::new(builder.build()) as Box<dyn NoiseModel>
        }))
        .map_err(|payload| {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown validation failure");
            anyhow!("invalid general-noise configuration: {message}")
        })
    }
}

#[derive(Clone, Copy)]
enum MeasurementKind {
    Bool,
    Leakage,
}

struct MeasurementResult {
    id: u64,
    kind: MeasurementKind,
}

struct GeneralNoiseErrorModel {
    model: Box<dyn NoiseModel>,
    builder: ByteMessageBuilder,
    last_operation_end: Vec<u64>,
    local_groups: Vec<BTreeSet<usize>>,
}

impl GeneralNoiseErrorModel {
    fn new(model: Box<dyn NoiseModel>, n_qubits: usize, config: &Config) -> Result<Self> {
        for group in &config.measurement.local_groups {
            if let Some(qubit) = group.iter().find(|qubit| **qubit >= n_qubits) {
                bail!(
                    "local crosstalk group contains qubit {qubit}, but the simulation has {n_qubits} qubits"
                );
            }
        }
        Ok(Self {
            model,
            builder: ByteMessage::quantum_operations_builder(),
            last_operation_end: vec![0; n_qubits],
            local_groups: config
                .measurement
                .local_groups
                .iter()
                .map(|group| group.iter().copied().collect())
                .collect(),
        })
    }

    fn qubit(&self, id: u64) -> Result<usize> {
        let qubit = usize::try_from(id).context("qubit index does not fit usize")?;
        if qubit >= self.last_operation_end.len() {
            bail!(
                "qubit {id} is out of bounds for {} qubits",
                self.last_operation_end.len()
            );
        }
        Ok(qubit)
    }

    fn add_idle_before(&mut self, qubit: usize, start: u64) -> Result<()> {
        let previous = self.last_operation_end[qubit];
        if start < previous {
            bail!(
                "operation on qubit {qubit} starts at {start}ns before its previous operation ended at {previous}ns"
            );
        }
        if start > previous {
            let seconds = std::time::Duration::from_nanos(start - previous).as_secs_f64();
            self.builder.idle(seconds, &[qubit]);
        }
        Ok(())
    }

    fn local_victims(&self, measured: &BTreeSet<usize>) -> Vec<usize> {
        self.local_groups
            .iter()
            .filter(|group| !group.is_disjoint(measured))
            .flat_map(BTreeSet::iter)
            .filter(|qubit| !measured.contains(qubit))
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn process_message(
        &mut self,
        message: ByteMessage,
        simulator: &mut dyn SimulatorInterface,
    ) -> Result<ByteMessage> {
        let mut stage = self
            .model
            .start(message)
            .map_err(|error| anyhow!(error.to_string()))?;
        loop {
            match stage {
                EngineStage::NeedsProcessing(operations) => {
                    let output = SeleneSimulator::process(simulator, &operations)
                        .map_err(|error| anyhow!(error.to_string()))?;
                    stage = self
                        .model
                        .continue_processing(output)
                        .map_err(|error| anyhow!(error.to_string()))?;
                }
                EngineStage::Complete(output) => return Ok(output),
            }
        }
    }
}

impl ErrorModelInterface for GeneralNoiseErrorModel {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }

    fn shot_start(&mut self, _shot_id: u64, error_seed: u64) -> Result<()> {
        self.model.set_seed(error_seed);
        self.last_operation_end.fill(0);
        self.builder.reset();
        let _ = self.builder.for_quantum_operations();
        Ok(())
    }

    fn shot_end(&mut self) -> Result<()> {
        self.model
            .reset()
            .map_err(|error| anyhow!(error.to_string()))
    }

    fn handle_operations(
        &mut self,
        operations: BatchOperation,
        simulator: &mut dyn SimulatorInterface,
    ) -> Result<BatchResult> {
        let timing = operations
            .runtime_source()
            .ok_or_else(|| anyhow!("PECOS general noise expects a runtime operation batch"))?;
        let start: u64 = timing.start().into();
        let end: u64 = timing.end().into();
        let measured = operations
            .iter_ops()
            .filter_map(|operation| match operation {
                Operation::Measure { qubit_id, .. } | Operation::MeasureLeaked { qubit_id, .. } => {
                    usize::try_from(*qubit_id).ok()
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();

        let mut expected = Vec::new();
        let mut crosstalk_added = false;
        for operation in operations {
            if !crosstalk_added
                && matches!(
                    &operation,
                    Operation::Measure { .. } | Operation::MeasureLeaked { .. }
                )
            {
                self.builder
                    .meas_crosstalk_global_payload(&measured.iter().copied().collect::<Vec<_>>());
                let local = self.local_victims(&measured);
                if !local.is_empty() {
                    self.builder.meas_crosstalk_local_payload(&local);
                }
                crosstalk_added = true;
            }
            match operation {
                Operation::RXYGate {
                    qubit_id,
                    theta,
                    phi,
                } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start)?;
                    self.builder.r1xy(
                        Angle64::from_radians(theta),
                        Angle64::from_radians(phi),
                        &[qubit],
                    );
                    self.last_operation_end[qubit] = end;
                }
                Operation::RZGate { qubit_id, theta } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start)?;
                    self.builder.rz(Angle64::from_radians(theta), &[qubit]);
                    self.last_operation_end[qubit] = end;
                }
                Operation::RZZGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => {
                    let first = self.qubit(qubit_id_1)?;
                    let second = self.qubit(qubit_id_2)?;
                    self.add_idle_before(first, start)?;
                    self.add_idle_before(second, start)?;
                    self.builder
                        .rzz(Angle64::from_radians(theta), &[(first, second)]);
                    self.last_operation_end[first] = end;
                    self.last_operation_end[second] = end;
                }
                Operation::Measure {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start)?;
                    self.builder.mz(&[qubit]);
                    self.last_operation_end[qubit] = end;
                    expected.push(MeasurementResult {
                        id: result_id,
                        kind: MeasurementKind::Bool,
                    });
                }
                Operation::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start)?;
                    self.builder.measure_leakages(&[qubit]);
                    self.last_operation_end[qubit] = end;
                    expected.push(MeasurementResult {
                        id: result_id,
                        kind: MeasurementKind::Leakage,
                    });
                }
                Operation::Reset { qubit_id } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start)?;
                    self.builder.pz(&[qubit]);
                    self.last_operation_end[qubit] = end;
                }
                Operation::RPPGate { .. } => {
                    bail!(
                        "RPP operations do not yet have a PECOS general-noise gate representation"
                    );
                }
                Operation::Custom { custom_tag, .. } => {
                    bail!(
                        "custom Selene runtime operation {custom_tag} has no device-neutral PECOS meaning"
                    );
                }
                _ => bail!("unsupported Selene runtime operation"),
            }
        }

        if self.builder.message_count() == 0 {
            return Ok(BatchResult::default());
        }
        let message = self.builder.build();
        self.builder.reset();
        let _ = self.builder.for_quantum_operations();
        let output = self.process_message(message, simulator)?;
        let outcomes = output
            .outcomes()
            .map_err(|error| anyhow!(error.to_string()))?;
        if outcomes.len() != expected.len() {
            bail!(
                "PECOS returned {} user outcomes for {} Selene measurements",
                outcomes.len(),
                expected.len()
            );
        }
        let mut result = BatchResult::default();
        for (measurement, outcome) in expected.into_iter().zip(outcomes) {
            match measurement.kind {
                MeasurementKind::Bool if outcome <= 1 => {
                    result.set_bool_result(measurement.id, outcome == 1);
                }
                MeasurementKind::Leakage if outcome <= 2 => {
                    result.set_u64_result(measurement.id, u64::from(outcome));
                }
                MeasurementKind::Bool => bail!("PECOS returned non-boolean outcome {outcome}"),
                MeasurementKind::Leakage => {
                    bail!("PECOS returned invalid leakage outcome {outcome}")
                }
            }
        }
        Ok(result)
    }

    fn get_metric(&mut self, _nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
        Ok(None)
    }
}

#[derive(Default)]
struct GeneralNoiseFactory;

impl ErrorModelInterfaceFactory for GeneralNoiseFactory {
    type Interface = GeneralNoiseErrorModel;

    fn init(
        self: Arc<Self>,
        n_qubits: u64,
        error_model_args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        if error_model_args.len() != 2 {
            bail!(
                "PECOS general noise expects one JSON configuration argument, got {}",
                error_model_args.len().saturating_sub(1)
            );
        }
        let config: Config = serde_json::from_str(error_model_args[1].as_ref())
            .context("could not parse PECOS general-noise JSON configuration")?;
        let model = config.build_model()?;
        Ok(Box::new(GeneralNoiseErrorModel::new(
            model,
            usize::try_from(n_qubits).context("qubit count does not fit usize")?,
            &config,
        )?))
    }
}

export_error_model_plugin!(crate::GeneralNoiseFactory);

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn rich_configuration_builds_the_current_pecos_model() {
        let json = r#"{
            "preparation":{
                "probability":0.001,
                "leakage_ratio":0.2,
                "average_crosstalk_probability":0.0001
            },
            "measurement":{
                "p0_to_1":0.002,
                "p1_to_0":0.003,
                "global_crosstalk_probability":0.0001,
                "local_crosstalk_probability":0.0002,
                "crosstalk_model":{
                    "0->0":0.9,"0->1":0.1,
                    "1->0":0.2,"1->1":0.8
                },
                "local_groups":[[0,1],[2,3]]
            },
            "single_qubit":{"average_infidelity":0.0001},
            "two_qubit":{
                "average_infidelity":0.001,
                "angle_coefficients":[1.0,0.0,1.0,0.0],
                "angle_power":1.0,
                "idle_after_gate":0.000005
            },
            "idle":{
                "linear_rate":0.01,
                "sin_squared_rate":0.02,
                "coherent_rate":0.03
            },
            "scaling":{"overall":0.5,"leakage":0.75,"emission":1.25},
            "noiseless_gates":["RZ"]
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        config.build_model().unwrap();
    }

    #[test]
    fn unknown_configuration_fields_are_rejected() {
        let error = serde_json::from_str::<Config>(r#"{"quantinuum_device":"H2"}"#)
            .expect_err("device-specific fields must not enter the generic API");
        assert!(error.to_string().contains("unknown field"));
    }
}
