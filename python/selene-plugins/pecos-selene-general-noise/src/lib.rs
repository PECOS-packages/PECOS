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
    use std::collections::BTreeSet;
    use std::f64::consts::PI;

    use anyhow::Result;
    use pecos_core::Angle64;
    use pecos_engines::prelude::{ByteMessage, EngineStage, NoiseModel};
    use selene_core::error_model::BatchResult;
    use selene_core::error_model::interface::ErrorModelInterface;
    use selene_core::runtime::{BatchOperation, Operation};
    use selene_core::simulator::SimulatorInterface;
    use selene_core::time::{Duration, Instant};
    use selene_core::utils::MetricValue;

    use super::{Config, GeneralNoiseErrorModel};
    use crate::simulator::SeleneSimulator;

    #[derive(Default)]
    struct ZeroSimulator;

    impl SimulatorInterface for ZeroSimulator {
        fn exit(&mut self) -> Result<()> {
            Ok(())
        }

        fn shot_start(&mut self, _shot_id: u64, _seed: u64) -> Result<()> {
            Ok(())
        }

        fn shot_end(&mut self) -> Result<()> {
            Ok(())
        }

        fn handle_operations(&mut self, operations: BatchOperation) -> Result<BatchResult> {
            let mut result = BatchResult::default();
            for operation in operations {
                match operation {
                    Operation::Measure { result_id, .. } => {
                        result.set_bool_result(result_id, false);
                    }
                    Operation::MeasureLeaked { result_id, .. } => {
                        result.set_u64_result(result_id, 0);
                    }
                    _ => {}
                }
            }
            Ok(result)
        }

        fn get_metric(&mut self, _nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
            Ok(None)
        }
    }

    #[derive(Default)]
    struct ClassicalSimulator {
        bits: Vec<bool>,
        received: Vec<Vec<Operation>>,
    }

    impl ClassicalSimulator {
        fn with_qubits(n_qubits: usize) -> Self {
            Self {
                bits: vec![false; n_qubits],
                received: Vec::new(),
            }
        }
    }

    impl SimulatorInterface for ClassicalSimulator {
        fn exit(&mut self) -> Result<()> {
            Ok(())
        }

        fn shot_start(&mut self, _shot_id: u64, _seed: u64) -> Result<()> {
            Ok(())
        }

        fn shot_end(&mut self) -> Result<()> {
            Ok(())
        }

        fn handle_operations(&mut self, operations: BatchOperation) -> Result<BatchResult> {
            self.received
                .push(operations.iter_ops().cloned().collect::<Vec<_>>());
            let mut result = BatchResult::default();
            for operation in operations {
                match operation {
                    Operation::Reset { qubit_id } => {
                        self.bits[usize::try_from(qubit_id).unwrap()] = false;
                    }
                    Operation::RXYGate {
                        qubit_id,
                        theta,
                        phi,
                    } if (theta.abs() - PI).abs() < 1e-12
                        && (phi.abs() < 1e-12 || (phi.abs() - PI / 2.0).abs() < 1e-12) =>
                    {
                        let qubit = usize::try_from(qubit_id).unwrap();
                        self.bits[qubit] = !self.bits[qubit];
                    }
                    Operation::Measure {
                        qubit_id,
                        result_id,
                    } => {
                        result.set_bool_result(
                            result_id,
                            self.bits[usize::try_from(qubit_id).unwrap()],
                        );
                    }
                    Operation::MeasureLeaked { result_id, .. } => {
                        result.set_u64_result(result_id, 0);
                    }
                    _ => {}
                }
            }
            Ok(result)
        }

        fn get_metric(&mut self, _nth_metric: u8) -> Result<Option<(String, MetricValue)>> {
            Ok(None)
        }
    }

    fn runtime_batch(operations: Vec<Operation>, start: u64, duration: u64) -> BatchOperation {
        BatchOperation::runtime(operations, Instant::from(start), Duration::from(duration))
    }

    fn build_error_model(config_json: &str, n_qubits: usize) -> GeneralNoiseErrorModel {
        let config: Config = serde_json::from_str(config_json).unwrap();
        GeneralNoiseErrorModel::new(config.build_model().unwrap(), n_qubits, &config).unwrap()
    }

    fn process_direct_model(
        model: &mut dyn NoiseModel,
        message: ByteMessage,
        simulator: &mut dyn SimulatorInterface,
    ) -> ByteMessage {
        let mut stage = model.start(message).unwrap();
        loop {
            match stage {
                EngineStage::NeedsProcessing(operations) => {
                    let output = SeleneSimulator::process(simulator, &operations).unwrap();
                    stage = model.continue_processing(output).unwrap();
                }
                EngineStage::Complete(output) => return output,
            }
        }
    }

    fn reference_message(
        operations: &[Operation],
        start: u64,
        end: u64,
        last_operation_end: &mut [u64],
        local_groups: &[BTreeSet<usize>],
    ) -> (ByteMessage, Vec<(u64, bool)>) {
        let mut builder = ByteMessage::quantum_operations_builder();
        let measured = operations
            .iter()
            .filter_map(|operation| match operation {
                Operation::Measure { qubit_id, .. } | Operation::MeasureLeaked { qubit_id, .. } => {
                    Some(usize::try_from(*qubit_id).unwrap())
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if !measured.is_empty() {
            builder.meas_crosstalk_global_payload(&measured.iter().copied().collect::<Vec<_>>());
            let local = local_groups
                .iter()
                .filter(|group| !group.is_disjoint(&measured))
                .flat_map(BTreeSet::iter)
                .filter(|qubit| !measured.contains(qubit))
                .copied()
                .collect::<BTreeSet<_>>();
            if !local.is_empty() {
                builder.meas_crosstalk_local_payload(&local.iter().copied().collect::<Vec<_>>());
            }
        }

        let mut expected = Vec::new();
        for operation in operations {
            let mut idle_before = |qubit: usize| {
                if start > last_operation_end[qubit] {
                    let seconds =
                        std::time::Duration::from_nanos(start - last_operation_end[qubit])
                            .as_secs_f64();
                    builder.idle(seconds, &[qubit]);
                }
                last_operation_end[qubit] = end;
            };
            match operation {
                Operation::RXYGate {
                    qubit_id,
                    theta,
                    phi,
                } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.r1xy(
                        Angle64::from_radians(*theta),
                        Angle64::from_radians(*phi),
                        &[qubit],
                    );
                }
                Operation::RZGate { qubit_id, theta } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.rz(Angle64::from_radians(*theta), &[qubit]);
                }
                Operation::RZZGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => {
                    let first = usize::try_from(*qubit_id_1).unwrap();
                    let second = usize::try_from(*qubit_id_2).unwrap();
                    idle_before(first);
                    idle_before(second);
                    builder.rzz(Angle64::from_radians(*theta), &[(first, second)]);
                }
                Operation::Measure {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.mz(&[qubit]);
                    expected.push((*result_id, false));
                }
                Operation::MeasureLeaked {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.measure_leakages(&[qubit]);
                    expected.push((*result_id, true));
                }
                Operation::Reset { qubit_id } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.pz(&[qubit]);
                }
                _ => unreachable!(),
            }
        }
        (builder.build(), expected)
    }

    fn next_random(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn leakage_measurement(config_json: &str) -> u64 {
        let config: Config = serde_json::from_str(config_json).unwrap();
        let mut error_model =
            GeneralNoiseErrorModel::new(config.build_model().unwrap(), 1, &config).unwrap();
        let mut simulator = ZeroSimulator;
        error_model.shot_start(0, 41).unwrap();
        error_model
            .handle_operations(
                BatchOperation::runtime(
                    vec![Operation::Reset { qubit_id: 0 }],
                    Instant::from(0),
                    Duration::from(1),
                ),
                &mut simulator,
            )
            .unwrap();
        let result = error_model
            .handle_operations(
                BatchOperation::runtime(
                    vec![Operation::MeasureLeaked {
                        qubit_id: 0,
                        result_id: 7,
                    }],
                    Instant::from(1),
                    Duration::from(1),
                ),
                &mut simulator,
            )
            .unwrap();
        assert!(result.bool_results.is_empty());
        assert_eq!(result.u64_results.len(), 1);
        assert_eq!(result.u64_results[0].result_id, 7);
        result.u64_results[0].value
    }

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
        let error = serde_json::from_str::<Config>(r#"{"device_profile":"example"}"#)
            .expect_err("device-specific fields must not enter the generic API");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn measure_leaked_reports_computational_and_leaked_outcomes() {
        assert_eq!(leakage_measurement("{}"), 0);
        assert_eq!(
            leakage_measurement(r#"{"preparation":{"probability":1.0,"leakage_ratio":1.0}}"#,),
            2
        );
    }

    #[test]
    fn simulator_bridge_translates_pecos_operations_in_order() {
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.pz(&[0]);
        builder.r1xy(
            Angle64::from_radians(0.75),
            Angle64::from_radians(-0.25),
            &[0],
        );
        builder.rz(Angle64::from_radians(0.5), &[1]);
        builder.rzz(Angle64::from_radians(-0.125), &[(0, 1)]);
        builder.mz(&[0, 1]);
        let message = builder.build();
        let mut simulator = ClassicalSimulator::with_qubits(2);

        let output = SeleneSimulator::process(&mut simulator, &message).unwrap();

        assert_eq!(output.outcomes().unwrap(), vec![0, 0]);
        assert_eq!(simulator.received.len(), 1);
        let operations = &simulator.received[0];
        assert_eq!(operations.len(), 6);
        assert!(matches!(operations[0], Operation::Reset { qubit_id: 0 }));
        assert!(matches!(
            operations[1],
            Operation::RXYGate { qubit_id: 0, .. }
        ));
        let Operation::RXYGate { theta, phi, .. } = operations[1] else {
            unreachable!()
        };
        assert!((theta - 0.75).abs() < 1e-12);
        assert!((phi + 0.25).abs() < 1e-12);
        assert!(matches!(
            operations[2],
            Operation::RZGate { qubit_id: 1, .. }
        ));
        let Operation::RZGate { theta, .. } = operations[2] else {
            unreachable!()
        };
        assert!((theta - 0.5).abs() < 1e-12);
        assert!(matches!(
            operations[3],
            Operation::RZZGate {
                qubit_id_1: 0,
                qubit_id_2: 1,
                ..
            }
        ));
        let Operation::RZZGate { theta, .. } = operations[3] else {
            unreachable!()
        };
        assert!((theta + 0.125).abs() < 1e-12);
        assert!(matches!(
            operations[4],
            Operation::Measure {
                qubit_id: 0,
                result_id: 0
            }
        ));
        assert!(matches!(
            operations[5],
            Operation::Measure {
                qubit_id: 1,
                result_id: 1
            }
        ));
    }

    #[test]
    fn runtime_timestamps_drive_every_idle_family() {
        let configurations = [
            r#"{"idle":{"linear_rate":1.0,"linear_model":{"X":1.0}}}"#,
            r#"{"idle":{"sin_squared_rate":1.5707963267948966,"sin_squared_model":{"X":1.0}}}"#,
            r#"{"idle":{"coherent_rate":3.141592653589793,"coherent_model":{"RX":1.0}}}"#,
        ];
        for config in configurations {
            let mut error_model = build_error_model(config, 1);
            let mut simulator = ClassicalSimulator::with_qubits(1);
            error_model.shot_start(0, 41).unwrap();
            error_model
                .handle_operations(
                    runtime_batch(vec![Operation::Reset { qubit_id: 0 }], 0, 1),
                    &mut simulator,
                )
                .unwrap();

            let result = error_model
                .handle_operations(
                    runtime_batch(
                        vec![Operation::Measure {
                            qubit_id: 0,
                            result_id: 9,
                        }],
                        1_000_000_001,
                        1,
                    ),
                    &mut simulator,
                )
                .unwrap();

            assert_eq!(result.bool_results.len(), 1);
            assert_eq!(result.bool_results[0].result_id, 9);
            assert!(result.bool_results[0].value, "idle configuration: {config}");
        }
    }

    #[test]
    fn runtime_timestamps_accumulate_per_qubit_between_operations() {
        let mut error_model = build_error_model(
            r#"{"idle":{"linear_rate":1.0,"linear_model":{"X":1.0}}}"#,
            1,
        );
        let mut simulator = ClassicalSimulator::with_qubits(1);
        error_model.shot_start(0, 43).unwrap();
        error_model
            .handle_operations(
                runtime_batch(vec![Operation::Reset { qubit_id: 0 }], 0, 1),
                &mut simulator,
            )
            .unwrap();
        error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::RZGate {
                        qubit_id: 0,
                        theta: 0.25,
                    }],
                    1_000_000_001,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();
        let result = error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::Measure {
                        qubit_id: 0,
                        result_id: 11,
                    }],
                    2_000_000_002,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();

        assert!(!result.bool_results[0].value);
    }

    #[test]
    fn seeded_randomized_traces_match_direct_general_noise_execution() {
        const N_QUBITS: usize = 3;
        const CONFIG: &str = r#"{
            "preparation":{"probability":0.17,"leakage_ratio":0.29},
            "measurement":{
                "p0_to_1":0.11,
                "p1_to_0":0.07,
                "global_crosstalk_probability":0.13,
                "local_crosstalk_probability":0.19,
                "crosstalk_model":{
                    "0->0":0.7,"0->1":0.2,"0->L":0.1,
                    "1->0":0.15,"1->1":0.75,"1->L":0.1
                },
                "local_groups":[[0,1],[1,2]]
            },
            "single_qubit":{
                "probability":0.23,
                "pauli_model":{"X":0.5,"Z":0.5},
                "emission_ratio":0.31,
                "emission_model":{"X":0.6,"L":0.4},
                "seepage_probability":0.27
            },
            "two_qubit":{
                "probability":0.21,
                "pauli_model":{"XI":0.4,"IZ":0.3,"ZZ":0.3},
                "emission_ratio":0.25,
                "emission_model":{"XI":0.5,"IL":0.25,"LI":0.25},
                "seepage_probability":0.33
            },
            "idle":{
                "linear_rate":500000.0,
                "linear_model":{"X":0.5,"Z":0.5},
                "sin_squared_rate":700000.0,
                "sin_squared_model":{"X":0.4,"L":0.2},
                "coherent_rate":300000.0,
                "coherent_model":{"RX":0.7,"RZ":0.2}
            }
        }"#;

        let config: Config = serde_json::from_str(CONFIG).unwrap();
        let mut direct_model = config.build_model().unwrap();
        let mut adapter =
            GeneralNoiseErrorModel::new(config.build_model().unwrap(), N_QUBITS, &config).unwrap();
        let mut direct_simulator = ClassicalSimulator::with_qubits(N_QUBITS);
        let mut adapter_simulator = ClassicalSimulator::with_qubits(N_QUBITS);
        let local_groups = [BTreeSet::from([0, 1]), BTreeSet::from([1, 2])];
        let mut reference_last_end = vec![0; N_QUBITS];
        let error_seed = 8_191;
        direct_model.set_seed(error_seed);
        adapter.shot_start(0, error_seed).unwrap();

        let mut random = 0x5eed_d1ff_e2e5_u64;
        let mut cursor = 0_u64;
        let mut result_id = 100_u64;
        let mut trace = vec![(
            vec![
                Operation::Reset { qubit_id: 0 },
                Operation::Reset { qubit_id: 1 },
                Operation::Reset { qubit_id: 2 },
            ],
            0,
            2,
        )];
        cursor += 2;
        for case_id in 0..96 {
            let gap = 1 + next_random(&mut random) % 17;
            let duration = 1 + next_random(&mut random) % 5;
            let start = cursor + gap;
            let first = next_random(&mut random) % N_QUBITS as u64;
            let second = (first + 1 + next_random(&mut random) % 2) % N_QUBITS as u64;
            let angle = match next_random(&mut random) % 5 {
                0 => -PI,
                1 => -PI / 2.0,
                2 => PI / 4.0,
                3 => PI / 2.0,
                _ => PI,
            };
            let mut operations = match next_random(&mut random) % 6 {
                0 => vec![Operation::Reset { qubit_id: first }],
                1 => vec![Operation::RXYGate {
                    qubit_id: first,
                    theta: angle,
                    phi: angle / 3.0,
                }],
                2 => vec![Operation::RZGate {
                    qubit_id: first,
                    theta: angle,
                }],
                3 => vec![Operation::RZZGate {
                    qubit_id_1: first,
                    qubit_id_2: second,
                    theta: angle,
                }],
                4 => {
                    let operation = Operation::Measure {
                        qubit_id: first,
                        result_id,
                    };
                    result_id += 1;
                    vec![operation]
                }
                _ => {
                    let operation = Operation::MeasureLeaked {
                        qubit_id: first,
                        result_id,
                    };
                    result_id += 1;
                    vec![operation]
                }
            };
            if case_id % 12 == 0 && !matches!(operations[0], Operation::RZZGate { .. }) {
                let parallel = match operations[0] {
                    Operation::Reset { .. } => Operation::Reset { qubit_id: second },
                    Operation::RXYGate { theta, phi, .. } => Operation::RXYGate {
                        qubit_id: second,
                        theta,
                        phi,
                    },
                    Operation::RZGate { theta, .. } => Operation::RZGate {
                        qubit_id: second,
                        theta,
                    },
                    Operation::Measure { .. } => {
                        let operation = Operation::Measure {
                            qubit_id: second,
                            result_id,
                        };
                        result_id += 1;
                        operation
                    }
                    Operation::MeasureLeaked { .. } => {
                        let operation = Operation::MeasureLeaked {
                            qubit_id: second,
                            result_id,
                        };
                        result_id += 1;
                        operation
                    }
                    _ => unreachable!(),
                };
                operations.push(parallel);
            }
            trace.push((operations, start, duration));
            cursor = start + duration;
        }

        for (operations, start, duration) in trace {
            let end = start + duration;
            let (reference, expected_results) = reference_message(
                &operations,
                start,
                end,
                &mut reference_last_end,
                &local_groups,
            );
            let adapter_result = adapter
                .handle_operations(
                    runtime_batch(operations, start, duration),
                    &mut adapter_simulator,
                )
                .unwrap();
            let direct_output =
                process_direct_model(direct_model.as_mut(), reference, &mut direct_simulator);

            if !expected_results.is_empty() {
                let outcomes = direct_output.outcomes().unwrap();
                assert_eq!(outcomes.len(), expected_results.len());
                for ((id, leakage), outcome) in expected_results.iter().zip(outcomes) {
                    if *leakage {
                        let actual = adapter_result
                            .u64_results
                            .iter()
                            .find(|result| result.result_id == *id)
                            .unwrap();
                        assert_eq!(actual.value, u64::from(outcome));
                    } else {
                        let actual = adapter_result
                            .bool_results
                            .iter()
                            .find(|result| result.result_id == *id)
                            .unwrap();
                        assert_eq!(actual.value, outcome == 1);
                    }
                }
            }
            assert_eq!(adapter_simulator.received, direct_simulator.received);
            assert_eq!(adapter_simulator.bits, direct_simulator.bits);
        }
    }

    #[test]
    fn adapter_rejects_invalid_runtime_contracts() {
        let config: Config =
            serde_json::from_str(r#"{"measurement":{"local_groups":[[0,2]]}}"#).unwrap();
        let error = GeneralNoiseErrorModel::new(config.build_model().unwrap(), 2, &config)
            .err()
            .expect("an out-of-bounds local group must fail");
        assert!(error.to_string().contains("contains qubit 2"));

        let mut error_model = build_error_model("{}", 1);
        let mut simulator = ClassicalSimulator::with_qubits(1);
        error_model.shot_start(0, 41).unwrap();
        error_model
            .handle_operations(
                runtime_batch(vec![Operation::Reset { qubit_id: 0 }], 10, 10),
                &mut simulator,
            )
            .unwrap();
        let error = error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::Measure {
                        qubit_id: 0,
                        result_id: 0,
                    }],
                    19,
                    1,
                ),
                &mut simulator,
            )
            .err()
            .expect("overlapping operations must fail");
        assert!(
            error
                .to_string()
                .contains("before its previous operation ended")
        );

        let mut error_model = build_error_model("{}", 1);
        let error = error_model
            .handle_operations(
                BatchOperation::error_model(vec![Operation::Reset { qubit_id: 0 }]),
                &mut simulator,
            )
            .err()
            .expect("an error-model batch must fail");
        assert!(
            error
                .to_string()
                .contains("expects a runtime operation batch")
        );

        let error = error_model
            .handle_operations(
                runtime_batch(vec![Operation::Reset { qubit_id: 1 }], 0, 1),
                &mut simulator,
            )
            .err()
            .expect("an out-of-bounds operation must fail");
        assert!(error.to_string().contains("qubit 1 is out of bounds"));

        let mut error_model = build_error_model("{}", 1);
        let error = error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::RPPGate {
                        qubit_id_1: 0,
                        qubit_id_2: 0,
                        theta: PI / 2.0,
                        phi: 0.0,
                    }],
                    0,
                    1,
                ),
                &mut simulator,
            )
            .err()
            .expect("an RPP operation must fail");
        assert!(error.to_string().contains("RPP operations do not yet have"));

        let mut error_model = build_error_model("{}", 1);
        let error = error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::Custom {
                        custom_tag: 17,
                        data: Vec::new().into_boxed_slice(),
                    }],
                    0,
                    1,
                ),
                &mut simulator,
            )
            .err()
            .expect("a custom operation must fail");
        assert!(
            error
                .to_string()
                .contains("no device-neutral PECOS meaning")
        );
    }
}
