//! Device-neutral Selene adapter for PECOS's general noise model.

mod simulator;

use std::collections::BTreeSet;

use anyhow::{Context, Result, anyhow, bail};
use pecos_core::Angle64;
use pecos_engines::prelude::*;
use selene_core::error_model::BatchResult;
use selene_core::error_model::interface::ErrorModelInterface;
use selene_core::runtime::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;
use selene_core::utils::MetricValue;
use serde::Deserialize;

use crate::simulator::SeleneSimulator;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PluginConfig {
    model: GeneralNoiseConfig,
    #[serde(default)]
    local_groups: Vec<Vec<usize>>,
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

/// Stateful Selene adapter for an engine-owned [`GeneralNoiseConfig`] and local topology.
///
/// This type implements [`ErrorModelInterface`]. The `pecos-selene-general-noise`
/// plugin exports it to Selene; downstream Rust plugins can wrap it to normalize
/// runtime-specific operations before applying generic noise. Each shot must
/// follow the Selene start/operations/end lifecycle. Pass complete runtime
/// batches with their original timing and measurement IDs; splitting batches can
/// change noise semantics. Custom operations remain errors. After an execution
/// error, discard the instance rather than reuse partial state.
pub struct GeneralNoiseErrorModel {
    model: Box<dyn NoiseModel>,
    builder: ByteMessageBuilder,
    last_operation_end: Vec<Option<u64>>,
    local_groups: Vec<BTreeSet<usize>>,
}

impl GeneralNoiseErrorModel {
    /// Construct from the model and local-group envelope emitted by
    /// `GeneralNoisePlugin.get_init_args()`.
    ///
    /// # Errors
    /// Returns an error for malformed JSON, unknown fields, invalid noise configuration,
    /// or local groups outside the simulation's capacity.
    pub fn from_plugin_json(n_qubits: usize, json: &str) -> Result<Self> {
        let config: PluginConfig = serde_json::from_str(json)
            .context("could not parse PECOS general-noise JSON configuration")?;
        Self::new(n_qubits, &config.model, &config.local_groups)
    }

    /// Construct from typed engine parameters and Selene local-crosstalk groups.
    ///
    /// This does not start a shot or interpret custom runtime operations; callers must
    /// explicitly normalize those operations and follow the Selene lifecycle.
    ///
    /// # Errors
    /// Returns an error for invalid model parameters or out-of-capacity local groups.
    pub fn new(
        n_qubits: usize,
        config: &GeneralNoiseConfig,
        local_groups: &[Vec<usize>],
    ) -> Result<Self> {
        let model = Box::new(
            config
                .build()
                .context("invalid general-noise configuration")?,
        );
        for group in local_groups {
            if let Some(qubit) = group.iter().find(|qubit| **qubit >= n_qubits) {
                bail!(
                    "local crosstalk group contains qubit {qubit}, but the simulation has {n_qubits} qubits"
                );
            }
        }
        Ok(Self {
            model,
            builder: ByteMessage::quantum_operations_builder(),
            last_operation_end: vec![None; n_qubits],
            local_groups: local_groups
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

    fn add_idle_before(
        &mut self,
        qubit: usize,
        start: u64,
        touched: &mut BTreeSet<usize>,
    ) -> Result<()> {
        if !touched.insert(qubit) {
            return Ok(());
        }
        // A qubit's first operation in a shot has no predecessor within that shot, so
        // there is no gap to model. Selene's runtime clock is monotonic across shots
        // (`SimpleRuntime::shot_start` is a no-op and its instant only advances), so
        // treating "no previous operation" as timestamp zero would charge the qubit
        // for every nanosecond of emulator time that elapsed in earlier shots.
        let Some(previous) = self.last_operation_end[qubit] else {
            return Ok(());
        };
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
        self.last_operation_end.fill(None);
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
        let mut touched = BTreeSet::new();
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
                    self.add_idle_before(qubit, start, &mut touched)?;
                    self.builder.rxy1q(
                        Angle64::try_from_radians(theta)?,
                        Angle64::try_from_radians(phi)?,
                        &[qubit],
                    );
                }
                Operation::RZGate { qubit_id, theta } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start, &mut touched)?;
                    self.builder.rz(Angle64::try_from_radians(theta)?, &[qubit]);
                }
                Operation::RZZGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                } => {
                    let first = self.qubit(qubit_id_1)?;
                    let second = self.qubit(qubit_id_2)?;
                    self.add_idle_before(first, start, &mut touched)?;
                    self.add_idle_before(second, start, &mut touched)?;
                    self.builder
                        .rzz(Angle64::try_from_radians(theta)?, &[(first, second)]);
                }
                Operation::Measure {
                    qubit_id,
                    result_id,
                } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start, &mut touched)?;
                    self.builder.mz(&[qubit]);
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
                    self.add_idle_before(qubit, start, &mut touched)?;
                    self.builder.measure_leakages(&[qubit]);
                    expected.push(MeasurementResult {
                        id: result_id,
                        kind: MeasurementKind::Leakage,
                    });
                }
                Operation::Reset { qubit_id } => {
                    let qubit = self.qubit(qubit_id)?;
                    self.add_idle_before(qubit, start, &mut touched)?;
                    self.builder.pz(&[qubit]);
                }
                Operation::RPPGate {
                    qubit_id_1,
                    qubit_id_2,
                    theta,
                    phi,
                } => {
                    let first = self.qubit(qubit_id_1)?;
                    let second = self.qubit(qubit_id_2)?;
                    if first == second {
                        bail!("RXYXY2Q requires two distinct qubits");
                    }
                    self.add_idle_before(first, start, &mut touched)?;
                    self.add_idle_before(second, start, &mut touched)?;
                    self.builder.rxyxy2q(
                        Angle64::try_from_radians(theta)?,
                        Angle64::try_from_radians(phi)?,
                        &[(first, second)],
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

        for qubit in touched {
            self.last_operation_end[qubit] = Some(end);
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::f64::consts::PI;

    use anyhow::Result;
    use pecos_core::Angle64;
    use pecos_engines::byte_message::GateType;
    use pecos_engines::prelude::{ByteMessage, EngineStage, NoiseModel};
    use selene_core::error_model::BatchResult;
    use selene_core::error_model::interface::ErrorModelInterface;
    use selene_core::runtime::{BatchOperation, Operation};
    use selene_core::simulator::SimulatorInterface;
    use selene_core::time::{Duration, Instant};
    use selene_core::utils::MetricValue;

    use super::{GeneralNoiseErrorModel, PluginConfig};
    use crate::simulator::SeleneSimulator;

    #[test]
    fn all_runtime_rotation_parameters_reject_nonfinite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for operation in [
                Operation::RZGate {
                    qubit_id: 0,
                    theta: value,
                },
                Operation::RZZGate {
                    qubit_id_1: 0,
                    qubit_id_2: 1,
                    theta: value,
                },
                Operation::RXYGate {
                    qubit_id: 0,
                    theta: value,
                    phi: 0.0,
                },
                Operation::RXYGate {
                    qubit_id: 0,
                    theta: 0.0,
                    phi: value,
                },
                Operation::RPPGate {
                    qubit_id_1: 0,
                    qubit_id_2: 1,
                    theta: value,
                    phi: 0.0,
                },
                Operation::RPPGate {
                    qubit_id_1: 0,
                    qubit_id_2: 1,
                    theta: 0.0,
                    phi: value,
                },
            ] {
                let mut model = build_error_model(r#"{"model":{}}"#, 2);
                model.shot_start(0, 17).unwrap();
                let error = model
                    .handle_operations(runtime_batch(vec![operation], 0, 1), &mut ZeroSimulator)
                    .err()
                    .expect("non-finite runtime angle must fail");
                assert!(
                    error
                        .to_string()
                        .contains(&format!("angle must be finite, got {value} radians")),
                    "{error}"
                );
            }
        }
    }

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
        GeneralNoiseErrorModel::from_plugin_json(n_qubits, config_json).unwrap()
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
        last_operation_end: &mut [Option<u64>],
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
        let mut touched = BTreeSet::new();
        for operation in operations {
            let mut idle_before = |qubit: usize| {
                if !touched.insert(qubit) {
                    return;
                }
                if let Some(previous) = last_operation_end[qubit]
                    && start > previous
                {
                    let seconds = std::time::Duration::from_nanos(start - previous).as_secs_f64();
                    builder.idle(seconds, &[qubit]);
                }
            };
            match operation {
                Operation::RXYGate {
                    qubit_id,
                    theta,
                    phi,
                } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.rxy1q(
                        Angle64::try_from_radians(*theta).unwrap(),
                        Angle64::try_from_radians(*phi).unwrap(),
                        &[qubit],
                    );
                }
                Operation::RZGate { qubit_id, theta } => {
                    let qubit = usize::try_from(*qubit_id).unwrap();
                    idle_before(qubit);
                    builder.rz(Angle64::try_from_radians(*theta).unwrap(), &[qubit]);
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
                    builder.rzz(
                        Angle64::try_from_radians(*theta).unwrap(),
                        &[(first, second)],
                    );
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
        for qubit in touched {
            last_operation_end[qubit] = Some(end);
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
        let mut error_model = GeneralNoiseErrorModel::from_plugin_json(1, config_json).unwrap();
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

    fn hook_outcomes(site: &str, hooks: &serde_json::Value, two: bool) -> Vec<u64> {
        let config = serde_json::json!({
            if two { "two_qubit" } else { "single_qubit" }: { site: hooks }
        });
        let mut model = build_error_model(&serde_json::json!({"model": config}).to_string(), 2);
        let mut simulator = ZeroSimulator;
        model.shot_start(0, 41).unwrap();
        model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::Reset { qubit_id: 0 },
                        Operation::Reset { qubit_id: 1 },
                    ],
                    0,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();
        let gate = if two {
            Operation::RZZGate {
                qubit_id_1: 0,
                qubit_id_2: 1,
                theta: 0.5,
            }
        } else {
            Operation::RZGate {
                qubit_id: 0,
                theta: 0.5,
            }
        };
        let internal = model
            .handle_operations(runtime_batch(vec![gate], 1, 1), &mut simulator)
            .unwrap();
        assert!(internal.bool_results.is_empty());
        assert!(internal.u64_results.is_empty());
        let output = model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::MeasureLeaked {
                            qubit_id: 0,
                            result_id: 7,
                        },
                        Operation::MeasureLeaked {
                            qubit_id: 1,
                            result_id: 8,
                        },
                    ],
                    2,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();
        assert!(output.bool_results.is_empty());
        assert_eq!(
            output
                .u64_results
                .iter()
                .map(|item| item.result_id)
                .collect::<Vec<_>>(),
            vec![7, 8]
        );
        output.u64_results.iter().map(|item| item.value).collect()
    }

    #[test]
    fn gate_hooks_execute_in_order_and_route_internal_measurements() {
        let leak = serde_json::json!({"kind":"pauli_leakage","probability":1.0,"events":{"L":1.0}});
        let recover = serde_json::json!({"kind":"transition","probability":1.0,"transitions":{"L":{"0":1.0}}});
        for site in ["hooks_before", "hooks_after"] {
            for two in [false, true] {
                let leaked = if two { vec![2, 2] } else { vec![2, 0] };
                assert_eq!(hook_outcomes(site, &serde_json::json!([leak]), two), leaked);
                assert_eq!(
                    hook_outcomes(site, &serde_json::json!([leak, recover]), two),
                    vec![0, 0]
                );
                assert_eq!(
                    hook_outcomes(site, &serde_json::json!([recover, leak]), two),
                    leaked
                );
            }
        }
        let transition_leak = serde_json::json!({"kind":"transition","probability":1.0,"transitions":{"0->L":1.0,"1->L":1.0}});
        assert_eq!(
            hook_outcomes("hooks_before", &serde_json::json!([transition_leak]), false),
            vec![2, 0]
        );
    }

    #[test]
    fn two_qubit_hooks_support_joint_wildcards_and_distinct_legs() {
        for family in ["transition", "pauli_leakage"] {
            let (first, second, joint) = if family == "transition" {
                (
                    serde_json::json!({"kind":family,"probability":1.0,"transitions":{"0->L":1.0,"1->L":1.0}}),
                    serde_json::json!({"kind":family,"probability":0.0,"transitions":{"L->0":1.0}}),
                    serde_json::json!({"kind":family,"probability":1.0,"transitions":{"0*->L*":1.0,"1*->L*":1.0}}),
                )
            } else {
                (
                    serde_json::json!({"kind":family,"probability":1.0,"events":{"L":1.0}}),
                    serde_json::json!({"kind":family,"probability":0.0,"events":{"L":1.0}}),
                    serde_json::json!({"kind":family,"probability":1.0,"events":{"LI":1.0}}),
                )
            };
            for site in ["hooks_before", "hooks_after"] {
                assert_eq!(
                    hook_outcomes(
                        site,
                        &serde_json::json!([{"kind":"independent","first":first,"second":second}]),
                        true
                    ),
                    vec![2, 0]
                );
                assert_eq!(
                    hook_outcomes(site, &serde_json::json!([joint]), true),
                    vec![2, 0]
                );
            }
        }
    }

    #[test]
    fn measure_leaked_reports_computational_and_leaked_outcomes() {
        assert_eq!(leakage_measurement(r#"{"model":{}}"#), 0);
        assert_eq!(
            leakage_measurement(
                r#"{"model":{"preparation":{"probability":1.0,"leakage_ratio":1.0}}}"#,
            ),
            2
        );
    }

    #[test]
    fn simulator_bridge_translates_pecos_operations_in_order() {
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.pz(&[0]);
        builder.rxy1q(
            pecos_core::radians!(0.75),
            pecos_core::radians!(-0.25),
            &[0],
        );
        builder.rz(pecos_core::radians!(0.5), &[1]);
        builder.rzz(pecos_core::radians!(-0.125), &[(0, 1)]);
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
            r#"{"model":{"idle":{"linear_rate":1.0,"linear_model":{"X":1.0}}}}"#,
            r#"{"model":{"idle":{"sin_squared_rate":1.5707963267948966,"sin_squared_model":{"X":1.0}}}}"#,
            r#"{"model":{"idle":{"coherent_rate":3.141592653589793,"coherent_model":{"RX":1.0}}}}"#,
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
            r#"{"model":{"idle":{"linear_rate":1.0,"linear_model":{"X":1.0}}}}"#,
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
        const CONFIG: &str = r#"{"model":{
            "preparation":{"probability":0.17,"leakage_ratio":0.29},
            "measurement":{
                "p0_to_1":0.11,
                "p1_to_0":0.07,
                "global_crosstalk_probability":0.13,
                "local_crosstalk_probability":0.19,
                "crosstalk_model":{
                    "0->0":0.7,"0->1":0.2,"0->L":0.1,
                    "1->0":0.15,"1->1":0.75,"1->L":0.1
                }
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
        },"local_groups":[[0,1],[1,2]]}"#;

        let config: PluginConfig = serde_json::from_str(CONFIG).unwrap();
        let mut direct_model: Box<dyn NoiseModel> = Box::new(config.model.build().unwrap());
        let mut adapter = GeneralNoiseErrorModel::from_plugin_json(N_QUBITS, CONFIG).unwrap();
        let mut direct_simulator = ClassicalSimulator::with_qubits(N_QUBITS);
        let mut adapter_simulator = ClassicalSimulator::with_qubits(N_QUBITS);
        let local_groups = [BTreeSet::from([0, 1]), BTreeSet::from([1, 2])];
        let mut reference_last_end: Vec<Option<u64>> = vec![None; N_QUBITS];
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
    fn a_shot_does_not_inherit_the_previous_shots_clock() {
        // Selene's runtime clock is monotonic across shots: `SimpleRuntime::shot_start`
        // is a no-op and its instant only ever advances. A qubit's first operation in a
        // shot therefore has no predecessor within that shot and must not accrue idle
        // time measured from an earlier shot's timestamps.
        let config = r#"{"model":{"idle":{"linear_rate":1.0,"linear_model":{"X":1.0}}}}"#;
        let batch = || {
            vec![Operation::RXYGate {
                qubit_id: 0,
                theta: PI,
                phi: 0.0,
            }]
        };

        // One operation per qubit per batch, as the runtime schedules them.
        let mut error_model = build_error_model(config, 1);
        let span = 1_000_000_000;
        let run_shot = |model: &mut GeneralNoiseErrorModel, shot: u64, base: u64| {
            let mut simulator = ClassicalSimulator::with_qubits(1);
            model.shot_start(shot, 41).unwrap();
            model
                .handle_operations(runtime_batch(batch(), base, span), &mut simulator)
                .unwrap();
            model
                .handle_operations(
                    runtime_batch(
                        vec![Operation::Measure {
                            qubit_id: 0,
                            result_id: 0,
                        }],
                        base + span,
                        span,
                    ),
                    &mut simulator,
                )
                .unwrap();
            model.shot_end().unwrap();
            simulator.received
        };

        let first = run_shot(&mut error_model, 0, 0);
        let second = run_shot(&mut error_model, 1, 2 * span);

        assert_eq!(
            first, second,
            "the same circuit under the same seed must reach the simulator identically \
             regardless of how much emulator time preceded the shot"
        );
    }

    #[test]
    fn prepared_qubits_do_not_leak_across_shots() {
        // `prepared_qubits` is the global measurement-crosstalk victim pool. A qubit
        // prepared in an earlier shot must not be a victim in a later shot that never
        // prepared it, or shots stop being independent and any rate averaged over them
        // depends on shot ordering.
        //
        // The oracle is a freshly constructed model with no history running the same
        // shot: whatever the correct emission is, both must produce it identically.
        let config = r#"{"model":{"measurement":{"global_crosstalk_probability":1.0,"crosstalk_model":{"0->1":1.0,"1->0":1.0}}}}"#;
        let later_shot = || {
            vec![Operation::Measure {
                qubit_id: 0,
                result_id: 0,
            }]
        };

        let mut with_history = build_error_model(config, 2);
        let mut discarded = ClassicalSimulator::with_qubits(2);
        with_history.shot_start(0, 41).unwrap();
        with_history
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::Reset { qubit_id: 1 },
                        Operation::Measure {
                            qubit_id: 0,
                            result_id: 0,
                        },
                    ],
                    0,
                    1,
                ),
                &mut discarded,
            )
            .unwrap();
        with_history.shot_end().unwrap();

        let mut history_simulator = ClassicalSimulator::with_qubits(2);
        with_history.shot_start(1, 41).unwrap();
        with_history
            .handle_operations(runtime_batch(later_shot(), 1, 1), &mut history_simulator)
            .unwrap();
        with_history.shot_end().unwrap();

        let mut fresh = build_error_model(config, 2);
        let mut fresh_simulator = ClassicalSimulator::with_qubits(2);
        fresh.shot_start(1, 41).unwrap();
        fresh
            .handle_operations(runtime_batch(later_shot(), 1, 1), &mut fresh_simulator)
            .unwrap();
        fresh.shot_end().unwrap();

        assert_eq!(
            history_simulator.received, fresh_simulator.received,
            "a shot must not depend on qubits prepared in earlier shots"
        );
    }

    #[test]
    fn any_reset_returns_a_leaked_qubit_to_the_computational_subspace() {
        // Declaring a gate noiseless suppresses its faults, not the physics it
        // performs. The bridge lowers PZ to a real reset on the simulator, so a
        // noiseless PZ that failed to clear the model's leakage record would leave
        // the model and the simulator disagreeing about the qubit's state.
        // Both reset paths must clear leakage: the ordinary one through the
        // preparation fault handler, and the one that bypasses faults entirely.
        for config in [
            r#"{"model":{"single_qubit":{"probability":1.0,"emission_ratio":1.0,"emission_model":{"L":1.0}},"scaling":{"leakage":1.0}}}"#,
            r#"{"model":{"single_qubit":{"probability":1.0,"emission_ratio":1.0,"emission_model":{"L":1.0}},"scaling":{"leakage":1.0},"noiseless_gates":["PZ"]}}"#,
        ] {
            let mut error_model = build_error_model(config, 1);
            let mut simulator = ClassicalSimulator::with_qubits(1);
            error_model.shot_start(0, 41).unwrap();

            // A one-qubit gate at full infidelity and full leakage ratio leaks the qubit.
            error_model
                .handle_operations(
                    runtime_batch(
                        vec![Operation::RXYGate {
                            qubit_id: 0,
                            theta: PI,
                            phi: 0.0,
                        }],
                        0,
                        1,
                    ),
                    &mut simulator,
                )
                .unwrap();
            let leaked = error_model
                .handle_operations(
                    runtime_batch(
                        vec![Operation::MeasureLeaked {
                            qubit_id: 0,
                            result_id: 0,
                        }],
                        1,
                        1,
                    ),
                    &mut simulator,
                )
                .unwrap();
            assert_eq!(
                leaked.u64_results[0].value, 2,
                "the gate should have leaked the qubit: {config}"
            );

            error_model
                .handle_operations(
                    runtime_batch(vec![Operation::Reset { qubit_id: 0 }], 2, 1),
                    &mut simulator,
                )
                .unwrap();
            let after_reset = error_model
                .handle_operations(
                    runtime_batch(
                        vec![Operation::MeasureLeaked {
                            qubit_id: 0,
                            result_id: 1,
                        }],
                        3,
                        1,
                    ),
                    &mut simulator,
                )
                .unwrap();
            assert_eq!(
                after_reset.u64_results[0].value, 0,
                "reset must return the qubit to the computational subspace: {config}"
            );
        }
    }

    // The three tests below assert emissions directly rather than against
    // `reference_message`. That helper is a transcription of `handle_operations`, so a
    // mistake made in both is invisible to the differential trace; these behaviours had
    // no other guard.

    #[test]
    fn a_sub_second_gap_still_produces_idle() {
        // A gap shorter than one second must not be truncated away. At 0.5s with a
        // linear rate of 2.0 the idle probability is exactly 1.0, so the X is
        // deterministic; truncating the gap to whole seconds would yield 0.0 and emit
        // nothing.
        let mut error_model = build_error_model(
            r#"{"model":{"idle":{"linear_rate":2.0,"linear_model":{"X":1.0}}}}"#,
            1,
        );
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
                        result_id: 0,
                    }],
                    500_000_001,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();
        assert!(
            result.bool_results[0].value,
            "a 0.5s gap must produce idle noise, not be truncated to zero seconds"
        );
    }

    #[test]
    fn a_measured_qubit_is_not_its_own_crosstalk_victim() {
        // Local crosstalk targets the rest of a measured qubit's group, never the
        // measured qubit itself, whose outcome is already governed by readout error.
        let mut error_model = build_error_model(
            r#"{"model":{"measurement":{"local_crosstalk_probability":1.0,"crosstalk_model":{"0->1":1.0,"1->0":1.0}}},"local_groups":[[0,1]]}"#,
            2,
        );
        let mut simulator = ClassicalSimulator::with_qubits(2);
        error_model.shot_start(0, 41).unwrap();
        error_model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::Reset { qubit_id: 0 },
                        Operation::Reset { qubit_id: 1 },
                    ],
                    0,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();
        simulator.received.clear();
        error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::Measure {
                        qubit_id: 0,
                        result_id: 0,
                    }],
                    1,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();

        let disturbed_qubits = simulator
            .received
            .iter()
            .flatten()
            .filter_map(|operation| match operation {
                Operation::RXYGate { qubit_id, .. } | Operation::RZGate { qubit_id, .. } => {
                    Some(*qubit_id)
                }
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        assert!(
            !disturbed_qubits.contains(&0),
            "the measured qubit must not be a victim of its own measurement crosstalk: {:?}",
            simulator.received
        );
        assert!(
            disturbed_qubits.contains(&1),
            "the rest of the measured qubit's group must be disturbed: {:?}",
            simulator.received
        );
    }

    #[test]
    fn a_leakage_measurement_also_triggers_crosstalk() {
        // MeasureLeaked is a measurement: it disturbs its neighbours exactly as an
        // ordinary measurement does.
        let mut error_model = build_error_model(
            r#"{"model":{"measurement":{"local_crosstalk_probability":1.0,"crosstalk_model":{"0->1":1.0,"1->0":1.0}}},"local_groups":[[0,1]]}"#,
            2,
        );
        let mut simulator = ClassicalSimulator::with_qubits(2);
        error_model.shot_start(0, 41).unwrap();
        error_model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::Reset { qubit_id: 0 },
                        Operation::Reset { qubit_id: 1 },
                    ],
                    0,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();
        simulator.received.clear();
        error_model
            .handle_operations(
                runtime_batch(
                    vec![Operation::MeasureLeaked {
                        qubit_id: 0,
                        result_id: 0,
                    }],
                    1,
                    1,
                ),
                &mut simulator,
            )
            .unwrap();

        let disturbed_one = simulator.received.iter().flatten().any(|operation| {
            matches!(
                operation,
                Operation::RXYGate { qubit_id: 1, .. } | Operation::RZGate { qubit_id: 1, .. }
            )
        });
        assert!(
            disturbed_one,
            "a leakage measurement must trigger measurement crosstalk: {:?}",
            simulator.received
        );
    }

    /// Residual global phase of every unitary bridge arm, pinned as a value.
    ///
    /// Direction convention, stated here and nowhere else:
    ///
    /// ```text
    ///     U_pecos = e^{i phi} * U_emitted
    /// ```
    ///
    /// `phi` is what this test returns and what the table below pins.
    ///
    /// The PECOS side is PECOS's *executable* dense matrix, `UnitaryRep::to_matrix()`,
    /// never a formula written here -- a hand-written formula is a second
    /// implementation and can only agree with itself. The Selene side is transcribed
    /// from Selene's own reference `selene-ext/simulators/quest/python/
    /// gate_definitions.py` at the pinned revision: `RZ(a) = diag(e^{-ia/2}, e^{ia/2})`,
    /// `RX(t) = [[cos t/2, -i sin t/2], [-i sin t/2, cos t/2]]`, `RXY(t, p) = RZ(p) RX(t)
    /// RZ(-p)`, and `RZZ(a) = diag(e^{-ia/2}, e^{ia/2}, e^{ia/2}, e^{-ia/2})`.
    ///
    /// Three facts the pinned values record rather than hide:
    ///
    /// - PECOS's dense matrices and simulators both halve the signed angle representative,
    ///   so negative-angle rotations agree with Selene's signed textbook definitions.
    /// - Selene's Rust `QuEST` simulator at the same revision scales `exp(i a/2)` out of
    ///   `RZZ`, differing from Selene's own reference definition above. Under that one
    ///   simulator the two-qubit arms carry an additional `-a/2` that the reference does
    ///   not. That is Selene-internal, and unobservable on an error-model path where
    ///   nothing is controlled. This test pins the documented reference.
    /// - PECOS's named two-qubit Pauli roots are conventional phase-fixed matrices, so
    ///   `SP^2 = P` exactly, while Selene's reference set provides only the rotation
    ///   representatives. These arms therefore carry the convention phase. It is
    ///   unobservable here because nothing on this error-model path is controlled.
    ///
    /// Beyond the phase, the test asserts `U_pecos == e^{i phi} U_emitted` entrywise, so
    /// an arm that is wrong by more than a phase (the former `H` arm was wrong by `Z`)
    /// fails here rather than being reported as a phase. It also asserts every emitted
    /// operation targets the qubits the PECOS gate named.
    #[test]
    fn residual_phase_per_arm() {
        use nalgebra::DMatrix;
        use num_complex::Complex64;
        use pecos_core::UnitaryRep;
        use pecos_core::unitary_rep::RotationType;
        use pecos_quantum::ToMatrix;
        use smallvec::{SmallVec, smallvec};

        fn c(re: f64, im: f64) -> Complex64 {
            Complex64::new(re, im)
        }
        fn selene_rz(a: f64) -> DMatrix<Complex64> {
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    c(0.0, -a / 2.0).exp(),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, a / 2.0).exp(),
                ],
            )
        }
        fn selene_rx(t: f64) -> DMatrix<Complex64> {
            let (s, co) = ((t / 2.0).sin(), (t / 2.0).cos());
            DMatrix::from_row_slice(2, 2, &[c(co, 0.0), c(0.0, -s), c(0.0, -s), c(co, 0.0)])
        }
        fn selene_rxy(t: f64, p: f64) -> DMatrix<Complex64> {
            selene_rz(p) * selene_rx(t) * selene_rz(-p)
        }
        fn selene_rzz(a: f64) -> DMatrix<Complex64> {
            DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
                c(0.0, -a / 2.0).exp(),
                c(0.0, a / 2.0).exp(),
                c(0.0, a / 2.0).exp(),
                c(0.0, -a / 2.0).exp(),
            ]))
        }
        /// Emitted operations composed in execution order -- Selene applies a batch
        /// first to last, so each later operation multiplies on the left. Every
        /// operation must target one of `targets`.
        fn emitted_matrix(ops: &[Operation], targets: &[u64]) -> DMatrix<Complex64> {
            let dim = 1 << targets.len();
            let mut total = DMatrix::identity(dim, dim);
            for op in ops {
                let (step, touched): (DMatrix<Complex64>, Vec<u64>) = match *op {
                    Operation::RXYGate {
                        qubit_id,
                        theta,
                        phi,
                    } => (selene_rxy(theta, phi), vec![qubit_id]),
                    Operation::RZGate { qubit_id, theta } => (selene_rz(theta), vec![qubit_id]),
                    Operation::RZZGate {
                        qubit_id_1,
                        qubit_id_2,
                        theta,
                    } => (selene_rzz(theta), vec![qubit_id_1, qubit_id_2]),
                    ref other => panic!("phase table only covers unitary arms, got {other:?}"),
                };
                assert_eq!(
                    touched, targets,
                    "emitted {op:?} does not act on the PECOS gate's targets {targets:?}"
                );
                assert_eq!(step.nrows(), dim, "arm emitted an op of the wrong width");
                total = step * total;
            }
            total
        }
        /// Returns phi with `U_pecos == e^{i phi} U_emitted`, or the entrywise mismatch.
        fn residual_phase(
            pecos: &DMatrix<Complex64>,
            emitted: &DMatrix<Complex64>,
        ) -> Result<f64, String> {
            if pecos
                .iter()
                .chain(emitted.iter())
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err("non-finite entry".into());
            }
            let (i, j) = (0..pecos.nrows())
                .flat_map(|i| (0..pecos.ncols()).map(move |j| (i, j)))
                .find(|&(i, j)| pecos[(i, j)].norm() > 1e-9 && emitted[(i, j)].norm() > 1e-9)
                .ok_or("no jointly nonzero entry")?;
            let ratio = pecos[(i, j)] / emitted[(i, j)];
            if (ratio.norm() - 1.0).abs() > 1e-9 {
                return Err(format!("ratio is not unit modulus: {ratio}"));
            }
            let rebuilt = emitted * ratio;
            let worst = (pecos - &rebuilt)
                .iter()
                .map(|z| z.norm())
                .fold(0.0, f64::max);
            if worst > 1e-9 {
                return Err(format!(
                    "differs by more than a global phase (max entry error {worst:.3e})"
                ));
            }
            Ok(ratio.arg())
        }

        /// One row of the pinned table.
        struct Arm {
            name: &'static str,
            build: fn(&mut pecos_engines::prelude::ByteMessageBuilder),
            pecos: UnitaryRep,
            /// `phi` in `U_pecos = e^{i phi} U_emitted`.
            expected: f64,
        }
        let q1: SmallVec<[usize; 3]> = smallvec![0];
        let q2: SmallVec<[usize; 3]> = smallvec![0, 1];
        let named = |g: GateType, q: &SmallVec<[usize; 3]>| UnitaryRep::gate(g, q.clone());
        let rot = |r: RotationType, a: f64, q: &SmallVec<[usize; 3]>| {
            UnitaryRep::rotation(r, Angle64::try_from_radians(a).unwrap(), q.clone())
        };
        let arm = |name, build, pecos, expected| Arm {
            name,
            build,
            pecos,
            expected,
        };
        let cases = vec![
            // Named single-qubit gates.
            arm(
                "X",
                |b| {
                    b.x(&[0]);
                },
                named(GateType::X, &q1),
                PI / 2.0,
            ),
            arm(
                "Y",
                |b| {
                    b.y(&[0]);
                },
                named(GateType::Y, &q1),
                PI / 2.0,
            ),
            arm(
                "Z",
                |b| {
                    b.z(&[0]);
                },
                named(GateType::Z, &q1),
                PI / 2.0,
            ),
            arm(
                "H",
                |b| {
                    b.h(&[0]);
                },
                named(GateType::H, &q1),
                PI / 2.0,
            ),
            arm(
                "SZ",
                |b| {
                    b.sz(&[0]);
                },
                named(GateType::SZ, &q1),
                PI / 4.0,
            ),
            arm(
                "SZdg",
                |b| {
                    b.szdg(&[0]);
                },
                named(GateType::SZdg, &q1),
                -PI / 4.0,
            ),
            arm(
                "T",
                |b| {
                    b.t(&[0]);
                },
                named(GateType::T, &q1),
                PI / 8.0,
            ),
            arm(
                "Tdg",
                |b| {
                    b.tdg(&[0]);
                },
                named(GateType::Tdg, &q1),
                -PI / 8.0,
            ),
            // Parameterised single-qubit rotations, positive and negative angles.
            arm(
                "RX(+0.37)",
                |b| {
                    b.rx(pecos_core::radians!(0.37), &[0]);
                },
                rot(RotationType::RX, 0.37, &q1),
                0.0,
            ),
            arm(
                "RX(-0.37)",
                |b| {
                    b.rx(pecos_core::radians!(-0.37), &[0]);
                },
                rot(RotationType::RX, -0.37, &q1),
                0.0,
            ),
            arm(
                "RY(+0.37)",
                |b| {
                    b.ry(pecos_core::radians!(0.37), &[0]);
                },
                rot(RotationType::RY, 0.37, &q1),
                0.0,
            ),
            arm(
                "RY(-0.37)",
                |b| {
                    b.ry(pecos_core::radians!(-0.37), &[0]);
                },
                rot(RotationType::RY, -0.37, &q1),
                0.0,
            ),
            arm(
                "RZ(+0.37)",
                |b| {
                    b.rz(pecos_core::radians!(0.37), &[0]);
                },
                rot(RotationType::RZ, 0.37, &q1),
                0.0,
            ),
            arm(
                "RZ(-0.37)",
                |b| {
                    b.rz(pecos_core::radians!(-0.37), &[0]);
                },
                rot(RotationType::RZ, -0.37, &q1),
                0.0,
            ),
            arm(
                "RXY1Q(+0.37, -0.91)",
                |b| {
                    b.rxy1q(
                        pecos_core::radians!(0.37),
                        pecos_core::radians!(-0.91),
                        &[0],
                    );
                },
                UnitaryRep::Gate(
                    pecos_core::Unitary::RXY1Q {
                        theta: pecos_core::radians!(0.37),
                        phi: pecos_core::radians!(-0.91),
                    },
                    q1.clone(),
                ),
                0.0,
            ),
            // Two-qubit.
            arm(
                "SZZ",
                |b| {
                    b.szz(&[(0, 1)]);
                },
                named(GateType::SZZ, &q2),
                PI / 4.0,
            ),
            arm(
                "SZZdg",
                |b| {
                    b.szzdg(&[(0, 1)]);
                },
                named(GateType::SZZdg, &q2),
                -PI / 4.0,
            ),
            arm(
                "RZZ(+0.37)",
                |b| {
                    b.rzz(pecos_core::radians!(0.37), &[(0, 1)]);
                },
                rot(RotationType::RZZ, 0.37, &q2),
                0.0,
            ),
            arm(
                "RZZ(-0.37)",
                |b| {
                    b.rzz(pecos_core::radians!(-0.37), &[(0, 1)]);
                },
                rot(RotationType::RZZ, -0.37, &q2),
                0.0,
            ),
        ];

        for Arm {
            name,
            build,
            pecos,
            expected,
        } in cases
        {
            let targets: Vec<u64> = pecos.qubits().iter().map(|&q| q as u64).collect();
            let mut builder = ByteMessage::quantum_operations_builder();
            build(&mut builder);
            let message = builder.build();
            let mut simulator = ClassicalSimulator::with_qubits(targets.len());
            SeleneSimulator::process(&mut simulator, &message).unwrap();
            let ops = simulator.received.concat();
            let emitted = emitted_matrix(&ops, &targets);
            let source = pecos.to_matrix().inner().clone();
            let phi = residual_phase(&source, &emitted)
                .unwrap_or_else(|why| panic!("{name}: {why}; emitted {ops:?}"));
            let delta = (phi - expected).rem_euclid(2.0 * PI);
            let delta = delta.min(2.0 * PI - delta);
            assert!(
                delta < 1e-9,
                "{name}: residual phase {phi:.6} rad, expected {expected:.6} rad (U_pecos = e^{{i phi}} U_emitted)"
            );
        }
    }

    #[test]
    fn rpp_round_trip_preserves_one_gate_and_both_qubit_times() {
        let mut model = build_error_model(r#"{"model":{}}"#, 3);
        let mut sim = ClassicalSimulator::with_qubits(3);
        model.shot_start(0, 41).unwrap();
        model
            .handle_operations(
                runtime_batch(
                    vec![Operation::RPPGate {
                        qubit_id_1: 2,
                        qubit_id_2: 0,
                        theta: -0.73,
                        phi: 0.41,
                    }],
                    20,
                    7,
                ),
                &mut sim,
            )
            .unwrap();
        assert_eq!(model.last_operation_end, vec![Some(27), None, Some(27)]);
        let received = sim.received.concat();
        assert_eq!(received.len(), 1);
        let Operation::RPPGate {
            qubit_id_1,
            qubit_id_2,
            theta,
            phi,
        } = received[0]
        else {
            panic!("expected a single RPP operation, got {received:?}");
        };
        assert_eq!((qubit_id_1, qubit_id_2), (2, 0));
        assert!((theta + 0.73).abs() < 1e-12);
        assert!((phi - 0.41).abs() < 1e-12);
        // Either target is busy until 27ns, so a gate starting at 26ns
        // must fail whichever of the two qubits it uses.
        for qubit_id in [0, 2] {
            assert!(
                model
                    .handle_operations(
                        runtime_batch(
                            vec![Operation::RZGate {
                                qubit_id,
                                theta: 0.5,
                            }],
                            26,
                            1
                        ),
                        &mut sim
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn rpp_uses_one_two_qubit_noise_channel() {
        // Force an XI error for each two-qubit gate and Z for every
        // single-qubit gate. An RPP should produce one XI error and no Z
        // errors from its internal basis changes.
        let mut model = build_error_model(
            r#"{"model":{
            "two_qubit":{"probability":1.0,"pauli_model":{"XI":1.0}},
            "single_qubit":{"probability":1.0,"pauli_model":{"Z":1.0}}
        }}"#,
            2,
        );
        let mut sim = ClassicalSimulator::with_qubits(2);
        model.shot_start(0, 41).unwrap();
        model
            .handle_operations(
                runtime_batch(
                    vec![Operation::RPPGate {
                        qubit_id_1: 0,
                        qubit_id_2: 1,
                        theta: 0.73,
                        phi: 0.41,
                    }],
                    0,
                    1,
                ),
                &mut sim,
            )
            .unwrap();
        let received = sim.received.concat();
        assert_eq!(received.len(), 2, "{received:?}");
        assert_eq!(
            received
                .iter()
                .filter(|op| matches!(op, Operation::RPPGate { .. }))
                .count(),
            1
        );
        assert_eq!(
            received
                .iter()
                .filter(|op| matches!(op, Operation::RXYGate { theta, phi, .. }
            if (*theta - PI).abs() < 1e-12 && phi.abs() < 1e-12))
                .count(),
            1
        );
    }

    #[test]
    fn same_qubit_batch_charges_idle_once_and_advances_to_batch_end() {
        let mut model = build_error_model(
            r#"{"model":{"idle":{"coherent_rate":1.0,"coherent_model":{"RZ":1.0}}}}"#,
            1,
        );
        let mut simulator = ClassicalSimulator::with_qubits(1);
        model.shot_start(0, 41).unwrap();
        model
            .handle_operations(
                runtime_batch(vec![Operation::Reset { qubit_id: 0 }], 0, 100_000_000),
                &mut simulator,
            )
            .unwrap();
        simulator.received.clear();
        model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::RXYGate {
                            qubit_id: 0,
                            theta: 0.2,
                            phi: 0.0,
                        },
                        Operation::Measure {
                            qubit_id: 0,
                            result_id: 0,
                        },
                    ],
                    350_000_000,
                    100_000_000,
                ),
                &mut simulator,
            )
            .unwrap();
        let idle_angles = |simulator: &ClassicalSimulator| {
            simulator
                .received
                .iter()
                .flatten()
                .filter_map(|operation| {
                    if let Operation::RZGate { qubit_id: 0, theta } = operation {
                        Some(*theta)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        let angles = idle_angles(&simulator);
        assert_eq!(angles.len(), 1);
        assert!((angles[0] - 0.25).abs() < 1e-12);
        assert_eq!(model.last_operation_end, vec![Some(450_000_000)]);
        simulator.received.clear();
        model
            .handle_operations(
                runtime_batch(
                    vec![Operation::Measure {
                        qubit_id: 0,
                        result_id: 1,
                    }],
                    575_000_000,
                    100_000_000,
                ),
                &mut simulator,
            )
            .unwrap();
        let angles = idle_angles(&simulator);
        assert_eq!(angles.len(), 1);
        assert!((angles[0] - 0.125).abs() < 1e-12);
    }

    #[test]
    fn same_qubit_batch_with_rzz_preserves_idle_gaps() {
        assert_same_qubit_batch_idle(
            Operation::RZZGate {
                qubit_id_1: 0,
                qubit_id_2: 1,
                theta: 0.3,
            },
            &[(0, 0.25), (1, 0.2)],
            &[Some(450_000_000), Some(450_000_000)],
        );
    }

    #[test]
    fn same_qubit_batch_with_reset_preserves_idle_gap() {
        assert_same_qubit_batch_idle(
            Operation::Reset { qubit_id: 0 },
            &[(0, 0.25)],
            &[Some(450_000_000), Some(150_000_000)],
        );
    }

    fn assert_same_qubit_batch_idle(
        second: Operation,
        expected_idle: &[(u64, f64)],
        expected_ends: &[Option<u64>],
    ) {
        let mut model = build_error_model(
            r#"{"model":{"idle":{"coherent_rate":1.0,"coherent_model":{"RZ":1.0}}}}"#,
            2,
        );
        let mut simulator = ClassicalSimulator::with_qubits(2);
        model.shot_start(0, 41).unwrap();
        for (qubit_id, start) in [(0, 0), (1, 50_000_000)] {
            model
                .handle_operations(
                    runtime_batch(vec![Operation::Reset { qubit_id }], start, 100_000_000),
                    &mut simulator,
                )
                .unwrap();
        }
        simulator.received.clear();
        model
            .handle_operations(
                runtime_batch(
                    vec![
                        Operation::RXYGate {
                            qubit_id: 0,
                            theta: 0.2,
                            phi: 0.0,
                        },
                        second,
                    ],
                    350_000_000,
                    100_000_000,
                ),
                &mut simulator,
            )
            .unwrap();
        let idle = simulator
            .received
            .iter()
            .flatten()
            .filter_map(|operation| {
                if let Operation::RZGate { qubit_id, theta } = operation {
                    Some((*qubit_id, *theta))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(idle.len(), expected_idle.len());
        for ((qubit, angle), (expected_qubit, expected_angle)) in idle.iter().zip(expected_idle) {
            assert_eq!(qubit, expected_qubit);
            assert!((angle - expected_angle).abs() < 1e-12);
        }
        assert_eq!(model.last_operation_end, expected_ends);
    }

    #[test]
    fn adapter_rejects_invalid_runtime_contracts() {
        let error =
            GeneralNoiseErrorModel::from_plugin_json(2, r#"{"model":{},"local_groups":[[0,2]]}"#)
                .err()
                .expect("an out-of-bounds local group must fail");
        assert!(error.to_string().contains("contains qubit 2"));

        let mut error_model = build_error_model(r#"{"model":{}}"#, 1);
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

        let mut error_model = build_error_model(r#"{"model":{}}"#, 1);
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

        let mut error_model = build_error_model(r#"{"model":{}}"#, 1);
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
            .expect("an RPP operation on the same qubit twice must fail");
        assert!(error.to_string().contains("requires two distinct qubits"));

        let mut error_model = build_error_model(r#"{"model":{}}"#, 1);
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
    #[test]
    fn invalid_model_configuration_retains_adapter_context() {
        let error = GeneralNoiseErrorModel::from_plugin_json(
            1,
            r#"{"model":{"single_qubit":{"probability":1.5}}}"#,
        )
        .err()
        .expect("invalid model must fail");
        assert_eq!(error.to_string(), "invalid general-noise configuration");
        assert_eq!(
            error.root_cause().to_string(),
            "Input error: Probability must be between 0 and 1, got 1.5"
        );
    }

    #[test]
    fn plugin_envelope_is_strict_and_topology_stays_in_adapter() {
        for json in [
            "{}",
            r#"{"model":{},"unknown":1}"#,
            r#"{"model":{"measurement":{"local_groups":[[0]]}}}"#,
            r#"{"model":{},"local_groups":[[-1]]}"#,
            r#"{"model":{},"local_groups":[[2]]}"#,
        ] {
            assert!(
                GeneralNoiseErrorModel::from_plugin_json(2, json).is_err(),
                "{json}"
            );
        }
        let without_groups =
            GeneralNoiseErrorModel::from_plugin_json(2, r#"{"model":{}}"#).unwrap();
        assert_eq!(without_groups.local_groups, [] as [BTreeSet<usize>; 0]);
        let envelope =
            r#"{"model":{"preparation":{"probability":0.1}},"local_groups":[[0,1],[1]]}"#;
        let from_json = GeneralNoiseErrorModel::from_plugin_json(2, envelope).unwrap();
        let config =
            pecos_engines::GeneralNoiseConfig::from_json(r#"{"preparation":{"probability":0.1}}"#)
                .unwrap();
        let typed = GeneralNoiseErrorModel::new(2, &config, &[vec![0, 1], vec![1]]).unwrap();
        assert_eq!(from_json.local_groups, typed.local_groups);
        assert_eq!(
            typed.local_groups,
            vec![BTreeSet::from([0, 1]), BTreeSet::from([1])]
        );
        assert!(GeneralNoiseErrorModel::new(1, &config, &[vec![1]]).is_err());
    }
}
