//! Downstream use of the public constructor, including a replacement FFI factory.
use anyhow::Result;
use pecos_selene_general_noise::GeneralNoiseErrorModel;
use selene_core::error_model::BatchResult;
use selene_core::error_model::interface::ErrorModelInterface;
use selene_core::runtime::{BatchOperation, Operation};
use selene_core::simulator::SimulatorInterface;
use selene_core::time::{Duration, Instant};
use selene_core::utils::MetricValue;

#[derive(Default)]
struct RecordingSimulator(Vec<Operation>);
impl SimulatorInterface for RecordingSimulator {
    fn exit(&mut self) -> Result<()> {
        Ok(())
    }
    fn shot_start(&mut self, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn shot_end(&mut self) -> Result<()> {
        Ok(())
    }
    fn handle_operations(&mut self, batch: BatchOperation) -> Result<BatchResult> {
        let mut result = BatchResult::default();
        for op in batch {
            if let Operation::Measure { result_id, .. } = op {
                result.set_bool_result(result_id, false);
            }
            self.0.push(op);
        }
        Ok(result)
    }
    fn get_metric(&mut self, _: u8) -> Result<Option<(String, MetricValue)>> {
        Ok(None)
    }
}
fn batch(ops: Vec<Operation>, start: u64) -> BatchOperation {
    BatchOperation::runtime(ops, Instant::from(start), Duration::from(1))
}

#[test]
fn constructor_validates_configuration_and_capacity() {
    for config in [
        "{",
        r#"{"unknown":1}"#,
        r#"{"preparation":{"probability":2.0}}"#,
        r#"{"measurement":{"local_groups":[[0,1]]}}"#,
    ] {
        assert!(
            GeneralNoiseErrorModel::from_json(1, config).is_err(),
            "{config}"
        );
    }
}

#[test]
fn library_retains_native_idle_and_measurement_identity() {
    let mut model = GeneralNoiseErrorModel::from_json(
        1,
        r#"{"idle":{"linear_rate":2.0,"linear_model":{"X":1.0}}}"#,
    )
    .unwrap();
    let mut simulator = RecordingSimulator::default();
    model.shot_start(7, 11).unwrap();
    model
        .handle_operations(
            batch(vec![Operation::Reset { qubit_id: 0 }], 0),
            &mut simulator,
        )
        .unwrap();
    let result = model
        .handle_operations(
            batch(
                vec![Operation::Measure {
                    qubit_id: 0,
                    result_id: 101,
                }],
                500_000_001,
            ),
            &mut simulator,
        )
        .unwrap();
    assert_eq!(result.bool_results.len(), 1);
    assert_eq!(result.bool_results[0].result_id, 101);
    assert!(simulator.0.iter().any(|op| matches!(op, Operation::RXYGate { theta, .. } if (theta.abs()-std::f64::consts::PI).abs()<1e-12)));
    model.shot_end().unwrap();
    simulator.0.clear();
    model.shot_start(8, 13).unwrap();
    model
        .handle_operations(
            batch(
                vec![Operation::Measure {
                    qubit_id: 0,
                    result_id: 102,
                }],
                2_000_000_000,
            ),
            &mut simulator,
        )
        .unwrap();
    assert!(
        !simulator
            .0
            .iter()
            .any(|op| matches!(op, Operation::RXYGate { .. }))
    );
    model.shot_end().unwrap();
}

#[test]
fn constructor_does_not_acknowledge_custom_operations() {
    let mut model = GeneralNoiseErrorModel::from_json(1, "{}").unwrap();
    let mut simulator = RecordingSimulator::default();
    model.shot_start(0, 3).unwrap();
    let error = model
        .handle_operations(
            batch(
                vec![Operation::Custom {
                    custom_tag: 42,
                    data: vec![1].into(),
                }],
                0,
            ),
            &mut simulator,
        )
        .err()
        .expect("custom operation must be rejected");
    assert!(
        error
            .to_string()
            .contains("no device-neutral PECOS meaning")
    );
    assert_eq!(simulator.0, [] as [Operation; 0]);
}

// This consumer supplies its own entry points. Linking it with the dependency's
// exports would produce duplicate symbols, so this is also a composition check.
#[cfg(not(feature = "plugin-export"))]
#[derive(Default)]
struct ConsumerFactory;
#[cfg(not(feature = "plugin-export"))]
impl selene_core::error_model::interface::ErrorModelInterfaceFactory for ConsumerFactory {
    type Interface = GeneralNoiseErrorModel;
    fn init(
        self: std::sync::Arc<Self>,
        n_qubits: u64,
        args: &[impl AsRef<str>],
    ) -> Result<Box<Self::Interface>> {
        anyhow::ensure!(args.len() == 2, "expected one configuration argument");
        Ok(Box::new(GeneralNoiseErrorModel::from_json(
            usize::try_from(n_qubits)?,
            args[1].as_ref(),
        )?))
    }
}
#[cfg(not(feature = "plugin-export"))]
selene_core::export_error_model_plugin!(crate::ConsumerFactory);
