use std::any::Any;

use pecos_core::errors::PecosError;
use pecos_engines::{
    ByteMessage, ClassicalControlEngineBuilder, ClassicalEngine, ControlEngine, Engine, EngineStage,
};
use pecos_neo::prelude::*;
use pecos_neo::tool::{
    importance_sampling, monte_carlo, path_enumeration, sim_neo, sim_neo_builder, sparse_stab,
    subset_simulation,
};
use pecos_results::Shot;

#[derive(Clone)]
struct ContinueFailureEngine;

impl Engine for ContinueFailureEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, _input: ()) -> Result<Shot, PecosError> {
        Ok(Shot::default())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}

impl ClassicalEngine for ContinueFailureEngine {
    fn num_qubits(&self) -> usize {
        1
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        Ok(ByteMessage::create_empty())
    }

    fn handle_measurements(&mut self, _message: ByteMessage) -> Result<(), PecosError> {
        Ok(())
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        Ok(Shot::default())
    }

    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for ContinueFailureEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, _input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        let commands = ByteMessage::quantum_operations_builder()
            .h(&[0])
            .mz(&[0])
            .build();
        Ok(EngineStage::NeedsProcessing(commands))
    }

    fn continue_processing(
        &mut self,
        _result: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        Err(PecosError::Processing("stub continue failure".into()))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}

#[derive(Clone)]
struct ContinueFailureBuilder;

impl ClassicalControlEngineBuilder for ContinueFailureBuilder {
    type Engine = ContinueFailureEngine;

    fn build(self) -> Result<ContinueFailureEngine, PecosError> {
        Ok(ContinueFailureEngine)
    }
}

#[derive(Clone)]
struct BuildFailureBuilder;

impl ClassicalControlEngineBuilder for BuildFailureBuilder {
    type Engine = ContinueFailureEngine;

    fn build(self) -> Result<ContinueFailureEngine, PecosError> {
        Err(PecosError::Processing("stub builder failure".into()))
    }
}

#[test]
fn sequential_classical_error_is_returned() {
    let error = sim_neo_builder()
        .with_engine(ContinueFailureBuilder)
        .quantum(sparse_stab())
        .qubits(1)
        .sampling(monte_carlo(3))
        .run()
        .expect_err("classical engine failure should be returned");

    assert!(error.to_string().contains("stub continue failure"));
}

#[test]
fn sequential_quantum_error_is_returned() {
    let circuit = CommandBuilder::new().rz(&[0], 0.3).build();
    let error = sim_neo(circuit)
        .quantum(sparse_stab())
        .sampling(monte_carlo(1))
        .run()
        .expect_err("non-Clifford execution failure should be returned");

    assert!(error.to_string().contains("has a non-Clifford angle"));
}

#[test]
fn parallel_classical_error_is_returned() {
    let error = sim_neo_builder()
        .with_engine(ContinueFailureBuilder)
        .quantum(sparse_stab())
        .qubits(1)
        .sampling(monte_carlo(8).workers(4))
        .run()
        .expect_err("parallel classical engine failure should be returned");

    assert!(error.to_string().contains("stub continue failure"));
}

#[test]
fn startup_error_is_returned_on_every_run() {
    let mut simulation = sim_neo_builder()
        .with_engine(BuildFailureBuilder)
        .quantum(sparse_stab())
        .qubits(1)
        .sampling(monte_carlo(1))
        .build();

    for _ in 0..2 {
        let error = simulation
            .run()
            .expect_err("classical engine build failure should be returned on every run");
        assert!(error.to_string().contains("stub builder failure"));
    }
}

#[test]
fn event_handler_gate_error_is_returned() {
    let injected_gate = CommandBuilder::new()
        .rz(&[0], 0.3)
        .build()
        .iter()
        .next()
        .expect("injected circuit should contain one gate")
        .clone();
    let handlers = EventHandlers::new()
        .on_before_gate(move |_| NoiseResponse::inject_gate(injected_gate.clone()));

    let error = sim_neo(CommandBuilder::new().h(&[0]).build())
        .quantum(sparse_stab())
        .event_handlers(handlers)
        .sampling(monte_carlo(1))
        .run()
        .expect_err("event-handler gate failure should be returned");

    assert!(error.to_string().contains("has a non-Clifford angle"));
}

#[test]
#[should_panic(expected = "Importance sampling cannot execute static circuit gate RZ")]
fn importance_sampling_rejects_non_clifford_circuit_at_build() {
    let _ = sim_neo(CommandBuilder::new().rz(&[0], 0.3).build())
        .quantum(sparse_stab())
        .sampling(importance_sampling(1))
        .build();
}

#[test]
#[should_panic(expected = "Path enumeration cannot execute static circuit gate RZ")]
fn path_enumeration_rejects_non_clifford_circuit_at_build() {
    let _ = sim_neo(CommandBuilder::new().rz(&[0], 0.3).build())
        .quantum(sparse_stab())
        .sampling(path_enumeration(1))
        .build();
}

#[test]
#[should_panic(expected = "Subset simulation cannot execute static circuit gate RZ")]
fn subset_simulation_rejects_non_clifford_circuit_at_build() {
    let _ = sim_neo(CommandBuilder::new().rz(&[0], 0.3).build())
        .quantum(sparse_stab())
        .sampling(subset_simulation(1).score(|_| 0.0).failure(|_| false))
        .build();
}

#[test]
#[should_panic(expected = "Subset simulation requires samples_per_level > 0")]
fn subset_simulation_rejects_zero_samples_at_build() {
    let _ = sim_neo(CommandBuilder::new().mz(&[0]).build())
        .quantum(sparse_stab())
        .sampling(subset_simulation(0).score(|_| 0.0).failure(|_| false))
        .build();
}
