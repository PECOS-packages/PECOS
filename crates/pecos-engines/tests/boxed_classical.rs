use pecos_core::errors::PecosError;
use pecos_engines::{
    ByteMessage, ClassicalControlEngine, ClassicalEngine, ControlEngine, Engine, EngineStage, Shot,
};
use std::any::Any;

#[derive(Clone)]
struct StandaloneEngine {
    fail: bool,
    calls: u32,
}

impl Engine for StandaloneEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, (): ()) -> Result<Shot, PecosError> {
        self.calls += 1;
        if self.fail {
            return Err(PecosError::Processing("standalone failure".to_string()));
        }
        let mut shot = Shot::default();
        shot.data.insert(
            "calls".to_string(),
            pecos_engines::shot_results::Data::U32(self.calls),
        );
        Ok(shot)
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.calls = 0;
        Ok(())
    }
}

impl ClassicalEngine for StandaloneEngine {
    fn num_qubits(&self) -> usize {
        0
    }
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        panic!("standalone processing must dispatch to Engine::process")
    }
    fn handle_measurements(&mut self, _: ByteMessage) -> Result<(), PecosError> {
        panic!("standalone processing must dispatch to Engine::process")
    }
    fn get_results(&self) -> Result<Shot, PecosError> {
        panic!("standalone processing must dispatch to Engine::process")
    }
    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for StandaloneEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        panic!("the box must preserve the concrete engine's standalone policy")
    }
    fn continue_processing(
        &mut self,
        _: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        panic!("the box must preserve the concrete engine's standalone policy")
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Engine::reset(self)
    }
}

#[test]
fn boxed_process_preserves_standalone_state_and_results() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(StandaloneEngine {
        fail: false,
        calls: 0,
    });
    for expected in [1, 2] {
        assert_eq!(
            engine.process(()).unwrap().data["calls"].as_u32(),
            Some(expected)
        );
    }
}

#[test]
fn boxed_process_preserves_standalone_errors() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(StandaloneEngine {
        fail: true,
        calls: 0,
    });
    let result = engine.process(());
    let Err(PecosError::Processing(message)) = result else {
        panic!("expected the concrete standalone error, got {result:?}");
    };
    assert_eq!(message, "standalone failure");
}
