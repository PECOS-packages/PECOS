use pecos_core::errors::PecosError;
use pecos_engines::{ClassicalControlEngine, Engine, Shot};
use pecos_phir_json::{setup_phir_json_engine, v0_1::engine::PhirJsonEngine};
use serde_json::{Value, json};

const QUANTUM_ERROR: &str = "PhirJsonEngine::process(()) cannot execute quantum commands; use start()/continue_processing() with a quantum engine, or the simulation builder.";

fn program(ops: Vec<Value>) -> String {
    let mut declarations = vec![
        json!({"data":"qvar_define","data_type":"qubits","variable":"q","size":2}),
        json!({"data":"cvar_define","data_type":"i32","variable":"m","size":2}),
    ];
    declarations.extend(ops);
    json!({"format":"PHIR/JSON","version":"0.1.0","ops":declarations}).to_string()
}

fn boxed(json: &str) -> Box<dyn ClassicalControlEngine> {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), json).unwrap();
    setup_phir_json_engine(file.path()).unwrap()
}

fn assert_quantum_error(result: Result<Shot, PecosError>) {
    let Err(PecosError::Processing(message)) = result else {
        panic!("expected a quantum processing error, got {result:?}");
    };
    assert_eq!(message, QUANTUM_ERROR);
}

fn rejects(ops: Vec<Value>) {
    let json = program(ops);
    assert_quantum_error(PhirJsonEngine::from_json(&json).unwrap().process(()));
    assert_quantum_error(boxed(&json).process(()));
}

#[test]
fn standalone_rejects_gate() {
    rejects(vec![json!({"qop":"X","args":[["q",0]]})]);
}

#[test]
fn standalone_rejects_measurement() {
    rejects(vec![
        json!({"qop":"Measure","args":[["q",0]],"returns":[["m",0]]}),
    ]);
}

#[test]
fn standalone_rejects_taken_quantum_branch() {
    rejects(vec![
        json!({"block":"if","condition":{"cop":"==","args":[1,1]},
        "true_branch":[{"qop":"X","args":[["q",0]]}]}),
    ]);
}

#[test]
fn standalone_rejects_machine_command() {
    for mop in ["Idle", "Delay", "Transport"] {
        rejects(vec![
            json!({"mop":mop,"args":[["q",0]],"duration":[1.0,"ms"]}),
        ]);
    }
}

#[test]
fn standalone_rejects_quantum_after_empty_yield() {
    rejects(vec![
        json!({"cop":"Result","args":["m"],"returns":["before"]}),
        json!({"qop":"X","args":[["q",0]]}),
    ]);
}

#[test]
fn standalone_allows_untaken_quantum_branch() {
    let json = program(vec![
        json!({"block":"if","condition":{"cop":"==","args":[1,0]},
            "true_branch":[{"qop":"X","args":[["q",0]]}],
            "false_branch":[{"cop":"=","args":[3],"returns":["m"]}]}),
        json!({"cop":"Result","args":["m"],"returns":["out"]}),
    ]);
    for shot in [
        PhirJsonEngine::from_json(&json)
            .unwrap()
            .process(())
            .unwrap(),
        boxed(&json).process(()).unwrap(),
    ] {
        assert_eq!(shot.data["out"].as_u32(), Some(3));
    }
}

#[test]
fn standalone_allows_multiple_exports_and_barrier() {
    let json = program(vec![
        json!({"cop":"=","args":[3],"returns":["m"]}),
        json!({"cop":"Result","args":["m"],"returns":["whole"]}),
        json!({"meta":"barrier","args":[["q",0]]}),
        json!({"cop":"Result","args":[["m",0]],"returns":["low"]}),
        json!({"cop":"Result","args":[["m",1]],"returns":["high"]}),
    ]);
    for shot in [
        PhirJsonEngine::from_json(&json)
            .unwrap()
            .process(())
            .unwrap(),
        boxed(&json).process(()).unwrap(),
    ] {
        assert_eq!(
            ["whole", "low", "high"].map(|name| shot.data[name].as_u32()),
            [Some(3), Some(1), Some(1)]
        );
    }
}

#[cfg(feature = "wasm")]
mod foreign {
    use super::*;
    use pecos_phir_json::v0_1::foreign_objects::ForeignObject;
    use std::any::Any;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    // The processor clones foreign objects before calling them. Share the counter
    // so replay is observable even across those clones.
    #[derive(Clone, Debug)]
    struct CountingForeign(Arc<AtomicUsize>);

    impl ForeignObject for CountingForeign {
        fn clone_box(&self) -> Box<dyn ForeignObject> {
            Box::new(self.clone())
        }
        fn init(&mut self) -> Result<(), PecosError> {
            Ok(())
        }
        fn new_instance(&mut self) -> Result<(), PecosError> {
            Ok(())
        }
        fn get_funcs(&self) -> Vec<String> {
            vec!["add".to_string()]
        }
        fn exec(&mut self, _name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![args.iter().sum()])
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn engine(quantum: bool) -> (PhirJsonEngine, Arc<AtomicUsize>) {
        let mut ops = vec![
            json!({"cop":"ffcall","function":"add","args":[7,3],"returns":["sum"]}),
            json!({"cop":"Result","args":["sum"],"returns":["out"]}),
            json!({"cop":"Result","args":[["sum",1]],"returns":["bit"]}),
        ];
        if quantum {
            ops.push(json!({"qop":"X","args":[["q",0]]}));
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut engine = PhirJsonEngine::from_json(&program(ops)).unwrap();
        engine.set_foreign_object(Box::new(CountingForeign(Arc::clone(&calls))));
        (engine, calls)
    }

    #[test]
    fn standalone_foreign_call_runs_once_across_exports() {
        for boxed in [false, true] {
            let (mut engine, calls) = engine(false);
            let shot = if boxed {
                let mut engine: Box<dyn ClassicalControlEngine> = Box::new(engine);
                engine.process(())
            } else {
                engine.process(())
            }
            .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(shot.data["out"].as_u32(), Some(10));
            assert_eq!(shot.data["bit"].as_u32(), Some(1));
        }
    }

    #[test]
    fn standalone_rejects_quantum_after_foreign_call() {
        for boxed in [false, true] {
            let (mut engine, calls) = engine(true);
            let result = if boxed {
                let mut engine: Box<dyn ClassicalControlEngine> = Box::new(engine);
                engine.process(())
            } else {
                engine.process(())
            };
            assert_quantum_error(result);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
        }
    }
}
