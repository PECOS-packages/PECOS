use super::*;

fn bell(quantum: bool) -> PlironBellEngine {
    PlironBellEngine {
        msg: if quantum {
            bell_message()
        } else {
            ByteMessage::create_empty()
        },
        num_qubits: 2,
        sent: false,
        outcomes: vec![1, 1],
    }
}

fn adaptive(quantum: bool) -> PlironAdaptiveEngine {
    PlironAdaptiveEngine {
        plan: AdaptivePlan {
            batch1: vec![],
            cond_outcome_idx: 0,
            cond_target: 0,
            batch2: if quantum { vec![Cmd::X(0)] } else { vec![] },
        },
        stage: 0,
        b1: vec![1],
        b2: vec![1],
    }
}

fn conditional(quantum: bool) -> PlironIfEngine {
    PlironIfEngine {
        batch1: vec![],
        cond_outcome_idx: 0,
        // Stale b1 selects a measurement whose stale b2 result is one.
        // A fresh run takes the empty else branch and exports zero.
        then_cmds: vec![Cmd::Mz(0, 7)],
        else_cmds: if quantum { vec![Cmd::X(0)] } else { vec![] },
        post: vec![],
        export: vec![7],
        stage: 0,
        b1: vec![1],
        b2: vec![1],
    }
}

#[test]
fn boxed_bell_drains_empty_batch_and_replaces_stale_results() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(bell(false));
    assert_eq!(engine.get_results().unwrap().data["c"].as_u32(), Some(3));
    assert_eq!(engine.process(()).unwrap().data["c"].as_u32(), Some(0));
    let concrete = engine.as_any().downcast_ref::<PlironBellEngine>().unwrap();
    assert!(concrete.sent);
    assert!(concrete.outcomes.is_empty());
}

#[test]
fn boxed_adaptive_drains_both_batches_and_replaces_stale_results() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(adaptive(false));
    let stale = engine.get_results().unwrap();
    assert_eq!(
        ["mid", "final"].map(|key| stale.data[key].as_u32()),
        [Some(1), Some(1)]
    );
    let shot = engine.process(()).unwrap();
    assert_eq!(
        ["mid", "final"].map(|key| shot.data[key].as_u32()),
        [Some(0), Some(0)]
    );
    let concrete = engine
        .as_any()
        .downcast_ref::<PlironAdaptiveEngine>()
        .unwrap();
    assert_eq!(concrete.stage, 2);
    assert!(concrete.b1.is_empty());
    assert!(concrete.b2.is_empty());
}

#[test]
fn boxed_if_drains_both_batches_and_replaces_stale_results() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(conditional(false));
    assert_eq!(engine.get_results().unwrap().data["r7"].as_u32(), Some(1));
    assert_eq!(engine.process(()).unwrap().data["r7"].as_u32(), Some(0));
    let concrete = engine.as_any().downcast_ref::<PlironIfEngine>().unwrap();
    assert_eq!(concrete.stage, 2);
    assert!(concrete.b1.is_empty());
    assert!(concrete.b2.is_empty());
}

fn assert_quantum_error(mut engine: Box<dyn ClassicalControlEngine>) {
    let result = engine.process(());
    let Err(PecosError::Processing(message)) = result else {
        panic!("expected quantum processing error, got {result:?}");
    };
    assert_eq!(
        message,
        "Box<dyn ClassicalControlEngine>::process(()) cannot execute quantum commands; use start()/continue_processing() with a quantum engine, or the simulation builder."
    );
}

#[test]
fn boxed_bell_rejects_quantum_commands() {
    assert_quantum_error(Box::new(bell(true)));
}

#[test]
fn boxed_adaptive_rejects_quantum_after_empty_batch() {
    assert_quantum_error(Box::new(adaptive(true)));
}

#[test]
fn boxed_if_rejects_quantum_after_empty_batch() {
    assert_quantum_error(Box::new(conditional(true)));
}
