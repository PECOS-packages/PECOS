//! Opt-in interpreter attachment scoped to one logical Monte Carlo worker.
use pecos_core::errors::PecosError;
use pecos_engines::monte_carlo::{MonteCarloEngine, WorkerScope};
use pecos_engines::shot_results::ShotVec;
use pyo3::Python;

struct PythonWorkerContext;

impl WorkerScope for PythonWorkerContext {
    fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        // Python thread-state teardown can run finalizers after the last adapter
        // callback. Keep reentry rejection alive through attachment AND teardown:
        // a finalizer must not wait on the built simulation's caller-held mutex.
        let _callbacks = crate::scheduled_adapter::CallbackScope::enter();
        // Both guards stay on this worker. Detaching retains its thread state,
        // so nested callback attachments reuse it while Rust runs without the GIL.
        Python::try_attach(|py| py.detach(work))
            .ok_or_else(|| PecosError::Processing("Could not attach Python worker context".into()))
    }
}

pub(crate) fn run(
    engine: &mut MonteCarloEngine,
    shots: usize,
    workers: usize,
    enabled: bool,
) -> Result<ShotVec, PecosError> {
    if enabled {
        engine.run_with_workers_in_scope(shots, workers, &PythonWorkerContext)
    } else {
        engine.run_with_workers(shots, workers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::RngManageable;
    use pecos_engines::ByteMessage;
    use pecos_engines::engine_system::{ControlEngine, EngineStage};
    use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    use pecos_engines::noise::{NoiseModel, PassThroughNoiseModel};
    use pyo3::prelude::*;
    use pyo3::types::PyModule;
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    struct CallbackNoise {
        identity: usize,
        inner: PassThroughNoiseModel,
        callback: Arc<Py<PyAny>>,
        active: bool,
        panic: bool,
    }
    impl Clone for CallbackNoise {
        fn clone(&self) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
            Self {
                identity: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                inner: self.inner.clone(),
                callback: self.callback.clone(),
                active: false,
                panic: self.panic,
            }
        }
    }
    impl CallbackNoise {
        fn call(&self, name: &str) -> Result<(), PecosError> {
            Python::attach(|py| {
                self.callback
                    .call1(py, (self.identity, name))
                    .map(|_| ())
                    .map_err(|error| PecosError::Processing(error.to_string()))
            })
        }
    }
    impl Drop for CallbackNoise {
        fn drop(&mut self) {
            if self.active {
                self.call("drop").unwrap();
            }
        }
    }
    impl RngManageable for CallbackNoise {
        type Rng = pecos_random::PecosRng;
        fn set_rng(&mut self, rng: Self::Rng) {
            self.inner.set_rng(rng);
        }
        fn rng(&self) -> &Self::Rng {
            self.inner.rng()
        }
        fn rng_mut(&mut self) -> &mut Self::Rng {
            self.inner.rng_mut()
        }
    }
    impl NoiseModel for CallbackNoise {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }
    impl ControlEngine for CallbackNoise {
        type Input = ByteMessage;
        type Output = ByteMessage;
        type EngineInput = ByteMessage;
        type EngineOutput = ByteMessage;
        fn reset(&mut self) -> Result<(), PecosError> {
            self.active = true;
            self.call("reset")?;
            self.inner.reset()
        }
        fn start(
            &mut self,
            input: ByteMessage,
        ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
            assert!(!self.panic, "intentional callback worker panic");
            self.call("start")?;
            self.inner.start(input)
        }
        fn continue_processing(
            &mut self,
            result: ByteMessage,
        ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
            self.call("continue")?;
            self.inner.continue_processing(result)
        }
    }

    fn fixture(py: Python<'_>, panic: bool) -> (Bound<'_, PyModule>, MonteCarloEngine) {
        static NEXT_MODULE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let name = std::ffi::CString::new(format!(
            "worker_context_test_{}",
            NEXT_MODULE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ))
        .unwrap();
        let module = PyModule::from_code(
            py,
            c"\
import threading
from contextvars import ContextVar
local = threading.local()
value = ContextVar('value', default=0)
rows = []
finalized = []
fail = False
class Marker:
    def __del__(self):
        finalized.append(threading.get_ident())
def callback(identity, label):
    local.n = getattr(local, 'n', 0) + 1
    if not hasattr(local, 'marker'):
        local.marker = Marker()
    value.set(value.get() + 1)
    rows.append((identity, label, local.n, value.get()))
    if fail and label == 'start':
        raise ValueError('intentional callback error')
",
            c"worker_context_test.py",
            &name,
        )
        .unwrap();
        let noise = CallbackNoise {
            identity: 0,
            inner: PassThroughNoiseModel::new(),
            callback: Arc::new(module.getattr("callback").unwrap().unbind()),
            active: false,
            panic,
        };
        let circuit = ByteMessage::quantum_operations_builder()
            .pz(&[0])
            .h(&[0])
            .mz(&[0])
            .build();
        let engine = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(ExternalClassicalEngine::new_with_circuit(circuit)))
            .with_quantum_engine(Box::new(pecos_engines::quantum::StateVecEngine::new(2)))
            .with_noise_model(Box::new(noise))
            .build();
        (module, engine)
    }

    type Row = (u64, String, usize, usize);
    fn rows(module: &Bound<'_, PyModule>) -> Vec<Row> {
        module.getattr("rows").unwrap().extract().unwrap()
    }

    #[test]
    fn python_worker_context_lifetime_covers_reset_callbacks_and_engine_drop() {
        Python::initialize();
        Python::attach(|py| {
            for enabled in [false, true] {
                let (module, engine) = fixture(py, false);
                // Exercise the reusable Python-facing object and its GIL release.
                let simulation = Py::new(
                    py,
                    crate::engine_builders::PyQisControlSimulation {
                        inner: Arc::new(Mutex::new(engine)),
                        python_worker_context: enabled,
                        temp_dir: None,
                        operation_trace_dir: None,
                    },
                )
                .unwrap();
                for _ in 0..2 {
                    module
                        .getattr("rows")
                        .unwrap()
                        .call_method0("clear")
                        .unwrap();
                    module
                        .getattr("finalized")
                        .unwrap()
                        .call_method0("clear")
                        .unwrap();
                    simulation
                        .call_method1(py, "run_with_workers", (6, 2))
                        .unwrap();
                    let records = rows(&module);
                    let mut per_thread = std::collections::BTreeMap::<u64, Vec<&Row>>::new();
                    for record in &records {
                        per_thread.entry(record.0).or_default().push(record);
                    }
                    assert_eq!(per_thread.len(), 2);
                    for events in per_thread.values() {
                        assert_eq!(events.first().unwrap().1, "reset");
                        assert_eq!(events.last().unwrap().1, "drop");
                        assert!(events.iter().any(|e| e.1 == "continue"));
                        for (index, event) in events.iter().enumerate() {
                            let expected = if enabled { index + 1 } else { 1 };
                            assert_eq!(event.2, expected);
                            assert_eq!(event.3, expected);
                        }
                    }
                    let finalized: Vec<u64> =
                        module.getattr("finalized").unwrap().extract().unwrap();
                    assert_eq!(finalized.len(), if enabled { 2 } else { records.len() });
                    assert_eq!(
                        module
                            .getattr("value")
                            .unwrap()
                            .call_method0("get")
                            .unwrap()
                            .extract::<usize>()
                            .unwrap(),
                        0
                    );
                }
            }
        });
    }

    #[test]
    fn python_worker_context_failure_and_panic_clean_up_and_allow_next_run() {
        Python::initialize();
        Python::attach(|py| {
            for enabled in [false, true] {
                for panic in [false, true] {
                    let (module, mut engine) = fixture(py, panic);
                    module.setattr("fail", !panic).unwrap();
                    let error = py.detach(|| run(&mut engine, 4, 4, enabled)).unwrap_err();
                    assert!(error.to_string().contains(if panic {
                        "intentional callback worker panic"
                    } else {
                        "intentional callback error"
                    }));
                    assert_eq!(rows(&module).iter().filter(|r| r.1 == "drop").count(), 4);
                    let finalized: Vec<u64> =
                        module.getattr("finalized").unwrap().extract().unwrap();
                    assert_eq!(
                        finalized.len(),
                        if enabled { 4 } else { rows(&module).len() }
                    );
                    // Fresh worker state after failure, using a clone of the template.
                    engine
                        .hybrid_engine_template
                        .quantum_system
                        .noise_model_mut()
                        .as_any_mut()
                        .downcast_mut::<CallbackNoise>()
                        .unwrap()
                        .panic = false;
                    module.setattr("fail", false).unwrap();
                    let mut clone = engine.clone();
                    assert_eq!(
                        py.detach(|| run(&mut clone, 2, 1, enabled)).unwrap().len(),
                        2
                    );
                }
            }
        });
    }
}
