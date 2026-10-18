//! Compatibility Python API backed entirely by the native PHIR-JSON engine.
use crate::prelude::PhirJsonEngine as RustPhirJsonEngine;
use crate::prelude::*;
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

#[pyclass(module = "pecos_rslib", from_py_object)]
#[derive(Debug)]
pub struct PhirJsonEngine {
    engine: Mutex<RustPhirJsonEngine>,
    // The legacy Python method accepts one outcome at a time; the native engine
    // accepts the complete batch. No register interpretation belongs here.
    pending_outcomes: Vec<usize>,
    expected_outcomes: usize,
}

impl Clone for PhirJsonEngine {
    fn clone(&self) -> Self {
        Self {
            engine: Mutex::new(self.engine.lock().clone()),
            pending_outcomes: self.pending_outcomes.clone(),
            expected_outcomes: self.expected_outcomes,
        }
    }
}

fn python_error(error: PecosError) -> PyErr {
    if matches!(error, PecosError::External(_)) {
        return crate::errors::simulation_error(error, "PHIR interpreter failed");
    }
    PyValueError::new_err(format!(
        "{error}; supply a valid program or measurement batch supported by the native PHIR-JSON engine"
    ))
}

#[pymethods]
impl PhirJsonEngine {
    /// Create an engine using native PHIR-JSON validation.
    ///
    /// # Errors
    /// Returns ValueError for invalid PHIR-JSON.
    #[new]
    pub fn py_new(phir_json: &str) -> PyResult<Self> {
        Ok(Self {
            engine: Mutex::new(RustPhirJsonEngine::from_json(phir_json).map_err(python_error)?),
            pending_outcomes: Vec::new(),
            expected_outcomes: 0,
        })
    }

    /// Compatibility constructor: no Python schema validator is used.
    /// Native validation always applies.
    ///
    /// # Errors
    /// Returns ValueError for invalid PHIR-JSON.
    #[staticmethod]
    pub fn create_with_validation_disabled(phir_json: &str) -> PyResult<Self> {
        Self::py_new(phir_json)
    }

    #[getter]
    fn results_dict(&self) -> PyResult<BTreeMap<String, u32>> {
        self.get_results()
    }

    /// Return the next native command batch as legacy Python command dictionaries.
    ///
    /// # Errors
    /// Returns ValueError if execution fails or the previous batch needs outcomes.
    pub fn process_program(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        if self.expected_outcomes != 0 {
            return Err(PyValueError::new_err(format!(
                "Received {} of {} measurement outcomes; call handle_measurement for each pending outcome before process_program",
                self.pending_outcomes.len(),
                self.expected_outcomes
            )));
        }
        let message = self.generate_commands().map_err(python_error)?;
        let mut result = Vec::new();
        for gate in message.quantum_ops().map_err(python_error)? {
            let command = PyDict::new(py);
            let params = PyDict::new(py);
            let name = match gate.gate_type {
                GateType::MZ => "Measure".to_string(),
                GateType::PZ => "Prep".to_string(),
                _ => gate.gate_type.to_string(),
            };
            let angles: Vec<_> = gate.angles.iter().map(Angle64::to_radians).collect();
            if let [theta] = angles.as_slice() {
                params.set_item("theta", theta)?;
            } else if !angles.is_empty() {
                params.set_item("angles", angles)?;
            }
            if gate.gate_type == GateType::MZ {
                params.set_item("result_id", self.expected_outcomes)?;
                self.expected_outcomes += gate.qubits.len();
            }
            command.set_item("gate_type", name)?;
            command.set_item("params", params)?;
            command.set_item(
                "qubits",
                gate.qubits.iter().map(|q| **q).collect::<Vec<_>>(),
            )?;
            result.push(command.into_any().unbind());
        }
        Ok(result)
    }

    /// Deliver one outcome in command order, forwarding complete batches natively.
    ///
    /// # Errors
    /// Returns ValueError for an invalid outcome or absent pending measurement.
    pub fn handle_measurement(&mut self, outcome: u32) -> PyResult<()> {
        if outcome > 1 || self.expected_outcomes == 0 {
            return Err(PyValueError::new_err(format!(
                "Measurement outcome {outcome} with {} pending outcomes is invalid; call process_program first and supply one 0 or 1 per measurement",
                self.expected_outcomes
            )));
        }
        self.pending_outcomes.push(outcome as usize);
        if self.pending_outcomes.len() == self.expected_outcomes {
            let mut builder = ByteMessage::outcomes_builder();
            builder.add_outcomes(&self.pending_outcomes);
            self.engine
                .lock()
                .handle_measurements(builder.build())
                .map_err(python_error)?;
            self.pending_outcomes.clear();
            self.expected_outcomes = 0;
        }
        Ok(())
    }

    /// Return native results in the legacy u32 dictionary representation.
    ///
    /// # Errors
    /// Returns ValueError if a result cannot be represented as u32.
    pub fn get_results(&self) -> PyResult<BTreeMap<String, u32>> {
        self.engine.lock().get_results().map_err(python_error)?.data.into_iter().map(|(name, data)| {
            let value = data.as_u32().ok_or_else(|| PyValueError::new_err(format!(
                "Result {name:?} has value {data:?}, which cannot be represented as u32; use the native simulation results API for wider or noninteger values"
            )))?;
            Ok((name, value))
        }).collect()
    }
}

impl ClassicalEngine for PhirJsonEngine {
    fn num_qubits(&self) -> usize {
        self.engine.lock().num_qubits()
    }
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        self.engine.lock().generate_commands()
    }
    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        self.engine.lock().handle_measurements(message)
    }
    fn get_results(&self) -> Result<Shot, PecosError> {
        self.engine.lock().get_results()
    }
    fn compile(&self) -> Result<(), PecosError> {
        self.engine.lock().compile()
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.pending_outcomes.clear();
        self.expected_outcomes = 0;
        ClassicalEngine::reset(&mut *self.engine.lock())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl ControlEngine for PhirJsonEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn reset(&mut self) -> Result<(), PecosError> {
        ClassicalEngine::reset(self)
    }
    fn start(&mut self, input: ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.pending_outcomes.clear();
        self.expected_outcomes = 0;
        self.engine.lock().start(input)
    }
    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.engine.lock().continue_processing(measurements)
    }
}

impl Engine for PhirJsonEngine {
    type Input = ();
    type Output = Shot;
    fn process(&mut self, input: ()) -> Result<Shot, PecosError> {
        self.engine.lock().process(input)
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        ClassicalEngine::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_commands_and_results_match_native() {
        let source = serde_json::json!({"format":"PHIR/JSON", "version":"0.1.0", "ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":4},
            {"data":"cvar_define","data_type":"u32","variable":"m","size":4},
            {"qop":"H","args":[["q",0],["q",2]]},
            {"qop":"SZZ","args":[[["q",0],["q",1]],[["q",2],["q",3]]]},
            {"qop":"RZZ","angles":[[0.37],"rad"],"args":[[["q",0],["q",1]],[["q",2],["q",3]]]},
            {"qop":"CX","args":[[["q",0],["q",1]],[["q",2],["q",3]]]},
            {"qop":"X","args":[["q",1],["q",3]]},
            {"qop":"RZ","angles":[[0.25],"pi"],"args":[["q",0],["q",2]]},
            {"qop":"Measure","args":[["q",0],["q",1],["q",2],["q",3]],"returns":[["m",3],["m",2],["m",1],["m",0]]},
            {"cop":"Result","args":["m"],"returns":["out"]}
        ]}).to_string();
        let mut bridge = PhirJsonEngine::py_new(&source).unwrap();
        let mut native = RustPhirJsonEngine::from_json(&source).unwrap();
        for _ in 0..2 {
            ClassicalEngine::reset(&mut bridge).unwrap();
            ClassicalEngine::reset(&mut native).unwrap();
            assert_eq!(bridge.num_qubits(), native.num_qubits());
            assert_eq!(
                bridge.generate_commands().unwrap().quantum_ops().unwrap(),
                native.generate_commands().unwrap().quantum_ops().unwrap()
            );
            let mut builder = ByteMessage::outcomes_builder();
            builder.add_outcomes(&[1, 0, 1, 1]);
            bridge.handle_measurements(builder.build()).unwrap();
            let mut builder = ByteMessage::outcomes_builder();
            builder.add_outcomes(&[1, 0, 1, 1]);
            native.handle_measurements(builder.build()).unwrap();
            assert_eq!(
                bridge.generate_commands().unwrap().quantum_ops().unwrap(),
                native.generate_commands().unwrap().quantum_ops().unwrap()
            );
            assert_eq!(
                ClassicalEngine::get_results(&bridge).unwrap(),
                native.get_results().unwrap()
            );
            assert_eq!(bridge.get_results().unwrap().get("out"), Some(&11));
        }
        // Exercise the same broadcast, including RZZ, through independent
        // state-vector backends with identical measurement seeds.
        let mut bridge = PhirJsonEngine::py_new(&source).unwrap();
        let mut native = RustPhirJsonEngine::from_json(&source).unwrap();
        let mut bridge_quantum = pecos_engines::StateVecEngine::with_seed(4, 1314);
        let mut native_quantum = pecos_engines::StateVecEngine::with_seed(4, 1314);
        let bridge_commands = bridge.generate_commands().unwrap();
        let native_commands = native.generate_commands().unwrap();
        bridge
            .handle_measurements(bridge_quantum.process(bridge_commands).unwrap())
            .unwrap();
        native
            .handle_measurements(native_quantum.process(native_commands).unwrap())
            .unwrap();
        assert!(bridge.generate_commands().unwrap().is_empty().unwrap());
        assert!(native.generate_commands().unwrap().is_empty().unwrap());
        assert_eq!(
            ClassicalEngine::get_results(&bridge).unwrap(),
            native.get_results().unwrap()
        );
        Python::initialize();
        Python::attach(|py| {
            let mut bridge = PhirJsonEngine::py_new(&source).unwrap();
            let commands = bridge.process_program(py).unwrap();
            let measurements: Vec<_> = commands
                .iter()
                .filter(|command| {
                    command
                        .bind(py)
                        .get_item("gate_type")
                        .unwrap()
                        .extract::<String>()
                        .unwrap()
                        == "Measure"
                })
                .collect();
            assert_eq!(measurements.len(), 4);
            for (index, command) in measurements.iter().enumerate() {
                assert_eq!(
                    command
                        .bind(py)
                        .get_item("params")
                        .unwrap()
                        .get_item("result_id")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    index
                );
            }
            bridge.handle_measurement(1).unwrap();
            assert!(
                bridge
                    .process_program(py)
                    .unwrap_err()
                    .to_string()
                    .contains("1 of 4")
            );
            for outcome in [0, 1, 1] {
                bridge.handle_measurement(outcome).unwrap();
            }
            assert!(bridge.process_program(py).unwrap().is_empty());
            assert_eq!(bridge.results_dict().unwrap().get("out"), Some(&11));
        });
    }
}
