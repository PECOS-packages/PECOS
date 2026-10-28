// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Gate registration system for ahead-of-time custom gate definitions.
//!
//! This module provides a registry where users can define custom gates with
//! decompositions into base gates. Registered gates are decomposed at simulation
//! time, not at circuit construction time.

use crate::errors::PecosError;
use crate::gate_type::GateType;
use crate::value::Value;
use crate::{Angle64, QubitId};
use std::collections::HashMap;

/// A concrete decomposition step: (`gate_type`, qubits, angles, metadata).
pub type ConcreteStep = (GateType, Vec<QubitId>, Vec<Angle64>, HashMap<String, Value>);

/// The signature of a gate: its quantum and angle arities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateSignature {
    pub quantum_arity: usize,
    pub angle_arity: usize,
}

/// Where a decomposition step gets its angle value.
#[derive(Debug, Clone, PartialEq)]
pub enum AngleSource {
    /// Use the i-th angle from the parent gate's input angles.
    Input(u8),
    /// A fixed angle value.
    Fixed(Angle64),
    /// Negate the i-th input angle.
    NegInput(u8),
}

/// A single gate in a decomposition sequence.
/// Qubit indices are positional -- index 0 is the first qubit of the custom gate, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct DecompStep {
    pub gate_type: GateType,
    pub qubit_indices: Vec<u8>,
    pub angles: Vec<AngleSource>,
    pub metadata: HashMap<String, Value>,
}

/// Definition of a registered custom gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GateDefinition {
    pub name: String,
    pub quantum_arity: usize,
    pub angle_arity: usize,
    pub decomposition: Vec<DecompStep>,
}

/// Registry mapping gate names to definitions with decompositions.
#[derive(Debug, Clone, Default)]
pub struct GateRegistry {
    gates: HashMap<String, GateDefinition>,
}

impl GateRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, def: GateDefinition) {
        self.gates.insert(def.name.clone(), def);
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&GateDefinition> {
        self.gates.get(name)
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.gates.contains_key(name)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Extract signatures from all registered gates.
    #[must_use]
    pub fn signatures(&self) -> HashMap<String, GateSignature> {
        self.gates
            .iter()
            .map(|(name, def)| {
                (
                    name.clone(),
                    GateSignature {
                        quantum_arity: def.quantum_arity,
                        angle_arity: def.angle_arity,
                    },
                )
            })
            .collect()
    }

    /// Expand a custom gate into concrete (`GateType`, qubits, angles, metadata) tuples.
    /// Returns None if not registered or decomposition is empty.
    ///
    /// # Errors
    /// Returns an input error if operand counts or decomposition references are invalid.
    pub fn decompose(
        &self,
        name: &str,
        qubits: &[QubitId],
        input_angles: &[Angle64],
    ) -> Result<Option<Vec<ConcreteStep>>, PecosError> {
        let Some(def) = self.gates.get(name) else {
            return Ok(None);
        };
        if qubits.len() != def.quantum_arity || input_angles.len() != def.angle_arity {
            return Err(PecosError::Input(format!(
                "Gate {name} expects {} qubits and {} angles; got {} qubits and {} angles",
                def.quantum_arity,
                def.angle_arity,
                qubits.len(),
                input_angles.len()
            )));
        }
        // Definitions can be registered directly, without going through the builder.
        for step in &def.decomposition {
            validate_step_qubits(name, def.quantum_arity, &step.qubit_indices)?;
            validate_step_arity(
                name,
                step.gate_type,
                step.qubit_indices.len(),
                step.angles.len(),
            )?;
            for source in &step.angles {
                if let AngleSource::Input(index) | AngleSource::NegInput(index) = source
                    && usize::from(*index) >= def.angle_arity
                {
                    return Err(PecosError::Input(format!(
                        "Gate {name} step references angle index {index}; expected an index below {} (got {} input angles)",
                        def.angle_arity,
                        input_angles.len()
                    )));
                }
            }
        }
        if def.decomposition.is_empty() {
            return Ok(None);
        }
        let mut result = Vec::with_capacity(def.decomposition.len());
        for step in &def.decomposition {
            let concrete_qubits: Vec<QubitId> = step
                .qubit_indices
                .iter()
                .map(|&idx| qubits[idx as usize])
                .collect();
            let concrete_angles: Vec<Angle64> = step
                .angles
                .iter()
                .map(|src| match src {
                    AngleSource::Input(i) => input_angles[*i as usize],
                    AngleSource::Fixed(a) => *a,
                    AngleSource::NegInput(i) => -input_angles[*i as usize],
                })
                .collect();
            result.push((
                step.gate_type,
                concrete_qubits,
                concrete_angles,
                step.metadata.clone(),
            ));
        }
        Ok(Some(result))
    }
}

fn validate_step_arity(
    name: &str,
    gate: GateType,
    qubit_count: usize,
    angle_count: usize,
) -> Result<(), PecosError> {
    // These command/payload types report a compatibility arity of 1, not a
    // required operand count. Their actual width belongs to the concrete step.
    let variable_width = matches!(
        gate,
        GateType::Custom
            | GateType::Channel
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::TrackedPauliMeta
    );
    if !variable_width && qubit_count != gate.quantum_arity() {
        return Err(PecosError::Input(format!(
            "Gate {name} step {gate} expects {} qubits; got {qubit_count}",
            gate.quantum_arity()
        )));
    }
    // Custom gates have a registry-defined angle arity, as in `Gate::new`.
    if gate != GateType::Custom && angle_count != gate.angle_arity() {
        return Err(PecosError::Input(format!(
            "Gate {name} step {gate} expects {} angle sources; got {angle_count}",
            gate.angle_arity()
        )));
    }
    Ok(())
}

fn validate_step_qubits(name: &str, arity: usize, indices: &[u8]) -> Result<(), PecosError> {
    for &index in indices {
        if usize::from(index) >= arity {
            return Err(PecosError::Input(format!(
                "Gate {name} step references qubit index {index}; expected an index below declared arity {arity}"
            )));
        }
    }
    Ok(())
}

/// Builder for constructing gate definitions with a fluent API.
pub struct GateDefinitionBuilder {
    name: String,
    quantum_arity: usize,
    angle_arity: usize,
    decomposition: Vec<DecompStep>,
}

impl GateDefinitionBuilder {
    #[must_use]
    pub fn new(name: impl Into<String>, quantum_arity: usize) -> Self {
        Self {
            name: name.into(),
            quantum_arity,
            angle_arity: 0,
            decomposition: Vec::new(),
        }
    }

    #[must_use]
    pub fn angle_arity(mut self, arity: usize) -> Self {
        self.angle_arity = arity;
        self
    }

    /// Add a non-parameterized step.
    ///
    /// # Errors
    /// Rejects out-of-range qubit positions and incorrect step operand/angle counts.
    pub fn step(mut self, gate_type: GateType, qubit_indices: &[u8]) -> Result<Self, PecosError> {
        self.add_step(gate_type, qubit_indices, &[], HashMap::new())?;
        Ok(self)
    }

    /// Add a parameterized step.
    ///
    /// # Errors
    /// Rejects out-of-range qubit positions and incorrect step operand/angle counts.
    pub fn step_with_angles(
        mut self,
        gate_type: GateType,
        qubit_indices: &[u8],
        angles: &[AngleSource],
    ) -> Result<Self, PecosError> {
        self.add_step(gate_type, qubit_indices, angles, HashMap::new())?;
        Ok(self)
    }

    /// Add a step with metadata.
    ///
    /// # Errors
    /// Rejects out-of-range qubit positions and incorrect step operand/angle counts.
    pub fn step_with_metadata(
        mut self,
        gate_type: GateType,
        qubit_indices: &[u8],
        angles: &[AngleSource],
        metadata: HashMap<String, Value>,
    ) -> Result<Self, PecosError> {
        self.add_step(gate_type, qubit_indices, angles, metadata)?;
        Ok(self)
    }

    /// Append a step, preserving the builder on failure.
    ///
    /// # Errors
    /// Rejects out-of-range qubit positions and incorrect step operand/angle counts before appending.
    pub fn add_step(
        &mut self,
        gate_type: GateType,
        qubit_indices: &[u8],
        angles: &[AngleSource],
        metadata: HashMap<String, Value>,
    ) -> Result<(), PecosError> {
        validate_step_qubits(&self.name, self.quantum_arity, qubit_indices)?;
        validate_step_arity(&self.name, gate_type, qubit_indices.len(), angles.len())?;
        self.decomposition.push(DecompStep {
            gate_type,
            qubit_indices: qubit_indices.to_vec(),
            angles: angles.to_vec(),
            metadata,
        });
        Ok(())
    }

    #[must_use]
    pub fn build(self) -> GateDefinition {
        GateDefinition {
            name: self.name,
            quantum_arity: self.quantum_arity,
            angle_arity: self.angle_arity,
            decomposition: self.decomposition,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_round_trip() {
        let def = GateDefinitionBuilder::new("MY_SWAP", 2)
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .step(GateType::CX, &[1, 0])
            .expect("valid step")
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .build();

        assert_eq!(def.name, "MY_SWAP");
        assert_eq!(def.quantum_arity, 2);
        assert_eq!(def.angle_arity, 0);
        assert_eq!(def.decomposition.len(), 3);
        assert_eq!(def.decomposition[0].gate_type, GateType::CX);
        assert_eq!(def.decomposition[0].qubit_indices, vec![0, 1]);
        assert_eq!(def.decomposition[1].qubit_indices, vec![1, 0]);
    }

    #[test]
    fn test_decompose_positional_qubits() {
        let mut registry = GateRegistry::new();
        let def = GateDefinitionBuilder::new("MY_SWAP", 2)
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .step(GateType::CX, &[1, 0])
            .expect("valid step")
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .build();
        registry.register(def);

        let qubits = [QubitId::from(5usize), QubitId::from(10usize)];
        let result = registry
            .decompose("MY_SWAP", &qubits, &[])
            .expect("valid decomposition")
            .expect("registered decomposition");
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].1,
            vec![QubitId::from(5usize), QubitId::from(10usize)]
        );
        assert_eq!(
            result[1].1,
            vec![QubitId::from(10usize), QubitId::from(5usize)]
        );
        assert_eq!(
            result[2].1,
            vec![QubitId::from(5usize), QubitId::from(10usize)]
        );
        assert!(result[0].3.is_empty());
    }

    #[test]
    fn test_angle_source_resolution() {
        let mut registry = GateRegistry::new();
        let fixed_angle = Angle64::QUARTER_TURN;
        let def = GateDefinitionBuilder::new("CRZ_LIKE", 2)
            .angle_arity(1)
            .step_with_angles(GateType::RZ, &[1], &[AngleSource::Input(0)])
            .expect("valid step")
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .step_with_angles(GateType::RZ, &[1], &[AngleSource::NegInput(0)])
            .expect("valid step")
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .step_with_angles(GateType::RZ, &[0], &[AngleSource::Fixed(fixed_angle)])
            .expect("valid step")
            .build();
        registry.register(def);

        let qubits = [QubitId::from(0usize), QubitId::from(1usize)];
        let input_angle = Angle64::from_turn_ratio(1, 8);
        let result = registry
            .decompose("CRZ_LIKE", &qubits, &[input_angle])
            .expect("valid decomposition")
            .expect("registered decomposition");

        assert_eq!(result.len(), 5);
        assert_eq!(result[0].2, vec![input_angle]);
        assert_eq!(result[1].2, []);
        assert_eq!(result[2].2, vec![-input_angle]);
        assert_eq!(result[4].2, vec![fixed_angle]);
    }

    #[test]
    fn test_step_with_metadata() {
        let mut registry = GateRegistry::new();
        let mut meta = HashMap::new();
        meta.insert("duration".to_string(), Value::Float(100.0));
        meta.insert("label".to_string(), Value::String("fast".to_string()));
        meta.insert("count".to_string(), Value::Int(3));
        meta.insert("noisy".to_string(), Value::Bool(true));

        let def = GateDefinitionBuilder::new("ANNOTATED", 1)
            .step_with_metadata(GateType::H, &[0], &[], meta)
            .expect("valid step")
            .build();
        registry.register(def);

        let result = registry
            .decompose("ANNOTATED", &[QubitId::from(0usize)], &[])
            .expect("valid decomposition")
            .expect("registered decomposition");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].3.get("duration"), Some(&Value::Float(100.0)));
        assert_eq!(
            result[0].3.get("label"),
            Some(&Value::String("fast".to_string()))
        );
        assert_eq!(result[0].3.get("count"), Some(&Value::Int(3)));
        assert_eq!(result[0].3.get("noisy"), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_empty_decomposition_returns_none() {
        let mut registry = GateRegistry::new();
        let def = GateDefinitionBuilder::new("EMPTY", 1).build();
        registry.register(def);

        let result = registry.decompose("EMPTY", &[QubitId::from(0usize)], &[]);
        assert!(result.expect("valid input").is_none());
    }

    #[test]
    fn test_unregistered_gate_returns_none() {
        let registry = GateRegistry::new();
        let result = registry.decompose("NONEXISTENT", &[QubitId::from(0usize)], &[]);
        assert!(result.expect("valid input").is_none());
    }

    #[test]
    fn test_signatures() {
        let mut registry = GateRegistry::new();
        let def1 = GateDefinitionBuilder::new("MY_SWAP", 2)
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .build();
        let def2 = GateDefinitionBuilder::new("MY_RZ", 1)
            .angle_arity(1)
            .step_with_angles(GateType::RZ, &[0], &[AngleSource::Input(0)])
            .expect("valid step")
            .build();
        registry.register(def1);
        registry.register(def2);

        let sigs = registry.signatures();
        assert_eq!(sigs.len(), 2);
        assert_eq!(
            sigs.get("MY_SWAP"),
            Some(&GateSignature {
                quantum_arity: 2,
                angle_arity: 0
            })
        );
        assert_eq!(
            sigs.get("MY_RZ"),
            Some(&GateSignature {
                quantum_arity: 1,
                angle_arity: 1
            })
        );
    }

    #[test]
    fn test_registry_operations() {
        let mut registry = GateRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("SWAP"));

        let def = GateDefinitionBuilder::new("SWAP", 2)
            .step(GateType::CX, &[0, 1])
            .expect("valid step")
            .build();
        registry.register(def);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("SWAP"));
        assert!(registry.get("SWAP").is_some());
        assert_eq!(registry.get("SWAP").unwrap().quantum_arity, 2);
    }
    #[test]
    fn decompose_rejects_wrong_input_counts() {
        let mut registry = GateRegistry::new();
        registry.register(
            GateDefinitionBuilder::new("ROT", 1)
                .angle_arity(1)
                .step_with_angles(GateType::RZ, &[0], &[AngleSource::Input(0)])
                .expect("valid step")
                .build(),
        );
        for (qubits, angles) in [
            (vec![QubitId(0)], vec![]),
            (vec![], vec![Angle64::ZERO]),
            (vec![QubitId(0), QubitId(1)], vec![Angle64::ZERO]),
            (vec![QubitId(0)], vec![Angle64::ZERO, Angle64::ZERO]),
        ] {
            let error = registry
                .decompose("ROT", &qubits, &angles)
                .expect_err("invalid input")
                .to_string();
            assert!(error.contains("Gate ROT expects 1 qubits and 1 angles"));
            assert!(error.contains(&format!(
                "got {} qubits and {} angles",
                qubits.len(),
                angles.len()
            )));
        }
    }

    #[test]
    fn decompose_rejects_invalid_definition_references() {
        for (indices, angles, expected) in [
            (vec![1], vec![], "qubit index 1"),
            (vec![0], vec![AngleSource::Input(1)], "angle index 1"),
            (vec![0], vec![AngleSource::NegInput(1)], "angle index 1"),
        ] {
            let mut registry = GateRegistry::new();
            registry.register(GateDefinition {
                name: "BAD".into(),
                quantum_arity: 1,
                angle_arity: 1,
                decomposition: vec![DecompStep {
                    gate_type: GateType::RZ,
                    qubit_indices: indices,
                    angles,
                    metadata: HashMap::new(),
                }],
            });
            let error = registry
                .decompose("BAD", &[QubitId(0)], &[Angle64::ZERO])
                .expect_err("invalid definition")
                .to_string();
            assert!(error.contains("Gate BAD"));
            assert!(error.contains(expected));
            assert!(error.contains("below"));
        }
    }

    #[test]
    fn builder_rejects_invalid_step_positions() {
        let results = [
            GateDefinitionBuilder::new("BAD", 1).step(GateType::X, &[1]),
            GateDefinitionBuilder::new("BAD", 1).step_with_angles(GateType::RX, &[1], &[]),
            GateDefinitionBuilder::new("BAD", 1).step_with_metadata(
                GateType::X,
                &[1],
                &[],
                HashMap::new(),
            ),
        ];
        for result in results {
            let error = result
                .err()
                .expect("invalid position must fail")
                .to_string();
            assert!(error.contains("Gate BAD step references qubit index 1"));
            assert!(error.contains("declared arity 1"));
        }
        let mut builder = GateDefinitionBuilder::new("RECOVER", 1);
        assert!(
            builder
                .add_step(GateType::X, &[1], &[], HashMap::new())
                .is_err()
        );
        builder
            .add_step(GateType::X, &[0], &[], HashMap::new())
            .expect("valid position");
        assert_eq!(builder.build().decomposition.len(), 1);
    }
    #[test]
    fn builder_rejects_step_arity_mismatches() {
        for (gate, indices, angles, expected) in [
            (
                GateType::RZ,
                vec![0],
                vec![],
                "step RZ expects 1 angle sources; got 0",
            ),
            (
                GateType::CX,
                vec![0],
                vec![],
                "step CX expects 2 qubits; got 1",
            ),
            (
                GateType::H,
                vec![0, 1],
                vec![],
                "step H expects 1 qubits; got 2",
            ),
            (
                GateType::RZ,
                vec![0],
                vec![AngleSource::Fixed(Angle64::ZERO); 2],
                "step RZ expects 1 angle sources; got 2",
            ),
        ] {
            // All fluent entry points delegate to the same append validation.
            for method in 0..3 {
                let builder = GateDefinitionBuilder::new("HR", 2)
                    .step(GateType::H, &[0])
                    .expect("valid prefix");
                let result = match method {
                    0 if angles.is_empty() => builder.step(gate, &indices),
                    0 | 1 => builder.step_with_angles(gate, &indices, &angles),
                    _ => builder.step_with_metadata(gate, &indices, &angles, HashMap::new()),
                };
                let error = result.err().expect("malformed step must fail").to_string();
                assert!(error.contains("Gate HR"));
                assert!(error.contains(expected), "{error}");
            }
        }
    }

    #[test]
    fn direct_registration_rejects_step_arity_mismatches() {
        for (gate, expected) in [
            (GateType::RZ, "step RZ expects 1 angle sources; got 0"),
            (GateType::CX, "step CX expects 2 qubits; got 1"),
        ] {
            let mut definition = GateDefinitionBuilder::new("HR", 1)
                .step(GateType::H, &[0])
                .expect("valid prefix")
                .build();
            definition.decomposition.push(DecompStep {
                gate_type: gate,
                qubit_indices: vec![0],
                angles: vec![],
                metadata: HashMap::new(),
            });
            let mut registry = GateRegistry::new();
            registry.register(definition);
            let error = registry
                .decompose("HR", &[QubitId(0)], &[])
                .expect_err("direct definitions must also validate steps")
                .to_string();
            assert!(error.contains("Gate HR"));
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn variable_width_steps_preserve_concrete_width() {
        for gate in [
            GateType::Custom,
            GateType::Channel,
            GateType::MeasCrosstalkGlobalPayload,
            GateType::MeasCrosstalkLocalPayload,
            GateType::TrackedPauliMeta,
        ] {
            for indices in [vec![], vec![0, 1]] {
                let mut builder = GateDefinitionBuilder::new("VARIABLE", 2);
                builder
                    .add_step(gate, &indices, &[], HashMap::new())
                    .expect("variable width");
                let with_angle = builder.add_step(
                    gate,
                    &indices,
                    &[AngleSource::Fixed(Angle64::ZERO)],
                    HashMap::new(),
                );
                if gate == GateType::Custom {
                    // Custom steps carry a registry-defined number of angles.
                    with_angle.expect("custom steps accept angle sources");
                } else {
                    let error = with_angle
                        .expect_err("payload steps take no angle sources")
                        .to_string();
                    assert!(error.contains("expects 0 angle sources; got 1"));
                }
                let mut registry = GateRegistry::new();
                registry.register(builder.build());
                let steps = registry
                    .decompose("VARIABLE", &[QubitId(0), QubitId(1)], &[])
                    .expect("valid input")
                    .expect("registered decomposition");
                let expected_steps = if gate == GateType::Custom { 2 } else { 1 };
                assert_eq!(steps.len(), expected_steps);
                assert!(steps.iter().all(|step| step.1.len() == indices.len()));
            }
        }
    }
}
