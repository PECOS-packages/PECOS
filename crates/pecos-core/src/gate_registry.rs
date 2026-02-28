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
    #[must_use]
    pub fn decompose(
        &self,
        name: &str,
        qubits: &[QubitId],
        input_angles: &[Angle64],
    ) -> Option<Vec<ConcreteStep>> {
        let def = self.gates.get(name)?;
        if def.decomposition.is_empty() {
            return None;
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
        Some(result)
    }
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

    #[must_use]
    pub fn step(mut self, gate_type: GateType, qubit_indices: &[u8]) -> Self {
        self.decomposition.push(DecompStep {
            gate_type,
            qubit_indices: qubit_indices.to_vec(),
            angles: Vec::new(),
            metadata: HashMap::new(),
        });
        self
    }

    #[must_use]
    pub fn step_with_angles(
        mut self,
        gate_type: GateType,
        qubit_indices: &[u8],
        angles: &[AngleSource],
    ) -> Self {
        self.decomposition.push(DecompStep {
            gate_type,
            qubit_indices: qubit_indices.to_vec(),
            angles: angles.to_vec(),
            metadata: HashMap::new(),
        });
        self
    }

    #[must_use]
    pub fn step_with_metadata(
        mut self,
        gate_type: GateType,
        qubit_indices: &[u8],
        angles: &[AngleSource],
        metadata: HashMap<String, Value>,
    ) -> Self {
        self.decomposition.push(DecompStep {
            gate_type,
            qubit_indices: qubit_indices.to_vec(),
            angles: angles.to_vec(),
            metadata,
        });
        self
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
            .step(GateType::CX, &[1, 0])
            .step(GateType::CX, &[0, 1])
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
            .step(GateType::CX, &[1, 0])
            .step(GateType::CX, &[0, 1])
            .build();
        registry.register(def);

        let qubits = [QubitId::from(5usize), QubitId::from(10usize)];
        let result = registry.decompose("MY_SWAP", &qubits, &[]).unwrap();
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
        let fixed_angle = Angle64::from_turns(0.25);
        let def = GateDefinitionBuilder::new("CRZ_LIKE", 2)
            .angle_arity(1)
            .step_with_angles(GateType::RZ, &[1], &[AngleSource::Input(0)])
            .step(GateType::CX, &[0, 1])
            .step_with_angles(GateType::RZ, &[1], &[AngleSource::NegInput(0)])
            .step(GateType::CX, &[0, 1])
            .step_with_angles(GateType::RZ, &[0], &[AngleSource::Fixed(fixed_angle)])
            .build();
        registry.register(def);

        let qubits = [QubitId::from(0usize), QubitId::from(1usize)];
        let input_angle = Angle64::from_turns(0.125);
        let result = registry
            .decompose("CRZ_LIKE", &qubits, &[input_angle])
            .unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result[0].2, vec![input_angle]);
        assert!(result[1].2.is_empty());
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
            .build();
        registry.register(def);

        let result = registry
            .decompose("ANNOTATED", &[QubitId::from(0usize)], &[])
            .unwrap();
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
        assert!(result.is_none());
    }

    #[test]
    fn test_unregistered_gate_returns_none() {
        let registry = GateRegistry::new();
        let result = registry.decompose("NONEXISTENT", &[QubitId::from(0usize)], &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_signatures() {
        let mut registry = GateRegistry::new();
        let def1 = GateDefinitionBuilder::new("MY_SWAP", 2)
            .step(GateType::CX, &[0, 1])
            .build();
        let def2 = GateDefinitionBuilder::new("MY_RZ", 1)
            .angle_arity(1)
            .step_with_angles(GateType::RZ, &[0], &[AngleSource::Input(0)])
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
            .build();
        registry.register(def);

        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("SWAP"));
        assert!(registry.get("SWAP").is_some());
        assert_eq!(registry.get("SWAP").unwrap().quantum_arity, 2);
    }
}
