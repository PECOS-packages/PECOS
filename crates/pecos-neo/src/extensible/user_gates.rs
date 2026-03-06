//! User-defined gates with runtime registration.
//!
//! This module allows users to define custom gates at runtime with:
//! - Dynamic gate ID allocation
//! - Decomposition definition using the builder pattern
//! - Integration with the plugin system

use super::{
    DecompOp, DecompositionRegistry, GateCategory, GateId, GatePlugin, GateSpec, GateSupportSet,
    gates,
};
use std::any::TypeId;

/// Builder for defining a custom gate.
///
/// # Example
///
/// ```
/// use pecos_neo::extensible::*;
///
/// let builder = UserGateBuilder::new("MY_GATE")
///     .qubits(2)
///     .category(GateCategory::TwoQubitUnitary)
///     .requires([gates::H, gates::CX])
///     .decomposition(vec![
///         DecompOp::gate1(gates::H, 0),
///         DecompOp::gate2(gates::CX, 0, 1),
///         DecompOp::gate1(gates::H, 0),
///     ]);
/// ```
pub struct UserGateBuilder {
    name: String,
    spec: GateSpec,
    requires: GateSupportSet,
    decomposition: Option<Vec<DecompOp>>,
}

impl UserGateBuilder {
    /// Create a new user gate builder with the given name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            spec: GateSpec::new("").with_category(GateCategory::SingleQubitUnitary),
            requires: GateSupportSet::new(),
            decomposition: None,
        }
    }

    /// Set the number of qubits this gate operates on.
    #[must_use]
    pub fn qubits(mut self, count: u8) -> Self {
        self.spec = self.spec.with_quantum_arity(count);
        // Update category based on qubit count
        self.spec = self.spec.with_category(match count {
            1 => GateCategory::SingleQubitUnitary,
            2 => GateCategory::TwoQubitUnitary,
            _ => GateCategory::MultiQubitUnitary,
        });
        self
    }

    /// Set the number of angle parameters.
    #[must_use]
    pub fn angles(mut self, count: u8) -> Self {
        self.spec = self.spec.with_angle_arity(count);
        self
    }

    /// Set the gate category.
    #[must_use]
    pub fn category(mut self, category: GateCategory) -> Self {
        self.spec = self.spec.with_category(category);
        self
    }

    /// Set the gates required for decomposition.
    #[must_use]
    pub fn requires<I: IntoIterator<Item = GateId>>(mut self, gates: I) -> Self {
        self.requires = GateSupportSet::from_iter(gates);
        self
    }

    /// Set the decomposition operations.
    #[must_use]
    pub fn decomposition(mut self, ops: Vec<DecompOp>) -> Self {
        self.decomposition = Some(ops);
        self
    }

    /// Build and return the gate definition.
    #[must_use]
    pub fn build(self) -> UserGateDefinition {
        UserGateDefinition {
            name: self.name,
            spec: self.spec,
            requires: self.requires,
            decomposition: self.decomposition,
        }
    }
}

/// A user-defined gate definition.
#[derive(Clone)]
pub struct UserGateDefinition {
    /// Gate name.
    pub name: String,
    /// Gate specification.
    pub spec: GateSpec,
    /// Required gates for decomposition.
    pub requires: GateSupportSet,
    /// Decomposition operations (if any).
    pub decomposition: Option<Vec<DecompOp>>,
}

impl std::fmt::Debug for UserGateDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserGateDefinition")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field(
                "decomposition",
                &self
                    .decomposition
                    .as_ref()
                    .map(|d| format!("{} ops", d.len())),
            )
            .finish_non_exhaustive()
    }
}

/// Plugin for registering user-defined gates.
///
/// # Example
///
/// ```
/// use pecos_neo::extensible::*;
///
/// let plugin = UserGatesPlugin::new()
///     .define(
///         UserGateBuilder::new("HCX")
///             .qubits(2)
///             .requires([gates::H, gates::CX])
///             .decomposition(vec![
///                 DecompOp::gate1(gates::H, 0),
///                 DecompOp::gate2(gates::CX, 0, 1),
///             ])
///             .build()
///     );
///
/// let registry = PluginLoader::new()
///     .with_plugin(CoreGatesPlugin)
///     .with_plugin(plugin)
///     .build()
///     .unwrap();
/// ```
#[derive(Default)]
pub struct UserGatesPlugin {
    definitions: Vec<UserGateDefinition>,
    next_id: u16,
}

impl UserGatesPlugin {
    /// Create a new empty user gates plugin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            next_id: gates::USER_GATE_START,
        }
    }

    /// Add a user-defined gate.
    #[must_use]
    pub fn define(mut self, definition: UserGateDefinition) -> Self {
        self.definitions.push(definition);
        self
    }

    /// Add multiple user-defined gates.
    #[must_use]
    pub fn define_all<I: IntoIterator<Item = UserGateDefinition>>(
        mut self,
        definitions: I,
    ) -> Self {
        self.definitions.extend(definitions);
        self
    }

    /// Get the number of gates defined.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.definitions.len()
    }
}

impl GatePlugin for UserGatesPlugin {
    fn name(&self) -> &'static str {
        "user-gates"
    }

    fn dependencies(&self) -> Vec<TypeId> {
        vec![TypeId::of::<super::CoreGatesPlugin>()]
    }

    fn build(&self, registry: &mut DecompositionRegistry) {
        let mut current_id = self.next_id;

        for def in &self.definitions {
            let gate_id = GateId(current_id);
            current_id += 1;

            if let Some(ops) = &def.decomposition {
                registry.register_dynamic(gate_id, def.requires.clone(), ops.clone());
            } else {
                // Native user gate (no decomposition)
                registry.register_native(gate_id);
            }
        }
    }
}

/// Registry for user-defined gates with ID tracking.
///
/// This provides a more complete solution for managing user gates,
/// including looking up gates by name and allocating unique IDs.
#[derive(Clone)]
pub struct UserGateRegistry {
    /// Definitions by name.
    by_name: std::collections::HashMap<String, (GateId, UserGateDefinition)>,
    /// Next available ID.
    next_id: u16,
}

impl Default for UserGateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UserGateRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            by_name: std::collections::HashMap::new(),
            next_id: gates::USER_GATE_START,
        }
    }

    /// Register a user gate and return its ID.
    pub fn register(&mut self, definition: UserGateDefinition) -> GateId {
        let gate_id = GateId(self.next_id);
        self.next_id += 1;
        self.by_name
            .insert(definition.name.clone(), (gate_id, definition));
        gate_id
    }

    /// Get a gate ID by name.
    #[must_use]
    pub fn get_id(&self, name: &str) -> Option<GateId> {
        self.by_name.get(name).map(|(id, _)| *id)
    }

    /// Get a gate definition by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&UserGateDefinition> {
        self.by_name.get(name).map(|(_, def)| def)
    }

    /// Get a gate definition by ID.
    #[must_use]
    pub fn get_by_id(&self, id: GateId) -> Option<&UserGateDefinition> {
        self.by_name
            .values()
            .find(|(gate_id, _)| *gate_id == id)
            .map(|(_, def)| def)
    }

    /// Check if a gate is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Get the number of registered gates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Apply all registered gates to a decomposition registry.
    pub fn apply_to(&self, registry: &mut DecompositionRegistry) {
        for (gate_id, def) in self.by_name.values() {
            if let Some(ops) = &def.decomposition {
                registry.register_dynamic(*gate_id, def.requires.clone(), ops.clone());
            } else {
                registry.register_native(*gate_id);
            }
        }
    }

    /// Create a plugin from this registry.
    #[must_use]
    pub fn to_plugin(&self) -> UserGatesPlugin {
        let mut plugin = UserGatesPlugin::new();
        for (_, def) in self.by_name.values() {
            plugin = plugin.define(def.clone());
        }
        plugin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_gate_builder_basic() {
        let def = UserGateBuilder::new("MY_GATE").qubits(2).build();

        assert_eq!(def.name, "MY_GATE");
        assert_eq!(def.spec.quantum_arity, 2);
        assert_eq!(def.spec.category, GateCategory::TwoQubitUnitary);
    }

    #[test]
    fn test_user_gate_builder_with_decomposition() {
        let def = UserGateBuilder::new("HCX")
            .qubits(2)
            .requires([gates::H, gates::CX])
            .decomposition(vec![
                DecompOp::gate1(gates::H, 0),
                DecompOp::gate2(gates::CX, 0, 1),
            ])
            .build();

        assert_eq!(def.name, "HCX");
        assert!(def.requires.contains(gates::H));
        assert!(def.requires.contains(gates::CX));
        assert!(def.decomposition.is_some());
        assert_eq!(def.decomposition.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_user_gate_builder_angles() {
        let def = UserGateBuilder::new("MY_ROT")
            .qubits(1)
            .angles(1)
            .category(GateCategory::SingleQubitUnitary)
            .build();

        assert_eq!(def.spec.angle_arity, 1);
    }

    #[test]
    fn test_user_gates_plugin() {
        let plugin = UserGatesPlugin::new()
            .define(
                UserGateBuilder::new("GATE1")
                    .qubits(1)
                    .requires([gates::H])
                    .decomposition(vec![DecompOp::gate1(gates::H, 0)])
                    .build(),
            )
            .define(
                UserGateBuilder::new("GATE2")
                    .qubits(2)
                    .requires([gates::CX])
                    .decomposition(vec![DecompOp::gate2(gates::CX, 0, 1)])
                    .build(),
            );

        assert_eq!(plugin.gate_count(), 2);
        assert_eq!(plugin.name(), "user-gates");
    }

    #[test]
    fn test_user_gate_registry() {
        let mut registry = UserGateRegistry::new();

        let id1 = registry.register(UserGateBuilder::new("GATE1").qubits(1).build());

        let id2 = registry.register(UserGateBuilder::new("GATE2").qubits(2).build());

        assert!(id1.is_user_defined());
        assert!(id2.is_user_defined());
        assert_ne!(id1, id2);

        assert_eq!(registry.get_id("GATE1"), Some(id1));
        assert_eq!(registry.get_id("GATE2"), Some(id2));
        assert_eq!(registry.get_id("GATE3"), None);

        assert!(registry.contains("GATE1"));
        assert!(!registry.contains("GATE3"));

        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_user_gate_registry_apply() {
        let mut user_registry = UserGateRegistry::new();

        user_registry.register(
            UserGateBuilder::new("MY_SWAP")
                .qubits(2)
                .requires([gates::CX])
                .decomposition(vec![
                    DecompOp::gate2(gates::CX, 0, 1),
                    DecompOp::gate2(gates::CX, 1, 0),
                    DecompOp::gate2(gates::CX, 0, 1),
                ])
                .build(),
        );

        let mut decomp_registry = DecompositionRegistry::new();
        user_registry.apply_to(&mut decomp_registry);

        let gate_id = user_registry.get_id("MY_SWAP").unwrap();
        assert!(decomp_registry.contains(gate_id));

        let sim_support = GateSupportSet::from_iter([gates::CX]);
        assert!(decomp_registry.can_execute(gate_id, &sim_support));
    }

    #[test]
    fn test_user_gate_native() {
        let mut registry = UserGateRegistry::new();

        // Native user gate (no decomposition, simulator must support directly)
        let id = registry.register(UserGateBuilder::new("NATIVE_GATE").qubits(1).build());

        let mut decomp_registry = DecompositionRegistry::new();
        registry.apply_to(&mut decomp_registry);

        assert!(decomp_registry.contains(id));
        assert!(decomp_registry.is_native(id));
    }

    #[test]
    fn test_user_gate_registry_get_by_id() {
        let mut registry = UserGateRegistry::new();

        let id = registry.register(UserGateBuilder::new("TEST_GATE").qubits(1).build());

        let def = registry.get_by_id(id).unwrap();
        assert_eq!(def.name, "TEST_GATE");
    }

    #[test]
    fn test_user_gate_registry_to_plugin() {
        let mut registry = UserGateRegistry::new();

        registry.register(UserGateBuilder::new("GATE1").qubits(1).build());

        registry.register(UserGateBuilder::new("GATE2").qubits(2).build());

        let plugin = registry.to_plugin();
        assert_eq!(plugin.gate_count(), 2);
    }
}
