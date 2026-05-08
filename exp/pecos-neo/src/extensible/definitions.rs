// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Unified gate definitions.
//!
//! Combines gate specifications, decompositions, and noise parameters into a single
//! structure that can be passed to simulators, noise models, and runners.
//!
//! # Design Principles
//!
//! - **Define once, use everywhere**: Gate properties defined in one place
//! - **O(1) lookups**: All hot-path access is array-indexed by `GateId`
//! - **Simple ownership**: Pass by reference, clone only if needed
//! - **Validation at build time**: Invalid configurations caught early
//! - **Uniform treatment**: Core and custom gates look the same to consumers
//!
//! # Example
//!
//! ```
//! use pecos_neo::extensible::*;
//!
//! // Build gate definitions with the builder pattern
//! let defs = GateDefinitions::builder()
//!     // Core gates are pre-loaded
//!     // Define a custom gate
//!     .define_gate("MySWAP", GateSpec::new("MySWAP")
//!         .with_quantum_arity(2)
//!         .with_category(GateCategory::TwoQubitUnitary))
//!     // Set noise for the custom gate
//!     .with_noise("MySWAP", 0.02)
//!     .build()
//!     .unwrap();
//!
//! // Look up gate by name
//! let my_swap = defs.id_by_name("MySWAP").unwrap();
//! assert_eq!(defs.quantum_arity(my_swap), Some(2));
//! assert_eq!(defs.error_probability(my_swap), 0.02);
//! ```

use super::noise_integration::GateNoiseParams;
use super::{DecompEntry, GateCategory, GateId, GateSpec, GateSupportSet, gates};

// ============================================================================
// GateDefinitions - The unified container
// ============================================================================

/// Unified gate definitions.
///
/// Contains all information about gates: specs, decompositions, and noise parameters.
/// All lookups are O(1) via array indexing by `GateId`.
///
/// This is a value type - cheap to clone due to internal structure.
#[derive(Clone, Debug)]
pub struct GateDefinitions {
    /// Gate specifications indexed by `GateId`.
    /// Index 0-255 = core gates, 256+ = user gates.
    specs: Vec<Option<GateSpec>>,

    /// Gate decompositions indexed by `GateId`.
    decompositions: Vec<Option<DecompEntry>>,

    /// Noise parameters indexed by `GateId`.
    noise: Vec<Option<GateNoiseParams>>,

    /// Category-based noise defaults.
    category_noise: [Option<GateNoiseParams>; 8],

    /// Global noise default.
    global_noise: Option<GateNoiseParams>,

    /// Name to ID mapping (sorted for binary search).
    /// Only used for name lookups, not in hot path.
    name_to_id: Vec<(&'static str, GateId)>,

    /// Next available user gate ID.
    next_user_id: u16,
}

impl Default for GateDefinitions {
    fn default() -> Self {
        Self::new()
    }
}

impl GateDefinitions {
    /// Create new definitions with core gates pre-populated.
    #[must_use]
    pub fn new() -> Self {
        let mut defs = Self {
            specs: vec![None; 256],
            decompositions: vec![None; 256],
            noise: vec![None; 256],
            category_noise: [None, None, None, None, None, None, None, None],
            global_noise: None,
            name_to_id: Vec::new(),
            next_user_id: 256,
        };
        defs.init_core_gates();
        defs
    }

    /// Create a builder for constructing definitions.
    #[must_use]
    pub fn builder() -> GateDefinitionsBuilder {
        GateDefinitionsBuilder::new()
    }

    // ========================================================================
    // Spec access (O(1))
    // ========================================================================

    /// Get the spec for a gate. O(1).
    #[inline]
    #[must_use]
    pub fn spec(&self, id: GateId) -> Option<&GateSpec> {
        self.specs.get(id.0 as usize).and_then(|s| s.as_ref())
    }

    /// Get the category for a gate. O(1).
    #[inline]
    #[must_use]
    pub fn category(&self, id: GateId) -> Option<GateCategory> {
        self.spec(id).map(|s| s.category)
    }

    /// Get quantum arity for a gate. O(1).
    #[inline]
    #[must_use]
    pub fn quantum_arity(&self, id: GateId) -> Option<u8> {
        self.spec(id).map(|s| s.quantum_arity)
    }

    /// Check if a gate is single-qubit. O(1).
    #[inline]
    #[must_use]
    pub fn is_single_qubit(&self, id: GateId) -> bool {
        self.quantum_arity(id) == Some(1)
    }

    /// Check if a gate is two-qubit. O(1).
    #[inline]
    #[must_use]
    pub fn is_two_qubit(&self, id: GateId) -> bool {
        self.quantum_arity(id) == Some(2)
    }

    // ========================================================================
    // Decomposition access (O(1))
    // ========================================================================

    /// Get decomposition entry for a gate. O(1).
    #[inline]
    #[must_use]
    pub fn decomposition(&self, id: GateId) -> Option<&DecompEntry> {
        self.decompositions
            .get(id.0 as usize)
            .and_then(|d| d.as_ref())
    }

    /// Check if a gate has a decomposition. O(1).
    #[inline]
    #[must_use]
    pub fn has_decomposition(&self, id: GateId) -> bool {
        self.decomposition(id).is_some()
    }

    /// Get what gates are required to decompose this gate. O(1).
    #[must_use]
    pub fn decomposition_requires(&self, id: GateId) -> Option<&GateSupportSet> {
        self.decomposition(id).map(|d| &d.requires)
    }

    // ========================================================================
    // Noise access (O(1))
    // ========================================================================

    /// Get noise parameters for a gate. O(1).
    ///
    /// Lookup priority:
    /// 1. Per-gate config
    /// 2. Category default (if spec available)
    /// 3. Global default
    #[must_use]
    pub fn noise_params(&self, id: GateId) -> Option<&GateNoiseParams> {
        // Per-gate first
        if let Some(Some(params)) = self.noise.get(id.0 as usize) {
            return Some(params);
        }

        // Category default
        if let Some(spec) = self.spec(id) {
            let cat_idx = category_to_index(spec.category);
            if let Some(params) = &self.category_noise[cat_idx] {
                return Some(params);
            }
        }

        // Global default
        self.global_noise.as_ref()
    }

    /// Get error probability for a gate. O(1).
    #[must_use]
    pub fn error_probability(&self, id: GateId) -> f64 {
        self.noise_params(id).map_or(0.0, |p| p.error_probability)
    }

    // ========================================================================
    // Name lookup (O(log n) - not hot path)
    // ========================================================================

    /// Look up gate ID by name. O(log n).
    #[must_use]
    pub fn id_by_name(&self, name: &str) -> Option<GateId> {
        self.name_to_id
            .binary_search_by_key(&name, |(n, _)| *n)
            .ok()
            .map(|idx| self.name_to_id[idx].1)
    }

    /// Get gate name by ID. O(1).
    #[must_use]
    pub fn name(&self, id: GateId) -> Option<&'static str> {
        self.spec(id).map(|s| s.name)
    }

    // ========================================================================
    // Registration
    // ========================================================================

    /// Register a user-defined gate. Returns its ID.
    pub fn register(&mut self, spec: GateSpec) -> GateId {
        let id = GateId(self.next_user_id);
        self.next_user_id += 1;

        // Ensure capacity
        let idx = id.0 as usize;
        if idx >= self.specs.len() {
            self.specs.resize(idx + 1, None);
            self.decompositions.resize(idx + 1, None);
            self.noise.resize(idx + 1, None);
        }

        // Store spec
        let name = spec.name;
        self.specs[idx] = Some(spec);

        // Add to name index (keep sorted)
        let insert_pos = self
            .name_to_id
            .binary_search_by_key(&name, |(n, _)| *n)
            .unwrap_or_else(|pos| pos);
        self.name_to_id.insert(insert_pos, (name, id));

        id
    }

    /// Set decomposition for a gate.
    pub fn set_decomposition(&mut self, id: GateId, entry: DecompEntry) {
        let idx = id.0 as usize;
        if idx >= self.decompositions.len() {
            self.decompositions.resize(idx + 1, None);
        }
        self.decompositions[idx] = Some(entry);
    }

    /// Set noise parameters for a gate.
    pub fn set_noise(&mut self, id: GateId, params: GateNoiseParams) {
        let idx = id.0 as usize;
        if idx >= self.noise.len() {
            self.noise.resize(idx + 1, None);
        }
        self.noise[idx] = Some(params);
    }

    /// Set noise error probability for a gate.
    pub fn set_noise_error(&mut self, id: GateId, error_probability: f64) {
        self.set_noise(id, GateNoiseParams::with_error(error_probability));
    }

    /// Set category-based noise default.
    pub fn set_category_noise(&mut self, category: GateCategory, params: GateNoiseParams) {
        self.category_noise[category_to_index(category)] = Some(params);
    }

    /// Set global noise default.
    pub fn set_global_noise(&mut self, params: GateNoiseParams) {
        self.global_noise = Some(params);
    }

    // ========================================================================
    // Core gate initialization
    // ========================================================================

    fn init_core_gates(&mut self) {
        // Single-qubit Paulis
        self.set_core_spec(
            gates::I,
            GateSpec::new("I").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::X,
            GateSpec::new("X").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::Y,
            GateSpec::new("Y").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::Z,
            GateSpec::new("Z").with_category(GateCategory::SingleQubitUnitary),
        );

        // Single-qubit Cliffords
        self.set_core_spec(
            gates::H,
            GateSpec::new("H").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::F,
            GateSpec::new("F").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::Fdg,
            GateSpec::new("Fdg").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::SX,
            GateSpec::new("SX").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::SXdg,
            GateSpec::new("SXdg").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::SY,
            GateSpec::new("SY").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::SYdg,
            GateSpec::new("SYdg").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::SZ,
            GateSpec::new("SZ").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::SZdg,
            GateSpec::new("SZdg").with_category(GateCategory::SingleQubitUnitary),
        );

        // T gates
        self.set_core_spec(
            gates::T,
            GateSpec::new("T").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::Tdg,
            GateSpec::new("Tdg").with_category(GateCategory::SingleQubitUnitary),
        );

        // Single-qubit rotations
        self.set_core_spec(
            gates::RX,
            GateSpec::new("RX")
                .with_angle_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::RY,
            GateSpec::new("RY")
                .with_angle_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::RZ,
            GateSpec::new("RZ")
                .with_angle_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::U,
            GateSpec::new("U")
                .with_angle_arity(3)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core_spec(
            gates::R1XY,
            GateSpec::new("R1XY")
                .with_angle_arity(2)
                .with_category(GateCategory::SingleQubitUnitary),
        );

        // Two-qubit gates
        self.set_core_spec(
            gates::CX,
            GateSpec::new("CX")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::CY,
            GateSpec::new("CY")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::CZ,
            GateSpec::new("CZ")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SWAP,
            GateSpec::new("SWAP")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SXX,
            GateSpec::new("SXX")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SXXdg,
            GateSpec::new("SXXdg")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SYY,
            GateSpec::new("SYY")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SYYdg,
            GateSpec::new("SYYdg")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SZZ,
            GateSpec::new("SZZ")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::SZZdg,
            GateSpec::new("SZZdg")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        // Two-qubit parameterized
        self.set_core_spec(
            gates::RXX,
            GateSpec::new("RXX")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::RYY,
            GateSpec::new("RYY")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core_spec(
            gates::RZZ,
            GateSpec::new("RZZ")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        // Three-qubit
        self.set_core_spec(
            gates::CCX,
            GateSpec::new("CCX")
                .with_quantum_arity(3)
                .with_category(GateCategory::MultiQubitUnitary),
        );

        // Measurement
        self.set_core_spec(
            gates::MZ,
            GateSpec::new("MZ")
                .with_returns_result(true)
                .with_category(GateCategory::Measurement),
        );

        // Preparation
        self.set_core_spec(
            gates::PZ,
            GateSpec::new("PZ").with_category(GateCategory::Preparation),
        );

        // Idle
        self.set_core_spec(
            gates::IDLE,
            GateSpec::new("Idle")
                .with_param_arity(1)
                .with_category(GateCategory::Idle),
        );
    }

    fn set_core_spec(&mut self, id: GateId, spec: GateSpec) {
        debug_assert!(id.is_core(), "set_core_spec called with user gate");
        let name = spec.name;
        self.specs[id.0 as usize] = Some(spec);

        // Add to name index
        let insert_pos = self
            .name_to_id
            .binary_search_by_key(&name, |(n, _)| *n)
            .unwrap_or_else(|pos| pos);
        self.name_to_id.insert(insert_pos, (name, id));
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for constructing `GateDefinitions`.
#[derive(Default)]
pub struct GateDefinitionsBuilder {
    defs: GateDefinitions,
    errors: Vec<String>,
}

impl GateDefinitionsBuilder {
    /// Create a new builder with core gates pre-loaded.
    #[must_use]
    pub fn new() -> Self {
        Self {
            defs: GateDefinitions::new(),
            errors: Vec::new(),
        }
    }

    /// Define a new user gate with its spec.
    #[must_use]
    pub fn define_gate(mut self, name: &'static str, spec: GateSpec) -> Self {
        // Validate name matches
        if spec.name != name {
            self.errors.push(format!(
                "Gate name mismatch: define_gate('{}') but spec.name='{}'",
                name, spec.name
            ));
        }
        self.defs.register(spec);
        self
    }

    /// Set decomposition for a gate by name.
    #[must_use]
    pub fn with_decomposition(mut self, gate_name: &str, entry: DecompEntry) -> Self {
        if let Some(id) = self.defs.id_by_name(gate_name) {
            self.defs.set_decomposition(id, entry);
        } else {
            self.errors
                .push(format!("Unknown gate for decomposition: {gate_name}"));
        }
        self
    }

    /// Set decomposition for a gate by ID.
    #[must_use]
    pub fn with_decomposition_id(mut self, id: GateId, entry: DecompEntry) -> Self {
        self.defs.set_decomposition(id, entry);
        self
    }

    /// Set noise for a gate by name.
    #[must_use]
    pub fn with_noise(mut self, gate_name: &str, error_probability: f64) -> Self {
        if let Some(id) = self.defs.id_by_name(gate_name) {
            self.defs.set_noise_error(id, error_probability);
        } else {
            self.errors
                .push(format!("Unknown gate for noise: {gate_name}"));
        }
        self
    }

    /// Set noise for a gate by ID.
    #[must_use]
    pub fn with_noise_id(mut self, id: GateId, error_probability: f64) -> Self {
        self.defs.set_noise_error(id, error_probability);
        self
    }

    /// Set category-based noise default.
    #[must_use]
    pub fn with_category_noise(mut self, category: GateCategory, error_probability: f64) -> Self {
        self.defs
            .set_category_noise(category, GateNoiseParams::with_error(error_probability));
        self
    }

    /// Set global noise default.
    #[must_use]
    pub fn with_global_noise(mut self, error_probability: f64) -> Self {
        self.defs
            .set_global_noise(GateNoiseParams::with_error(error_probability));
        self
    }

    /// Build the definitions, returning errors if any.
    ///
    /// # Errors
    /// Returns `GateDefinitionsError` if any gate definitions have validation errors.
    pub fn build(self) -> Result<GateDefinitions, GateDefinitionsError> {
        if self.errors.is_empty() {
            Ok(self.defs)
        } else {
            Err(GateDefinitionsError {
                messages: self.errors,
            })
        }
    }

    /// Build the definitions, panicking on errors.
    ///
    /// # Panics
    /// Panics if any gate definitions have validation errors.
    #[must_use]
    pub fn build_or_panic(self) -> GateDefinitions {
        self.build().expect("GateDefinitions build failed")
    }
}

/// Error building gate definitions.
#[derive(Debug, Clone)]
pub struct GateDefinitionsError {
    pub messages: Vec<String>,
}

impl std::fmt::Display for GateDefinitionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for msg in &self.messages {
            writeln!(f, "- {msg}")?;
        }
        Ok(())
    }
}

impl std::error::Error for GateDefinitionsError {}

// ============================================================================
// Helpers
// ============================================================================

/// Convert category to array index.
fn category_to_index(category: GateCategory) -> usize {
    match category {
        GateCategory::SingleQubitUnitary => 0,
        GateCategory::TwoQubitUnitary => 1,
        GateCategory::MultiQubitUnitary => 2,
        GateCategory::Preparation => 3,
        GateCategory::Measurement => 4,
        GateCategory::Idle => 5,
        GateCategory::QubitManagement => 6,
        GateCategory::Custom(_) => 7,
    }
}

// ============================================================================
// Simulator gate execution trait
// ============================================================================

use pecos_core::{Angle64, QubitId};

/// Trait for simulators that can execute gates natively.
///
/// Simulators implement this to provide optimized gate implementations.
/// If a simulator doesn't support a gate natively, the runner falls back
/// to decomposition using `GateDefinitions`.
pub trait GateExecutor {
    /// Check if this executor supports a gate natively.
    fn supports(&self, gate_id: GateId) -> bool;

    /// Try to execute a gate natively.
    ///
    /// Returns `true` if the gate was executed, `false` if not supported.
    /// The caller should fall back to decomposition if `false` is returned.
    fn try_execute(&mut self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> bool;
}

/// Marker for simulators that don't provide native gate optimization.
///
/// All gates are executed via decomposition.
pub struct NoNativeGates;

impl GateExecutor for NoNativeGates {
    fn supports(&self, _gate_id: GateId) -> bool {
        false
    }

    fn try_execute(&mut self, _gate_id: GateId, _qubits: &[QubitId], _angles: &[Angle64]) -> bool {
        false
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_core_gates() {
        let defs = GateDefinitions::new();

        // Core gates should be present
        assert!(defs.spec(gates::H).is_some());
        assert!(defs.spec(gates::CX).is_some());
        assert!(defs.spec(gates::MZ).is_some());

        // Check properties
        assert_eq!(
            defs.category(gates::H),
            Some(GateCategory::SingleQubitUnitary)
        );
        assert_eq!(defs.quantum_arity(gates::CX), Some(2));
        assert!(defs.is_single_qubit(gates::H));
        assert!(defs.is_two_qubit(gates::CX));
    }

    #[test]
    fn test_name_lookup() {
        let defs = GateDefinitions::new();

        assert_eq!(defs.id_by_name("H"), Some(gates::H));
        assert_eq!(defs.id_by_name("CX"), Some(gates::CX));
        assert_eq!(defs.id_by_name("NonExistent"), None);

        assert_eq!(defs.name(gates::H), Some("H"));
    }

    #[test]
    fn test_register_custom_gate() {
        let mut defs = GateDefinitions::new();

        let my_gate = defs.register(
            GateSpec::new("MyGate")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        assert!(my_gate.is_user_defined());
        assert_eq!(defs.name(my_gate), Some("MyGate"));
        assert_eq!(defs.quantum_arity(my_gate), Some(2));
        assert_eq!(defs.id_by_name("MyGate"), Some(my_gate));
    }

    #[test]
    fn test_noise_lookup_priority() {
        let mut defs = GateDefinitions::new();

        // Set up noise at all three levels
        defs.set_global_noise(GateNoiseParams::with_error(0.1));
        defs.set_category_noise(
            GateCategory::SingleQubitUnitary,
            GateNoiseParams::with_error(0.01),
        );
        defs.set_noise_error(gates::H, 0.001);

        // H has per-gate config
        assert_eq!(defs.error_probability(gates::H), 0.001);

        // X uses category default
        assert_eq!(defs.error_probability(gates::X), 0.01);

        // CX uses global default (different category)
        assert_eq!(defs.error_probability(gates::CX), 0.1);
    }

    #[test]
    fn test_builder() {
        let defs = GateDefinitions::builder()
            .define_gate(
                "CustomGate",
                GateSpec::new("CustomGate")
                    .with_quantum_arity(1)
                    .with_category(GateCategory::SingleQubitUnitary),
            )
            .with_noise("CustomGate", 0.005)
            .with_category_noise(GateCategory::TwoQubitUnitary, 0.02)
            .build()
            .unwrap();

        let id = defs.id_by_name("CustomGate").unwrap();
        assert_eq!(defs.error_probability(id), 0.005);
        assert_eq!(defs.error_probability(gates::CX), 0.02);
    }

    #[test]
    fn test_builder_error() {
        let result = GateDefinitions::builder()
            .with_noise("NonExistentGate", 0.01)
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.messages[0].contains("Unknown gate"));
    }
}
