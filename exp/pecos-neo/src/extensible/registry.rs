//! Gate registry for managing gate specifications.

use super::{GateCategory, GateId, GateSpec, gates};

/// Registry of gate specifications.
///
/// NOT a global - this is scoped to a simulation context (World, Tool, etc.).
/// All hot-path lookups are O(1) via direct indexing.
pub struct GateRegistry {
    /// Core gate specs (indices 0-255, pre-populated)
    core_specs: Box<[Option<GateSpec>; 256]>,

    /// User gate specs (indices >= 256)
    user_specs: Vec<GateSpec>,

    /// Name to ID lookup - sorted for binary search
    /// Only used at parse time, never in hot path
    name_index: Vec<(&'static str, GateId)>,
}

impl Default for GateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl GateRegistry {
    /// Create a new registry with core gates pre-populated.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            core_specs: Box::new([const { None }; 256]),
            user_specs: Vec::new(),
            name_index: Vec::new(),
        };
        registry.init_core_gates();
        registry
    }

    /// Initialize all core gate specifications.
    fn init_core_gates(&mut self) {
        // Single-qubit Paulis
        self.set_core(
            gates::I,
            &GateSpec::new("I").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::X,
            &GateSpec::new("X").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::Y,
            &GateSpec::new("Y").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::Z,
            &GateSpec::new("Z").with_category(GateCategory::SingleQubitUnitary),
        );

        // Single-qubit Cliffords
        self.set_core(
            gates::H,
            &GateSpec::new("H").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::SX,
            &GateSpec::new("SX").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::SXdg,
            &GateSpec::new("SXdg").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::SY,
            &GateSpec::new("SY").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::SYdg,
            &GateSpec::new("SYdg").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::SZ,
            &GateSpec::new("SZ").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::SZdg,
            &GateSpec::new("SZdg").with_category(GateCategory::SingleQubitUnitary),
        );

        // T gates
        self.set_core(
            gates::T,
            &GateSpec::new("T").with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::Tdg,
            &GateSpec::new("Tdg").with_category(GateCategory::SingleQubitUnitary),
        );

        // Single-qubit rotations
        self.set_core(
            gates::RX,
            &GateSpec::new("RX")
                .with_angle_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::RY,
            &GateSpec::new("RY")
                .with_angle_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::RZ,
            &GateSpec::new("RZ")
                .with_angle_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::U,
            &GateSpec::new("U")
                .with_angle_arity(3)
                .with_category(GateCategory::SingleQubitUnitary),
        );
        self.set_core(
            gates::R1XY,
            &GateSpec::new("R1XY")
                .with_angle_arity(2)
                .with_category(GateCategory::SingleQubitUnitary),
        );

        // Two-qubit gates
        self.set_core(
            gates::CX,
            &GateSpec::new("CX")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::CY,
            &GateSpec::new("CY")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::CZ,
            &GateSpec::new("CZ")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::SWAP,
            &GateSpec::new("SWAP")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::ISWAP,
            &GateSpec::new("ISWAP")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        // Two-qubit Clifford rotations
        self.set_core(
            gates::SXX,
            &GateSpec::new("SXX")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::SXXdg,
            &GateSpec::new("SXXdg")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::SYY,
            &GateSpec::new("SYY")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::SYYdg,
            &GateSpec::new("SYYdg")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::SZZ,
            &GateSpec::new("SZZ")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::SZZdg,
            &GateSpec::new("SZZdg")
                .with_quantum_arity(2)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        // Two-qubit parameterized gates
        self.set_core(
            gates::CRZ,
            &GateSpec::new("CRZ")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::RXX,
            &GateSpec::new("RXX")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::RYY,
            &GateSpec::new("RYY")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );
        self.set_core(
            gates::RZZ,
            &GateSpec::new("RZZ")
                .with_quantum_arity(2)
                .with_angle_arity(1)
                .with_category(GateCategory::TwoQubitUnitary),
        );

        // Three-qubit gates
        self.set_core(
            gates::CCX,
            &GateSpec::new("CCX")
                .with_quantum_arity(3)
                .with_category(GateCategory::MultiQubitUnitary),
        );
        self.set_core(
            gates::CCZ,
            &GateSpec::new("CCZ")
                .with_quantum_arity(3)
                .with_category(GateCategory::MultiQubitUnitary),
        );
        self.set_core(
            gates::CSWAP,
            &GateSpec::new("CSWAP")
                .with_quantum_arity(3)
                .with_category(GateCategory::MultiQubitUnitary),
        );

        // Measurement
        self.set_core(
            gates::MZ,
            &GateSpec::new("MZ")
                .with_returns_result(true)
                .with_category(GateCategory::Measurement),
        );
        self.set_core(
            gates::MEASURE_LEAKED,
            &GateSpec::new("MeasureLeaked")
                .with_returns_result(true)
                .with_category(GateCategory::Measurement),
        );
        self.set_core(
            gates::MEASURE_FREE,
            &GateSpec::new("MeasureFree")
                .with_returns_result(true)
                .with_category(GateCategory::Measurement),
        );

        // State preparation
        self.set_core(
            gates::PZ,
            &GateSpec::new("PZ").with_category(GateCategory::Preparation),
        );
        self.set_core(
            gates::PX,
            &GateSpec::new("PrepX").with_category(GateCategory::Preparation),
        );
        self.set_core(
            gates::PY,
            &GateSpec::new("PrepY").with_category(GateCategory::Preparation),
        );

        // Qubit management
        self.set_core(
            gates::QALLOC,
            &GateSpec::new("QAlloc").with_category(GateCategory::QubitManagement),
        );
        self.set_core(
            gates::QFREE,
            &GateSpec::new("QFree").with_category(GateCategory::QubitManagement),
        );

        // Idle
        self.set_core(
            gates::IDLE,
            &GateSpec::new("Idle")
                .with_param_arity(1) // duration
                .with_category(GateCategory::Idle),
        );

        // Sort name index for binary search
        self.name_index.sort_by(|a, b| a.0.cmp(b.0));
    }

    /// Set a core gate specification.
    fn set_core(&mut self, id: GateId, spec: &GateSpec) {
        debug_assert!(id.is_core(), "set_core called with user-defined gate ID");
        let idx = id.0 as usize;
        self.core_specs[idx] = Some(spec.clone());

        // Add to name index
        self.name_index.push((spec.name, id));
    }

    /// Register a user-defined gate, returns its ID.
    pub fn register(&mut self, spec: GateSpec) -> GateId {
        #[allow(clippy::cast_possible_truncation)] // user gate count fits in u16
        let id = GateId(256 + self.user_specs.len() as u16);
        let name = spec.name;
        self.user_specs.push(spec);

        // Maintain sorted order for binary search
        let pos = self
            .name_index
            .binary_search_by_key(&name, |(n, _)| *n)
            .unwrap_or_else(|p| p);
        self.name_index.insert(pos, (name, id));

        id
    }

    /// Look up spec by ID - O(1) direct indexing.
    #[inline]
    #[must_use]
    pub fn get(&self, id: GateId) -> Option<&GateSpec> {
        if id.is_core() {
            self.core_specs[id.0 as usize].as_ref()
        } else {
            self.user_specs.get((id.0 - 256) as usize)
        }
    }

    /// Check if a gate ID is registered.
    #[inline]
    #[must_use]
    pub fn contains(&self, id: GateId) -> bool {
        self.get(id).is_some()
    }

    /// Look up ID by name - O(log n) binary search.
    /// Only used at parse time, never in simulation hot path.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<GateId> {
        self.name_index
            .binary_search_by_key(&name, |(n, _)| *n)
            .ok()
            .map(|i| self.name_index[i].1)
    }

    /// Get the number of registered user gates.
    #[must_use]
    pub fn user_gate_count(&self) -> usize {
        self.user_specs.len()
    }

    /// Iterate over all registered gate IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = GateId> + '_ {
        self.name_index.iter().map(|(_, id)| *id)
    }

    /// Iterate over all registered (id, spec) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (GateId, &GateSpec)> + '_ {
        self.name_index
            .iter()
            .filter_map(|(_, id)| self.get(*id).map(|spec| (*id, spec)))
    }
}
