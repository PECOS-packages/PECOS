//! Gate specification - describes the properties of a gate type.

/// Semantic category of a gate for noise model matching.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum GateCategory {
    /// Single-qubit unitary (X, Y, Z, H, RZ, etc.)
    #[default]
    SingleQubitUnitary,
    /// Two-qubit unitary (CX, CZ, RZZ, etc.)
    TwoQubitUnitary,
    /// Multi-qubit unitary (CCX, etc.)
    MultiQubitUnitary,
    /// State preparation
    Preparation,
    /// Measurement
    Measurement,
    /// Idle/wait operation
    Idle,
    /// Qubit allocation/deallocation
    QubitManagement,
    /// User-defined category
    Custom(u8),
}

/// Gate specification - describes what a gate IS.
///
/// This contains the metadata about a gate type, not an instance of a gate.
#[derive(Clone, Debug, PartialEq)]
pub struct GateSpec {
    /// Human-readable name (e.g., `"H"`, `"CX"`, `"MyRotation"`)
    pub name: &'static str,

    /// Number of qubits this gate operates on
    pub quantum_arity: u8,

    /// Number of angle parameters (Angle64)
    pub angle_arity: u8,

    /// Number of other parameters (f64, e.g., duration for Idle)
    pub param_arity: u8,

    /// Whether this gate produces measurement outcomes
    pub returns_result: bool,

    /// Semantic category for noise model matching
    pub category: GateCategory,
}

impl Default for GateSpec {
    fn default() -> Self {
        GateSpec {
            name: "",
            quantum_arity: 1,
            angle_arity: 0,
            param_arity: 0,
            returns_result: false,
            category: GateCategory::SingleQubitUnitary,
        }
    }
}

impl GateSpec {
    /// Create a new gate spec with the given name.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        GateSpec {
            name,
            ..Default::default()
        }
    }

    /// Set the quantum arity (number of qubits).
    #[must_use]
    pub fn with_quantum_arity(mut self, arity: u8) -> Self {
        self.quantum_arity = arity;
        self
    }

    /// Set the angle arity (number of `Angle64` parameters).
    #[must_use]
    pub fn with_angle_arity(mut self, arity: u8) -> Self {
        self.angle_arity = arity;
        self
    }

    /// Set the param arity (number of f64 parameters).
    #[must_use]
    pub fn with_param_arity(mut self, arity: u8) -> Self {
        self.param_arity = arity;
        self
    }

    /// Set whether this gate returns measurement results.
    #[must_use]
    pub fn with_returns_result(mut self, returns: bool) -> Self {
        self.returns_result = returns;
        self
    }

    /// Set the gate category.
    #[must_use]
    pub fn with_category(mut self, category: GateCategory) -> Self {
        self.category = category;
        self
    }

    /// Check if this is a single-qubit gate.
    #[must_use]
    pub fn is_single_qubit(&self) -> bool {
        self.quantum_arity == 1
    }

    /// Check if this is a two-qubit gate.
    #[must_use]
    pub fn is_two_qubit(&self) -> bool {
        self.quantum_arity == 2
    }

    /// Check if this gate is parameterized (has angles).
    #[must_use]
    pub fn is_parameterized(&self) -> bool {
        self.angle_arity > 0
    }
}
