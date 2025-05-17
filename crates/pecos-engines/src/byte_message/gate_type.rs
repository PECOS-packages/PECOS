use std::fmt;

/// FFI-friendly representation of quantum gate types
///
/// This enum is designed to be FFI-friendly with a C-compatible memory layout.
/// It represents the same gate types as the core `GateType` enum but with a more
/// predictable memory layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateType {
    X = 1,
    Y = 2,
    Z = 3,
    H = 4,
    CX = 5,
    SZZ = 6,
    RZ = 7,
    R1XY = 8,
    Measure = 9,
    Prep = 10,
    RZZ = 11,
    SZZdg = 12,
    Idle = 13,
    U = 14,
}

impl From<u8> for GateType {
    fn from(value: u8) -> Self {
        match value {
            1 => GateType::X,
            2 => GateType::Y,
            3 => GateType::Z,
            4 => GateType::H,
            5 => GateType::CX,
            6 => GateType::SZZ,
            7 => GateType::RZ,
            8 => GateType::R1XY,
            9 => GateType::Measure,
            10 => GateType::Prep,
            11 => GateType::RZZ,
            12 => GateType::SZZdg,
            13 => GateType::Idle,
            14 => GateType::U,
            _ => panic!("Invalid gate type ID: {value}"),
        }
    }
}

impl From<GateType> for u8 {
    fn from(gate_type: GateType) -> Self {
        gate_type as u8
    }
}

impl fmt::Display for GateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateType::X => write!(f, "X"),
            GateType::Y => write!(f, "Y"),
            GateType::Z => write!(f, "Z"),
            GateType::H => write!(f, "H"),
            GateType::CX => write!(f, "CX"),
            GateType::SZZ => write!(f, "SZZ"),
            GateType::RZ => write!(f, "RZ"),
            GateType::R1XY => write!(f, "R1XY"),
            GateType::Measure => write!(f, "Measure"),
            GateType::Prep => write!(f, "Prep"),
            GateType::RZZ => write!(f, "RZZ"),
            GateType::SZZdg => write!(f, "SZZdg"),
            GateType::Idle => write!(f, "Idle"),
            GateType::U => write!(f, "U"),
        }
    }
}

/// Represents a quantum gate with its type, parameters, and target qubits
///
/// This struct is designed to replace `QuantumCommand` with a more FFI-friendly
/// representation. It contains all the information needed to represent a quantum
/// gate operation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumGate {
    /// The type of the gate
    pub gate_type: GateType,
    /// The qubits the gate acts on
    pub qubits: Vec<usize>,
    /// Optional parameters for parameterized gates
    pub params: Vec<f64>,
    /// Optional result ID for measurement gates
    pub result_id: Option<usize>,
    /// Whether the gate should have noise applied to it
    pub noiseless: bool,
    // TODO: encode noiseless in the byte representation...
}

impl QuantumGate {
    /// Create a new quantum gate
    #[must_use]
    pub fn new(
        gate_type: GateType,
        qubits: Vec<usize>,
        params: Vec<f64>,
        result_id: Option<usize>,
    ) -> Self {
        Self {
            gate_type,
            qubits,
            params,
            result_id,
            noiseless: false,
        }
    }

    /// Create a new X gate
    #[must_use]
    pub fn x(qubit: usize) -> Self {
        Self::new(GateType::X, vec![qubit], vec![], None)
    }

    /// Create a new Y gate
    #[must_use]
    pub fn y(qubit: usize) -> Self {
        Self::new(GateType::Y, vec![qubit], vec![], None)
    }

    /// Create a new Z gate
    #[must_use]
    pub fn z(qubit: usize) -> Self {
        Self::new(GateType::Z, vec![qubit], vec![], None)
    }

    /// Create a new H gate
    #[must_use]
    pub fn h(qubit: usize) -> Self {
        Self::new(GateType::H, vec![qubit], vec![], None)
    }

    /// Create a new CX gate
    #[must_use]
    pub fn cx(control: usize, target: usize) -> Self {
        Self::new(GateType::CX, vec![control, target], vec![], None)
    }

    /// Create a new SZZ gate
    #[must_use]
    pub fn szz(qubit1: usize, qubit2: usize) -> Self {
        Self::new(GateType::SZZ, vec![qubit1, qubit2], vec![], None)
    }

    /// Create a new `SZZdg` gate
    #[must_use]
    pub fn szzdg(qubit1: usize, qubit2: usize) -> Self {
        Self::new(GateType::SZZdg, vec![qubit1, qubit2], vec![], None)
    }

    /// Create a new RZZ gate
    #[must_use]
    pub fn rzz(theta: f64, qubit1: usize, qubit2: usize) -> Self {
        Self::new(GateType::RZZ, vec![qubit1, qubit2], vec![theta], None)
    }

    /// Create a new RZ gate
    #[must_use]
    pub fn rz(theta: f64, qubit: usize) -> Self {
        Self::new(GateType::RZ, vec![qubit], vec![theta], None)
    }

    /// Create a new R1XY gate
    #[must_use]
    pub fn r1xy(theta: f64, phi: f64, qubit: usize) -> Self {
        Self::new(GateType::R1XY, vec![qubit], vec![theta, phi], None)
    }

    /// Create a new U gate
    #[must_use]
    pub fn u(theta: f64, phi: f64, lambda: f64, qubit: usize) -> Self {
        Self::new(GateType::U, vec![qubit], vec![theta, phi, lambda], None)
    }

    /// Create a new Measure gate
    #[must_use]
    pub fn measure(qubit: usize, result_id: usize) -> Self {
        Self::new(GateType::Measure, vec![qubit], vec![], Some(result_id))
    }

    #[must_use]
    pub fn prep(qubit: usize) -> Self {
        Self::new(GateType::Prep, vec![qubit], vec![], None)
    }

    /// Create a new Idle gate for qubits idling for a specific duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the idle period in seconds
    /// * `qubits` - The qubits that are idling
    ///
    /// # Returns
    ///
    /// A new Idle gate with the specified parameters
    #[must_use]
    pub fn idle(duration: f64, qubits: Vec<usize>) -> Self {
        Self::new(GateType::Idle, qubits, vec![duration], None)
    }

    /// Returns the duration of an idle gate, or 0.0 if not an idle gate
    #[must_use]
    pub fn idle_duration(&self) -> f64 {
        if self.gate_type == GateType::Idle && !self.params.is_empty() {
            self.params[0]
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn set_noiseless(mut self) -> Self {
        self.noiseless = true;
        self
    }

    #[must_use]
    pub fn set_noisy(mut self) -> Self {
        self.noiseless = false;
        self
    }

    #[must_use]
    pub fn is_noiseless(&self) -> bool {
        self.noiseless
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_type_id_conversion() {
        assert_eq!(GateType::X as u8, 1);
        assert_eq!(GateType::Y as u8, 2);
        assert_eq!(GateType::Z as u8, 3);
        assert_eq!(GateType::H as u8, 4);
        assert_eq!(GateType::CX as u8, 5);
        assert_eq!(GateType::SZZ as u8, 6);
        assert_eq!(GateType::RZ as u8, 7);
        assert_eq!(GateType::R1XY as u8, 8);
        assert_eq!(GateType::Measure as u8, 9);

        assert_eq!(GateType::from(1u8), GateType::X);
        assert_eq!(GateType::from(2u8), GateType::Y);
        assert_eq!(GateType::from(3u8), GateType::Z);
        assert_eq!(GateType::from(4u8), GateType::H);
        assert_eq!(GateType::from(5u8), GateType::CX);
        assert_eq!(GateType::from(6u8), GateType::SZZ);
        assert_eq!(GateType::from(7u8), GateType::RZ);
        assert_eq!(GateType::from(8u8), GateType::R1XY);
        assert_eq!(GateType::from(9u8), GateType::Measure);
    }

    #[test]
    fn test_quantum_gate_creation() {
        let x_gate = QuantumGate::x(0);
        assert_eq!(x_gate.gate_type, GateType::X);
        assert_eq!(x_gate.qubits, vec![0]);
        assert!(x_gate.params.is_empty());
        assert_eq!(x_gate.result_id, None);

        let rz_gate = QuantumGate::rz(0.5, 1);
        assert_eq!(rz_gate.gate_type, GateType::RZ);
        assert_eq!(rz_gate.qubits, vec![1]);
        assert_eq!(rz_gate.params, vec![0.5]);
        assert_eq!(rz_gate.result_id, None);

        let measure_gate = QuantumGate::measure(2, 42);
        assert_eq!(measure_gate.gate_type, GateType::Measure);
        assert_eq!(measure_gate.qubits, vec![2]);
        assert!(measure_gate.params.is_empty());
        assert_eq!(measure_gate.result_id, Some(42));
    }
}
