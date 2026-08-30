//! Gate type enumeration for quantum operations
//!
//! This module provides the `GateType` enum which represents the different
//! types of quantum gates supported by the byte protocol.

use std::fmt;

/// FFI-friendly representation of quantum gate types
///
/// This enum is designed to be FFI-friendly with a C-compatible memory layout.
/// It represents the same gate types as the core `GateType` enum but with a more
/// predictable memory layout.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GateType {
    I = 0b00,
    X = 0b01,
    Z = 0b10,
    Y = 0b11,
    /// sqrt(X) gate
    SX = 4,
    /// sqrt(X)-dagger gate
    SXdg = 5,
    /// sqrt(Y) gate
    SY = 6,
    /// sqrt(Y)-dagger gate
    SYdg = 7,
    SZ = 8,
    SZdg = 9,
    /// Hadamard gate, occupying the H1 slot in the H1-H6 family.
    H = 10,
    // H2 = 11
    // H3 = 12
    // H4 = 13
    // H5 = 14
    // H6 = 15
    /// F gate (face gate), occupying the F1 slot in the F1-F4 family.
    F = 16,
    /// F-dagger gate
    Fdg = 17,
    // F2 = 18
    // F2dg = 19
    // F3 = 20
    // F3dg = 21
    // F4 = 22
    // F4dg = 23
    RX = 30,
    RY = 31,
    RZ = 32,
    T = 33,
    Tdg = 34,
    // Other T-like gates?
    U = 35,
    RXY1Q = 36,

    CX = 50,
    CY = 51,
    CZ = 52,
    /// sqrt(XX) gate
    SXX = 53,
    /// sqrt(XX)-dagger gate
    SXXdg = 54,
    /// sqrt(YY) gate
    SYY = 55,
    /// sqrt(YY)-dagger gate
    SYYdg = 56,
    SZZ = 57,
    SZZdg = 58,
    SWAP = 59,
    // iSWAP = 60
    // G = 61
    /// Controlled-RZ gate (2 qubits, 1 angle parameter)
    CRZ = 70,
    /// Controlled-H gate (2 qubits)
    CH = 71,
    /// RXX rotation gate
    RXX = 80,
    /// RYY rotation gate
    RYY = 81,
    RZZ = 82,
    /// General 2-qubit Pauli rotation: exp(-i/2 * (a*XX + b*YY + c*ZZ))
    RXXRYYRZZ = 83,
    /// General 2-qubit unitary via KAK decomposition
    U2q = 84,
    /// Toffoli gate (CCX, 3 qubits)
    CCX = 90,

    /// Measure in the X basis.
    MX = 100,
    // MnX = 101
    // MY = 102
    // MnY = 103
    // MZ = 104
    MZ = 104,
    // MnZ = 105
    MeasureLeaked = 105,
    /// Measure and free the qubit (destructive measurement)
    MeasureFree = 106,
    /// Measure +Z, then prepare |0> (measure-and-prepare; the MP* family)
    MPZ = 107,
    // TODO: MPauli instead of the other variants?
    /// Prepare the +1 eigenstate of X.
    PX = 130,
    // PNX = 131
    // PY = 132
    // PNY = 133
    // PZ = 134
    PZ = 134,
    // PNZ
    /// Allocate a qubit in the |0⟩ state
    QAlloc = 135,
    /// Free/deallocate a qubit
    QFree = 136,
    Idle = 200,
    /// Meta-gate: tracked-Pauli annotation for fault tracking.
    ///
    /// This gate carries a Pauli string but has no effect on quantum state.
    /// Its position in the circuit determines which faults can flip the tracked Pauli
    /// (only faults before this node are relevant). The propagator uses it as a
    /// backward propagation start point.
    ///
    /// The Pauli string is encoded in `params`: each param encodes
    /// `qubit * 4 + pauli_type` where `pauli_type` is 1=X, 2=Y, 3=Z.
    TrackedPauliMeta = 210,
    MeasCrosstalkGlobalPayload = 218,
    MeasCrosstalkLocalPayload = 219,
    /// Typed channel operation embedded in an annotated/noisy circuit.
    ///
    /// The concrete channel payload is stored on [`crate::Gate`], not in the
    /// numeric gate type. Ideal circuits should not contain this gate type; it
    /// represents compiled noise annotations or explicit channel placement.
    Channel = 220,
    /// Custom/unrecognized gate type, with actual name stored in metadata
    Custom = 255,
}

/// Row-major 2x2 complex matrix stored as `(real, imaginary)` pairs.
///
/// The entries are `[a.re, a.im, b.re, b.im, c.re, c.im, d.re, d.im]` for
/// `[[a, b], [c, d]]`.
pub type SingleQubitGateMatrix = [f64; 8];

/// Converts a canonical f64 matrix to f32 in a constant context.
///
/// Canonical gate entries are finite zero or normal-range values. Conversion
/// uses IEEE-754 round-to-nearest, ties-to-even; an out-of-contract future
/// entry fails constant evaluation instead of silently changing semantics.
#[must_use]
pub const fn single_qubit_matrix_to_f32(matrix: SingleQubitGateMatrix) -> [f32; 8] {
    let mut converted = [0.0; 8];
    let mut index = 0;
    while index < matrix.len() {
        converted[index] = finite_normal_f64_to_f32(matrix[index]);
        index += 1;
    }
    converted
}

const fn finite_normal_f64_to_f32(value: f64) -> f32 {
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { 0 } else { 1_u32 << 31 };
    let exponent_bits = ((bits >> 52) & 0x7ff).to_le_bytes();
    let exponent = i32::from_le_bytes([exponent_bits[0], exponent_bits[1], 0, 0]);
    let fraction = bits & ((1_u64 << 52) - 1);

    if exponent == 0 {
        assert!(fraction == 0, "canonical matrix contains a subnormal f64");
        return f32::from_bits(sign);
    }
    assert!(
        exponent != 0x7ff,
        "canonical matrix contains a non-finite f64"
    );

    let mut target_exponent = exponent - 1023 + 127;
    assert!(
        target_exponent > 0 && target_exponent < 0xff,
        "canonical matrix entry is outside normal f32 range"
    );

    let significand = (1_u64 << 52) | fraction;
    let discarded_mask = (1_u64 << 29) - 1;
    let discarded = significand & discarded_mask;
    let halfway = 1_u64 << 28;
    let mut rounded = significand >> 29;
    if discarded > halfway || (discarded == halfway && rounded & 1 == 1) {
        rounded += 1;
    }
    if rounded == 1_u64 << 24 {
        rounded >>= 1;
        target_exponent += 1;
        assert!(target_exponent < 0xff, "canonical matrix overflows f32");
    }

    let rounded_bytes = rounded.to_le_bytes();
    let target_fraction =
        u32::from_le_bytes([rounded_bytes[0], rounded_bytes[1], rounded_bytes[2], 0])
            & ((1_u32 << 23) - 1);
    let target_exponent = u32::from_le_bytes(target_exponent.to_le_bytes());
    f32::from_bits(sign | (target_exponent << 23) | target_fraction)
}

/// Named, phase-fixed single-qubit gates with canonical matrices.
///
/// Non-dagger gates precede dagger gates so consumers that identify matrices
/// projectively can retain the non-dagger preference for self-inverse gates.
pub const NAMED_SINGLE_QUBIT_GATES: [GateType; 15] = [
    GateType::I,
    GateType::X,
    GateType::Y,
    GateType::Z,
    GateType::H,
    GateType::F,
    GateType::Fdg,
    GateType::SX,
    GateType::SXdg,
    GateType::SY,
    GateType::SYdg,
    GateType::SZ,
    GateType::SZdg,
    GateType::T,
    GateType::Tdg,
];

const SX_MATRIX: SingleQubitGateMatrix = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5];

// Conventional phase-fixed sqrt(Y), not RY(pi/2):
// (1/2) [[1+i, -1-i], [1+i, 1+i]].
const SY_MATRIX: SingleQubitGateMatrix = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5];

// Exact adjoint of the conventional phase-fixed sqrt(Y):
// (1/2) [[1-i, 1-i], [-1+i, 1-i]].
const SY_DAGGER_MATRIX: SingleQubitGateMatrix = [0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5];

const SZ_MATRIX: SingleQubitGateMatrix = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

const H_MATRIX: SingleQubitGateMatrix = [
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    -std::f64::consts::FRAC_1_SQRT_2,
    0.0,
];

const T_MATRIX: SingleQubitGateMatrix = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];

const X_MATRIX: SingleQubitGateMatrix = multiply_2x2(SX_MATRIX, SX_MATRIX);
const Y_MATRIX: SingleQubitGateMatrix = multiply_2x2(SY_MATRIX, SY_MATRIX);
const Z_MATRIX: SingleQubitGateMatrix = multiply_2x2(SZ_MATRIX, SZ_MATRIX);
const I_MATRIX: SingleQubitGateMatrix = multiply_2x2(Z_MATRIX, Z_MATRIX);
// F is F1. The factor i selects the order-three representative:
// F = i * SZ * SX.
const F_MATRIX: SingleQubitGateMatrix = scale_2x2(multiply_2x2(SZ_MATRIX, SX_MATRIX), 0.0, 1.0);

const fn multiply_complex(lhs_re: f64, lhs_im: f64, rhs_re: f64, rhs_im: f64) -> (f64, f64) {
    (
        lhs_re * rhs_re - lhs_im * rhs_im,
        lhs_re * rhs_im + lhs_im * rhs_re,
    )
}

const fn add_complex(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 + rhs.0, lhs.1 + rhs.1)
}

pub(crate) const fn multiply_2x2(
    lhs: SingleQubitGateMatrix,
    rhs: SingleQubitGateMatrix,
) -> SingleQubitGateMatrix {
    let a = add_complex(
        multiply_complex(lhs[0], lhs[1], rhs[0], rhs[1]),
        multiply_complex(lhs[2], lhs[3], rhs[4], rhs[5]),
    );
    let b = add_complex(
        multiply_complex(lhs[0], lhs[1], rhs[2], rhs[3]),
        multiply_complex(lhs[2], lhs[3], rhs[6], rhs[7]),
    );
    let c = add_complex(
        multiply_complex(lhs[4], lhs[5], rhs[0], rhs[1]),
        multiply_complex(lhs[6], lhs[7], rhs[4], rhs[5]),
    );
    let d = add_complex(
        multiply_complex(lhs[4], lhs[5], rhs[2], rhs[3]),
        multiply_complex(lhs[6], lhs[7], rhs[6], rhs[7]),
    );
    [a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1]
}

pub(crate) const fn scale_2x2(
    matrix: SingleQubitGateMatrix,
    phase_re: f64,
    phase_im: f64,
) -> SingleQubitGateMatrix {
    let a = multiply_complex(matrix[0], matrix[1], phase_re, phase_im);
    let b = multiply_complex(matrix[2], matrix[3], phase_re, phase_im);
    let c = multiply_complex(matrix[4], matrix[5], phase_re, phase_im);
    let d = multiply_complex(matrix[6], matrix[7], phase_re, phase_im);
    [a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1]
}

const fn conjugate_imaginary(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { -value }
}

const fn adjoint_2x2(matrix: SingleQubitGateMatrix) -> SingleQubitGateMatrix {
    [
        matrix[0],
        conjugate_imaginary(matrix[1]),
        matrix[4],
        conjugate_imaginary(matrix[5]),
        matrix[2],
        conjugate_imaginary(matrix[3]),
        matrix[6],
        conjugate_imaginary(matrix[7]),
    ]
}

impl From<u8> for GateType {
    fn from(value: u8) -> Self {
        match value {
            0 => GateType::I,
            1 => GateType::X,
            2 => GateType::Z,
            3 => GateType::Y,
            4 => GateType::SX,
            5 => GateType::SXdg,
            6 => GateType::SY,
            7 => GateType::SYdg,
            8 => GateType::SZ,
            9 => GateType::SZdg,
            10 => GateType::H,
            16 => GateType::F,
            17 => GateType::Fdg,
            30 => GateType::RX,
            31 => GateType::RY,
            32 => GateType::RZ,
            33 => GateType::T,
            34 => GateType::Tdg,
            35 => GateType::U,
            36 => GateType::RXY1Q,
            50 => GateType::CX,
            51 => GateType::CY,
            52 => GateType::CZ,
            53 => GateType::SXX,
            54 => GateType::SXXdg,
            55 => GateType::SYY,
            56 => GateType::SYYdg,
            57 => GateType::SZZ,
            58 => GateType::SZZdg,
            59 => GateType::SWAP,
            70 => GateType::CRZ,
            71 => GateType::CH,
            80 => GateType::RXX,
            81 => GateType::RYY,
            82 => GateType::RZZ,
            83 => GateType::RXXRYYRZZ,
            84 => GateType::U2q,
            90 => GateType::CCX,
            100 => GateType::MX,
            104 => GateType::MZ,
            105 => GateType::MeasureLeaked,
            106 => GateType::MeasureFree,
            107 => GateType::MPZ,
            130 => GateType::PX,
            134 => GateType::PZ,
            135 => GateType::QAlloc,
            136 => GateType::QFree,
            200 => GateType::Idle,
            218 => GateType::MeasCrosstalkGlobalPayload,
            219 => GateType::MeasCrosstalkLocalPayload,
            210 => GateType::TrackedPauliMeta,
            220 => GateType::Channel,
            255 => GateType::Custom,
            _ => panic!("Invalid gate type ID: {value}"),
        }
    }
}

impl GateType {
    /// Returns the canonical phase-fixed matrix for a named single-qubit gate.
    ///
    /// Matrices are row-major `(real, imaginary)` pairs in the order described
    /// by [`SingleQubitGateMatrix`]. Parameterized rotations deliberately have
    /// no entry: `RP(theta) = exp(-i theta P / 2)` remains a separate family.
    ///
    /// The roots use conventional phases, so `SX^2 = X`, `SY^2 = Y`,
    /// `SZ^2 = Z`, `T^2 = SZ`, and every dagger is the exact adjoint of its
    /// partner. `H` is H1 and satisfies `H^2 = I`; `F` is F1 and uses
    /// `F = i * SZ * SX`, so `F^3 = I`. In particular:
    ///
    /// - `SX = exp(i*pi/4) RX(pi/2)`
    /// - `SY = exp(i*pi/4) RY(pi/2)`
    /// - `SZ = exp(i*pi/4) RZ(pi/2)`
    /// - `T = exp(i*pi/8) RZ(pi/4)`
    ///
    /// The dagger gates have the conjugate phase multiplying the corresponding
    /// negative-angle rotation.
    #[must_use]
    pub const fn canonical_1q_matrix(self) -> Option<SingleQubitGateMatrix> {
        match self {
            GateType::I => Some(I_MATRIX),
            GateType::X => Some(X_MATRIX),
            GateType::Y => Some(Y_MATRIX),
            GateType::Z => Some(Z_MATRIX),
            GateType::SX => Some(SX_MATRIX),
            GateType::SXdg => Some(adjoint_2x2(SX_MATRIX)),
            GateType::SY => Some(SY_MATRIX),
            GateType::SYdg => Some(SY_DAGGER_MATRIX),
            GateType::SZ => Some(SZ_MATRIX),
            GateType::SZdg => Some(adjoint_2x2(SZ_MATRIX)),
            GateType::H => Some(H_MATRIX),
            GateType::F => Some(F_MATRIX),
            GateType::Fdg => Some(adjoint_2x2(F_MATRIX)),
            GateType::T => Some(T_MATRIX),
            GateType::Tdg => Some(adjoint_2x2(T_MATRIX)),
            _ => None,
        }
    }

    /// Returns true if this gate type is a meta-gate (annotation, not physical).
    ///
    /// Meta-gates have a position in the DAG but do not affect quantum state
    /// and should not create fault locations or receive noise.
    ///
    /// Idle-duration accounting depends on this predicate:
    /// `TickCircuit::fill_idle_gates` treats ticks whose batches are all
    /// meta as zero physical duration. Any new meta gate type MUST be added
    /// here, or it will manufacture phantom idle periods in
    /// idle-duration-driven noise models.
    #[must_use]
    pub const fn is_meta(self) -> bool {
        matches!(self, GateType::TrackedPauliMeta)
    }

    /// Returns true if this gate consumes a measurement record.
    ///
    /// A measurement record is a slot in the classical result stream, named by a
    /// [`MeasId`](crate::MeasId). Every site that allocates, counts, or resolves
    /// records must agree on which gates get one, or two different measurements
    /// end up sharing a record. Use this predicate rather than spelling the gate
    /// types out, so there is one place to change.
    ///
    /// `MeasureLeaked` is a measurement to `Gate::validate` and collapses the
    /// qubit, but it produces no classical result and so consumes no record.
    /// Deciding otherwise means changing this function and nothing else.
    #[must_use]
    pub const fn consumes_measurement_record(self) -> bool {
        matches!(
            self,
            GateType::MX | GateType::MZ | GateType::MeasureFree | GateType::MPZ
        )
    }

    /// Returns the number of angle parameters this gate type requires
    ///
    /// # Returns
    ///
    /// The number of floating-point angle parameters needed for this gate type
    #[must_use]
    pub const fn classical_arity(self) -> usize {
        match self {
            // Gates with no parameters
            GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::H
            | GateType::F
            | GateType::Fdg
            | GateType::T
            | GateType::Tdg
            | GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::CH
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
            | GateType::CCX
            | GateType::MX
            | GateType::MZ
            | GateType::MeasureLeaked
            | GateType::MeasureFree
            | GateType::MPZ
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::Channel
            | GateType::PX
            | GateType::PZ
            | GateType::QAlloc
            | GateType::QFree
            | GateType::Custom
            | GateType::TrackedPauliMeta => 0,

            // Gates with one parameter
            GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::CRZ
            | GateType::Idle => 1,

            // Gates with two parameters
            GateType::RXY1Q => 2,

            // Gates with three parameters
            GateType::U | GateType::RXXRYYRZZ => 3,

            // Gates with fifteen parameters (KAK decomposition)
            GateType::U2q => 15,
        }
    }

    /// Returns the number of qubits this gate type operates on
    ///
    /// # Returns
    ///
    /// The number of qubits this gate type requires. Variable-arity
    /// payload/meta gates return 1 for compatibility with validation code; the
    /// concrete gate stores the actual qubit count.
    #[must_use]
    pub const fn quantum_arity(self) -> usize {
        match self {
            // Single-qubit gates
            GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::H
            | GateType::F
            | GateType::Fdg
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::T
            | GateType::Tdg
            | GateType::RXY1Q
            | GateType::U
            | GateType::MX
            | GateType::MZ
            | GateType::MeasureLeaked
            | GateType::MeasureFree
            | GateType::MPZ
            | GateType::PX
            | GateType::PZ
            | GateType::QAlloc
            | GateType::QFree
            | GateType::Idle
            | GateType::Custom
            // Payload/meta gates are variable-arity but return 1 here because
            // validation checks `is_multiple_of(quantum_arity())`, and any
            // count is a multiple of 1. The actual qubit count is in the gate.
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::Channel
            | GateType::TrackedPauliMeta => 1,

            // Two-qubit gates
            GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::CH
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
            | GateType::CRZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::RXXRYYRZZ
            | GateType::U2q => 2,

            // Three-qubit gates
            GateType::CCX => 3,
        }
    }

    /// Returns the number of gates represented by a command with `qubit_count`
    /// qubits.
    ///
    /// Most gate commands are batchable: a command with 4 qubits and arity 2
    /// represents two gates. Payload/meta gates are annotations, not physical
    /// gates. Variable-arity custom/channel gates are counted as one
    /// command-level gate.
    #[must_use]
    pub const fn num_gates(self, qubit_count: usize) -> usize {
        if matches!(
            self,
            GateType::MeasCrosstalkGlobalPayload
                | GateType::MeasCrosstalkLocalPayload
                | GateType::TrackedPauliMeta
        ) {
            return 0;
        }
        if matches!(self, GateType::Custom | GateType::Channel) {
            return 1;
        }
        qubit_count / self.quantum_arity()
    }

    /// Returns the number of angle parameters this gate type requires.
    ///
    /// This is separate from `classical_arity()` which includes all classical parameters.
    /// For example, `Idle` has `classical_arity() = 1` (duration) but `angle_arity() = 0`.
    #[must_use]
    pub const fn angle_arity(self) -> usize {
        match self {
            // Rotation gates with angle parameters
            GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::CRZ => 1,
            GateType::RXY1Q => 2,
            GateType::U | GateType::RXXRYYRZZ => 3,
            GateType::U2q => 15,
            // All other gates have no angle parameters
            _ => 0,
        }
    }

    /// Returns whether this gate type requires angle parameters
    #[must_use]
    pub const fn is_parameterized(self) -> bool {
        self.classical_arity() > 0
    }

    /// Returns whether this gate type operates on a single qubit
    #[must_use]
    pub const fn is_single_qubit(self) -> bool {
        self.quantum_arity() == 1
    }

    /// Returns whether this gate type operates on two qubits
    #[must_use]
    pub const fn is_two_qubit(self) -> bool {
        self.quantum_arity() == 2
    }

    /// Returns whether this gate is a crosstalk payload gate
    #[must_use]
    pub const fn is_crosstalk_payload(self) -> bool {
        matches!(
            self,
            GateType::MeasCrosstalkGlobalPayload | GateType::MeasCrosstalkLocalPayload
        )
    }
}

impl fmt::Display for GateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateType::I => write!(f, "I"),
            GateType::X => write!(f, "X"),
            GateType::Y => write!(f, "Y"),
            GateType::Z => write!(f, "Z"),
            GateType::SX => write!(f, "SX"),
            GateType::SXdg => write!(f, "SXdg"),
            GateType::SY => write!(f, "SY"),
            GateType::SYdg => write!(f, "SYdg"),
            GateType::SZ => write!(f, "SZ"),
            GateType::SZdg => write!(f, "SZdg"),
            GateType::H => write!(f, "H"),
            GateType::F => write!(f, "F"),
            GateType::Fdg => write!(f, "Fdg"),
            GateType::RX => write!(f, "RX"),
            GateType::RY => write!(f, "RY"),
            GateType::RZ => write!(f, "RZ"),
            GateType::T => write!(f, "T"),
            GateType::Tdg => write!(f, "Tdg"),
            GateType::U => write!(f, "U"),
            GateType::RXY1Q => write!(f, "RXY1Q"),
            GateType::CX => write!(f, "CX"),
            GateType::CY => write!(f, "CY"),
            GateType::CZ => write!(f, "CZ"),
            GateType::CH => write!(f, "CH"),
            GateType::SXX => write!(f, "SXX"),
            GateType::SXXdg => write!(f, "SXXdg"),
            GateType::SYY => write!(f, "SYY"),
            GateType::SYYdg => write!(f, "SYYdg"),
            GateType::SZZ => write!(f, "SZZ"),
            GateType::SZZdg => write!(f, "SZZdg"),
            GateType::RXX => write!(f, "RXX"),
            GateType::RYY => write!(f, "RYY"),
            GateType::SWAP => write!(f, "SWAP"),
            GateType::CRZ => write!(f, "CRZ"),
            GateType::RZZ => write!(f, "RZZ"),
            GateType::RXXRYYRZZ => write!(f, "RXXRYYRZZ"),
            GateType::U2q => write!(f, "U2q"),
            GateType::CCX => write!(f, "CCX"),
            GateType::MX => write!(f, "MX"),
            GateType::MZ => write!(f, "MZ"),
            GateType::MeasureLeaked => write!(f, "MeasureLeaked"),
            GateType::MeasureFree => write!(f, "MeasureFree"),
            GateType::MPZ => write!(f, "MPZ"),
            GateType::PX => write!(f, "PX"),
            GateType::PZ => write!(f, "PZ"),
            GateType::QAlloc => write!(f, "QAlloc"),
            GateType::QFree => write!(f, "QFree"),
            GateType::Idle => write!(f, "Idle"),
            GateType::MeasCrosstalkGlobalPayload => write!(f, "MeasCrosstalkGlobalPayload"),
            GateType::MeasCrosstalkLocalPayload => write!(f, "MeasCrosstalkLocalPayload"),
            GateType::Channel => write!(f, "Channel"),
            GateType::Custom => write!(f, "Custom"),
            GateType::TrackedPauliMeta => write!(f, "TrackedPauli"),
        }
    }
}

impl std::str::FromStr for GateType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Try exact match first for multi-word aliases with specific casing
        match s {
            "init |+>" | "Init |+>" => return Ok(GateType::PX),
            "init |0>" | "Init |0>" => return Ok(GateType::PZ),
            "measure X" => return Ok(GateType::MX),
            "measure Z" => return Ok(GateType::MZ),
            _ => {}
        }

        // Case-insensitive match for all standard gate names
        let upper = s.to_ascii_uppercase();
        match upper.as_str() {
            "I" => Ok(GateType::I),
            "X" => Ok(GateType::X),
            "Y" => Ok(GateType::Y),
            "Z" => Ok(GateType::Z),
            "H" => Ok(GateType::H),
            "F" => Ok(GateType::F),
            "FDG" => Ok(GateType::Fdg),
            "SX" | "Q" => Ok(GateType::SX),
            "SXDG" | "QD" => Ok(GateType::SXdg),
            "SY" | "R" => Ok(GateType::SY),
            "SYDG" | "RD" => Ok(GateType::SYdg),
            "SZ" | "S" => Ok(GateType::SZ),
            "SZDG" | "SD" | "SDG" => Ok(GateType::SZdg),
            "T" => Ok(GateType::T),
            "TDG" => Ok(GateType::Tdg),
            "RX" => Ok(GateType::RX),
            "RY" => Ok(GateType::RY),
            "RZ" => Ok(GateType::RZ),
            "RXY1Q" | "R1XY" => Ok(GateType::RXY1Q),
            "U" => Ok(GateType::U),
            "CX" | "CNOT" => Ok(GateType::CX),
            "CY" => Ok(GateType::CY),
            "CZ" => Ok(GateType::CZ),
            "CH" => Ok(GateType::CH),
            "SXX" => Ok(GateType::SXX),
            "SXXDG" => Ok(GateType::SXXdg),
            "SYY" => Ok(GateType::SYY),
            "SYYDG" => Ok(GateType::SYYdg),
            "SZZ" => Ok(GateType::SZZ),
            "SZZDG" => Ok(GateType::SZZdg),
            "RXX" => Ok(GateType::RXX),
            "RYY" => Ok(GateType::RYY),
            "RZZ" => Ok(GateType::RZZ),
            "RXXRYYRZZ" => Ok(GateType::RXXRYYRZZ),
            "U2Q" => Ok(GateType::U2q),
            "CRZ" => Ok(GateType::CRZ),
            "CCX" | "TOFFOLI" => Ok(GateType::CCX),
            "SWAP" => Ok(GateType::SWAP),
            "MX" | "MEASURE X" => Ok(GateType::MX),
            "MEASURE" | "MZ" | "MEASURE Z" => Ok(GateType::MZ),
            "MEASUREFREE" | "MZFREE" => Ok(GateType::MeasureFree),
            "MEASURELEAKED" => Ok(GateType::MeasureLeaked),
            "MPZ" => Ok(GateType::MPZ),
            "PX" | "INIT |+>" => Ok(GateType::PX),
            "PREP" | "PZ" | "INIT" | "INIT |0>" | "RESET" => Ok(GateType::PZ),
            "QALLOC" => Ok(GateType::QAlloc),
            "QFREE" => Ok(GateType::QFree),
            "IDLE" => Ok(GateType::Idle),
            "TRACKEDPAULI" | "TRACKEDPAULIMETA" | "TP" => Ok(GateType::TrackedPauliMeta),
            "MEASCROSSTALKGLOBALPAYLOAD" | "MEAS_CROSSTALK_GLOBAL_PAYLOAD" => {
                Ok(GateType::MeasCrosstalkGlobalPayload)
            }
            "MEASCROSSTALKLOCALPAYLOAD" | "MEAS_CROSSTALK_LOCAL_PAYLOAD" => {
                Ok(GateType::MeasCrosstalkLocalPayload)
            }
            "CHANNEL" => Ok(GateType::Channel),
            _ => Err(format!("Unknown gate type: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOSURE_TOLERANCE: f64 = 1e-14;

    fn canonical_table() -> [SingleQubitGateMatrix; NAMED_SINGLE_QUBIT_GATES.len()] {
        NAMED_SINGLE_QUBIT_GATES.map(|gate| {
            gate.canonical_1q_matrix()
                .expect("named single-qubit gate must have a canonical matrix")
        })
    }

    fn table_matrix(
        table: &[SingleQubitGateMatrix; NAMED_SINGLE_QUBIT_GATES.len()],
        gate: GateType,
    ) -> SingleQubitGateMatrix {
        let index = NAMED_SINGLE_QUBIT_GATES
            .iter()
            .position(|candidate| *candidate == gate)
            .expect("gate must be present in canonical table");
        table[index]
    }

    fn matrix_power(matrix: SingleQubitGateMatrix, exponent: usize) -> SingleQubitGateMatrix {
        let mut result = I_MATRIX;
        for _ in 0..exponent {
            result = multiply_2x2(result, matrix);
        }
        result
    }

    fn max_matrix_error(lhs: SingleQubitGateMatrix, rhs: SingleQubitGateMatrix) -> f64 {
        lhs.into_iter()
            .zip(rhs)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max)
    }

    fn closure_relations(
        table: &[SingleQubitGateMatrix; NAMED_SINGLE_QUBIT_GATES.len()],
    ) -> Vec<(&'static str, SingleQubitGateMatrix, SingleQubitGateMatrix)> {
        let matrix = |gate| table_matrix(table, gate);
        let mut relations = vec![
            (
                "SX^2 = X",
                matrix_power(matrix(GateType::SX), 2),
                matrix(GateType::X),
            ),
            (
                "SXdg^2 = X",
                matrix_power(matrix(GateType::SXdg), 2),
                matrix(GateType::X),
            ),
            (
                "SY^2 = Y",
                matrix_power(matrix(GateType::SY), 2),
                matrix(GateType::Y),
            ),
            (
                "SYdg^2 = Y",
                matrix_power(matrix(GateType::SYdg), 2),
                matrix(GateType::Y),
            ),
            (
                "SZ^2 = Z",
                matrix_power(matrix(GateType::SZ), 2),
                matrix(GateType::Z),
            ),
            (
                "SZdg^2 = Z",
                matrix_power(matrix(GateType::SZdg), 2),
                matrix(GateType::Z),
            ),
            (
                "T^2 = SZ",
                matrix_power(matrix(GateType::T), 2),
                matrix(GateType::SZ),
            ),
            (
                "Tdg^2 = SZdg",
                matrix_power(matrix(GateType::Tdg), 2),
                matrix(GateType::SZdg),
            ),
            (
                "T^4 = Z",
                matrix_power(matrix(GateType::T), 4),
                matrix(GateType::Z),
            ),
            (
                "T^8 = I",
                matrix_power(matrix(GateType::T), 8),
                matrix(GateType::I),
            ),
            (
                "H^2 = I",
                matrix_power(matrix(GateType::H), 2),
                matrix(GateType::I),
            ),
            (
                "X^2 = I",
                matrix_power(matrix(GateType::X), 2),
                matrix(GateType::I),
            ),
            (
                "Y^2 = I",
                matrix_power(matrix(GateType::Y), 2),
                matrix(GateType::I),
            ),
            (
                "Z^2 = I",
                matrix_power(matrix(GateType::Z), 2),
                matrix(GateType::I),
            ),
            (
                "F = i * SZ * SX",
                matrix(GateType::F),
                scale_2x2(
                    multiply_2x2(matrix(GateType::SZ), matrix(GateType::SX)),
                    0.0,
                    1.0,
                ),
            ),
            (
                "F^3 = I",
                matrix_power(matrix(GateType::F), 3),
                matrix(GateType::I),
            ),
            (
                "Fdg^3 = I",
                matrix_power(matrix(GateType::Fdg), 3),
                matrix(GateType::I),
            ),
        ];

        for (gate, dagger, label) in [
            (GateType::SX, GateType::SXdg, "SXdg = SX adjoint"),
            (GateType::SY, GateType::SYdg, "SYdg = SY adjoint"),
            (GateType::SZ, GateType::SZdg, "SZdg = SZ adjoint"),
            (GateType::F, GateType::Fdg, "Fdg = F adjoint"),
            (GateType::T, GateType::Tdg, "Tdg = T adjoint"),
        ] {
            relations.push((label, matrix(dagger), adjoint_2x2(matrix(gate))));
        }

        relations
    }

    fn closure_holds(table: &[SingleQubitGateMatrix; NAMED_SINGLE_QUBIT_GATES.len()]) -> bool {
        closure_relations(table)
            .into_iter()
            .all(|(_, lhs, rhs)| max_matrix_error(lhs, rhs) <= CLOSURE_TOLERANCE)
    }

    #[test]
    fn canonical_single_qubit_matrices_close_phase_exactly() {
        let table = canonical_table();
        for (label, actual, expected) in closure_relations(&table) {
            let error = max_matrix_error(actual, expected);
            assert!(
                error <= CLOSURE_TOLERANCE,
                "{label}: max entrywise error {error:e} exceeds phase-exact tolerance \
                 {CLOSURE_TOLERANCE:e}; actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn closure_guard_rejects_a_phase_mutation_of_every_canonical_matrix() {
        let table = canonical_table();
        let phase_re = std::f64::consts::FRAC_1_SQRT_2;
        let phase_im = std::f64::consts::FRAC_1_SQRT_2;

        for (index, gate) in NAMED_SINGLE_QUBIT_GATES.into_iter().enumerate() {
            let mut mutant = table;
            for entry in mutant[index].as_chunks_mut::<2>().0 {
                let phased = multiply_complex(entry[0], entry[1], phase_re, phase_im);
                entry[0] = phased.0;
                entry[1] = phased.1;
            }
            assert!(
                !closure_holds(&mutant),
                "exp(i*pi/4) phase mutation of {gate:?} escaped every closure relation"
            );
        }
    }

    #[test]
    fn test_gate_type_id_conversion() {
        assert_eq!(GateType::I as u8, 0);
        assert_eq!(GateType::X as u8, 1);
        assert_eq!(GateType::Z as u8, 2);
        assert_eq!(GateType::Y as u8, 3);
        assert_eq!(GateType::H as u8, 10);
        assert_eq!(GateType::F as u8, 16);
        assert_eq!(GateType::Fdg as u8, 17);
        assert_eq!(GateType::CX as u8, 50);
        assert_eq!(GateType::SXX as u8, 53);
        assert_eq!(GateType::SXXdg as u8, 54);
        assert_eq!(GateType::SYY as u8, 55);
        assert_eq!(GateType::SYYdg as u8, 56);
        assert_eq!(GateType::SZZ as u8, 57);
        assert_eq!(GateType::RZ as u8, 32);
        assert_eq!(GateType::RXY1Q as u8, 36);
        assert_eq!(GateType::MZ as u8, 104);
        assert_eq!(GateType::MeasureLeaked as u8, 105);
        assert_eq!(GateType::MeasureFree as u8, 106);
        assert_eq!(GateType::PZ as u8, 134);
        assert_eq!(GateType::QAlloc as u8, 135);
        assert_eq!(GateType::QFree as u8, 136);
        assert_eq!(GateType::Idle as u8, 200);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload as u8, 218);
        assert_eq!(GateType::MeasCrosstalkLocalPayload as u8, 219);
        assert_eq!(GateType::Channel as u8, 220);
        assert_eq!(GateType::Custom as u8, 255);

        assert_eq!(GateType::from(0u8), GateType::I);
        assert_eq!(GateType::from(1u8), GateType::X);
        assert_eq!(GateType::from(2u8), GateType::Z);
        assert_eq!(GateType::from(3u8), GateType::Y);
        assert_eq!(GateType::from(10u8), GateType::H);
        assert_eq!(GateType::from(16u8), GateType::F);
        assert_eq!(GateType::from(17u8), GateType::Fdg);
        assert_eq!(GateType::from(50u8), GateType::CX);
        assert_eq!(GateType::from(53u8), GateType::SXX);
        assert_eq!(GateType::from(54u8), GateType::SXXdg);
        assert_eq!(GateType::from(55u8), GateType::SYY);
        assert_eq!(GateType::from(56u8), GateType::SYYdg);
        assert_eq!(GateType::from(57u8), GateType::SZZ);
        assert_eq!(GateType::from(32u8), GateType::RZ);
        assert_eq!(GateType::from(36u8), GateType::RXY1Q);
        assert_eq!(GateType::from(104u8), GateType::MZ);
        assert_eq!(GateType::from(105u8), GateType::MeasureLeaked);
        assert_eq!(GateType::from(106u8), GateType::MeasureFree);
        assert_eq!(GateType::from(134u8), GateType::PZ);
        assert_eq!(GateType::from(135u8), GateType::QAlloc);
        assert_eq!(GateType::from(136u8), GateType::QFree);
        assert_eq!(GateType::from(200u8), GateType::Idle);
        assert_eq!(GateType::from(218u8), GateType::MeasCrosstalkGlobalPayload);
        assert_eq!(GateType::from(219u8), GateType::MeasCrosstalkLocalPayload);
        assert_eq!(GateType::from(220u8), GateType::Channel);
        assert_eq!(GateType::from(255u8), GateType::Custom);
    }

    #[test]
    fn test_from_str() {
        use std::str::FromStr;

        // Standard names
        assert_eq!(GateType::from_str("H").unwrap(), GateType::H);
        assert_eq!(GateType::from_str("X").unwrap(), GateType::X);
        assert_eq!(GateType::from_str("CX").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("F").unwrap(), GateType::F);
        assert_eq!(GateType::from_str("Fdg").unwrap(), GateType::Fdg);
        assert_eq!(GateType::from_str("SXX").unwrap(), GateType::SXX);
        assert_eq!(GateType::from_str("SXXdg").unwrap(), GateType::SXXdg);
        assert_eq!(GateType::from_str("SYY").unwrap(), GateType::SYY);
        assert_eq!(GateType::from_str("SYYdg").unwrap(), GateType::SYYdg);
        assert_eq!(GateType::from_str("Channel").unwrap(), GateType::Channel);
        assert_eq!(GateType::from_str("SWAP").unwrap(), GateType::SWAP);
        assert_eq!(GateType::from_str("CCX").unwrap(), GateType::CCX);
        assert_eq!(GateType::from_str("RXY1Q").unwrap(), GateType::RXY1Q);
        assert_eq!(GateType::from_str("R1XY").unwrap(), GateType::RXY1Q);
        assert!(GateType::from_str("U1q").is_err());
        assert_eq!(GateType::RXY1Q.to_string(), "RXY1Q");
        assert_eq!(
            GateType::from_str("MeasCrosstalkGlobalPayload").unwrap(),
            GateType::MeasCrosstalkGlobalPayload
        );
        assert_eq!(
            GateType::from_str("MeasCrosstalkLocalPayload").unwrap(),
            GateType::MeasCrosstalkLocalPayload
        );

        // Aliases
        assert_eq!(GateType::from_str("CNOT").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("Q").unwrap(), GateType::SX);
        assert_eq!(GateType::from_str("S").unwrap(), GateType::SZ);
        assert_eq!(GateType::from_str("TOFFOLI").unwrap(), GateType::CCX);
        assert_eq!(GateType::from_str("init |0>").unwrap(), GateType::PZ);
        assert_eq!(
            GateType::from_str("meas_crosstalk_global_payload").unwrap(),
            GateType::MeasCrosstalkGlobalPayload
        );
        assert_eq!(
            GateType::from_str("meas_crosstalk_local_payload").unwrap(),
            GateType::MeasCrosstalkLocalPayload
        );

        // Case-insensitive matching
        assert_eq!(GateType::from_str("h").unwrap(), GateType::H);
        assert_eq!(GateType::from_str("cx").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("Cx").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("cX").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("cnot").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("Cnot").unwrap(), GateType::CX);
        assert_eq!(GateType::from_str("fdg").unwrap(), GateType::Fdg);
        assert_eq!(GateType::from_str("sxxdg").unwrap(), GateType::SXXdg);
        assert_eq!(GateType::from_str("r").unwrap(), GateType::SY);
        assert_eq!(GateType::from_str("R").unwrap(), GateType::SY);
        assert_eq!(GateType::from_str("q").unwrap(), GateType::SX);
        assert_eq!(GateType::from_str("s").unwrap(), GateType::SZ);
        assert_eq!(GateType::from_str("toffoli").unwrap(), GateType::CCX);
        assert_eq!(GateType::from_str("Toffoli").unwrap(), GateType::CCX);

        // Unknown
        assert!(GateType::from_str("FOOBAR").is_err());
    }

    #[test]
    fn test_classical_arity() {
        // Gates with no parameters
        assert_eq!(GateType::I.classical_arity(), 0);
        assert_eq!(GateType::X.classical_arity(), 0);
        assert_eq!(GateType::Y.classical_arity(), 0);
        assert_eq!(GateType::Z.classical_arity(), 0);
        assert_eq!(GateType::H.classical_arity(), 0);
        assert_eq!(GateType::CX.classical_arity(), 0);
        assert_eq!(GateType::SZZ.classical_arity(), 0);
        assert_eq!(GateType::SZZdg.classical_arity(), 0);
        assert_eq!(GateType::MZ.classical_arity(), 0);
        assert_eq!(GateType::MeasureLeaked.classical_arity(), 0);
        assert_eq!(GateType::MeasureFree.classical_arity(), 0);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload.classical_arity(), 0);
        assert_eq!(GateType::MeasCrosstalkLocalPayload.classical_arity(), 0);
        assert_eq!(GateType::Channel.classical_arity(), 0);
        assert_eq!(GateType::PZ.classical_arity(), 0);
        assert_eq!(GateType::QAlloc.classical_arity(), 0);
        assert_eq!(GateType::QFree.classical_arity(), 0);

        // Gates with one parameter
        assert_eq!(GateType::RZ.classical_arity(), 1);
        assert_eq!(GateType::RZZ.classical_arity(), 1);
        assert_eq!(GateType::Idle.classical_arity(), 1);

        // Gates with two parameters
        assert_eq!(GateType::RXY1Q.classical_arity(), 2);

        // Gates with three parameters
        assert_eq!(GateType::U.classical_arity(), 3);
    }

    #[test]
    fn test_quantum_arity() {
        // Single-qubit gates
        assert_eq!(GateType::I.quantum_arity(), 1);
        assert_eq!(GateType::X.quantum_arity(), 1);
        assert_eq!(GateType::Y.quantum_arity(), 1);
        assert_eq!(GateType::Z.quantum_arity(), 1);
        assert_eq!(GateType::H.quantum_arity(), 1);
        assert_eq!(GateType::RZ.quantum_arity(), 1);
        assert_eq!(GateType::RXY1Q.quantum_arity(), 1);
        assert_eq!(GateType::U.quantum_arity(), 1);
        assert_eq!(GateType::MZ.quantum_arity(), 1);
        assert_eq!(GateType::MeasureLeaked.quantum_arity(), 1);
        assert_eq!(GateType::MeasureFree.quantum_arity(), 1);
        assert_eq!(GateType::PZ.quantum_arity(), 1);
        assert_eq!(GateType::QAlloc.quantum_arity(), 1);
        assert_eq!(GateType::QFree.quantum_arity(), 1);
        assert_eq!(GateType::Idle.quantum_arity(), 1);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload.quantum_arity(), 1);
        assert_eq!(GateType::MeasCrosstalkLocalPayload.quantum_arity(), 1);
        assert_eq!(GateType::Channel.quantum_arity(), 1);

        // Two-qubit gates
        assert_eq!(GateType::CX.quantum_arity(), 2);
        assert_eq!(GateType::SZZ.quantum_arity(), 2);
        assert_eq!(GateType::SZZdg.quantum_arity(), 2);
        assert_eq!(GateType::RZZ.quantum_arity(), 2);
    }

    #[test]
    fn test_num_gates() {
        assert_eq!(GateType::H.num_gates(4), 4);
        assert_eq!(GateType::CX.num_gates(4), 2);
        assert_eq!(GateType::CCX.num_gates(6), 2);
        assert_eq!(GateType::Custom.num_gates(2), 1);
        assert_eq!(GateType::Channel.num_gates(2), 1);
        assert_eq!(GateType::TrackedPauliMeta.num_gates(3), 0);
        assert_eq!(GateType::MeasCrosstalkGlobalPayload.num_gates(3), 0);
        assert_eq!(GateType::MeasCrosstalkLocalPayload.num_gates(3), 0);
    }

    #[test]
    fn test_tracked_pauli_meta_gate_type_contract() {
        assert_eq!(
            "TrackedPauli".parse::<GateType>().unwrap(),
            GateType::TrackedPauliMeta
        );
        assert_eq!(
            "TrackedPauliMeta".parse::<GateType>().unwrap(),
            GateType::TrackedPauliMeta
        );
        assert_eq!(
            "TP".parse::<GateType>().unwrap(),
            GateType::TrackedPauliMeta
        );

        assert_eq!(GateType::TrackedPauliMeta.to_string(), "TrackedPauli");
        assert_eq!(GateType::TrackedPauliMeta as u8, 210);
        assert!(GateType::TrackedPauliMeta.is_meta());
        assert_eq!(GateType::TrackedPauliMeta.classical_arity(), 0);
        assert_eq!(GateType::TrackedPauliMeta.quantum_arity(), 1);
        assert_eq!(GateType::TrackedPauliMeta.num_gates(4), 0);
    }

    #[test]
    fn test_is_parameterized() {
        // Non-parameterized gates
        assert!(!GateType::I.is_parameterized());
        assert!(!GateType::X.is_parameterized());
        assert!(!GateType::Y.is_parameterized());
        assert!(!GateType::Z.is_parameterized());
        assert!(!GateType::H.is_parameterized());
        assert!(!GateType::CX.is_parameterized());
        assert!(!GateType::SZZ.is_parameterized());
        assert!(!GateType::SZZdg.is_parameterized());
        assert!(!GateType::MZ.is_parameterized());
        assert!(!GateType::MeasureLeaked.is_parameterized());
        assert!(!GateType::MeasureFree.is_parameterized());
        assert!(!GateType::MeasCrosstalkGlobalPayload.is_parameterized());
        assert!(!GateType::MeasCrosstalkLocalPayload.is_parameterized());
        assert!(!GateType::PZ.is_parameterized());
        assert!(!GateType::QAlloc.is_parameterized());
        assert!(!GateType::QFree.is_parameterized());

        // Parameterized gates
        assert!(GateType::RZ.is_parameterized());
        assert!(GateType::RZZ.is_parameterized());
        assert!(GateType::RXY1Q.is_parameterized());
        assert!(GateType::U.is_parameterized());
        assert!(GateType::Idle.is_parameterized());
    }

    #[test]
    fn test_is_single_qubit() {
        // Single-qubit gates
        assert!(GateType::I.is_single_qubit());
        assert!(GateType::X.is_single_qubit());
        assert!(GateType::Y.is_single_qubit());
        assert!(GateType::Z.is_single_qubit());
        assert!(GateType::H.is_single_qubit());
        assert!(GateType::RZ.is_single_qubit());
        assert!(GateType::RXY1Q.is_single_qubit());
        assert!(GateType::U.is_single_qubit());
        assert!(GateType::MZ.is_single_qubit());
        assert!(GateType::MeasureLeaked.is_single_qubit());
        assert!(GateType::MeasureFree.is_single_qubit());
        assert!(GateType::PZ.is_single_qubit());
        assert!(GateType::QAlloc.is_single_qubit());
        assert!(GateType::QFree.is_single_qubit());
        assert!(GateType::Idle.is_single_qubit());
        assert!(GateType::MeasCrosstalkGlobalPayload.is_single_qubit());
        assert!(GateType::MeasCrosstalkLocalPayload.is_single_qubit());

        // Two-qubit gates
        assert!(!GateType::CX.is_single_qubit());
        assert!(!GateType::SZZ.is_single_qubit());
        assert!(!GateType::SZZdg.is_single_qubit());
        assert!(!GateType::RZZ.is_single_qubit());
    }

    #[test]
    fn test_is_two_qubit() {
        // Single-qubit gates
        assert!(!GateType::I.is_two_qubit());
        assert!(!GateType::X.is_two_qubit());
        assert!(!GateType::Y.is_two_qubit());
        assert!(!GateType::Z.is_two_qubit());
        assert!(!GateType::H.is_two_qubit());
        assert!(!GateType::RZ.is_two_qubit());
        assert!(!GateType::RXY1Q.is_two_qubit());
        assert!(!GateType::U.is_two_qubit());
        assert!(!GateType::MZ.is_two_qubit());
        assert!(!GateType::MeasureLeaked.is_two_qubit());
        assert!(!GateType::MeasureFree.is_two_qubit());
        assert!(!GateType::PZ.is_two_qubit());
        assert!(!GateType::QAlloc.is_two_qubit());
        assert!(!GateType::QFree.is_two_qubit());
        assert!(!GateType::Idle.is_two_qubit());
        assert!(!GateType::MeasCrosstalkGlobalPayload.is_two_qubit());
        assert!(!GateType::MeasCrosstalkLocalPayload.is_two_qubit());

        // Two-qubit gates
        assert!(GateType::CX.is_two_qubit());
        assert!(GateType::SZZ.is_two_qubit());
        assert!(GateType::SZZdg.is_two_qubit());
        assert!(GateType::RZZ.is_two_qubit());
    }

    #[test]
    fn test_arity_usage_examples() {
        // Example usage of arity methods for validation
        let gate_type = GateType::RZZ;

        // Check parameter requirements
        assert_eq!(
            gate_type.classical_arity(),
            1,
            "RZZ requires 1 angle parameter"
        );
        assert!(gate_type.is_parameterized(), "RZZ is parameterized");

        // Check qubit requirements
        assert_eq!(gate_type.quantum_arity(), 2, "RZZ operates on 2 qubits");
        assert!(gate_type.is_two_qubit(), "RZZ is a two-qubit gate");
        assert!(
            !gate_type.is_single_qubit(),
            "RZZ is not a single-qubit gate"
        );

        // Example of using arity for validation
        let params = [1.57]; // One angle parameter
        let qubits = [0, 1]; // Two qubits

        // Validate parameter count
        assert_eq!(params.len(), gate_type.classical_arity());

        // Validate qubit count
        assert_eq!(qubits.len(), gate_type.quantum_arity());
    }
}
