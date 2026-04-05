// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Type definitions for the HUGR interpreter engine.
//!
//! This module contains all the data structures used by the HUGR engine:
//!
//! - **Quantum operations**: [`QuantumOp`] - extracted gate metadata
//! - **Classical operations**: [`ClassicalOp`], [`ClassicalOpType`] - arithmetic/logic ops
//! - **Classical values**: [`ClassicalValue`] - runtime values (bool, int, float, tuple, etc.)
//! - **Control flow metadata**: [`ConditionalInfo`], [`CfgInfo`], [`TailLoopInfo`], [`FuncDefnInfo`]
//! - **Active execution state**: [`ActiveCaseInfo`], [`ActiveCfgInfo`], [`ActiveTailLoopInfo`], [`ActiveCallInfo`]
//! - **Result capture**: [`CapturedResult`], [`ResultValue`]
//! - **Support types**: [`FutureState`], [`ContainerType`], [`RngContextState`]

use std::collections::BTreeSet;

use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use tket::hugr::Node;

// --- Wire Tracking Types ---

/// Key for tracking qubit wire flow: (node, `output_port_index`).
///
/// Used to map output ports to qubit IDs as values flow through the HUGR graph.
pub type WireKey = (Node, usize);

// --- State Grouping Types (ECS-like) ---

use std::collections::BTreeMap;

/// State for tracking wire values through the HUGR graph.
///
/// This groups all wire-related state: qubit mappings, classical values,
/// and qubit arrays. The propagation system uses this to track values
/// as they flow from output ports to connected input ports.
#[derive(Debug, Default, Clone)]
pub struct WireState {
    /// Map from (node, `output_port`) to qubit ID for tracking wire flow.
    pub wire_to_qubit: BTreeMap<WireKey, QubitId>,
    /// Classical wire values: tracks bool/integer/float values flowing through wires.
    pub classical_values: BTreeMap<WireKey, ClassicalValue>,
    /// Maps array wire keys to lists of qubit IDs for qubit arrays.
    pub qubit_arrays: BTreeMap<WireKey, Vec<QubitId>>,
    /// Next available qubit ID.
    pub next_qubit_id: usize,
}

impl WireState {
    /// Reset all wire state for a new execution.
    pub fn reset(&mut self) {
        self.wire_to_qubit.clear();
        self.classical_values.clear();
        self.qubit_arrays.clear();
        self.next_qubit_id = 0;
    }
}

/// State for tracking measurements and their results.
///
/// This groups all measurement-related state: mappings from nodes to qubits,
/// accumulated results, and output wire mappings for classical values.
#[derive(Debug, Default, Clone)]
pub struct MeasurementState {
    /// Measurement mappings: maps measurement index to (node, `qubit_id`).
    pub mappings: Vec<(Node, QubitId)>,
    /// Measurement results stored by qubit ID.
    pub results: BTreeMap<QubitId, u32>,
    /// Map from measurement node to the wire key where its classical output goes.
    pub output_wires: BTreeMap<Node, WireKey>,
    /// Number of measurements processed so far.
    pub processed_count: usize,
}

impl MeasurementState {
    /// Reset all measurement state for a new execution.
    pub fn reset(&mut self) {
        self.mappings.clear();
        self.results.clear();
        self.output_wires.clear();
        self.processed_count = 0;
    }
}

/// State for extension features (futures, RNG, shot tracking, global phase).
///
/// This groups state for various extension operations that aren't part of
/// core quantum or classical computation.
#[derive(Debug, Clone)]
pub struct ExtensionState {
    /// Active Futures (lazy measurement handles).
    pub futures: BTreeMap<FutureId, FutureState>,
    /// Next available Future ID.
    pub next_future_id: FutureId,
    /// Active RNG contexts.
    pub rng_contexts: BTreeMap<RngContextId, RngContextState>,
    /// Next available RNG context ID.
    pub next_rng_context_id: RngContextId,
    /// Current shot number (0-indexed).
    pub current_shot: u64,
    /// Accumulated global phase (in half-turns).
    pub global_phase: f64,
}

impl Default for ExtensionState {
    fn default() -> Self {
        Self {
            futures: BTreeMap::new(),
            next_future_id: 0,
            rng_contexts: BTreeMap::new(),
            next_rng_context_id: 0,
            current_shot: 0,
            global_phase: 0.0,
        }
    }
}

impl ExtensionState {
    /// Reset extension state for a new execution.
    pub fn reset(&mut self) {
        self.futures.clear();
        self.next_future_id = 0;
        self.rng_contexts.clear();
        self.next_rng_context_id = 0;
        // Note: current_shot is NOT reset - it tracks across executions
        self.global_phase = 0.0;
    }
}

/// Unique identifier for a Future value (lazy measurement result).
pub type FutureId = usize;

/// Unique identifier for an RNG context.
pub type RngContextId = usize;

// --- Quantum Operation Types ---

/// Information about a quantum operation extracted from HUGR.
///
/// This struct captures the essential metadata needed to execute a quantum gate:
/// the gate type, number of qubit inputs/outputs, and any rotation parameters.
#[derive(Debug, Clone)]
pub struct QuantumOp {
    /// The HUGR node (kept for debugging).
    #[allow(dead_code)]
    pub node: Node,
    /// The PECOS gate type.
    pub gate_type: GateType,
    /// Number of qubit input ports.
    pub num_qubit_inputs: usize,
    /// Number of qubit output ports.
    pub num_qubit_outputs: usize,
    /// Extracted rotation parameters (in radians).
    pub params: Vec<f64>,
}

// --- Classical Operation Types ---

/// Type of classical operation.
///
/// Covers logic operations, integer/float arithmetic, comparisons, bitwise ops,
/// type conversions, and tuple handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Some variants not yet used but needed for complete classical op support
pub enum ClassicalOpType {
    // Logic operations
    And,
    Or,
    Not,
    Xor,
    Eq,
    // Integer arithmetic
    Iadd,
    Isub,
    Imul,
    Idiv,
    Imod,
    Ineg,
    Iabs,
    // Integer comparisons
    Ieq,
    Ine,
    Ilt,
    Ile,
    Igt,
    Ige,
    // Integer bitwise
    Iand,
    Ior,
    Ixor,
    Inot,
    Ishl,
    Ishr,
    // Float arithmetic
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    Fneg,
    Fabs,
    Ffloor,
    Fceil,
    // Float comparisons
    Feq,
    Fne,
    Flt,
    Fle,
    Fgt,
    Fge,
    // Conversions
    ConvertIntToFloat,
    ConvertFloatToInt,
    // Constants
    ConstInt,
    ConstFloat,
    ConstBool,
    // Tuple operations
    MakeTuple,
    UnpackTuple,
}

/// Classical operation extracted from HUGR.
///
/// Contains metadata about a classical computation node including its type,
/// port counts, and for integer operations, the bit width and signedness.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for complete classical op support
pub struct ClassicalOp {
    /// The HUGR node.
    pub node: Node,
    /// The operation type.
    pub op_type: ClassicalOpType,
    /// Number of input ports.
    pub num_inputs: usize,
    /// Number of output ports.
    pub num_outputs: usize,
    /// For integer operations: bit width and signedness.
    /// Format: (`log_width`, `is_signed`) where width = `2^log_width` bits
    pub int_info: Option<(u8, bool)>,
    /// Constant value (for const operations).
    pub const_value: Option<ClassicalValue>,
}

// --- Classical Value Types ---

/// Represents a classical value that can flow through wires.
///
/// This is the runtime representation of classical data in the HUGR interpreter.
/// Supports various primitive types plus composite types (tuples, arrays) and
/// special handles (futures for lazy measurements, RNG contexts).
///
/// # Type Coercion
///
/// Several accessor methods provide implicit type coercion:
/// - [`as_bool`](Self::as_bool): Treats 0 as false for numeric types
/// - [`as_int`](Self::as_int): Converts floats by truncation
/// - [`as_float`](Self::as_float): Converts integers with possible precision loss
/// - [`as_rotation`](Self::as_rotation): Floats can be interpreted as rotations
#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalValue {
    /// Boolean value
    Bool(bool),
    /// Signed 64-bit integer
    Int(i64),
    /// Unsigned 64-bit integer
    UInt(u64),
    /// 64-bit floating point
    Float(f64),
    /// Tuple of values
    Tuple(Vec<ClassicalValue>),
    /// Array of values
    Array(Vec<ClassicalValue>),
    /// Future handle (for lazy measurements)
    Future(FutureId),
    /// Rotation angle (in half-turns, i.e., multiples of pi)
    Rotation(f64),
    /// RNG context handle
    RngContext(RngContextId),
    /// Qubit reference (for storing qubits in arrays)
    QubitRef(QubitId),
}

impl ClassicalValue {
    /// Convert to u32 for backward compatibility with control flow decisions.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Self::Bool(b) => Some(u32::from(*b)),
            Self::Int(i) => u32::try_from(*i).ok(),
            Self::UInt(u) => u32::try_from(*u).ok(),
            Self::Float(_)
            | Self::Tuple(_)
            | Self::Array(_)
            | Self::Future(_)
            | Self::Rotation(_)
            | Self::RngContext(_)
            | Self::QubitRef(_) => None,
        }
    }

    /// Try to interpret as boolean.
    ///
    /// Numeric types are coerced: 0 is false, non-zero is true.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::Int(i) => Some(*i != 0),
            Self::UInt(u) => Some(*u != 0),
            Self::Float(f) => Some(*f != 0.0),
            Self::Tuple(_)
            | Self::Array(_)
            | Self::Future(_)
            | Self::Rotation(_)
            | Self::RngContext(_)
            | Self::QubitRef(_) => None,
        }
    }

    /// Try to interpret as signed integer.
    ///
    /// Floats are truncated toward zero.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Bool(b) => Some(i64::from(*b)),
            Self::Int(i) => Some(*i),
            Self::UInt(u) => i64::try_from(*u).ok(),
            Self::Float(f) => Some(*f as i64),
            Self::Tuple(_)
            | Self::Array(_)
            | Self::Future(_)
            | Self::Rotation(_)
            | Self::RngContext(_)
            | Self::QubitRef(_) => None,
        }
    }

    /// Try to interpret as unsigned integer.
    ///
    /// Floats are truncated toward zero. Negative values return None.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            Self::Bool(b) => Some(u64::from(*b)),
            Self::Int(i) => u64::try_from(*i).ok(),
            Self::UInt(u) => Some(*u),
            Self::Float(f) => Some(*f as u64),
            Self::Tuple(_)
            | Self::Array(_)
            | Self::Future(_)
            | Self::Rotation(_)
            | Self::RngContext(_)
            | Self::QubitRef(_) => None,
        }
    }

    /// Try to interpret as float.
    ///
    /// Integers may lose precision for large values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Self::Int(i) => Some(*i as f64),
            Self::UInt(u) => Some(*u as f64),
            Self::Float(f) => Some(*f),
            Self::Rotation(r) => Some(*r), // Rotation can be interpreted as float (half-turns)
            Self::Tuple(_)
            | Self::Array(_)
            | Self::Future(_)
            | Self::RngContext(_)
            | Self::QubitRef(_) => None,
        }
    }

    /// Try to interpret as rotation (in half-turns).
    ///
    /// Floats can be interpreted as rotations.
    #[must_use]
    pub fn as_rotation(&self) -> Option<f64> {
        match self {
            Self::Rotation(r) => Some(*r),
            Self::Float(f) => Some(*f), // Float can be interpreted as rotation
            _ => None,
        }
    }

    /// Try to interpret as tuple.
    #[must_use]
    pub fn as_tuple(&self) -> Option<&[ClassicalValue]> {
        match self {
            Self::Tuple(v) => Some(v),
            _ => None,
        }
    }

    /// Get a specific element from a tuple.
    #[must_use]
    pub fn tuple_get(&self, index: usize) -> Option<&ClassicalValue> {
        match self {
            Self::Tuple(v) => v.get(index),
            _ => None,
        }
    }

    /// Try to interpret as array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[ClassicalValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Get a specific element from an array.
    #[must_use]
    pub fn array_get(&self, index: usize) -> Option<&ClassicalValue> {
        match self {
            Self::Array(v) => v.get(index),
            _ => None,
        }
    }

    /// Get the length of an array.
    #[must_use]
    pub fn array_len(&self) -> Option<usize> {
        match self {
            Self::Array(v) => Some(v.len()),
            _ => None,
        }
    }
}

// --- Result Capture Types ---

/// A captured result from a tket.result operation.
///
/// Used to record named output values from HUGR programs for later retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturedResult {
    /// The label for this result (from the operation parameters).
    pub label: String,
    /// The captured value.
    pub value: ResultValue,
}

/// Value types that can be captured via tket.result operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultValue {
    /// Boolean value (from `result_bool`).
    Bool(bool),
    /// Signed integer (from `result_int`).
    Int(i64),
    /// Unsigned integer (from `result_uint`).
    UInt(u64),
    /// Floating-point value (from `result_f64`).
    Float(f64),
    /// Array of booleans (from `result_array_bool`).
    ArrayBool(Vec<bool>),
    /// Array of signed integers (from `result_array_int`).
    ArrayInt(Vec<i64>),
    /// Array of unsigned integers (from `result_array_uint`).
    ArrayUInt(Vec<u64>),
    /// Array of floats (from `result_array_f64`).
    ArrayFloat(Vec<f64>),
}

// --- Future/Lazy Measurement Types ---

/// State of a Future (lazy measurement result).
///
/// Futures represent deferred measurement results that may not be available
/// immediately. They transition from Pending to Resolved when the quantum
/// backend provides the measurement outcome.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Forward-looking implementation for HUGR programs with lazy measurements
pub enum FutureState {
    /// The measurement has been queued but result not yet available.
    Pending {
        /// The measurement node that created this Future.
        measurement_node: Node,
        /// The qubit that was measured.
        qubit: QubitId,
        /// Index in `measurement_mappings` for result retrieval.
        measurement_index: usize,
    },
    /// The measurement result is available.
    Resolved(u32),
}

// --- Container Type Classification ---

/// Container type for determining wire mapping behavior.
///
/// Different HUGR container types have different port mapping semantics.
/// This classification is used when tracing wires through container boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerType {
    /// DFG: Input port N -> Input node output N, Output node input N -> Output port N
    Dfg,
    /// Case: Similar to DFG, but is child of a Conditional
    Case,
    /// Conditional: Control input unpacks Sum type; data inputs pass through
    Conditional,
    /// `TailLoop`: Complex - has `CONTINUE_TAG/BREAK_TAG` handling
    TailLoop,
    /// `FuncDefn`: Function definition, similar to DFG
    FuncDefn,
    /// Call: Function call, maps to `FuncDefn`'s Input/Output
    Call,
    /// CFG: Control flow graph with basic blocks
    Cfg,
    /// `DataflowBlock`: Basic block inside a CFG
    DataflowBlock,
    /// Other: Unknown container type
    Other,
}

// --- RNG Support Types ---

/// State of an RNG context for random number generation.
///
/// Uses a simple xorshift64 PRNG for reproducible random numbers.
#[derive(Debug, Clone)]
pub struct RngContextState {
    /// The seed used to initialize this context.
    #[allow(dead_code)]
    pub seed: u64,
    /// Simple PRNG state (xorshift64).
    pub state: u64,
}

impl RngContextState {
    /// Create a new RNG context with the given seed.
    ///
    /// If the seed is 0, a default non-zero value is used since xorshift64
    /// would otherwise always produce 0.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // xorshift64 requires non-zero state
        let state = if seed == 0 {
            0x1234_5678_9ABC_DEF0
        } else {
            seed
        };
        Self { seed, state }
    }

    /// Generate the next random u64 value using xorshift64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for double precision
        let bits = self.next_u64() >> 11;
        #[allow(clippy::cast_precision_loss)]
        let result = bits as f64 / (1u64 << 53) as f64;
        result
    }
}

// --- Conditional Control Flow Types ---

/// Information about a Conditional node for control flow.
///
/// A Conditional node contains multiple Case children. At runtime, a control
/// value (typically from a measurement) selects which Case to execute.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for conditional execution
pub struct ConditionalInfo {
    /// The Conditional node in the HUGR.
    pub node: Node,
    /// Case children nodes, indexed by branch index.
    pub cases: Vec<Node>,
    /// Number of qubit inputs to the conditional.
    pub num_qubit_inputs: usize,
    /// Number of qubit outputs from the conditional.
    pub num_qubit_outputs: usize,
}

/// Information about a Case being actively processed.
#[derive(Debug, Clone)]
pub struct ActiveCaseInfo {
    /// The parent Conditional node.
    pub conditional_node: Node,
    /// All quantum operation nodes inside this Case.
    pub ops_in_case: BTreeSet<Node>,
}

// --- CFG Control Flow Types ---

/// Information about a CFG (Control Flow Graph) node.
///
/// CFG nodes contain `DataflowBlock` children that represent basic blocks.
/// Control flow between blocks is determined by Sum types at port 0 of
/// each block's output, with the tag value selecting the successor.
#[derive(Debug, Clone)]
pub struct CfgInfo {
    /// The CFG node in the HUGR (kept for future diagnostics).
    #[allow(dead_code)]
    pub node: Node,
    /// Entry block (first `DataflowBlock` child).
    pub entry_block: Node,
    /// Exit block (`ExitBlock` child).
    pub exit_block: Node,
    /// All `DataflowBlock` children indexed by node.
    pub blocks: std::collections::BTreeMap<Node, DataflowBlockInfo>,
    /// Number of input values to the CFG (kept for wire validation).
    #[allow(dead_code)]
    pub num_inputs: usize,
    /// Number of output values from the CFG (kept for wire validation).
    #[allow(dead_code)]
    pub num_outputs: usize,
}

/// Information about a `DataflowBlock` within a CFG.
#[derive(Debug, Clone)]
pub struct DataflowBlockInfo {
    /// The `DataflowBlock` node (kept for diagnostics).
    #[allow(dead_code)]
    pub node: Node,
    /// Number of input values for this block (kept for wire validation).
    #[allow(dead_code)]
    pub num_inputs: usize,
    /// Number of successor blocks (from `sum_rows.len()`) (kept for validation).
    #[allow(dead_code)]
    pub num_successors: usize,
    /// Successor block nodes indexed by Sum tag.
    pub successors: Vec<Node>,
    /// All quantum operation nodes inside this block.
    pub quantum_ops: BTreeSet<Node>,
    /// All Call nodes inside this block.
    pub call_nodes: BTreeSet<Node>,
    /// All Conditional nodes inside this block.
    pub conditional_nodes: BTreeSet<Node>,
    /// All tket.bool operation nodes inside this block.
    pub bool_ops: BTreeSet<Node>,
    /// All classical operation nodes inside this block (arithmetic, comparisons, etc.).
    pub classical_ops: BTreeSet<Node>,
    /// All extension operation nodes inside this block (tket.result, tket.qsystem, etc.).
    /// These are extension ops that are not tracked elsewhere (not quantum, bool, or classical).
    pub extension_ops: BTreeSet<Node>,
    /// All `TailLoop` nodes inside this block.
    pub tailloop_nodes: BTreeSet<Node>,
    /// Input node inside this block (kept for future wire tracing).
    #[allow(dead_code)]
    pub input_node: Option<Node>,
    /// Output node inside this block (kept for future wire tracing).
    #[allow(dead_code)]
    pub output_node: Option<Node>,
}

/// Information about a CFG being actively processed.
#[derive(Debug, Clone)]
pub struct ActiveCfgInfo {
    /// The CFG node (kept for diagnostics).
    #[allow(dead_code)]
    pub cfg_node: Node,
    /// Currently executing block.
    pub current_block: Node,
    /// Blocks that have been fully processed.
    pub completed_blocks: BTreeSet<Node>,
}

// --- Function Call Types ---

/// Information about a `FuncDefn` (function definition) node.
#[derive(Debug, Clone)]
pub struct FuncDefnInfo {
    /// The `FuncDefn` node.
    #[allow(dead_code)]
    pub node: Node,
    /// The function name.
    #[allow(dead_code)]
    pub name: String,
    /// Input node inside the `FuncDefn`.
    pub input_node: Node,
    /// Output node inside the `FuncDefn`.
    pub output_node: Node,
    /// The CFG inside the `FuncDefn` (if any).
    pub cfg_node: Option<Node>,
    /// Number of input parameters.
    pub num_inputs: usize,
    /// Number of output values.
    pub num_outputs: usize,
}

/// Information about an active Call being executed.
#[derive(Debug, Clone)]
pub struct ActiveCallInfo {
    /// The Call node.
    #[allow(dead_code)]
    pub call_node: Node,
    /// The `FuncDefn` being called.
    pub func_defn_node: Node,
}

// --- TailLoop Control Flow Types ---

/// Information about a `TailLoop` node.
///
/// `TailLoop` executes its body repeatedly until the body outputs `BREAK_TAG` (1).
/// On `CONTINUE_TAG` (0), the body is re-executed with updated values.
///
/// # Port Layout
///
/// - Inputs: `just_inputs` (not iterated) + `rest` (iterated)
/// - Body Input node: `just_inputs` + `rest` (from previous iteration or initial)
/// - Body Output node: Sum(CONTINUE: `rest`, BREAK: `just_outputs` + `rest`)
/// - Outputs: `just_outputs` + `rest` (from BREAK)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Some fields reserved for future use
pub struct TailLoopInfo {
    /// The `TailLoop` node in the HUGR.
    pub node: Node,
    /// Input node inside the `TailLoop` body.
    pub input_node: Node,
    /// Output node inside the `TailLoop` body.
    pub output_node: Node,
    /// Number of "just inputs" (only input, not iterated).
    pub just_inputs_count: usize,
    /// Number of "just outputs" (only output from BREAK).
    pub just_outputs_count: usize,
    /// Number of "rest" values (both input and output, iterated).
    pub rest_count: usize,
    /// All quantum operation nodes inside this `TailLoop` body.
    pub quantum_ops: BTreeSet<Node>,
    /// All Call nodes inside this `TailLoop` body.
    pub call_nodes: BTreeSet<Node>,
    /// All extension operation nodes inside this `TailLoop` body.
    pub extension_ops: BTreeSet<Node>,
    /// All classical operation nodes inside this `TailLoop` body.
    pub classical_ops: BTreeSet<Node>,
    /// All tket.bool operation nodes inside this `TailLoop` body.
    pub bool_ops: BTreeSet<Node>,
    /// All Conditional nodes inside this `TailLoop` body.
    pub conditional_nodes: BTreeSet<Node>,
    /// Total number of `TailLoop` input ports.
    pub num_inputs: usize,
    /// Total number of `TailLoop` output ports.
    pub num_outputs: usize,
}

/// Information about an active `TailLoop` being executed.
#[derive(Debug, Clone)]
pub struct ActiveTailLoopInfo {
    /// The `TailLoop` node.
    #[allow(dead_code)]
    pub tailloop_node: Node,
    /// Current iteration number (for debugging/limits).
    pub iteration: usize,
    /// Whether the body has been activated for current iteration.
    pub body_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_value_bool_coercion() {
        assert_eq!(ClassicalValue::Bool(true).as_bool(), Some(true));
        assert_eq!(ClassicalValue::Bool(false).as_bool(), Some(false));
        assert_eq!(ClassicalValue::Int(0).as_bool(), Some(false));
        assert_eq!(ClassicalValue::Int(1).as_bool(), Some(true));
        assert_eq!(ClassicalValue::Int(-1).as_bool(), Some(true));
        assert_eq!(ClassicalValue::UInt(0).as_bool(), Some(false));
        assert_eq!(ClassicalValue::UInt(1).as_bool(), Some(true));
        assert_eq!(ClassicalValue::Float(0.0).as_bool(), Some(false));
        assert_eq!(ClassicalValue::Float(0.5).as_bool(), Some(true));
    }

    #[test]
    fn test_classical_value_int_coercion() {
        assert_eq!(ClassicalValue::Int(42).as_int(), Some(42));
        assert_eq!(ClassicalValue::Int(-42).as_int(), Some(-42));
        assert_eq!(ClassicalValue::UInt(42).as_int(), Some(42));
        assert_eq!(ClassicalValue::Float(3.7).as_int(), Some(3));
        assert_eq!(ClassicalValue::Float(-3.7).as_int(), Some(-3));
        assert_eq!(ClassicalValue::Bool(true).as_int(), Some(1));
        assert_eq!(ClassicalValue::Bool(false).as_int(), Some(0));
    }

    #[test]
    fn test_classical_value_tuple() {
        let tuple = ClassicalValue::Tuple(vec![ClassicalValue::Int(1), ClassicalValue::Bool(true)]);
        assert!(tuple.as_tuple().is_some());
        assert_eq!(tuple.tuple_get(0), Some(&ClassicalValue::Int(1)));
        assert_eq!(tuple.tuple_get(1), Some(&ClassicalValue::Bool(true)));
        assert_eq!(tuple.tuple_get(2), None);
    }

    #[test]
    fn test_classical_value_array() {
        let array = ClassicalValue::Array(vec![
            ClassicalValue::Int(1),
            ClassicalValue::Int(2),
            ClassicalValue::Int(3),
        ]);
        assert!(array.as_array().is_some());
        assert_eq!(array.array_len(), Some(3));
        assert_eq!(array.array_get(1), Some(&ClassicalValue::Int(2)));
    }

    #[test]
    fn test_rng_context_state() {
        let mut rng = RngContextState::new(12345);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2); // Should produce different values

        let f = rng.next_f64();
        assert!((0.0..1.0).contains(&f)); // Should be in [0, 1)
    }
}
