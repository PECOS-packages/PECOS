/*!
HUGR Parser - Direct to PHIR

Converts HUGR programs into PHIR Modules suitable for execution by `PhirEngine`.

Uses the proper HUGR API (`extension_id`, `unqualified_id`) to identify quantum operations
instead of fragile debug-string matching.

Supports:
- All standard quantum gates (H, X, Y, Z, S, Sdg, T, Tdg, CX, CZ, SWAP, etc.)
- Parameterized rotation gates (Rx, Ry, Rz) with angle extraction
- Measurements with proper Result op emission
- Qubit allocation via `VarDefine` (no explicit Alloc instructions)

Scope: Straight-line quantum circuits (no classical control flow).
For programs with control flow, use `GuppyHugrEngine` directly.
*/

use crate::builtin_ops::{BuiltinOp, FuncOp, ModuleOp, VarDefineOp};
use crate::error::{PhirError, Result};
use crate::ops::{ClassicalOp, Operation, QuantumOp};
use crate::phir::{AttributeValue, Block, Instruction, SSAValue, Terminator};
use crate::types::{FunctionType, IntWidth, Type};
use log::debug;
use pecos_core::Angle64;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[cfg(feature = "hugr")]
use tket::hugr::ops::OpType;
#[cfg(feature = "hugr")]
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, NodeIndex, PortIndex};

// ========================================================================
// Public API
// ========================================================================

/// Parse HUGR bytes directly into PHIR representation.
///
/// Handles both HUGR Package envelope and JSON formats.
///
/// # Errors
///
/// Returns an error if parsing or conversion fails.
pub fn parse_hugr_bytes_to_phir(hugr_bytes: &[u8]) -> Result<ModuleOp> {
    use tket::hugr::envelope::read_envelope;
    use tket::hugr::extension::{ExtensionRegistry, prelude};
    use tket::hugr::std_extensions::{
        arithmetic::{conversions, float_ops, float_types, int_ops, int_types},
        collections, logic, ptr,
    };
    use tket_qsystem::extension::{futures, gpu, qsystem, result, wasm};

    // Create extension registry with all required extensions
    let extensions = ExtensionRegistry::new([
        prelude::PRELUDE.clone(),
        int_types::EXTENSION.clone(),
        int_ops::EXTENSION.clone(),
        float_types::EXTENSION.clone(),
        float_ops::EXTENSION.clone(),
        conversions::EXTENSION.clone(),
        logic::EXTENSION.clone(),
        ptr::EXTENSION.clone(),
        collections::list::EXTENSION.clone(),
        collections::array::EXTENSION.clone(),
        collections::static_array::EXTENSION.clone(),
        collections::borrow_array::EXTENSION.clone(),
        futures::EXTENSION.clone(),
        result::EXTENSION.clone(),
        qsystem::EXTENSION.clone(),
        tket::extension::rotation::ROTATION_EXTENSION.clone(),
        tket::extension::TKET_EXTENSION.clone(),
        tket::extension::TKET1_EXTENSION.clone(),
        tket::extension::bool::BOOL_EXTENSION.clone(),
        tket::extension::debug::DEBUG_EXTENSION.clone(),
        gpu::EXTENSION.clone(),
        wasm::EXTENSION.clone(),
    ]);

    if hugr_bytes.is_empty() {
        return Err(PhirError::internal("Empty HUGR input".to_string()));
    }

    let (_desc, package) = read_envelope(hugr_bytes, &extensions)
        .map_err(|e| PhirError::internal(format!("Failed to read HUGR: {e}")))?;

    let hugr = package
        .modules
        .first()
        .ok_or_else(|| PhirError::internal("Package contains no HUGR modules".to_string()))?;

    let mut converter = HugrToPhirConverter::new();
    converter.convert(hugr)
}

/// Parse HUGR string into PHIR representation.
///
/// Supports HUGR Package envelope format, direct HUGR JSON, and simplified test format.
///
/// # Errors
///
/// Returns an error if parsing or conversion fails.
pub fn parse_hugr_to_phir(hugr_str: &str) -> Result<ModuleOp> {
    match parse_hugr_bytes_to_phir(hugr_str.as_bytes()) {
        Ok(module) => Ok(module),
        Err(_) => parse_simplified_hugr_json(hugr_str),
    }
}

// ========================================================================
// Converter
// ========================================================================

#[cfg(feature = "hugr")]
struct HugrToPhirConverter {
    next_ssa: u32,
    /// Maps (`node_index`, `output_port_index`) -> `SSAValue` for wire tracking.
    wire_values: BTreeMap<(usize, usize), SSAValue>,
    /// SSA value for each qubit, indexed by allocation order.
    qubit_ssa: Vec<SSAValue>,
    /// Deferred measurements: (`qubit_ssa`, `bit_index`).
    deferred_measurements: Vec<(SSAValue, usize)>,
}

#[cfg(feature = "hugr")]
impl HugrToPhirConverter {
    fn new() -> Self {
        Self {
            next_ssa: 0,
            wire_values: BTreeMap::new(),
            qubit_ssa: Vec::new(),
            deferred_measurements: Vec::new(),
        }
    }

    fn fresh_ssa(&mut self) -> SSAValue {
        let v = SSAValue::new(self.next_ssa);
        self.next_ssa += 1;
        v
    }

    /// Record that (node, `output_port`) produces the given SSA value.
    fn map_wire(&mut self, node: Node, port: usize, value: SSAValue) {
        self.wire_values.insert((node.index(), port), value);
    }

    /// Resolve the SSA value feeding into `node`'s input `port`.
    fn resolve_wire(&self, hugr: &Hugr, node: Node, port: usize) -> Option<SSAValue> {
        let in_port = IncomingPort::from(port);
        let (src_node, src_port) = hugr.single_linked_output(node, in_port)?;
        self.wire_values
            .get(&(src_node.index(), src_port.index()))
            .copied()
    }

    /// Top-level conversion: HUGR -> PHIR Module.
    fn convert(&mut self, hugr: &Hugr) -> Result<ModuleOp> {
        let mut module = ModuleOp::new("hugr_module");

        // Find the container node (DFG or DataflowBlock) holding operations
        let dfg_node = self.find_operations_container(hugr).ok_or_else(|| {
            PhirError::internal(
                "No operations container (DFG or DataflowBlock) found in HUGR".to_string(),
            )
        })?;

        // Gather DFG children in topological order
        let children = self.topological_children(hugr, dfg_node);

        // Count qubits and measurements
        let num_qubits = count_ops(hugr, &children, "QAlloc");
        let num_measurements =
            count_ops(hugr, &children, "Measure") + count_ops(hugr, &children, "MeasureFree");

        // Build the main function
        let func_type = FunctionType {
            inputs: vec![],
            outputs: vec![],
            variadic: false,
        };
        let mut func = FuncOp::new("main", func_type);

        let block = func
            .entry_region_mut()
            .and_then(|r| r.entry_block_mut())
            .ok_or_else(|| PhirError::internal("Failed to get entry block".to_string()))?;

        // 1) VarDefine for quantum register
        if num_qubits > 0 {
            block.add_instruction(Instruction::new(
                Operation::Builtin(BuiltinOp::VarDefine(VarDefineOp::new(
                    "q".to_string(),
                    "qubits".to_string(),
                    num_qubits,
                ))),
                vec![],
                vec![],
                vec![],
            ));
        }

        // 2) VarDefine for measurement scratch register
        if num_measurements > 0 {
            block.add_instruction(Instruction::new(
                Operation::Builtin(BuiltinOp::VarDefine(VarDefineOp::new(
                    "m".to_string(),
                    "i64".to_string(),
                    num_measurements,
                ))),
                vec![],
                vec![],
                vec![],
            ));
            // VarDefine for result export register
            block.add_instruction(Instruction::new(
                Operation::Builtin(BuiltinOp::VarDefine(VarDefineOp::new(
                    "c".to_string(),
                    "i64".to_string(),
                    num_measurements,
                ))),
                vec![],
                vec![],
                vec![],
            ));
        }

        // 3) Assign SSA values to qubits (VarDefine handles allocation)
        for _ in 0..num_qubits {
            let ssa = self.fresh_ssa();
            self.qubit_ssa.push(ssa);
        }

        // 4) Process operations (measurements are deferred)
        self.process_children(hugr, &children, block)?;

        // 5) Emit deferred measurements
        let measurements = std::mem::take(&mut self.deferred_measurements);
        self.emit_measurements(block, &measurements)?;

        module.add_function(func);
        Ok(module)
    }

    /// Find the container node whose children are the quantum operations.
    ///
    /// Guppy-compiled HUGRs use: `Module -> FuncDefn -> CFG -> DataflowBlock`.
    /// Other tools may use: `Module -> FuncDefn -> DFG`.
    /// Handles both structures.
    fn find_operations_container(&self, hugr: &Hugr) -> Option<Node> {
        // First: look for FuncDefn and check its children
        for node in hugr.nodes() {
            if matches!(hugr.get_optype(node), OpType::FuncDefn(_)) {
                for child in hugr.children(node) {
                    match hugr.get_optype(child) {
                        // Direct DFG body (e.g. from tket or manual construction)
                        OpType::DFG(_) => return Some(child),
                        // CFG body (Guppy output): find the first DataflowBlock
                        OpType::CFG(_) => {
                            for cfg_child in hugr.children(child) {
                                if matches!(hugr.get_optype(cfg_child), OpType::DataflowBlock(_)) {
                                    return Some(cfg_child);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Fallback: find any DFG or DataflowBlock
        for node in hugr.nodes() {
            if matches!(
                hugr.get_optype(node),
                OpType::DFG(_) | OpType::DataflowBlock(_)
            ) {
                return Some(node);
            }
        }

        None
    }

    /// Get children of a container in topological order using a work-queue approach.
    fn topological_children(&self, hugr: &Hugr, container: Node) -> Vec<Node> {
        let children: Vec<Node> = hugr.children(container).collect();
        let child_set: BTreeSet<usize> = children.iter().map(|n| n.index()).collect();

        let mut processed: BTreeSet<usize> = BTreeSet::new();
        let mut result = Vec::new();
        let mut queue: VecDeque<Node> = VecDeque::new();

        // Seed with nodes whose internal predecessors are all satisfied
        for &child in &children {
            if self.all_internal_preds_ready(hugr, child, &processed, &child_set) {
                queue.push_back(child);
            }
        }

        while let Some(node) = queue.pop_front() {
            if processed.contains(&node.index()) {
                continue;
            }
            processed.insert(node.index());
            result.push(node);

            // Check if any unprocessed children are now ready
            for &child in &children {
                if !processed.contains(&child.index())
                    && self.all_internal_preds_ready(hugr, child, &processed, &child_set)
                {
                    queue.push_back(child);
                }
            }
        }

        result
    }

    /// Check if all predecessors of `node` that are within `container_set` have been processed.
    fn all_internal_preds_ready(
        &self,
        hugr: &Hugr,
        node: Node,
        processed: &BTreeSet<usize>,
        container_set: &BTreeSet<usize>,
    ) -> bool {
        let num_inputs = hugr.num_inputs(node);
        for port_idx in 0..num_inputs {
            let in_port = IncomingPort::from(port_idx);
            if let Some((src_node, _)) = hugr.single_linked_output(node, in_port)
                && container_set.contains(&src_node.index())
                && !processed.contains(&src_node.index())
            {
                return false;
            }
        }
        true
    }

    /// Process all children of a DFG, converting extension ops to PHIR instructions.
    fn process_children(
        &mut self,
        hugr: &Hugr,
        children: &[Node],
        block: &mut Block,
    ) -> Result<()> {
        let mut qubit_alloc_idx: usize = 0;

        for &node in children {
            let op = hugr.get_optype(node);

            // We only care about extension ops (quantum gates, QAlloc, etc.)
            let Some(ext_op) = op.as_extension_op() else {
                // Skip Input, Output, Const, LoadConstant, DFG, etc.
                continue;
            };

            let ext_id = ext_op.extension_id();
            let op_name = ext_op.unqualified_id().to_string();

            // Only handle tket.quantum extension ops
            if ext_id.as_ref() as &str != "tket.quantum" {
                debug!("Skipping non-quantum extension op: {ext_id}.{op_name}");
                continue;
            }

            let (num_q_in, num_q_out) = qubit_io_counts(&op_name);

            match op_name.as_str() {
                "QAlloc" => {
                    // Map the output wire to the next qubit SSA value
                    if qubit_alloc_idx < self.qubit_ssa.len() {
                        let ssa = self.qubit_ssa[qubit_alloc_idx];
                        self.map_wire(node, 0, ssa);
                        qubit_alloc_idx += 1;
                    }
                }

                "QFree" => {
                    // Qubit deallocation -- nothing to emit, just skip.
                    // The qubit wire is consumed.
                }

                "Measure" => {
                    // Resolve the qubit input
                    let qubit_ssa = self.resolve_wire(hugr, node, 0).ok_or_else(|| {
                        PhirError::internal(format!(
                            "Measure: unresolved qubit input at node {:?}",
                            node.index()
                        ))
                    })?;

                    // Output port 0 is the post-measurement qubit (same SSA value)
                    self.map_wire(node, 0, qubit_ssa);

                    // Output port 1 is the bool result (fresh SSA)
                    let bool_ssa = self.fresh_ssa();
                    self.map_wire(node, 1, bool_ssa);

                    // Defer the measurement
                    let bit_idx = self.deferred_measurements.len();
                    self.deferred_measurements.push((qubit_ssa, bit_idx));
                }

                "MeasureFree" => {
                    // Like Measure but also frees the qubit
                    let qubit_ssa = self.resolve_wire(hugr, node, 0).ok_or_else(|| {
                        PhirError::internal(format!(
                            "MeasureFree: unresolved qubit input at node {:?}",
                            node.index()
                        ))
                    })?;

                    // No qubit output (qubit is freed)
                    // Output port 0 is the bool result
                    let bool_ssa = self.fresh_ssa();
                    self.map_wire(node, 0, bool_ssa);

                    let bit_idx = self.deferred_measurements.len();
                    self.deferred_measurements.push((qubit_ssa, bit_idx));
                }

                "Reset" => {
                    let qubit_ssa = self.resolve_wire(hugr, node, 0).ok_or_else(|| {
                        PhirError::internal(format!(
                            "Reset: unresolved qubit input at node {:?}",
                            node.index()
                        ))
                    })?;

                    block.add_instruction(Instruction::new(
                        Operation::Quantum(QuantumOp::Reset),
                        vec![qubit_ssa],
                        vec![self.fresh_ssa()],
                        vec![Type::Qubit],
                    ));

                    // Output port 0: same qubit identity
                    self.map_wire(node, 0, qubit_ssa);
                }

                _ => {
                    // Standard quantum gate
                    let angle = if is_rotation_op(&op_name) {
                        self.extract_angle(hugr, node, num_q_in)
                    } else {
                        None
                    };

                    let Some(quantum_op) = hugr_name_to_quantum_op(&op_name, angle) else {
                        debug!("Skipping unsupported quantum op: {op_name}");
                        // Still map output wires so downstream ops can resolve
                        for port in 0..num_q_out {
                            if let Some(in_ssa) = self.resolve_wire(hugr, node, port) {
                                self.map_wire(node, port, in_ssa);
                            }
                        }
                        continue;
                    };

                    // Resolve qubit inputs
                    let mut operands = Vec::with_capacity(num_q_in);
                    for port in 0..num_q_in {
                        let ssa = self.resolve_wire(hugr, node, port).ok_or_else(|| {
                            PhirError::internal(format!(
                                "{op_name}: unresolved qubit input port {port} at node {:?}",
                                node.index()
                            ))
                        })?;
                        operands.push(ssa);
                    }

                    let results: Vec<SSAValue> =
                        operands.iter().map(|_| self.fresh_ssa()).collect();
                    let result_types = vec![Type::Qubit; results.len()];

                    block.add_instruction(Instruction::new(
                        Operation::Quantum(quantum_op),
                        operands.clone(),
                        results,
                        result_types,
                    ));

                    // Map output wires to the same qubit SSA values
                    // (qubit identity is preserved through gates)
                    for (port, &ssa) in operands.iter().enumerate() {
                        self.map_wire(node, port, ssa);
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract a rotation angle from the input wire at port `num_qubit_inputs`.
    ///
    /// Traces back through Const / `LoadConstant` / `from_halfturns_unchecked` chains.
    /// Returns the angle in radians.
    fn extract_angle(&self, hugr: &Hugr, node: Node, num_qubit_inputs: usize) -> Option<f64> {
        let angle_port = IncomingPort::from(num_qubit_inputs);
        let (src_node, _) = hugr.single_linked_output(node, angle_port)?;
        let (value, is_half_turns) = trace_const(hugr, src_node, 0)?;
        let full_turns = if is_half_turns { value * 0.5 } else { value };
        // Convert full turns to radians
        Some(full_turns * 2.0 * std::f64::consts::PI)
    }

    /// Emit Measure + Bitcast + Shl + Or + Result for all deferred measurements.
    fn emit_measurements(
        &mut self,
        block: &mut Block,
        measurements: &[(SSAValue, usize)],
    ) -> Result<()> {
        if measurements.is_empty() {
            return Ok(());
        }

        // Step 1: Emit all Measure instructions
        let mut meas_results: Vec<(SSAValue, usize)> = Vec::with_capacity(measurements.len());
        for &(qubit_ssa, bit_idx) in measurements {
            let meas_result = self.fresh_ssa();
            block.add_instruction(Instruction::new(
                Operation::Quantum(QuantumOp::Measure),
                vec![qubit_ssa],
                vec![meas_result],
                vec![Type::Bit],
            ));
            meas_results.push((meas_result, bit_idx));
        }

        // Step 2: Combine bits into a single integer and emit Result
        // Start with ConstInt(0)
        let zero_ssa = self.fresh_ssa();
        block.add_instruction(Instruction::new(
            Operation::Classical(ClassicalOp::ConstInt(0)),
            vec![],
            vec![zero_ssa],
            vec![Type::Int(IntWidth::I64)],
        ));

        let mut accum = zero_ssa;

        for &(meas_ssa, bit_idx) in &meas_results {
            // Bitcast measurement bit to i64
            let cast_ssa = self.fresh_ssa();
            block.add_instruction(Instruction::new(
                Operation::Classical(ClassicalOp::Bitcast),
                vec![meas_ssa],
                vec![cast_ssa],
                vec![Type::Int(IntWidth::I64)],
            ));

            // Shift left by bit_idx
            let shifted_ssa = self.fresh_ssa();
            let shift_amount = u32::try_from(bit_idx)
                .map_err(|_| PhirError::internal("bit index too large".to_string()))?;
            block.add_instruction(Instruction::new(
                Operation::Classical(ClassicalOp::Shl(shift_amount)),
                vec![cast_ssa],
                vec![shifted_ssa],
                vec![Type::Int(IntWidth::I64)],
            ));

            // Or with accumulator
            let or_ssa = self.fresh_ssa();
            block.add_instruction(Instruction::new(
                Operation::Classical(ClassicalOp::Or),
                vec![accum, shifted_ssa],
                vec![or_ssa],
                vec![Type::Int(IntWidth::I64)],
            ));

            accum = or_ssa;
        }

        // Emit Result instruction with export_name attribute
        let result_ssa = self.fresh_ssa();
        let mut result_instr = Instruction::new(
            Operation::Classical(ClassicalOp::Result),
            vec![accum],
            vec![result_ssa],
            vec![Type::Int(IntWidth::I64)],
        );
        result_instr.attributes.insert(
            "export_name".to_string(),
            AttributeValue::String("c".to_string()),
        );
        block.add_instruction(result_instr);

        Ok(())
    }
}

// ========================================================================
// Constant tracing (for rotation angles)
// ========================================================================

/// Recursively trace back through a node to find a constant float value.
///
/// Returns `(value, is_half_turns)` where `is_half_turns` indicates if the
/// value passed through `from_halfturns_unchecked`.
#[cfg(feature = "hugr")]
fn trace_const(hugr: &Hugr, node: Node, depth: usize) -> Option<(f64, bool)> {
    if depth > 20 {
        return None;
    }

    let op = hugr.get_optype(node);

    // Const node: extract the float value from debug representation
    if let OpType::Const(const_op) = op {
        return extract_const_float(const_op);
    }

    // LoadConstant: follow to the Const
    if matches!(op, OpType::LoadConstant(_)) {
        let const_port = IncomingPort::from(0);
        if let Some((const_node, _)) = hugr.single_linked_output(node, const_port) {
            return trace_const(hugr, const_node, depth + 1);
        }
    }

    // Extension ops: handle from_halfturns_unchecked, fdiv, fneg, etc.
    if let Some(ext_op) = op.as_extension_op() {
        let name = ext_op.unqualified_id().to_string();

        // Pass-through operations (UnpackTuple, MakeTuple, etc.) -- follow port 0
        if name == "UnpackTuple" || name == "MakeTuple" {
            let input_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, input_port) {
                return trace_const(hugr, src_node, depth + 1);
            }
        }

        if name == "from_halfturns_unchecked" {
            let float_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, float_port)
                && let Some((val, _)) = trace_const(hugr, src_node, depth + 1)
            {
                return Some((val, true));
            }
        }

        if name == "fdiv" {
            let num_port = IncomingPort::from(0);
            let denom_port = IncomingPort::from(1);
            if let (Some((num_node, _)), Some((denom_node, _))) = (
                hugr.single_linked_output(node, num_port),
                hugr.single_linked_output(node, denom_port),
            ) && let (Some((num_val, _)), Some((denom_val, _))) = (
                trace_const(hugr, num_node, depth + 1),
                trace_const(hugr, denom_node, depth + 1),
            ) && denom_val != 0.0
            {
                return Some((num_val / denom_val, false));
            }
        }

        if name == "fneg" {
            let input_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, input_port)
                && let Some((val, is_ht)) = trace_const(hugr, src_node, depth + 1)
            {
                return Some((-val, is_ht));
            }
        }

        if name == "convert_s" || name == "convert_u" {
            let input_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, input_port) {
                return trace_const(hugr, src_node, depth + 1);
            }
        }
    }

    None
}

/// Extract a float value from a HUGR Const node using its debug representation.
///
/// Handles patterns: `F64(number)`, `ConstF64 { value: V }`, `Tuple(number)`,
/// `ConstInt { ... value: V ... }`.
#[cfg(feature = "hugr")]
fn extract_const_float(const_op: &tket::hugr::ops::Const) -> Option<(f64, bool)> {
    let debug_str = format!("{const_op:?}");

    // Pattern: F64(number)
    if let Some(start) = debug_str.find("F64(") {
        let rest = &debug_str[start + 4..];
        if let Some(end) = rest.find(')')
            && let Ok(val) = rest[..end].parse::<f64>()
        {
            return Some((val, false));
        }
    }

    // Pattern: ConstF64 { value: V }
    if let Some(start) = debug_str.find("ConstF64 {")
        && let Some(val_start) = debug_str[start..].find("value:")
    {
        let rest = &debug_str[start + val_start + 6..];
        if let Some(val) = parse_leading_float(rest.trim()) {
            return Some((val, false));
        }
    }

    // Pattern: Tuple(number)
    if let Some(start) = debug_str.find("Tuple(") {
        let rest = &debug_str[start + 6..];
        if let Some(end) = rest.find(')')
            && let Ok(val) = rest[..end].parse::<f64>()
        {
            return Some((val, false));
        }
    }

    // Pattern: ConstInt { ... value: V ... }
    if let Some(start) = debug_str.find("ConstInt {")
        && let Some(val_start) = debug_str[start..].find("value:")
    {
        let rest = &debug_str[start + val_start + 6..];
        if let Some(val) = parse_leading_int(rest.trim()) {
            #[allow(clippy::cast_precision_loss)]
            return Some((val as f64, false));
        }
    }

    // Fallback: look for "value:" followed by a float
    if let Some(start) = debug_str.find("value:") {
        let rest = &debug_str[start + 6..];
        if let Some(val) = parse_leading_float(rest.trim()) {
            return Some((val, false));
        }
    }

    None
}

/// Parse a leading float from a string (stops at non-numeric characters).
fn parse_leading_float(s: &str) -> Option<f64> {
    let mut num_str = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+' {
            num_str.push(c);
        } else if !num_str.is_empty() {
            break;
        }
    }
    if num_str.is_empty() {
        return None;
    }
    num_str.parse::<f64>().ok()
}

/// Parse a leading integer from a string.
fn parse_leading_int(s: &str) -> Option<i64> {
    let mut num_str = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() || c == '-' {
            num_str.push(c);
        } else if !num_str.is_empty() {
            break;
        }
    }
    if num_str.is_empty() {
        return None;
    }
    num_str.parse::<i64>().ok()
}

// ========================================================================
// Gate mapping helpers
// ========================================================================

/// Map a HUGR quantum operation name to a PHIR `QuantumOp`.
///
/// For rotation gates, `angle` should contain the angle in radians.
/// When a rotation angle matches a known Clifford gate exactly (using `Angle64`
/// fixed-point comparison), the named Clifford gate is returned instead of
/// the generic rotation. This allows stabilizer simulators to handle these
/// circuits without requiring the state-vector backend.
fn hugr_name_to_quantum_op(name: &str, angle: Option<f64>) -> Option<QuantumOp> {
    match name {
        // Single-qubit gates
        "H" => Some(QuantumOp::H),
        "X" => Some(QuantumOp::X),
        "Y" => Some(QuantumOp::Y),
        "Z" => Some(QuantumOp::Z),
        "S" => Some(QuantumOp::S),
        "Sdg" => Some(QuantumOp::Sdg),
        "T" => Some(QuantumOp::T),
        "Tdg" => Some(QuantumOp::Tdg),
        // Rotation gates -- try to simplify to a named Clifford gate
        "Rx" => {
            let a = Angle64::from_radians(angle.unwrap_or(0.0));
            Some(simplify_rotation(
                pecos_core::gate_type::GateType::RX,
                a,
                QuantumOp::RX(a),
            ))
        }
        "Ry" => {
            let a = Angle64::from_radians(angle.unwrap_or(0.0));
            Some(simplify_rotation(
                pecos_core::gate_type::GateType::RY,
                a,
                QuantumOp::RY(a),
            ))
        }
        "Rz" => {
            let a = Angle64::from_radians(angle.unwrap_or(0.0));
            Some(simplify_rotation(
                pecos_core::gate_type::GateType::RZ,
                a,
                QuantumOp::RZ(a),
            ))
        }
        // Two-qubit gates
        "CX" => Some(QuantumOp::CX),
        "CY" => Some(QuantumOp::CY),
        "CZ" => Some(QuantumOp::CZ),
        "CH" => Some(QuantumOp::CH),
        "SWAP" => Some(QuantumOp::SWAP),
        "CRz" => Some(QuantumOp::RZZ(Angle64::from_radians(angle.unwrap_or(0.0)))),
        "ZZMax" => Some(QuantumOp::RZZ(Angle64::QUARTER_TURN)),
        // Three-qubit gates
        "Toffoli" | "CCX" => Some(QuantumOp::Toffoli),
        // Lifecycle ops handled in process_children; all else unknown
        _ => None,
    }
}

/// Try to simplify a rotation gate to a named Clifford gate using the shared
/// `pecos_core::try_simplify_rotation` utility. Falls back to the original
/// rotation `QuantumOp` for non-Clifford angles.
fn simplify_rotation(
    gate_type: pecos_core::gate_type::GateType,
    angle: Angle64,
    fallback: QuantumOp,
) -> QuantumOp {
    match pecos_core::try_simplify_rotation(gate_type, angle) {
        Some(clifford) => gate_type_to_quantum_op(clifford),
        None => fallback,
    }
}

/// Map a `GateType` Clifford result back to a `QuantumOp`.
fn gate_type_to_quantum_op(gt: pecos_core::gate_type::GateType) -> QuantumOp {
    use pecos_core::gate_type::GateType;
    match gt {
        GateType::I => QuantumOp::RZ(Angle64::ZERO), // identity
        GateType::X => QuantumOp::X,
        GateType::Y => QuantumOp::Y,
        GateType::Z => QuantumOp::Z,
        GateType::SZ => QuantumOp::S,
        GateType::SZdg => QuantumOp::Sdg,
        GateType::T => QuantumOp::T,
        GateType::Tdg => QuantumOp::Tdg,
        GateType::H => QuantumOp::H,
        _ => unreachable!("unexpected GateType from simplification: {gt}"),
    }
}

/// Return (`num_qubit_inputs`, `num_qubit_outputs`) for a HUGR quantum op.
fn qubit_io_counts(name: &str) -> (usize, usize) {
    match name {
        "QAlloc" => (0, 1),
        "QFree" | "MeasureFree" => (1, 0),
        // Single-qubit gates + Measure (1 qubit in, 1 qubit out; Measure also has a bool out)
        "Measure" | "H" | "X" | "Y" | "Z" | "S" | "Sdg" | "T" | "Tdg" | "V" | "Vdg" | "Rx"
        | "Ry" | "Rz" | "Reset" => (1, 1),
        // Two-qubit gates
        "CX" | "CY" | "CZ" | "CH" | "ZZMax" | "SWAP" | "CRz" => (2, 2),
        // Three-qubit gates
        "Toffoli" | "CCX" => (3, 3),
        _ => (0, 0),
    }
}

/// Whether this op is a rotation gate that takes an angle parameter.
fn is_rotation_op(name: &str) -> bool {
    matches!(name, "Rx" | "Ry" | "Rz" | "CRz")
}

/// Count how many children are extension ops with the given unqualified name.
#[cfg(feature = "hugr")]
fn count_ops(hugr: &Hugr, children: &[Node], target_name: &str) -> usize {
    children
        .iter()
        .filter(|&&node| {
            hugr.get_optype(node)
                .as_extension_op()
                .is_some_and(|ext| ext.unqualified_id() == target_name)
        })
        .count()
}

// ========================================================================
// Simplified JSON parser (for testing)
// ========================================================================

/// Parse simplified HUGR JSON format (for testing).
fn parse_simplified_hugr_json(json: &str) -> Result<ModuleOp> {
    let value: Value = serde_json::from_str(json)
        .map_err(|e| PhirError::internal(format!("Invalid JSON: {e}")))?;

    let mut module = ModuleOp::new("main");

    if let Some(ops) = value["operations"].as_array() {
        let func = FuncOp::new(
            "main",
            FunctionType {
                inputs: vec![],
                outputs: vec![],
                variadic: false,
            },
        );

        let mut block = Block::new(None);

        for (i, op) in ops.iter().enumerate() {
            if let Some(op_str) = op["op"].as_str() {
                let instr = match op_str {
                    "H" | "Hadamard" => {
                        let qubit = SSAValue::new(0);
                        Instruction::new(
                            Operation::Quantum(QuantumOp::H),
                            vec![qubit],
                            vec![qubit],
                            vec![Type::Qubit],
                        )
                    }
                    "CNOT" | "CX" => {
                        let control = SSAValue::new(0);
                        let target = SSAValue::new(1);
                        Instruction::new(
                            Operation::Quantum(QuantumOp::CX),
                            vec![control, target],
                            vec![control, target],
                            vec![Type::Qubit, Type::Qubit],
                        )
                    }
                    "Measure" => {
                        let qubit = SSAValue::new(0);
                        let result_id =
                            u32::try_from(i).expect("Operation index too large for u32") + 100;
                        let result = SSAValue::new(result_id);
                        Instruction::new(
                            Operation::Quantum(QuantumOp::Measure),
                            vec![qubit],
                            vec![result],
                            vec![Type::Bool],
                        )
                    }
                    _ => continue,
                };
                block.operations.push(instr);
            }
        }

        block.terminator = Some(Terminator::Return { values: vec![] });
        // Replace the default entry block
        let mut f = func;
        if let Some(region) = f.body.first_mut() {
            if region.blocks.is_empty() {
                region.blocks.push(block);
            } else {
                region.blocks[0] = block;
            }
        }
        module.add_function(f);
    }

    Ok(module)
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hugr_parsing_placeholder() {
        let simple_json = r#"{"operations": []}"#;
        let result = parse_simplified_hugr_json(simple_json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simplified_json_parsing() {
        let json = r#"
        {
            "operations": [
                {"op": "H", "qubit": 0},
                {"op": "Measure", "qubit": 0}
            ]
        }
        "#;

        let module = parse_simplified_hugr_json(json).unwrap();
        assert_eq!(module.name, "main");
    }

    #[test]
    fn test_gate_mapping_basic() {
        assert_eq!(hugr_name_to_quantum_op("H", None), Some(QuantumOp::H));
        assert_eq!(hugr_name_to_quantum_op("X", None), Some(QuantumOp::X));
        assert_eq!(hugr_name_to_quantum_op("CX", None), Some(QuantumOp::CX));
        assert_eq!(hugr_name_to_quantum_op("CZ", None), Some(QuantumOp::CZ));
        assert_eq!(hugr_name_to_quantum_op("SWAP", None), Some(QuantumOp::SWAP));
        assert_eq!(
            hugr_name_to_quantum_op("Toffoli", None),
            Some(QuantumOp::Toffoli)
        );
    }

    #[test]
    fn test_gate_mapping_rotation_clifford_simplification() {
        // pi/2 = S gate for RZ
        let pi_2 = Some(std::f64::consts::FRAC_PI_2);
        assert_eq!(hugr_name_to_quantum_op("Rz", pi_2), Some(QuantumOp::S));

        // pi = Z gate for RZ
        let pi = Some(std::f64::consts::PI);
        assert_eq!(hugr_name_to_quantum_op("Rz", pi), Some(QuantumOp::Z));

        // -pi/2 = Sdg gate for RZ
        let neg_pi_2 = Some(-std::f64::consts::FRAC_PI_2);
        assert_eq!(
            hugr_name_to_quantum_op("Rz", neg_pi_2),
            Some(QuantumOp::Sdg)
        );

        // pi/4 = T gate for RZ
        let pi_4 = Some(std::f64::consts::FRAC_PI_4);
        assert_eq!(hugr_name_to_quantum_op("Rz", pi_4), Some(QuantumOp::T));

        // -pi/4 = Tdg gate for RZ
        let neg_pi_4 = Some(-std::f64::consts::FRAC_PI_4);
        assert_eq!(
            hugr_name_to_quantum_op("Rz", neg_pi_4),
            Some(QuantumOp::Tdg)
        );

        // pi = X gate for RX
        assert_eq!(hugr_name_to_quantum_op("Rx", pi), Some(QuantumOp::X));

        // pi = Y gate for RY
        assert_eq!(hugr_name_to_quantum_op("Ry", pi), Some(QuantumOp::Y));

        // Non-Clifford angles stay as rotations
        let arbitrary = Some(1.23);
        assert_eq!(
            hugr_name_to_quantum_op("Rz", arbitrary),
            Some(QuantumOp::RZ(Angle64::from_radians(1.23)))
        );
        assert_eq!(
            hugr_name_to_quantum_op("Rx", arbitrary),
            Some(QuantumOp::RX(Angle64::from_radians(1.23)))
        );
    }

    #[test]
    fn test_qubit_io_counts() {
        assert_eq!(qubit_io_counts("QAlloc"), (0, 1));
        assert_eq!(qubit_io_counts("QFree"), (1, 0));
        assert_eq!(qubit_io_counts("Measure"), (1, 1));
        assert_eq!(qubit_io_counts("H"), (1, 1));
        assert_eq!(qubit_io_counts("CX"), (2, 2));
        assert_eq!(qubit_io_counts("Toffoli"), (3, 3));
    }

    #[test]
    fn test_is_rotation_op() {
        assert!(is_rotation_op("Rx"));
        assert!(is_rotation_op("Ry"));
        assert!(is_rotation_op("Rz"));
        assert!(is_rotation_op("CRz"));
        assert!(!is_rotation_op("H"));
        assert!(!is_rotation_op("CX"));
    }

    #[test]
    fn test_parse_leading_float() {
        assert_eq!(parse_leading_float("2.75)"), Some(2.75));
        assert_eq!(parse_leading_float("1.0, "), Some(1.0));
        assert_eq!(parse_leading_float("-2.5}"), Some(-2.5));
        assert_eq!(parse_leading_float("abc"), None);
    }

    #[test]
    fn test_parse_leading_int() {
        assert_eq!(parse_leading_int("42, "), Some(42));
        assert_eq!(parse_leading_int("-7}"), Some(-7));
        assert_eq!(parse_leading_int("abc"), None);
    }
}
