/*!
QIS LLVM IR Parser

Parses QIS-dialect LLVM IR text into a PHIR `Module` with QIS dialect `CustomOps`.

Supports two calling conventions:
- Selene style: `___rxy`, `___rz`, `___rzz`, `___qalloc`, etc.
- QIR style: `__quantum__qis__h__body`, `__quantum__qis__cnot__body`, etc.

QIR-style gates that are not hardware-native are decomposed into QIS
hardware-native ops (rxy, rz, rzz) using the same decomposition helpers
from `hugr_to_qis`.

Supports multi-block control flow: branches, conditional branches, switch
statements, and phi nodes (converted to MLIR-style block arguments).
*/

use crate::error::{PhirError, Result};
use crate::hugr_to_qis::{
    decompose_cx, decompose_h, emit_const_float, emit_qis_rxy, emit_qis_rz, emit_qis_rzz,
};
use crate::ops::{ClassicalOp, CustomOp, Operation};
use crate::phir::{
    Block, BlockArgument, BlockRef, Instruction, Module, Region, SSAValue, Terminator,
};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};

/// Parse QIS LLVM IR text into a PHIR `Module` with QIS dialect `CustomOps`.
///
/// Supports multi-block control flow (branches, conditional branches, switch,
/// phi nodes).
///
/// # Errors
///
/// Returns an error if the IR cannot be parsed.
pub fn parse_qis_llvm_ir(llvm_ir: &str) -> Result<Module> {
    let mut parser = QisIrParser::new();
    parser.parse(llvm_ir)
}

// ============================================================
// Intermediate types for multi-block parsing
// ============================================================

/// A block as parsed from LLVM IR, before phi-to-block-arg conversion.
struct RawBlock {
    label: String,
    phis: Vec<RawPhi>,
    instructions: Vec<Instruction>,
    terminator: RawTerminator,
}

/// A phi node as parsed from LLVM IR.
struct RawPhi {
    /// The result name, e.g. `%x`
    result_name: String,
    /// (`value_string`, `predecessor_label`) pairs
    incoming: Vec<(String, String)>,
}

/// A terminator as parsed from LLVM IR.
enum RawTerminator {
    RetVoid,
    #[allow(dead_code)]
    RetVal(String),
    Br(String),
    CondBr {
        cond: String,
        true_label: String,
        false_label: String,
    },
    Switch {
        value: String,
        default_label: String,
        cases: Vec<(i64, String)>,
    },
    Unreachable,
    /// No terminator found (entry block of a straight-line function).
    Missing,
}

// ============================================================
// Parser
// ============================================================

struct QisIrParser {
    /// Counter for generating fresh SSA values
    next_id: u32,
    /// Map from LLVM `%name` to PHIR `SSAValue`
    value_map: BTreeMap<String, SSAValue>,
    /// Set of declared QIS function names (from `declare` lines)
    qis_functions: BTreeSet<String>,
}

impl QisIrParser {
    fn new() -> Self {
        Self {
            next_id: 0,
            value_map: BTreeMap::new(),
            qis_functions: BTreeSet::new(),
        }
    }

    fn fresh_value(&mut self) -> SSAValue {
        let v = SSAValue::new(self.next_id);
        self.next_id += 1;
        v
    }

    /// Resolve or create an `SSAValue` for a named LLVM value (`%foo` or `%0`).
    fn resolve_value(&mut self, name: &str) -> SSAValue {
        if let Some(&v) = self.value_map.get(name) {
            return v;
        }
        let v = self.fresh_value();
        self.value_map.insert(name.to_string(), v);
        v
    }

    // ----------------------------------------------------------------
    // Top-level parse
    // ----------------------------------------------------------------

    fn parse(&mut self, ir: &str) -> Result<Module> {
        self.scan_declares(ir);

        let mut module = Module::new("qis_module");

        // Parse function bodies into raw blocks
        let raw_blocks = self.parse_define_bodies(ir)?;

        if raw_blocks.is_empty() {
            return Ok(module);
        }

        // Convert raw blocks to PHIR blocks (phi -> block args)
        let blocks = self.convert_raw_to_phir(&raw_blocks);

        // Determine region kind based on block count
        let kind = if blocks.len() <= 1 {
            crate::region_kinds::RegionKind::Graph
        } else {
            crate::region_kinds::RegionKind::SSACFG
        };

        module.body = Region {
            kind,
            attributes: BTreeMap::new(),
            blocks,
        };

        Ok(module)
    }

    // ----------------------------------------------------------------
    // Pass 1: scan declares
    // ----------------------------------------------------------------

    fn scan_declares(&mut self, ir: &str) {
        let re = Regex::new(r"declare\s+\S+\s+@([^\s(]+)").expect("valid regex");
        for line in ir.lines() {
            let trimmed = line.trim();
            if let Some(caps) = re.captures(trimmed) {
                let name = caps[1].to_string();
                if is_qis_function(&name) {
                    self.qis_functions.insert(name);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Pass 2: parse define bodies into RawBlocks
    // ----------------------------------------------------------------

    fn parse_define_bodies(&mut self, ir: &str) -> Result<Vec<RawBlock>> {
        let mut all_blocks = Vec::new();
        let mut in_define = false;
        let mut current_label: Option<String> = None;
        let mut current_phis = Vec::new();
        let mut current_instrs = Vec::new();
        let mut current_term: Option<RawTerminator> = None;
        let mut in_switch = false;
        let mut switch_value = String::new();
        let mut switch_default = String::new();
        let mut switch_cases: Vec<(i64, String)> = Vec::new();
        let case_re = Regex::new(r"i\d+\s+(-?\d+),\s*label\s+%(\S+)").expect("valid regex");

        for line in ir.lines() {
            let trimmed = line.trim();

            // Skip module-level lines
            if !in_define {
                if trimmed.starts_with("define ") {
                    in_define = true;
                    current_label = Some("entry".to_string());
                    current_phis.clear();
                    current_instrs.clear();
                    current_term = None;
                }
                continue;
            }

            // End of define
            if trimmed == "}" {
                // Flush last block
                if let Some(label) = current_label.take() {
                    all_blocks.push(RawBlock {
                        label,
                        phis: std::mem::take(&mut current_phis),
                        instructions: std::mem::take(&mut current_instrs),
                        terminator: current_term.take().unwrap_or(RawTerminator::Missing),
                    });
                }
                in_define = false;
                continue;
            }

            // Collecting switch cases
            if in_switch {
                if trimmed.starts_with(']') || trimmed == "]" {
                    current_term = Some(RawTerminator::Switch {
                        value: std::mem::take(&mut switch_value),
                        default_label: std::mem::take(&mut switch_default),
                        cases: std::mem::take(&mut switch_cases),
                    });
                    in_switch = false;
                    continue;
                }
                // Parse switch case line: `i32 42, label %bb3`
                if let Some(caps) = case_re.captures(trimmed)
                    && let Ok(val) = caps[1].parse::<i64>()
                {
                    switch_cases.push((val, caps[2].to_string()));
                }
                continue;
            }

            // Skip empty, comments, metadata
            if trimmed.is_empty() || trimmed.starts_with(';') || trimmed.starts_with('!') {
                continue;
            }

            // Block label: `entry:` or `bb1:   ; preds = %entry, %other`
            if let Some(label) = parse_block_label(trimmed) {
                // Flush previous block (skip if completely empty)
                if let Some(prev_label) = current_label.take() {
                    let has_content = !current_phis.is_empty()
                        || !current_instrs.is_empty()
                        || current_term.is_some();
                    if has_content {
                        all_blocks.push(RawBlock {
                            label: prev_label,
                            phis: std::mem::take(&mut current_phis),
                            instructions: std::mem::take(&mut current_instrs),
                            terminator: current_term.take().unwrap_or(RawTerminator::Missing),
                        });
                    }
                }
                current_label = Some(label);
                continue;
            }

            // Phi node: `%x = phi i32 [ %v1, %bb1 ], [ %v2, %bb2 ]`
            if let Some(phi) = parse_phi_line(trimmed) {
                current_phis.push(phi);
                continue;
            }

            // Terminators
            if let Some(term) = Self::parse_terminator_line(
                trimmed,
                &mut in_switch,
                &mut switch_value,
                &mut switch_default,
            ) {
                current_term = Some(term);
                continue;
            }

            // Regular instruction (call, icmp, etc.)
            // First, register any result name so SSA references work
            register_result_name(trimmed, &mut self.value_map, &mut self.next_id);

            if let Some(mut instrs) = self.parse_instruction(trimmed)? {
                current_instrs.append(&mut instrs);
            }
        }

        Ok(all_blocks)
    }

    /// Parse a terminator line. Returns `Some(RawTerminator)` if the line is a
    /// terminator, `None` otherwise.
    fn parse_terminator_line(
        line: &str,
        in_switch: &mut bool,
        switch_value: &mut String,
        switch_default: &mut String,
    ) -> Option<RawTerminator> {
        // ret void
        if line == "ret void" {
            return Some(RawTerminator::RetVoid);
        }

        // ret <type> <value>
        if line.starts_with("ret ") {
            let parts: Vec<&str> = line.splitn(3, ' ').collect();
            if parts.len() >= 3 {
                return Some(RawTerminator::RetVal(parts[2].to_string()));
            }
            return Some(RawTerminator::RetVoid);
        }

        // unreachable
        if line == "unreachable" {
            return Some(RawTerminator::Unreachable);
        }

        // Conditional branch: br i1 %cond, label %true, label %false
        let cond_br_re = Regex::new(r"br\s+i1\s+(%\S+),\s*label\s+%(\S+),\s*label\s+%(\S+)")
            .expect("valid regex");
        if let Some(caps) = cond_br_re.captures(line) {
            return Some(RawTerminator::CondBr {
                cond: caps[1].to_string(),
                true_label: caps[2].to_string(),
                false_label: caps[3].to_string(),
            });
        }

        // Unconditional branch: br label %target
        let br_re = Regex::new(r"br\s+label\s+%(\S+)").expect("valid regex");
        if let Some(caps) = br_re.captures(line) {
            return Some(RawTerminator::Br(caps[1].to_string()));
        }

        // Switch: switch i32 %val, label %default [
        let switch_re =
            Regex::new(r"switch\s+i\d+\s+(%\S+),\s*label\s+%(\S+)\s*\[").expect("valid regex");
        if let Some(caps) = switch_re.captures(line) {
            *switch_value = caps[1].to_string();
            *switch_default = caps[2].to_string();

            // Check if entire switch is on one line (contains `]`)
            if line.contains(']') {
                let mut cases = Vec::new();
                let case_re = Regex::new(r"i\d+\s+(-?\d+),\s*label\s+%(\S+)").expect("valid regex");
                // Find cases after the `[`
                if let Some(bracket_pos) = line.find('[') {
                    let cases_str = &line[bracket_pos + 1..];
                    for caps in case_re.captures_iter(cases_str) {
                        if let Ok(val) = caps[1].parse::<i64>() {
                            cases.push((val, caps[2].to_string()));
                        }
                    }
                }
                return Some(RawTerminator::Switch {
                    value: std::mem::take(switch_value),
                    default_label: std::mem::take(switch_default),
                    cases,
                });
            }

            // Multi-line switch -- caller will collect cases
            *in_switch = true;
            return None; // Don't return a terminator yet
        }

        None
    }

    // ----------------------------------------------------------------
    // Pass 3: convert RawBlocks to PHIR Blocks
    // ----------------------------------------------------------------

    fn convert_raw_to_phir(&mut self, raw_blocks: &[RawBlock]) -> Vec<Block> {
        // Build phi argument map:
        // phi_args[target_label][pred_label] = vec of SSAValues to pass
        let phi_args = self.build_phi_arg_map(raw_blocks);

        let mut blocks = Vec::with_capacity(raw_blocks.len());

        for raw in raw_blocks {
            // Block arguments from phi nodes
            let arguments: Vec<BlockArgument> = raw
                .phis
                .iter()
                .map(|phi| {
                    let value = self.resolve_value(&phi.result_name);
                    BlockArgument {
                        value,
                        ty: crate::types::Type::Unknown,
                        name: Some(phi.result_name.clone()),
                    }
                })
                .collect();

            // Convert terminator
            let terminator = self.convert_terminator(&raw.terminator, &raw.label, &phi_args);

            blocks.push(Block {
                label: Some(raw.label.clone()),
                arguments,
                attributes: BTreeMap::new(),
                operations: raw.instructions.clone(),
                terminator,
            });
        }

        blocks
    }

    /// Build a map: `target_label -> pred_label -> [SSAValue args]`
    ///
    /// For each target block with phi nodes, and each predecessor, collect the
    /// SSA values to pass as block arguments.
    fn build_phi_arg_map(
        &mut self,
        raw_blocks: &[RawBlock],
    ) -> BTreeMap<String, BTreeMap<String, Vec<SSAValue>>> {
        let mut map: BTreeMap<String, BTreeMap<String, Vec<SSAValue>>> = BTreeMap::new();

        for raw in raw_blocks {
            if raw.phis.is_empty() {
                continue;
            }

            let target_entry = map.entry(raw.label.clone()).or_default();

            for (phi_idx, phi) in raw.phis.iter().enumerate() {
                for (val_str, pred_label) in &phi.incoming {
                    let ssa = self.resolve_phi_value(val_str);
                    let pred_entry = target_entry.entry(pred_label.clone()).or_default();
                    // Ensure the vec is long enough
                    if pred_entry.len() <= phi_idx {
                        pred_entry.resize(phi_idx + 1, SSAValue::new(0));
                    }
                    pred_entry[phi_idx] = ssa;
                }
            }
        }

        map
    }

    /// Resolve a phi incoming value to an `SSAValue`.
    fn resolve_phi_value(&mut self, val_str: &str) -> SSAValue {
        if val_str.starts_with('%') {
            self.resolve_value(val_str)
        } else {
            // Constants (null, undef, true, false, integers, floats) and
            // unknown values all get a fresh SSA placeholder.
            self.fresh_value()
        }
    }

    /// Convert a `RawTerminator` to a PHIR `Terminator`, attaching phi args.
    fn convert_terminator(
        &mut self,
        raw: &RawTerminator,
        source_label: &str,
        phi_args: &BTreeMap<String, BTreeMap<String, Vec<SSAValue>>>,
    ) -> Option<Terminator> {
        match raw {
            RawTerminator::Missing => None,

            RawTerminator::RetVoid => Some(Terminator::Return { values: vec![] }),

            RawTerminator::RetVal(val) => {
                let ssa = self.resolve_value(val);
                Some(Terminator::Return { values: vec![ssa] })
            }

            RawTerminator::Br(target) => {
                let args = get_phi_args(phi_args, target, source_label);
                Some(Terminator::Branch {
                    target: BlockRef::Label(target.clone()),
                    args,
                })
            }

            RawTerminator::CondBr {
                cond,
                true_label,
                false_label,
            } => {
                let condition = self.resolve_value(cond);
                let true_args = get_phi_args(phi_args, true_label, source_label);
                let false_args = get_phi_args(phi_args, false_label, source_label);
                Some(Terminator::ConditionalBranch {
                    condition,
                    true_target: BlockRef::Label(true_label.clone()),
                    true_args,
                    false_target: BlockRef::Label(false_label.clone()),
                    false_args,
                })
            }

            RawTerminator::Switch {
                value,
                default_label,
                cases,
            } => {
                let val = self.resolve_value(value);
                let default_args = get_phi_args(phi_args, default_label, source_label);
                let phir_cases = cases
                    .iter()
                    .map(|(case_val, label)| {
                        let args = get_phi_args(phi_args, label, source_label);
                        crate::phir::SwitchCase {
                            value: *case_val,
                            target: BlockRef::Label(label.clone()),
                            args,
                        }
                    })
                    .collect();

                Some(Terminator::Switch {
                    value: val,
                    default_target: BlockRef::Label(default_label.clone()),
                    default_args,
                    cases: phir_cases,
                })
            }

            RawTerminator::Unreachable => Some(Terminator::Unreachable),
        }
    }

    // ----------------------------------------------------------------
    // Instruction parsing (unchanged from v1)
    // ----------------------------------------------------------------

    /// Parse a single LLVM IR instruction line into zero or more PHIR instructions.
    #[allow(clippy::too_many_lines)]
    fn parse_instruction(&mut self, line: &str) -> Result<Option<Vec<Instruction>>> {
        // ---- Call instructions ----
        let call_re =
            Regex::new(r"(?:(%[^\s]+)\s*=\s*)?call\s+\S+\s+@([^\s(]+)\(").expect("valid regex");

        if let Some(caps) = call_re.captures(line) {
            let result_name = caps.get(1).map(|m| m.as_str().to_string());
            let func_name = caps[2].to_string();

            if !is_qis_function(&func_name) {
                return Ok(None);
            }

            let args_start = caps.get(0).unwrap().end();
            let args_str = extract_balanced_parens(line, args_start);

            let args = self.parse_call_args(&args_str);
            let result = result_name.map(|n| self.resolve_value(&n));

            return Ok(Some(self.lower_qis_call(&func_name, &args, result)?));
        }

        // ---- Binary arithmetic: add, sub, mul, and, or, xor, shl, lshr, ashr ----
        let binop_re = Regex::new(
            r"^(%\S+)\s*=\s*(add|sub|mul|udiv|sdiv|urem|srem|and|or|xor|shl|lshr|ashr)\s+(?:nuw\s+)?(?:nsw\s+)?(?:exact\s+)?\S+\s+(%?\S+),\s*(%?\S+)$",
        )
        .expect("valid regex");
        if let Some(caps) = binop_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let op_name = &caps[2];
            let lhs = self.resolve_or_const(&caps[3]);
            let rhs = self.resolve_or_const(&caps[4]);
            let op = match op_name {
                "add" => ClassicalOp::Add,
                "sub" => ClassicalOp::Sub,
                "mul" => ClassicalOp::Mul,
                "udiv" | "sdiv" => ClassicalOp::Div,
                "urem" | "srem" => ClassicalOp::Mod,
                "and" => ClassicalOp::And,
                "or" => ClassicalOp::Or,
                "xor" => ClassicalOp::Xor,
                "shl" => ClassicalOp::Shl(0),
                "lshr" | "ashr" => ClassicalOp::Shr(0),
                _ => unreachable!(),
            };
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Classical(op),
                operands: vec![lhs, rhs],
                result_types: vec![crate::types::Type::Int(crate::types::IntWidth::I64)],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Integer comparison: icmp ----
        let icmp_re = Regex::new(
            r"^(%\S+)\s*=\s*icmp\s+(eq|ne|slt|sle|sgt|sge|ult|ule|ugt|uge)\s+\S+\s+(%?\S+),\s*(%?\S+)$",
        )
        .expect("valid regex");
        if let Some(caps) = icmp_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let pred = &caps[2];
            let lhs = self.resolve_or_const(&caps[3]);
            let rhs = self.resolve_or_const(&caps[4]);
            let op = match pred {
                "eq" => ClassicalOp::Eq,
                "ne" => ClassicalOp::Ne,
                "slt" | "ult" => ClassicalOp::Lt,
                "sle" | "ule" => ClassicalOp::Le,
                "sgt" | "ugt" => ClassicalOp::Gt,
                "sge" | "uge" => ClassicalOp::Ge,
                _ => unreachable!(),
            };
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Classical(op),
                operands: vec![lhs, rhs],
                result_types: vec![crate::types::Type::Bool],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Select: %r = select i1 %cond, <ty> %a, <ty> %b ----
        let select_re =
            Regex::new(r"^(%\S+)\s*=\s*select\s+i1\s+(%?\S+),\s*\S+\s+(%?\S+),\s*\S+\s+(%?\S+)$")
                .expect("valid regex");
        if let Some(caps) = select_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let cond = self.resolve_or_const(&caps[2]);
            let true_val = self.resolve_or_const(&caps[3]);
            let false_val = self.resolve_or_const(&caps[4]);
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Classical(ClassicalOp::Select),
                operands: vec![cond, true_val, false_val],
                result_types: vec![crate::types::Type::Int(crate::types::IntWidth::I64)],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Alloca: %r = alloca <type>, align <n> ----
        let alloca_re =
            Regex::new(r"^(%\S+)\s*=\s*alloca\s+(.+?)(?:,\s*align\s+\d+)?$").expect("valid regex");
        if let Some(caps) = alloca_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let ty = parse_llvm_type(&caps[2]);
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Memory(crate::ops::MemoryOp::Alloc(
                    crate::ops::AllocType::Scalar(ty.clone()),
                )),
                operands: vec![],
                result_types: vec![ty],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Load: %r = load <type>, <type>* %ptr, align <n> ----
        let load_re =
            Regex::new(r"^(%\S+)\s*=\s*load\s+(.+?),\s*\S+\s+(%\S+)(?:,\s*align\s+\d+)?$")
                .expect("valid regex");
        if let Some(caps) = load_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let ty = parse_llvm_type(&caps[2]);
            let ptr = self.resolve_value(&caps[3]);
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Memory(crate::ops::MemoryOp::Load),
                operands: vec![ptr],
                result_types: vec![ty],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Store: store <type> <val>, <type>* %ptr, align <n> ----
        let store_re = Regex::new(r"^store\s+\S+\s+(%?\S+),\s*\S+\s+(%\S+)(?:,\s*align\s+\d+)?$")
            .expect("valid regex");
        if let Some(caps) = store_re.captures(line) {
            let val = self.resolve_or_const(&caps[1]);
            let ptr = self.resolve_value(&caps[2]);
            return Ok(Some(vec![Instruction {
                results: vec![],
                operation: Operation::Memory(crate::ops::MemoryOp::Store),
                operands: vec![val, ptr],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Type casts: trunc, zext, sext, bitcast ----
        let cast_re =
            Regex::new(r"^(%\S+)\s*=\s*(?:trunc|zext|sext|bitcast)\s+\S+\s+(%\S+)\s+to\s+\S+$")
                .expect("valid regex");
        if let Some(caps) = cast_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let src = self.resolve_value(&caps[2]);
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Classical(ClassicalOp::Bitcast),
                operands: vec![src],
                result_types: vec![crate::types::Type::Int(crate::types::IntWidth::I64)],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        // ---- Aggregate ops: insertvalue, extractvalue ----
        // These are tracked for SSA value resolution but don't produce
        // meaningful PHIR ops for the quantum pipeline.
        let extract_re =
            Regex::new(r"^(%\S+)\s*=\s*extractvalue\s+.+\s+(%\S+),\s*(\d+)$").expect("valid regex");
        if let Some(caps) = extract_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let agg = self.resolve_value(&caps[2]);
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Classical(ClassicalOp::Assign),
                operands: vec![agg],
                result_types: vec![crate::types::Type::Unknown],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        let insertvalue_re =
            Regex::new(r"^(%\S+)\s*=\s*insertvalue\s+.+\s+(%?\S+),\s*\S+\s+(%?\S+),\s*\d+$")
                .expect("valid regex");
        if let Some(caps) = insertvalue_re.captures(line) {
            let result = self.resolve_value(&caps[1]);
            let _agg = self.resolve_or_const(&caps[2]);
            let _val = self.resolve_or_const(&caps[3]);
            return Ok(Some(vec![Instruction {
                results: vec![result],
                operation: Operation::Classical(ClassicalOp::Assign),
                operands: vec![],
                result_types: vec![crate::types::Type::Unknown],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            }]));
        }

        Ok(None)
    }

    /// Parse the comma-separated argument list of a call instruction.
    fn parse_call_args(&mut self, args_str: &str) -> Vec<Arg> {
        if args_str.trim().is_empty() {
            return vec![];
        }

        let mut args = Vec::new();
        let inttoptr_re = Regex::new(r"inttoptr\s*\(\s*i\d+\s+(\d+)\s+to\b").expect("valid regex");

        for part in split_args(args_str) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            // Handle `inttoptr (i64 N to %Qubit*)` -> treat as qubit index N
            if let Some(caps) = inttoptr_re.captures(part)
                && let Ok(idx) = caps[1].parse::<i64>()
            {
                args.push(Arg::Int(idx));
                continue;
            }

            if let Some((ty, val)) = part.rsplit_once(' ') {
                let ty = ty.trim();
                let val = val.trim();
                args.push(self.parse_arg(ty, val));
            }
        }
        args
    }

    fn parse_arg(&mut self, ty: &str, val: &str) -> Arg {
        if val.starts_with('%') {
            Arg::Ssa(self.resolve_value(val))
        } else if val == "null" || val == "zeroinitializer" {
            // QIR uses `null` for qubit 0
            Arg::Int(0)
        } else if ty.contains("double") || ty.contains("float") {
            let f = parse_float_literal(val);
            Arg::Float(f)
        } else if ty.contains("i1") || ty.contains("i32") || ty.contains("i64") {
            if let Ok(i) = val.parse::<i64>() {
                Arg::Int(i)
            } else {
                Arg::Ssa(self.resolve_value(val))
            }
        } else {
            Arg::Ssa(self.resolve_value(val))
        }
    }

    /// Lower a QIS function call to PHIR instructions.
    #[allow(clippy::too_many_lines)]
    fn lower_qis_call(
        &mut self,
        func_name: &str,
        args: &[Arg],
        result: Option<SSAValue>,
    ) -> Result<Vec<Instruction>> {
        let qis_name = normalize_qis_name(func_name);
        let mut out = Vec::new();

        match qis_name.as_str() {
            // ---- Selene-native ops (direct mapping) ----
            "qalloc" => {
                let r = result.unwrap_or_else(|| self.fresh_value());
                out.push(Instruction {
                    results: vec![r],
                    operation: Operation::Custom(CustomOp::new(
                        "qis",
                        "qalloc",
                        vec![],
                        BTreeMap::new(),
                    )),
                    operands: vec![],
                    result_types: vec![crate::types::Type::Qubit],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "qfree" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                out.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(CustomOp::new(
                        "qis",
                        "qfree",
                        vec![],
                        BTreeMap::new(),
                    )),
                    operands: vec![qubit],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "reset" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                out.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(CustomOp::new(
                        "qis",
                        "reset",
                        vec![],
                        BTreeMap::new(),
                    )),
                    operands: vec![qubit],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "rxy" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let theta = self.arg_to_ssa(&args[1], &mut out);
                let phi = self.arg_to_ssa(&args[2], &mut out);
                out.push(emit_qis_rxy(qubit, theta, phi));
            }

            "rz" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let angle = self.arg_to_ssa(&args[1], &mut out);
                out.push(emit_qis_rz(qubit, angle));
            }

            "rzz" => {
                let q1 = self.arg_to_ssa(&args[0], &mut out);
                let q2 = self.arg_to_ssa(&args[1], &mut out);
                let angle = self.arg_to_ssa(&args[2], &mut out);
                out.push(emit_qis_rzz(q1, q2, angle));
            }

            "measure" | "mz" | "m" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let r = result.unwrap_or_else(|| self.fresh_value());
                out.push(Instruction {
                    results: vec![r],
                    operation: Operation::Custom(CustomOp::new(
                        "qis",
                        "measure",
                        vec![],
                        BTreeMap::new(),
                    )),
                    operands: vec![qubit],
                    result_types: vec![crate::types::Type::Bool],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "lazy_measure" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let r = result.unwrap_or_else(|| self.fresh_value());
                out.push(Instruction {
                    results: vec![r],
                    operation: Operation::Custom(CustomOp::new(
                        "qis",
                        "lazy_measure",
                        vec![],
                        BTreeMap::new(),
                    )),
                    operands: vec![qubit],
                    result_types: vec![crate::types::Type::Future],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "read_future" | "read_result" => {
                let future = self.arg_to_ssa(&args[0], &mut out);
                let r = result.unwrap_or_else(|| self.fresh_value());
                out.push(Instruction {
                    results: vec![r],
                    operation: Operation::Custom(CustomOp::new(
                        "qis",
                        "read_future",
                        vec![],
                        BTreeMap::new(),
                    )),
                    operands: vec![future],
                    result_types: vec![crate::types::Type::Bool],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            // ---- QIR gates: decompose to hardware-native QIS ----
            "h" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                decompose_h(qubit, &mut out, &mut || self.fresh_value());
            }

            "x" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let pi_val = self.fresh_value();
                out.push(emit_const_float(pi_val, std::f64::consts::PI));
                let zero = self.fresh_value();
                out.push(emit_const_float(zero, 0.0));
                out.push(emit_qis_rxy(qubit, pi_val, zero));
            }

            "y" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let pi_val = self.fresh_value();
                out.push(emit_const_float(pi_val, std::f64::consts::PI));
                let half_pi = self.fresh_value();
                out.push(emit_const_float(half_pi, std::f64::consts::FRAC_PI_2));
                out.push(emit_qis_rxy(qubit, pi_val, half_pi));
            }

            "z" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let pi_val = self.fresh_value();
                out.push(emit_const_float(pi_val, std::f64::consts::PI));
                out.push(emit_qis_rz(qubit, pi_val));
            }

            "s" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let v = self.fresh_value();
                out.push(emit_const_float(v, std::f64::consts::FRAC_PI_2));
                out.push(emit_qis_rz(qubit, v));
            }

            "sdg" | "s_adj" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let v = self.fresh_value();
                out.push(emit_const_float(v, -std::f64::consts::FRAC_PI_2));
                out.push(emit_qis_rz(qubit, v));
            }

            "t" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let v = self.fresh_value();
                out.push(emit_const_float(v, std::f64::consts::FRAC_PI_4));
                out.push(emit_qis_rz(qubit, v));
            }

            "tdg" | "t_adj" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let v = self.fresh_value();
                out.push(emit_const_float(v, -std::f64::consts::FRAC_PI_4));
                out.push(emit_qis_rz(qubit, v));
            }

            "rx" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let theta = self.arg_to_ssa(&args[1], &mut out);
                let zero = self.fresh_value();
                out.push(emit_const_float(zero, 0.0));
                out.push(emit_qis_rxy(qubit, theta, zero));
            }

            "ry" => {
                let qubit = self.arg_to_ssa(&args[0], &mut out);
                let theta = self.arg_to_ssa(&args[1], &mut out);
                let half_pi = self.fresh_value();
                out.push(emit_const_float(half_pi, std::f64::consts::FRAC_PI_2));
                out.push(emit_qis_rxy(qubit, theta, half_pi));
            }

            "cx" | "cnot" => {
                let control = self.arg_to_ssa(&args[0], &mut out);
                let target = self.arg_to_ssa(&args[1], &mut out);
                decompose_cx(control, target, &mut out, &mut || self.fresh_value());
            }

            "cz" => {
                let q1 = self.arg_to_ssa(&args[0], &mut out);
                let q2 = self.arg_to_ssa(&args[1], &mut out);
                out.push(make_custom_op("qis", "cz", vec![q1, q2], vec![]));
            }

            "swap" => {
                let q1 = self.arg_to_ssa(&args[0], &mut out);
                let q2 = self.arg_to_ssa(&args[1], &mut out);
                out.push(make_custom_op("qis", "swap", vec![q1, q2], vec![]));
            }

            "cphase" | "cp" => {
                let q1 = self.arg_to_ssa(&args[0], &mut out);
                let q2 = self.arg_to_ssa(&args[1], &mut out);
                let angle = self.arg_to_ssa(&args[2], &mut out);
                out.push(make_custom_op("qis", "cphase", vec![q1, q2, angle], vec![]));
            }

            "zz" => {
                // QIS zz is equivalent to RZZ -- two qubit operands, no angle parameter
                // In PECOS, this maps to qis.rzz but the angle is implicit.
                // We treat it as a direct rzz with the two qubits; the angle will need
                // to be provided by the caller or resolved from context.
                let q1 = self.arg_to_ssa(&args[0], &mut out);
                let q2 = self.arg_to_ssa(&args[1], &mut out);
                // If a third argument (angle) is provided, use it; otherwise emit
                // as a 2-operand rzz and let downstream handle it.
                if args.len() > 2 {
                    let angle = self.arg_to_ssa(&args[2], &mut out);
                    out.push(emit_qis_rzz(q1, q2, angle));
                } else {
                    out.push(make_custom_op("qis", "rzz", vec![q1, q2], vec![]));
                }
            }

            // ---- QIR runtime operations ----
            "rt_result_allocate" => {
                // Allocate a result slot -- just track the SSA value
                let r = result.unwrap_or_else(|| self.fresh_value());
                out.push(Instruction {
                    results: vec![r],
                    operation: Operation::Classical(ClassicalOp::ConstInt(0)),
                    operands: vec![],
                    result_types: vec![crate::types::Type::Int(crate::types::IntWidth::I64)],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "rt_elide" => {
                // Runtime bookkeeping calls (result_record_output,
                // tuple_start/end, refcount, etc.) -- elide entirely.
            }

            other => {
                return Err(PhirError::internal(format!(
                    "qis_parser: unknown QIS function: {other} (original: {func_name})"
                )));
            }
        }

        Ok(out)
    }

    /// Resolve a value string to an `SSAValue`. If it looks like a `%name`,
    /// resolve it in the value map. If it looks like an integer literal, emit
    /// a `ConstInt` and return the SSA value.
    fn resolve_or_const(&mut self, val: &str) -> SSAValue {
        let val = val.trim();
        if val.starts_with('%') {
            return self.resolve_value(val);
        }
        if val == "true"
            || val == "false"
            || val == "null"
            || val == "undef"
            || val == "poison"
            || val == "zeroinitializer"
        {
            return self.fresh_value();
        }
        if val.parse::<i64>().is_ok() || val.parse::<f64>().is_ok() {
            return self.fresh_value();
        }
        // Unknown -- create a fresh value
        self.fresh_value()
    }

    /// Convert an `Arg` to an `SSAValue`, emitting a `ConstFloat`/`ConstInt` if
    /// the arg is a literal.
    fn arg_to_ssa(&mut self, arg: &Arg, out: &mut Vec<Instruction>) -> SSAValue {
        match arg {
            Arg::Ssa(v) => *v,
            Arg::Float(f) => {
                let v = self.fresh_value();
                out.push(emit_const_float(v, *f));
                v
            }
            Arg::Int(i) => {
                let v = self.fresh_value();
                out.push(Instruction {
                    results: vec![v],
                    operation: Operation::Classical(ClassicalOp::ConstInt(*i)),
                    operands: vec![],
                    result_types: vec![crate::types::Type::Int(crate::types::IntWidth::I64)],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
                v
            }
        }
    }
}

// ============================================================
// Free helper functions
// ============================================================

#[derive(Debug, Clone)]
enum Arg {
    Ssa(SSAValue),
    Float(f64),
    Int(i64),
}

/// Create a simple `CustomOp` instruction with given operands and results.
fn make_custom_op(
    dialect: &str,
    name: &str,
    operands: Vec<SSAValue>,
    results: Vec<SSAValue>,
) -> Instruction {
    Instruction {
        results,
        operation: Operation::Custom(CustomOp::new(dialect, name, vec![], BTreeMap::new())),
        operands,
        result_types: vec![],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    }
}

/// Look up phi args for a branch from `source_label` to `target_label`.
fn get_phi_args(
    phi_args: &BTreeMap<String, BTreeMap<String, Vec<SSAValue>>>,
    target_label: &str,
    source_label: &str,
) -> Vec<SSAValue> {
    phi_args
        .get(target_label)
        .and_then(|preds| preds.get(source_label))
        .cloned()
        .unwrap_or_default()
}

/// Extract a block label from a line like `entry:` or `bb1:   ; preds = ...`.
fn parse_block_label(line: &str) -> Option<String> {
    // Label must start at the beginning (not indented in standard LLVM IR, but
    // we trim, so just check the pattern).
    // A label is a word followed by `:`, optionally followed by a comment.
    let re = Regex::new(r"^([a-zA-Z_.][a-zA-Z0-9_.]*)\s*:").expect("valid regex");
    re.captures(line).map(|caps| caps[1].to_string())
}

/// Parse a phi instruction line.
///
/// `%x = phi i32 [ %v1, %bb1 ], [ %v2, %bb2 ]`
fn parse_phi_line(line: &str) -> Option<RawPhi> {
    // Check if this is a phi
    let phi_re = Regex::new(r"^(%\S+)\s*=\s*phi\s+\S+\s+(.+)$").expect("valid regex");
    let caps = phi_re.captures(line)?;

    let result_name = caps[1].to_string();
    let incoming_str = &caps[2];

    // Parse each `[ value, %label ]` pair
    let pair_re = Regex::new(r"\[\s*([^,]+),\s*%(\S+)\s*\]").expect("valid regex");
    let mut incoming = Vec::new();
    for pair_caps in pair_re.captures_iter(incoming_str) {
        let value = pair_caps[1].trim().to_string();
        let label = pair_caps[2].to_string();
        incoming.push((value, label));
    }

    if incoming.is_empty() {
        return None;
    }

    Some(RawPhi {
        result_name,
        incoming,
    })
}

/// If a line defines an SSA result (`%name = ...`), register it in the value
/// map so that forward references resolve correctly. Does not generate PHIR
/// instructions.
fn register_result_name(line: &str, value_map: &mut BTreeMap<String, SSAValue>, next_id: &mut u32) {
    let re = Regex::new(r"^(%\S+)\s*=").expect("valid regex");
    if let Some(caps) = re.captures(line) {
        let name = caps[1].to_string();
        if let std::collections::btree_map::Entry::Vacant(e) = value_map.entry(name) {
            e.insert(SSAValue::new(*next_id));
            *next_id += 1;
        }
    }
}

/// Extract the argument substring from position `start` up to the matching
/// closing paren, handling nested parentheses.
fn extract_balanced_parens(line: &str, start: usize) -> String {
    let bytes = line.as_bytes();
    let mut depth = 0usize;
    for i in start..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                if depth == 0 {
                    return line[start..i].to_string();
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    // Fallback: return everything from start
    line[start..].to_string()
}

/// Split a comma-separated arg list, respecting nested parens/brackets.
fn split_args(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Recognise whether a function name is a QIS function (Selene or QIR style).
fn is_qis_function(name: &str) -> bool {
    if name.starts_with("___") {
        return true;
    }
    if name.starts_with("__quantum__") {
        return true;
    }
    false
}

/// Normalise a QIS function name to a short canonical form.
///
/// `___rxy` -> `rxy`
/// `__quantum__qis__h__body` -> `h`
/// `__quantum__qis__cnot__body` -> `cnot`
/// `__quantum__rt__qubit_allocate` -> `qalloc`
/// `__quantum__rt__qubit_release` -> `qfree`
fn normalize_qis_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("___") {
        return rest.to_string();
    }

    // Runtime qubit management
    if name.contains("__rt__qubit_allocate") {
        return "qalloc".to_string();
    }
    if name.contains("__rt__qubit_release") {
        return "qfree".to_string();
    }

    // Runtime result/output operations -- elide
    if name.contains("__rt__result_get_one") || name.contains("__rt__result_get_zero") {
        return "rt_elide".to_string();
    }
    if name.contains("__rt__result_allocate") {
        return "rt_result_allocate".to_string();
    }
    if name.contains("__rt__result_record_output")
        || name.contains("__rt__int_record_output")
        || name.contains("__rt__tuple_start_record_output")
        || name.contains("__rt__tuple_end_record_output")
        || name.contains("__rt__result_update_reference_count")
        || name.contains("__rt__result_equal")
    {
        return "rt_elide".to_string();
    }

    // QIS gate operations
    let re = Regex::new(r"__quantum__qis__([a-z0-9_]+?)(?:__body|__adj)?$").expect("valid regex");
    if let Some(caps) = re.captures(name) {
        let gate = &caps[1];
        if name.ends_with("__adj") {
            return format!("{gate}_adj");
        }
        return gate.to_string();
    }

    name.to_string()
}

/// Map an LLVM type string to a PHIR `Type`.
fn parse_llvm_type(s: &str) -> crate::types::Type {
    let s = s.trim();
    match s {
        "i1" => crate::types::Type::Bool,
        "i8" | "i16" | "i32" | "i64" | "i128" => {
            crate::types::Type::Int(crate::types::IntWidth::I64)
        }
        "double" | "float" => crate::types::Type::Float(crate::types::FloatPrecision::F64),
        _ if s.starts_with('{') => crate::types::Type::Unknown,
        _ => crate::types::Type::Unknown,
    }
}

/// Parse a float literal, including LLVM hex floats (`0x3FF921FB54442D18`).
fn parse_float_literal(s: &str) -> f64 {
    if let Some(hex) = s.strip_prefix("0x")
        && let Ok(bits) = u64::from_str_radix(hex, 16)
    {
        return f64::from_bits(bits);
    }
    s.parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_float_literal_decimal() {
        assert!((parse_float_literal("1.234") - 1.234).abs() < 1e-12);
    }

    #[test]
    fn test_parse_float_literal_hex() {
        let val = parse_float_literal("0x3FF921FB54442D18");
        assert!((val - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_qis_name() {
        assert_eq!(normalize_qis_name("___rxy"), "rxy");
        assert_eq!(normalize_qis_name("___qalloc"), "qalloc");
        assert_eq!(normalize_qis_name("__quantum__qis__h__body"), "h");
        assert_eq!(normalize_qis_name("__quantum__qis__cnot__body"), "cnot");
        assert_eq!(normalize_qis_name("__quantum__qis__s__adj"), "s_adj");
        assert_eq!(
            normalize_qis_name("__quantum__rt__qubit_allocate"),
            "qalloc"
        );
        assert_eq!(normalize_qis_name("__quantum__rt__qubit_release"), "qfree");
    }

    #[test]
    fn test_parse_block_label() {
        assert_eq!(parse_block_label("entry:"), Some("entry".to_string()));
        assert_eq!(
            parse_block_label("bb1:   ; preds = %entry"),
            Some("bb1".to_string())
        );
        assert_eq!(parse_block_label("  %x = call ..."), None);
    }

    #[test]
    fn test_parse_phi_line() {
        let phi = parse_phi_line("%x = phi i32 [ %v1, %bb1 ], [ %v2, %bb2 ]").unwrap();
        assert_eq!(phi.result_name, "%x");
        assert_eq!(phi.incoming.len(), 2);
        assert_eq!(phi.incoming[0], ("%v1".to_string(), "bb1".to_string()));
        assert_eq!(phi.incoming[1], ("%v2".to_string(), "bb2".to_string()));
    }

    #[test]
    fn test_parse_phi_with_constants() {
        let phi = parse_phi_line("%x = phi i32 [ 0, %bb1 ], [ 1, %bb2 ]").unwrap();
        assert_eq!(phi.result_name, "%x");
        assert_eq!(phi.incoming[0].0, "0");
        assert_eq!(phi.incoming[1].0, "1");
    }

    // ---- Straight-line tests (backward compatible) ----

    #[test]
    fn test_parse_selene_style() {
        let ir = r"
declare void @___rz(i64, double)
declare void @___rxy(i64, double, double)
declare i64 @___qalloc()
declare void @___qfree(i64)
declare i1 @___measure(i64)

define void @main() {
entry:
  %q0 = call i64 @___qalloc()
  call void @___rz(i64 %q0, double 0x3FF921FB54442D18)
  call void @___rxy(i64 %q0, double 0x3FF921FB54442D18, double 0x0000000000000000)
  %m = call i1 @___measure(i64 %q0)
  call void @___qfree(i64 %q0)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();

        assert!(ops.contains(&"qis.qalloc".to_string()));
        assert!(ops.contains(&"qis.rz".to_string()));
        assert!(ops.contains(&"qis.rxy".to_string()));
        assert!(ops.contains(&"qis.measure".to_string()));
        assert!(ops.contains(&"qis.qfree".to_string()));
    }

    #[test]
    fn test_parse_qir_style_h_gate() {
        let ir = r"
declare void @__quantum__qis__h__body(%Qubit*)

define void @main() {
entry:
  call void @__quantum__qis__h__body(%Qubit* null)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let qis_ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .filter_map(|i| {
                if let Operation::Custom(c) = &i.operation {
                    Some(format!("{}.{}", c.dialect(), c.name()))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(qis_ops, vec!["qis.rz", "qis.rxy", "qis.rz"]);
    }

    #[test]
    fn test_parse_qir_style_cx_gate() {
        let ir = r"
declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)

define void @main() {
entry:
  call void @__quantum__qis__cnot__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let qis_ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .filter_map(|i| {
                if let Operation::Custom(c) = &i.operation {
                    Some(c.name().to_string())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(qis_ops, vec!["rxy", "rzz", "rz", "rxy"]);
    }

    // ---- Control flow tests ----

    #[test]
    fn test_unconditional_branch() {
        let ir = r"
declare i64 @___qalloc()
declare void @___qfree(i64)

define void @main() {
entry:
  %q = call i64 @___qalloc()
  br label %cleanup

cleanup:
  call void @___qfree(i64 %q)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        assert_eq!(module.body.blocks.len(), 2);
        assert_eq!(module.body.blocks[0].label, Some("entry".to_string()));
        assert_eq!(module.body.blocks[1].label, Some("cleanup".to_string()));

        // Entry block has a branch terminator
        let term = module.body.blocks[0].terminator.as_ref().unwrap();
        assert!(matches!(term, Terminator::Branch { target, .. }
            if *target == BlockRef::Label("cleanup".to_string())));

        // Cleanup block has the qfree instruction
        let ops: Vec<String> = module.body.blocks[1]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"qis.qfree".to_string()));
    }

    #[test]
    fn test_conditional_branch() {
        let ir = r"
declare i64 @___qalloc()
declare i1 @___measure(i64)
declare void @___rz(i64, double)
declare void @___qfree(i64)

define void @main() {
entry:
  %q = call i64 @___qalloc()
  %m = call i1 @___measure(i64 %q)
  br i1 %m, label %then, label %else

then:
  call void @___rz(i64 %q, double 0x3FF921FB54442D18)
  br label %merge

else:
  br label %merge

merge:
  call void @___qfree(i64 %q)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        assert_eq!(module.body.blocks.len(), 4);

        // Entry block ends with conditional branch
        let term = module.body.blocks[0].terminator.as_ref().unwrap();
        assert!(matches!(term, Terminator::ConditionalBranch {
            true_target: BlockRef::Label(t),
            false_target: BlockRef::Label(f),
            ..
        } if t == "then" && f == "else"));

        // "then" block has an rz gate
        let then_ops: Vec<String> = module.body.blocks[1]
            .operations
            .iter()
            .filter_map(|i| {
                if let Operation::Custom(c) = &i.operation {
                    Some(c.name().to_string())
                } else {
                    None
                }
            })
            .collect();
        assert!(then_ops.contains(&"rz".to_string()));
    }

    #[test]
    fn test_phi_nodes() {
        let ir = r"
declare i64 @___qalloc()
declare i1 @___measure(i64)
declare void @___rz(i64, double)
declare void @___qfree(i64)

define void @main() {
entry:
  %q = call i64 @___qalloc()
  %m = call i1 @___measure(i64 %q)
  br i1 %m, label %then, label %else

then:
  call void @___rz(i64 %q, double 0x3FF921FB54442D18)
  br label %merge

else:
  br label %merge

merge:
  %result = phi i1 [ %m, %then ], [ %m, %else ]
  call void @___qfree(i64 %q)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        assert_eq!(module.body.blocks.len(), 4);

        // Merge block should have a block argument from the phi
        let merge_block = &module.body.blocks[3];
        assert_eq!(merge_block.label, Some("merge".to_string()));
        assert_eq!(merge_block.arguments.len(), 1);
        assert_eq!(merge_block.arguments[0].name, Some("%result".to_string()));

        // The branch terminators going to merge should carry args
        let then_term = module.body.blocks[1].terminator.as_ref().unwrap();
        if let Terminator::Branch { args, .. } = then_term {
            assert_eq!(args.len(), 1);
        } else {
            panic!("expected Branch terminator on 'then' block");
        }
    }

    #[test]
    fn test_switch() {
        let ir = r"
declare i64 @___qalloc()
declare void @___rz(i64, double)
declare void @___qfree(i64)

define void @main() {
entry:
  %q = call i64 @___qalloc()
  switch i32 %q, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]

case0:
  call void @___rz(i64 %q, double 0x3FF921FB54442D18)
  br label %end

case1:
  br label %end

default:
  br label %end

end:
  call void @___qfree(i64 %q)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        assert_eq!(module.body.blocks.len(), 5);

        let term = module.body.blocks[0].terminator.as_ref().unwrap();
        if let Terminator::Switch {
            default_target,
            cases,
            ..
        } = term
        {
            assert_eq!(*default_target, BlockRef::Label("default".to_string()));
            assert_eq!(cases.len(), 2);
            assert_eq!(cases[0].value, 0);
            assert_eq!(cases[1].value, 1);
        } else {
            panic!("expected Switch terminator");
        }
    }

    // ---- Classical instruction tests ----

    #[test]
    fn test_parse_arithmetic() {
        let ir = r"
declare i64 @___qalloc()
declare void @___qfree(i64)

define void @main() {
entry:
  %q = call i64 @___qalloc()
  %a = add nsw i64 %q, 1
  %b = sub nsw i64 %a, 2
  %c = mul nuw nsw i64 %b, 3
  call void @___qfree(i64 %c)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"qis.qalloc".to_string()));
        assert!(ops.contains(&"arith.add".to_string()));
        assert!(ops.contains(&"arith.sub".to_string()));
        assert!(ops.contains(&"arith.mul".to_string()));
        assert!(ops.contains(&"qis.qfree".to_string()));
    }

    #[test]
    fn test_parse_icmp_and_select() {
        let ir = r"
declare i1 @___measure(i64)

define void @main() {
entry:
  %m = call i1 @___measure(i64 0)
  %cmp = icmp ne i64 %m, 0
  %val = select i1 %cmp, i64 42, i64 0
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"arith.ne".to_string()));
        assert!(ops.contains(&"arith.select".to_string()));
    }

    #[test]
    fn test_parse_alloca_load_store() {
        let ir = r"
declare i64 @___qalloc()

define void @main() {
entry:
  %ptr = alloca i64, align 8
  %q = call i64 @___qalloc()
  store i64 %q, i64* %ptr, align 8
  %loaded = load i64, i64* %ptr, align 8
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"memory.alloc".to_string()));
        assert!(ops.contains(&"memory.store".to_string()));
        assert!(ops.contains(&"memory.load".to_string()));
    }

    #[test]
    fn test_parse_trunc_zext() {
        let ir = r"
declare i64 @__quantum__rt__qubit_allocate()
declare void @__quantum__qis__h__body(i64)

define void @main() {
entry:
  %qubit_usize = call i64 @__quantum__rt__qubit_allocate()
  %qubit = trunc i64 %qubit_usize to i16
  %qubit_i64 = zext i16 %qubit to i64
  call void @__quantum__qis__h__body(i64 %qubit_i64)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"qis.qalloc".to_string()));
        assert!(ops.contains(&"arith.bitcast".to_string()));
    }

    #[test]
    fn test_parse_qir_mz_and_reset() {
        // Pattern from ArithmeticOps.Targeted.ll
        let ir = r"
declare void @__quantum__qis__x__body(%Qubit*)
declare void @__quantum__qis__mz__body(%Qubit*, %Result*)
declare void @__quantum__qis__reset__body(%Qubit*)
declare i1 @__quantum__qis__read_result__body(%Result*)

define void @main() {
entry:
  call void @__quantum__qis__x__body(%Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  call void @__quantum__qis__reset__body(%Qubit* null)
  %0 = call i1 @__quantum__qis__read_result__body(%Result* null)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        // x is decomposed to RXY
        assert!(ops.contains(&"qis.rxy".to_string()));
        // mz maps to measure
        assert!(ops.contains(&"qis.measure".to_string()));
        // reset
        assert!(ops.contains(&"qis.reset".to_string()));
        // read_result maps to read_future
        assert!(ops.contains(&"qis.read_future".to_string()));
    }

    #[test]
    fn test_parse_qir_m_with_result_id() {
        // Pattern from bell_state.ll: m takes qubit and result_id
        let ir = r"
declare i64 @__quantum__rt__qubit_allocate()
declare i64 @__quantum__rt__result_allocate()
declare i32 @__quantum__qis__m__body(i64, i64)

define void @main() {
entry:
  %q = call i64 @__quantum__rt__qubit_allocate()
  %result_id = call i64 @__quantum__rt__result_allocate()
  %m = call i32 @__quantum__qis__m__body(i64 %q, i64 %result_id)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"qis.qalloc".to_string()));
        assert!(ops.contains(&"qis.measure".to_string()));
    }

    #[test]
    fn test_parse_runtime_record_output() {
        // Runtime calls should be elided
        let ir = r"
declare void @__quantum__rt__tuple_start_record_output()
declare void @__quantum__rt__int_record_output(i64, i8*)
declare void @__quantum__rt__tuple_end_record_output()
declare void @__quantum__rt__result_record_output(i64, i8*)

define void @main() {
entry:
  call void @__quantum__rt__tuple_start_record_output()
  call void @__quantum__rt__int_record_output(i64 42, i8* null)
  call void @__quantum__rt__result_record_output(i64 0, i8* null)
  call void @__quantum__rt__tuple_end_record_output()
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        // All runtime calls should be elided -- no operations
        assert!(module.body.blocks[0].operations.is_empty());
    }

    #[test]
    fn test_parse_bell_state_pattern() {
        // Simplified version of the real bell_state.ll pattern
        let ir = r"
declare i64 @__quantum__rt__qubit_allocate()
declare void @__quantum__qis__h__body(i64)
declare void @__quantum__qis__cx__body(i64, i64)
declare i64 @__quantum__rt__result_allocate()
declare i32 @__quantum__qis__m__body(i64, i64)
declare void @__quantum__rt__result_record_output(i64, i8*)

define void @main() {
entry:
  %q0 = call i64 @__quantum__rt__qubit_allocate()
  %q1 = call i64 @__quantum__rt__qubit_allocate()
  call void @__quantum__qis__h__body(i64 %q0)
  call void @__quantum__qis__cx__body(i64 %q0, i64 %q1)
  %r0 = call i64 @__quantum__rt__result_allocate()
  %m0 = call i32 @__quantum__qis__m__body(i64 %q0, i64 %r0)
  call void @__quantum__rt__result_record_output(i64 %r0, i8* null)
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        let ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        // Should have: 2x qalloc, H decomposed (rz+rxy+rz), CX decomposed, result_allocate, measure
        // result_record_output should be elided
        assert!(ops.contains(&"qis.qalloc".to_string()));
        assert!(ops.contains(&"qis.rz".to_string()));
        assert!(ops.contains(&"qis.rxy".to_string()));
        assert!(ops.contains(&"qis.rzz".to_string()));
        assert!(ops.contains(&"qis.measure".to_string()));
        // No runtime record ops
        assert!(!ops.contains(&"rt_elide".to_string()));
    }

    #[test]
    fn test_parse_adaptive_circuit() {
        // From qprog.ll -- adaptive algorithm with measurement feedback
        let ir = r"
declare void @__quantum__qis__rz__body(double, i64)
declare void @__quantum__qis__rx__body(double, i64)
declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)

define void @main() {
entry:
  call void @__quantum__qis__rz__body(double 3.14159265359, i64 0)
  call void @__quantum__qis__rx__body(double 3.14159265359, i64 1)
  %m = call i32 @__quantum__qis__m__body(i64 0, i64 2)
  %should_x = icmp eq i32 %m, 1
  br i1 %should_x, label %apply_x, label %skip_x

apply_x:
  call void @__quantum__qis__x__body(i64 1)
  br label %done

skip_x:
  br label %done

done:
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        assert_eq!(module.body.blocks.len(), 4);

        // Entry block has rz, rx, measure, icmp, conditional branch
        let entry_ops: Vec<String> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(entry_ops.contains(&"qis.rz".to_string()));
        assert!(entry_ops.contains(&"qis.rxy".to_string())); // rx decomposes to rxy
        assert!(entry_ops.contains(&"qis.measure".to_string()));
        assert!(entry_ops.contains(&"arith.eq".to_string()));
    }

    #[test]
    fn test_parse_phi_with_arithmetic() {
        // Pattern from ArithmeticOps.Targeted.ll -- phi nodes with add/sub/mul
        let ir = r"
declare void @__quantum__qis__x__body(%Qubit*)
declare void @__quantum__qis__mz__body(%Qubit*, %Result*)
declare void @__quantum__qis__reset__body(%Qubit*)
declare i1 @__quantum__qis__read_result__body(%Result*)

define void @main() {
entry:
  call void @__quantum__qis__x__body(%Qubit* null)
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  call void @__quantum__qis__reset__body(%Qubit* null)
  %0 = call i1 @__quantum__qis__read_result__body(%Result* null)
  br i1 %0, label %then, label %cont

then:
  br label %cont

cont:
  %count = phi i64 [ 1, %then ], [ 0, %entry ]
  %next = add nuw nsw i64 %count, 1
  ret void
}
";
        let module = parse_qis_llvm_ir(ir).unwrap();
        assert_eq!(module.body.blocks.len(), 3);

        // Cont block should have block argument from phi and an add
        let cont = &module.body.blocks[2];
        assert_eq!(cont.arguments.len(), 1);
        let cont_ops: Vec<String> = cont.operations.iter().map(|i| i.operation.name()).collect();
        assert!(cont_ops.contains(&"arith.add".to_string()));
    }
}
