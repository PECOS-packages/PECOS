// Copyright 2024 The PECOS Developers
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

//! Classical interpreter for PHIR programs.
//!
//! This module provides a `PhirClassicalInterpreter` that matches the Python
//! `PhirClassicalInterpreter` protocol. It walks a PHIR program's AST, handles
//! classical operations inline, and yields batches of quantum/machine operations
//! at measurement boundaries.

use crate::v0_1::ast::{ArgItem, Expression, Operation, PHIRProgram, QubitArg, infer_size};
use crate::v0_1::environment::{DataType, Environment};
use crate::v0_1::expression::ExpressionEvaluator;
use crate::v0_1::foreign_objects::ForeignObject;
use crate::v0_1::name_resolver::resolve_sim_name;
use pecos_core::errors::PecosError;
use std::collections::BTreeMap;

/// A quantum operation yielded by the interpreter.
///
/// Matches the fields of Python's `QOp` class.
#[derive(Debug, Clone)]
pub struct YieldedQOp {
    /// Gate name (e.g. "H", "Measure", "RZ")
    pub name: String,
    /// Simulator-specific name (resolved via `name_resolver`)
    pub sim_name: String,
    /// Qubit IDs -- flat list for single-qubit args, nested for multi-qubit
    pub args: QOpArgs,
    /// Measurement return targets, e.g. [["m", 0], ["m", 1]]
    pub returns: Option<Vec<(String, usize)>>,
    /// Metadata dict
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// Rotation angles in radians
    pub angles: Option<Vec<f64>>,
}

/// Arguments for a quantum operation -- either flat qubit IDs or tupled pairs.
#[derive(Debug, Clone)]
pub enum QOpArgs {
    /// Single-qubit args: list of qubit IDs
    Single(Vec<usize>),
    /// Multi-qubit args: list of tuples of qubit IDs
    Multi(Vec<Vec<usize>>),
}

/// A machine operation yielded by the interpreter.
#[derive(Debug, Clone)]
pub struct YieldedMOp {
    pub name: String,
    pub args: Option<QOpArgs>,
    pub returns: Option<Vec<(String, usize)>>,
    pub metadata: Option<BTreeMap<String, serde_json::Value>>,
}

/// A yielded operation (quantum or machine).
#[derive(Debug, Clone)]
pub enum YieldedOp {
    QOp(YieldedQOp),
    MOp(YieldedMOp),
}

/// Metadata about a quantum variable (for resolving qubit args to IDs).
#[derive(Debug, Clone)]
struct QVarMeta {
    /// Starting global qubit ID for this variable
    start_id: usize,
}

/// Classical interpreter for PHIR programs.
///
/// Walks the PHIR AST, executes classical operations (assignment, Result mapping,
/// foreign function calls) inline, and yields batches of quantum/machine operations
/// at measurement boundaries. Matches the Python `PhirClassicalInterpreter` protocol.
pub struct PhirClassicalInterpreter {
    /// The parsed PHIR program
    program: Option<PHIRProgram>,
    /// Classical variable environment
    environment: Environment,
    /// Quantum variable metadata (name -> `QVarMeta`)
    qvar_meta: BTreeMap<String, QVarMeta>,
    /// Total number of qubits
    num_qubits: usize,
    /// Foreign object for `FFCalls`
    foreign_object: Option<Box<dyn ForeignObject>>,
}

impl PhirClassicalInterpreter {
    /// Creates a new interpreter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            program: None,
            environment: Environment::new(),
            qvar_meta: BTreeMap::new(),
            num_qubits: 0,
            foreign_object: None,
        }
    }

    /// Initialize the interpreter with a PHIR JSON program.
    ///
    /// Parses the JSON, extracts variable definitions, initializes the environment.
    /// Returns the number of qubits.
    ///
    /// # Errors
    ///
    /// Returns `PecosError::Input` if the JSON is invalid, the format/version is
    /// unsupported, or a variable definition is malformed.
    pub fn init(
        &mut self,
        json: &str,
        foreign_object: Option<Box<dyn ForeignObject>>,
    ) -> Result<usize, PecosError> {
        let program: PHIRProgram = serde_json::from_str(json)
            .map_err(|e| PecosError::Input(format!("Failed to parse PHIR JSON: {e}")))?;

        // Validate format
        if program.format != "PHIR/JSON" && program.format != "PHIR" {
            return Err(PecosError::Input(format!(
                "Unsupported PHIR format: {}",
                program.format
            )));
        }

        // Validate version < 0.2.0
        let version_parts: Vec<u32> = program
            .version
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        if version_parts.len() >= 2 && (version_parts[0] > 0 || version_parts[1] >= 2) {
            return Err(PecosError::Input(format!(
                "PHIR version {} not supported; only versions < 0.2.0 are supported",
                program.version
            )));
        }

        self.foreign_object = foreign_object;
        self.environment = Environment::new();
        self.qvar_meta.clear();
        self.num_qubits = 0;

        // Process variable definitions from the ops
        for op in &program.ops {
            if let Operation::VariableDefinition {
                data,
                data_type,
                variable,
                size,
            } = op
            {
                let resolved_size = infer_size(data_type, *size);
                match data.as_str() {
                    "qvar_define" if data_type == "qubits" => {
                        let start_id = self.num_qubits;
                        self.qvar_meta
                            .insert(variable.clone(), QVarMeta { start_id });
                        self.num_qubits += resolved_size;
                        // Don't add quantum vars to the classical environment --
                        // they live in qvar_meta only, matching Python behavior
                        // where qvar_meta and csym2id are separate namespaces.
                    }
                    "cvar_define" => {
                        let dt = data_type.parse::<DataType>()?;
                        if !self.environment.has_variable(variable) {
                            self.environment.add_variable(variable, dt, resolved_size)?;
                        }
                    }
                    _ => {}
                }
            }
        }

        self.program = Some(program);
        Ok(self.num_qubits)
    }

    /// Reset variable values for a new shot (keeps definitions).
    pub fn shot_reinit(&mut self) {
        self.environment.reset_values();
    }

    /// Dynamically add a classical variable (used by `HybridEngine` for `__q{i}__` vars).
    ///
    /// # Errors
    ///
    /// Returns `PecosError::Input` if the variable cannot be added to the environment.
    pub fn add_cvar(
        &mut self,
        name: &str,
        data_type: DataType,
        size: usize,
    ) -> Result<(), PecosError> {
        if !self.environment.has_variable(name) {
            self.environment.add_variable(name, data_type, size)?;
        }
        Ok(())
    }

    /// Get a reference to the program ops.
    #[must_use]
    pub fn program_ops(&self) -> &[Operation] {
        self.program.as_ref().map_or(&[], |p| p.ops.as_slice())
    }

    /// Take the program out temporarily for iteration.
    /// Returns the program ops and a guard that puts it back.
    fn take_program(&mut self) -> Option<PHIRProgram> {
        self.program.take()
    }

    /// Put the program back after iteration.
    fn restore_program(&mut self, program: PHIRProgram) {
        self.program = Some(program);
    }

    /// Execute a list of operations, returning an iterator that yields batches
    /// of quantum/machine ops at measurement boundaries.
    ///
    /// For the inner interpreter (noisy ops), pass the noisy op slice directly.
    pub fn execute_ops<'a>(&'a mut self, ops: &'a [Operation]) -> ExecuteIter<'a> {
        ExecuteIter::new(self, ops)
    }

    /// Execute the program, collecting all batches.
    ///
    /// This is a convenience method that handles the borrow issue by
    /// temporarily taking the program out. For the `PyO3` layer, a different
    /// approach is used (the iterator holds state separately).
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if any operation fails during execution.
    pub fn execute_program(&mut self) -> Result<Vec<Vec<YieldedOp>>, PecosError> {
        let program = self.take_program();
        let result = if let Some(ref prog) = program {
            let mut batches = Vec::new();
            let mut iter = ExecuteIter::new(self, &prog.ops);
            while let Some(batch) = iter.next_batch()? {
                batches.push(batch);
            }
            Ok(batches)
        } else {
            Ok(Vec::new())
        };
        if let Some(prog) = program {
            self.restore_program(prog);
        }
        result
    }

    /// Receive measurement results and store in classical variables.
    ///
    /// Each dict entry maps either:
    /// - `(cvar_name, bit_idx)` -> value (bit-level assignment)
    /// - `cvar_name` -> value (whole variable assignment)
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if a variable name is unknown or a bit index is out of range.
    pub fn receive_results(
        &mut self,
        results: &[BTreeMap<MeasKey, i64>],
    ) -> Result<(), PecosError> {
        for meas_dict in results {
            for (key, val) in meas_dict {
                match key {
                    MeasKey::Var(name) => {
                        #[allow(clippy::cast_sign_loss)]
                        self.assign_int_var(name, *val as u64)?;
                    }
                    MeasKey::Bit(name, idx) => {
                        self.assign_int_bit(name, *idx, *val)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract specific bits from classical state, filtering private vars.
    ///
    /// Returns `{(name, bit_idx): bit_value}` for each measurement bit,
    /// excluding variables that start with `__`.
    #[must_use]
    pub fn result_bits(
        &self,
        measurements: &[BTreeMap<(String, usize), i64>],
    ) -> BTreeMap<(String, usize), i64> {
        let mut result = BTreeMap::new();
        for meas_dict in measurements {
            for key in meas_dict.keys() {
                let (name, idx) = key;
                // Filter private vars
                if name.starts_with("__") {
                    continue;
                }
                if let Ok(bit) = self.environment.get_bit(name, *idx) {
                    result.insert((name.clone(), *idx), i64::from(bool::from(bit)));
                }
            }
        }
        result
    }

    /// Return all classical variable values.
    ///
    /// If `return_int` is true, returns integer values.
    /// If false, returns zero-padded binary strings.
    #[must_use]
    pub fn results(&self, return_int: bool) -> BTreeMap<String, ResultValue> {
        let mut result = BTreeMap::new();
        for info in self.environment.get_all_variables() {
            if info.data_type == DataType::Qubits {
                continue;
            }
            if let Some(val) = self.environment.get(&info.name) {
                if return_int {
                    let dtype_name = info.data_type.to_string();
                    if info.data_type.is_signed() {
                        result.insert(
                            info.name.clone(),
                            ResultValue::Int(val.as_i64(), dtype_name),
                        );
                    } else {
                        result.insert(
                            info.name.clone(),
                            ResultValue::UInt(val.as_u64(), dtype_name),
                        );
                    }
                } else {
                    // Match Python: format(cval, '0{size}b')
                    //
                    // Python stores values in PECOS signed dtypes. When the value
                    // is negative (determined by the TYPE's sign interpretation),
                    // format() shows "-" prefix + magnitude. We use as_i64() which
                    // sign-extends from the register's bit width -- this matches
                    // Python behavior because Python stores the sign-extended value
                    // in the full-width dtype.
                    let signed_val = val.as_i64();
                    let bits = if info.data_type.is_signed() && signed_val < 0 {
                        format!(
                            "-{:0>width$b}",
                            signed_val.unsigned_abs(),
                            width = info.size
                        )
                    } else {
                        format!("{:0>width$b}", val.as_u64(), width = info.size)
                    };
                    result.insert(info.name.clone(), ResultValue::BitString(bits));
                }
            }
        }
        result
    }

    /// Get the number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Resolve a `QubitArg` to integer qubit IDs.
    fn resolve_qubit_arg(&self, arg: &QubitArg) -> Result<Vec<usize>, PecosError> {
        match arg {
            QubitArg::SingleQubit((var, idx)) => {
                let meta = self
                    .qvar_meta
                    .get(var)
                    .ok_or_else(|| PecosError::Input(format!("Unknown quantum variable: {var}")))?;
                Ok(vec![meta.start_id + idx])
            }
            QubitArg::MultipleQubits(qubits) => {
                let mut ids = Vec::new();
                for (var, idx) in qubits {
                    let meta = self.qvar_meta.get(var).ok_or_else(|| {
                        PecosError::Input(format!("Unknown quantum variable: {var}"))
                    })?;
                    ids.push(meta.start_id + idx);
                }
                Ok(ids)
            }
        }
    }

    /// Convert a `QuantumOp` AST node to a `YieldedQOp`.
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if qubit args reference unknown quantum variables.
    pub fn make_qop(
        &self,
        qop_name: &str,
        angles: &Option<Vec<f64>>,
        args: &[QubitArg],
        returns: &[(String, usize)],
        metadata: &Option<BTreeMap<String, serde_json::Value>>,
    ) -> Result<YieldedQOp, PecosError> {
        // Resolve qubit args to integer IDs
        let mut is_multi = false;
        for arg in args {
            if matches!(arg, QubitArg::MultipleQubits(_)) {
                is_multi = true;
                break;
            }
        }

        let resolved_args = if is_multi {
            let mut multi = Vec::new();
            for arg in args {
                multi.push(self.resolve_qubit_arg(arg)?);
            }
            QOpArgs::Multi(multi)
        } else {
            let mut flat = Vec::new();
            for arg in args {
                flat.extend(self.resolve_qubit_arg(arg)?);
            }
            QOpArgs::Single(flat)
        };

        // Build metadata
        let mut meta = metadata.clone().unwrap_or_default();
        if let Some(angs) = angles {
            if angs.len() == 1 {
                meta.insert("angle".to_string(), serde_json::json!(angs[0]));
            } else {
                meta.insert("angles".to_string(), serde_json::json!(angs));
            }
        }

        // Build var_output for measurement compatibility
        if !returns.is_empty() {
            let flat_args: Vec<usize> = match &resolved_args {
                QOpArgs::Single(ids) => ids.clone(),
                QOpArgs::Multi(groups) => groups.iter().flatten().copied().collect(),
            };
            let mut var_output = serde_json::Map::new();
            for (q, r) in flat_args.iter().zip(returns.iter()) {
                var_output.insert(q.to_string(), serde_json::json!([r.0, r.1]));
            }
            meta.insert(
                "var_output".to_string(),
                serde_json::Value::Object(var_output),
            );
        }

        let sim_name = resolve_sim_name(qop_name, angles.as_deref());

        let ret = if returns.is_empty() {
            None
        } else {
            Some(returns.to_vec())
        };

        Ok(YieldedQOp {
            name: qop_name.to_string(),
            sim_name,
            args: resolved_args,
            returns: ret,
            metadata: meta,
            angles: angles.clone(),
        })
    }

    /// Convert a `MachineOp` AST node to a `YieldedMOp`.
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if qubit args reference unknown quantum variables.
    pub fn make_mop(
        &self,
        mop_name: &str,
        args: &Option<Vec<QubitArg>>,
        duration: &Option<(f64, String)>,
        metadata: &Option<BTreeMap<String, serde_json::Value>>,
    ) -> Result<YieldedMOp, PecosError> {
        let resolved_args = if let Some(qargs) = args {
            let mut is_multi = false;
            for arg in qargs {
                if matches!(arg, QubitArg::MultipleQubits(_)) {
                    is_multi = true;
                    break;
                }
            }
            if is_multi {
                let mut multi = Vec::new();
                for arg in qargs {
                    multi.push(self.resolve_qubit_arg(arg)?);
                }
                Some(QOpArgs::Multi(multi))
            } else {
                let mut flat = Vec::new();
                for arg in qargs {
                    flat.extend(self.resolve_qubit_arg(arg)?);
                }
                Some(QOpArgs::Single(flat))
            }
        } else {
            None
        };

        // Include duration in metadata if present
        let mut meta = metadata.clone();
        if let Some((val, unit)) = duration {
            let m = meta.get_or_insert_with(BTreeMap::new);
            m.insert("duration".to_string(), serde_json::json!([val, unit]));
        }

        Ok(YieldedMOp {
            name: mop_name.to_string(),
            args: resolved_args,
            returns: None,
            metadata: meta,
        })
    }

    /// Assign an integer value to a whole classical variable.
    ///
    /// The Environment's `BitValue` storage automatically masks to the declared
    /// bit width, so no manual masking is needed.
    fn assign_int_var(&mut self, name: &str, val: u64) -> Result<(), PecosError> {
        if self.environment.has_variable(name) {
            self.environment.set_raw(name, val)?;
        }
        Ok(())
    }

    /// Assign a bit value to a specific bit of a classical variable.
    fn assign_int_bit(&mut self, name: &str, idx: usize, val: i64) -> Result<(), PecosError> {
        if self.environment.has_variable(name) {
            self.environment.set_bit(name, idx, (val & 1) != 0)?;
        }
        Ok(())
    }

    /// Evaluate an expression using the current environment.
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if the expression references unknown variables or
    /// contains unsupported operations.
    pub fn eval_expr(&self, expr: &Expression) -> Result<i64, PecosError> {
        let mut evaluator = ExpressionEvaluator::new(&self.environment);
        let result = evaluator.eval_expr(expr)?;
        Ok(result.as_i64())
    }

    /// Evaluate an `ArgItem`.
    fn eval_arg(&self, arg: &ArgItem) -> Result<i64, PecosError> {
        let mut evaluator = ExpressionEvaluator::new(&self.environment);
        let result = evaluator.eval_arg(arg)?;
        Ok(result.as_i64())
    }

    /// Handle a classical operation (assignment, Result, `FFCall`).
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if the operation references unknown variables,
    /// a foreign function call fails, or the operation is unsupported.
    pub fn handle_cop(
        &mut self,
        cop: &str,
        args: &[ArgItem],
        returns: &[ArgItem],
        function: &Option<String>,
    ) -> Result<(), PecosError> {
        match cop {
            "=" => {
                // Evaluate ALL args first (before any assignment),
                // then assign to corresponding returns.
                // This matches Python: args = [eval(a) for a in op.args]
                let values: Vec<i64> =
                    args.iter()
                        .map(|arg| self.eval_arg(arg))
                        .collect::<Result<Vec<_>, _>>()?;

                for (val, ret) in values.into_iter().zip(returns.iter()) {
                    match ret {
                        ArgItem::Simple(var) => {
                            if !self.environment.has_variable(var) {
                                self.environment.add_variable(var, DataType::I32, 32)?;
                            }
                            #[allow(clippy::cast_sign_loss)]
                            self.environment.set_raw(var, val as u64)?;
                        }
                        ArgItem::Indexed((var, idx)) => {
                            if !self.environment.has_variable(var) {
                                self.environment.add_variable(var, DataType::I32, 32)?;
                            }
                            self.environment.set_bit(var, *idx, (val & 1) != 0)?;
                        }
                        _ => {
                            return Err(PecosError::Input(
                                "Assignment target must be a variable".to_string(),
                            ));
                        }
                    }
                }
            }
            "Result" => {
                // Map source register to destination register
                for (src_arg, dst_arg) in args.iter().zip(returns.iter()) {
                    let src_name = match src_arg {
                        ArgItem::Simple(name) | ArgItem::Indexed((name, _)) => name.clone(),
                        _ => {
                            return Err(PecosError::Input(
                                "Result source must be a variable".to_string(),
                            ));
                        }
                    };
                    let dst_name = match dst_arg {
                        ArgItem::Simple(name) | ArgItem::Indexed((name, _)) => name.clone(),
                        _ => {
                            return Err(PecosError::Input(
                                "Result destination must be a variable".to_string(),
                            ));
                        }
                    };
                    // Copy variable value from source to destination
                    self.environment.copy_variable(&src_name, &dst_name)?;
                }
            }
            "ffcall" => {
                let func_name = function
                    .as_ref()
                    .ok_or_else(|| PecosError::Input("FFCall missing function name".to_string()))?;

                let foreign_obj = self.foreign_object.as_ref().ok_or_else(|| {
                    PecosError::Input(format!(
                        "Trying to call foreign function `{func_name}` but no foreign object supplied!"
                    ))
                })?;

                // Evaluate arguments
                let mut call_args = Vec::new();
                for arg in args {
                    call_args.push(self.eval_arg(arg)?);
                }

                // Execute
                let mut fo = foreign_obj.clone_box();
                let result = fo.exec(func_name, &call_args)?;

                // Assign return values
                for (i, ret) in returns.iter().enumerate() {
                    if i < result.len() {
                        match ret {
                            ArgItem::Simple(var) => {
                                if !self.environment.has_variable(var) {
                                    self.environment.add_variable(var, DataType::I32, 32)?;
                                }
                                #[allow(clippy::cast_sign_loss)]
                                self.environment.set_raw(var, result[i] as u64)?;
                            }
                            ArgItem::Indexed((var, idx)) => {
                                if !self.environment.has_variable(var) {
                                    self.environment.add_variable(var, DataType::I32, 32)?;
                                }
                                self.environment.set_bit(var, *idx, (result[i] & 1) != 0)?;
                            }
                            _ => {
                                return Err(PecosError::Input(
                                    "FFCall return must be a variable".to_string(),
                                ));
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(PecosError::Input(format!(
                    "Unsupported classical operation: {cop}"
                )));
            }
        }
        Ok(())
    }
}

impl Default for PhirClassicalInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// Key for measurement results -- either a whole var or a specific bit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeasKey {
    Var(String),
    Bit(String, usize),
}

/// Result value -- either a typed integer or a bit string.
#[derive(Debug, Clone)]
pub enum ResultValue {
    /// Signed integer value with its PHIR data type name (e.g. "i32", "i64")
    Int(i64, String),
    /// Unsigned integer value with its PHIR data type name (e.g. "u32", "u64")
    UInt(u64, String),
    BitString(String),
}

// ── Execution iterator ──────────────────────────────────────────────

/// Stack frame for the iterative block flattener: a slice of operations and current index.
struct StackFrame<'a>(&'a [Operation], usize);

/// Iterator that walks the PHIR program and yields batches of quantum/machine
/// operations at measurement boundaries.
pub struct ExecuteIter<'a> {
    interp: &'a mut PhirClassicalInterpreter,
    /// Stack of operation slices being processed.
    stack: Vec<StackFrame<'a>>,
    /// Buffer of yielded ops accumulated until a measurement.
    buffer: Vec<YieldedOp>,
    /// Whether we've finished.
    done: bool,
}

impl<'a> ExecuteIter<'a> {
    fn new(interp: &'a mut PhirClassicalInterpreter, ops: &'a [Operation]) -> Self {
        Self {
            interp,
            stack: vec![StackFrame(ops, 0)],
            buffer: Vec::new(),
            done: false,
        }
    }

    /// Advance to the next yield point (measurement boundary or end of program).
    ///
    /// Returns `Some(batch)` with the buffered ops, or `None` when done.
    ///
    /// # Errors
    ///
    /// Returns `PecosError` if any operation fails during execution.
    pub fn next_batch(&mut self) -> Result<Option<Vec<YieldedOp>>, PecosError> {
        if self.done {
            return Ok(None);
        }

        loop {
            // Pop the top stack frame
            let Some(frame) = self.stack.last_mut() else {
                // Stack empty -- yield remaining buffer if non-empty
                self.done = true;
                if self.buffer.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(std::mem::take(&mut self.buffer)));
            };

            let StackFrame(ops, idx) = frame;

            if *idx >= ops.len() {
                // This frame is exhausted, pop it
                self.stack.pop();
                continue;
            }

            let op = &ops[*idx];
            *idx += 1;

            match op {
                Operation::VariableDefinition { .. }
                | Operation::DataExport { .. }
                | Operation::Comment { .. } => {
                    // Skip: var defs already processed during init,
                    // data exports and comments are no-ops
                }
                Operation::MetaInstruction { meta, .. } => {
                    if meta == "barrier" {
                        // Skip barriers (same as Python)
                    }
                }
                Operation::QuantumOp {
                    qop,
                    angles,
                    args,
                    returns,
                    metadata,
                } => {
                    let yielded = self.interp.make_qop(qop, angles, args, returns, metadata)?;
                    let is_measure = matches!(qop.as_str(), "measure Z" | "Measure" | "Measure +Z");
                    self.buffer.push(YieldedOp::QOp(yielded));

                    if is_measure {
                        return Ok(Some(std::mem::take(&mut self.buffer)));
                    }
                }
                Operation::MachineOp {
                    mop,
                    args,
                    duration,
                    metadata,
                } => {
                    let yielded = self.interp.make_mop(mop, args, duration, metadata)?;
                    self.buffer.push(YieldedOp::MOp(yielded));
                }
                Operation::ClassicalOp {
                    cop,
                    args,
                    returns,
                    function,
                    ..
                } => {
                    self.interp.handle_cop(cop, args, returns, function)?;
                }
                Operation::Block {
                    block,
                    ops: block_ops,
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => match block.as_str() {
                    "sequence" => {
                        self.stack.push(StackFrame(block_ops, 0));
                    }
                    "qparallel" => {
                        // Treat like sequence (Python does the same in _flatten_blocks)
                        self.stack.push(StackFrame(block_ops, 0));
                    }
                    "if" => {
                        let condition = condition.as_ref().ok_or_else(|| {
                            PecosError::Input("If block missing condition".to_string())
                        })?;
                        let cond_val = self.interp.eval_expr(condition)?;
                        if cond_val != 0 {
                            if let Some(tb) = true_branch {
                                self.stack.push(StackFrame(tb, 0));
                            }
                        } else if let Some(fb) = false_branch {
                            self.stack.push(StackFrame(fb, 0));
                        }
                    }
                    other => {
                        return Err(PecosError::Input(format!("Unknown block type: {other}")));
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_PROGRAM: &str = r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["c"]}
        ]
    }"#;

    #[test]
    fn test_init() {
        let mut interp = PhirClassicalInterpreter::new();
        let nq = interp.init(SIMPLE_PROGRAM, None).unwrap();
        assert_eq!(nq, 2);
        assert_eq!(interp.num_qubits(), 2);
    }

    #[test]
    fn test_execute_yields_batches() {
        let mut interp = PhirClassicalInterpreter::new();
        interp.init(SIMPLE_PROGRAM, None).unwrap();

        let batches = interp.execute_program().unwrap();
        // Should have one batch (H + Measure), Result handled inline
        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.len(), 2);
        assert!(matches!(&batch[0], YieldedOp::QOp(q) if q.name == "H"));
        assert!(matches!(&batch[1], YieldedOp::QOp(q) if q.name == "Measure"));
    }

    #[test]
    fn test_receive_results() {
        let mut interp = PhirClassicalInterpreter::new();
        interp.init(SIMPLE_PROGRAM, None).unwrap();

        // Simulate receiving measurement results
        let mut meas = BTreeMap::new();
        meas.insert(MeasKey::Bit("m".to_string(), 0), 1i64);
        meas.insert(MeasKey::Bit("m".to_string(), 1), 0i64);
        interp.receive_results(&[meas]).unwrap();

        let results = interp.results(true);
        let m_val = results.get("m").unwrap();
        match m_val {
            ResultValue::Int(v, _) => assert_eq!(*v, 1), // bit 0 = 1, bit 1 = 0 -> value = 1
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_result_bits_filters_private() {
        let mut interp = PhirClassicalInterpreter::new();
        interp.init(SIMPLE_PROGRAM, None).unwrap();

        // Add private var
        interp.add_cvar("__q0__", DataType::I64, 1).unwrap();

        // Set some values
        interp.environment.set_raw("m", 3).unwrap();
        interp.environment.set_raw("__q0__", 1).unwrap();

        let mut meas = BTreeMap::new();
        meas.insert(("m".to_string(), 0usize), 1i64);
        meas.insert(("__q0__".to_string(), 0usize), 1i64);

        let bits = interp.result_bits(&[meas]);
        // Should include m but NOT __q0__
        assert!(bits.contains_key(&("m".to_string(), 0)));
        assert!(!bits.contains_key(&("__q0__".to_string(), 0)));
    }
}
