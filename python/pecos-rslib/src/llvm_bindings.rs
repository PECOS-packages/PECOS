// Copyright 2024 The PECOS Developers
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

// Allow non-snake-case for functions that match Python's llvmlite API
#![allow(non_snake_case)]

//! Python bindings for LLVM IR generation
//!
//! This module provides Python classes for LLVM IR generation that are compatible
//! with Python's llvmlite API, enabling quantum IR code generation in Python.
//!
//! Usage in Python:
//! ```python
//! from pecos_rslib.llvm import ir, binding
//!
//! module = ir.Module("my_module")
//! # Create LLVM IR using a familiar API
//! ```

use pecos::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// Import inkwell types directly
use inkwell::context::Context;

// ============================================================================
// Comment Tracking
// ============================================================================

/// Represents a comment to be injected into the LLVM IR
#[derive(Clone, Debug)]
struct TrackedComment {
    /// The basic block name where this comment should appear
    block_name: String,
    /// The index of the instruction after which this comment should appear
    instruction_index: usize,
    /// The comment text
    text: String,
}

/// Global comment storage - maps module pointers to their comments
static GLOBAL_COMMENTS: OnceLock<Mutex<HashMap<usize, Vec<TrackedComment>>>> = OnceLock::new();

/// Get or initialize the global comments storage
fn global_comments() -> &'static Mutex<HashMap<usize, Vec<TrackedComment>>> {
    GLOBAL_COMMENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Inject comments into LLVM IR string at the appropriate positions
fn inject_comments(ir: &str, comments: &[TrackedComment]) -> String {
    let mut result = String::new();
    let mut current_block: Option<String> = None;
    let mut instruction_count = 0;

    // Group comments by (block_name, instruction_index) for efficient lookup
    let mut comment_map: HashMap<(String, usize), Vec<String>> = HashMap::new();
    for comment in comments {
        comment_map
            .entry((comment.block_name.clone(), comment.instruction_index))
            .or_default()
            .push(comment.text.clone());
    }

    for line in ir.lines() {
        let trimmed = line.trim();

        // Detect block label (e.g., "entry:" or "if.then:")
        if !trimmed.is_empty() && trimmed.ends_with(':') && !trimmed.starts_with(';') {
            // Extract block name (remove trailing colon and any attributes)
            if let Some(block_name) = trimmed.split(':').next() {
                current_block = Some(block_name.trim().to_string());
                instruction_count = 0;
            }
        }

        // Check if this line is an instruction
        // Instructions typically start with "%" (result) or are calls/stores/etc
        let is_instruction = !trimmed.is_empty()
            && !trimmed.starts_with(';')  // Not a comment
            && !trimmed.ends_with(':')    // Not a label
            && !trimmed.starts_with("declare ")  // Not a declaration
            && !trimmed.starts_with("define ")   // Not a function definition
            && !trimmed.starts_with("attributes ")  // Not attributes
            && !trimmed.contains("ModuleID")  // Not module metadata
            && !trimmed.starts_with('@')  // Not a global variable definition
            && !trimmed.starts_with('}')  // Not end of block
            && (trimmed.starts_with('%') || trimmed.starts_with("call ") || trimmed.starts_with("ret ") || trimmed.contains(" = "));

        // BEFORE adding an instruction, check if we should inject comments
        if is_instruction && let Some(ref block_name) = current_block {
            // Look for comments that should appear before this instruction
            if let Some(comment_texts) = comment_map.get(&(block_name.clone(), instruction_count)) {
                for comment_text in comment_texts {
                    // Inject the comment with proper indentation
                    result.push_str("  ");
                    result.push_str(comment_text);
                    result.push('\n');
                }
            }
            instruction_count += 1;
        }

        // Add the original line
        result.push_str(line);
        result.push('\n');
    }

    result
}

// ============================================================================
// Module - Core LLVM module with owned context
// ============================================================================

/// Python wrapper for LLVM Module
///
/// Provides LLVM module creation and management (compatible with llvmlite API)
#[pyclass(name = "Module")]
pub struct PyLLVMModule {
    // We use Box::leak for 'static lifetime (safe for FFI)
    context_ptr: *mut Context,
    module_ptr: *mut LLModule<'static>,
}

// SAFETY: Python's GIL ensures single-threaded access
unsafe impl Send for PyLLVMModule {}
unsafe impl Sync for PyLLVMModule {}

impl Drop for PyLLVMModule {
    fn drop(&mut self) {
        unsafe {
            if !self.module_ptr.is_null() {
                let _ = Box::from_raw(self.module_ptr);
            }
            if !self.context_ptr.is_null() {
                let _ = Box::from_raw(self.context_ptr);
            }
        }
    }
}

#[pymethods]
impl PyLLVMModule {
    #[new]
    fn new(name: &str) -> Self {
        // Create and leak context for 'static lifetime
        let context = Box::new(Context::create());
        let context_ptr = Box::into_raw(context);
        let context_ref: &'static Context = unsafe { &*context_ptr };

        // Create and leak module
        let module = Box::new(LLModule::new(context_ref, name));
        let module_ptr = Box::into_raw(module);

        // Initialize comment storage for this module
        let module_id = module_ptr as usize;
        global_comments()
            .lock()
            .unwrap()
            .insert(module_id, Vec::new());

        Self {
            context_ptr,
            module_ptr,
        }
    }

    /// Get module as LLVM IR string (mirrors str(module) in llvmlite)
    fn __str__(&self) -> String {
        let base_ir = unsafe { (*self.module_ptr).to_string() };
        let module_id = self.module_ptr as usize;

        // Get comments for this module
        let comments = global_comments()
            .lock()
            .unwrap()
            .get(&module_id)
            .cloned()
            .unwrap_or_default();

        let ir_with_comments = if comments.is_empty() {
            base_ir
        } else {
            inject_comments(&base_ir, &comments)
        };

        // Format compatibility layer: LLVM's AsmWriter hardcodes certain output formats
        // that differ from llvmlite's text generation. Since there are no API options to
        // control these formats, we apply minimal replacements for compatibility.
        //
        // These replacements are necessary because:
        // 1. LLVM's AsmWriter.cpp hardcodes: if (isIntegerTy(1)) Out << "true"/"false"
        //    llvmlite generates: "1"/"0"
        // 2. LLVM optimizes zero pointers to "null"
        //    llvmlite keeps explicit: "inttoptr (i64 0 to ...)"
        // 3. LLVM uses sequential SSA names: %0, %1, %2, ...
        //    llvmlite uses even numbers: %.2, %.4, %.6, ...
        //    (llvmlite's NameScope generates .1, .2, .3 but skips names, so unnamed values get even numbers)
        //
        // Both formats are semantically identical and valid LLVM IR.

        let ir = ir_with_comments
            .replace("i1 true", "i1 1")
            .replace("i1 false", "i1 0");

        // Replace "TYPE* null" with "TYPE* inttoptr (i64 0 to TYPE*)" for any pointer type
        // Handles both named types (%Qubit*) and built-in types (i8*, i64*, etc.)
        let null_ptr_re = Regex::new(r"(%?\w+\*) null").unwrap();
        let ir = null_ptr_re
            .replace_all(&ir, "$1 inttoptr (i64 0 to $1)")
            .to_string();

        // Replace LLVM's sequential SSA names (%0, %1, %2) with llvmlite's even-numbered names (%.2, %.4, %.6)
        // llvmlite's NameScope increments for all operations including comments.
        //
        // The formula needs to match llvmlite's behavior:
        // 1. llvmlite's function setup consumes .1
        // 2. In typical QIR generation, there's a "Generated using" comment that consumes .2
        // 3. Then permutation comments consume additional numbers
        // 4. llvmlite uses even numbers
        //
        // For gen_qir.py specifically, which adds a "Generated using" comment at the start:
        // - .1 is consumed by function setup
        // - .2 is consumed by "Generated using" comment
        // - First unnamed value gets .4 (skip .3 for next comment if any)
        // So: %0 → %.4, %1 → %.6, %2 → %.8, etc.
        // Formula: %n → %.{(n + 2) * 2}
        let ssa_re = Regex::new(r"%(\d+)([^0-9a-zA-Z_])").unwrap();
        ssa_re
            .replace_all(&ir, |caps: &regex::Captures| {
                let num: usize = caps[1].parse().unwrap();
                let suffix = &caps[2];
                // Offset by 2 to account for function setup (.1) and "Generated using" comment (.2)
                format!("%.{}{}", (num + 2) * 2, suffix)
            })
            .to_string()
    }

    /// Get module as LLVM IR string (mirrors repr(module) in llvmlite)
    #[allow(clippy::unused_self)]
    fn __repr__(&self) -> String {
        "<LLVM Module>".to_string()
    }

    /// Get the module's context property
    ///
    /// Returns a `PyModuleContext` that provides access to type creation methods
    #[getter]
    fn context(&self) -> PyModuleContext {
        PyModuleContext {
            context_ptr: self.context_ptr,
        }
    }

    /// Get global variables (stub for now - implement if needed)
    #[getter]
    #[allow(clippy::unused_self)]
    fn globals(&self) -> Vec<String> {
        // TODO: Implement if gen_qir.py needs it
        Vec::new()
    }

    /// Add a function to the module
    ///
    /// Mirrors `module.add_function(name`, `func_type`)
    fn add_function(&mut self, name: &str, func_type: &PyFunctionType) -> PyFunction {
        let module = unsafe { &mut *self.module_ptr };
        let context = unsafe { &*self.context_ptr };
        // Reconstruct the LLFunctionType from components
        let fn_ty = LLFunctionType::new_with_context(
            context,
            func_type.ret_type,
            &func_type.param_types,
            func_type.var_args,
        );
        let ll_function = module.add_function(name, fn_ty);
        PyFunction {
            function: ll_function.get(), // Get the underlying FunctionValue
            context_ptr: self.context_ptr,
            module_id: self.module_ptr as usize,
        }
    }

    /// Add a global variable to the module
    ///
    /// Mirrors ir.GlobalVariable(module, type, name)
    fn add_global(
        &mut self,
        name: &str,
        ty: PyAnyType,
        initializer: Option<PyLLValue>,
    ) -> PyGlobalVariable {
        let module = unsafe { &mut *self.module_ptr };
        let context = unsafe { &*self.context_ptr };
        let ll_type = ty.to_ll_type(context);
        let init_val = initializer.map(|v| v.value);
        let global = module.add_global(name, ll_type, init_val);
        PyGlobalVariable {
            global,
            context_ptr: self.context_ptr,
        }
    }
}

// ============================================================================
// ModuleContext - Provides type creation methods
// ============================================================================

/// Python wrapper for module.context
///
/// Provides access to type creation like `module.context.get_identified_type()`
#[pyclass(name = "ModuleContext", from_py_object)]
#[derive(Clone)]
pub struct PyModuleContext {
    context_ptr: *mut Context,
}

unsafe impl Send for PyModuleContext {}
unsafe impl Sync for PyModuleContext {}

#[pymethods]
impl PyModuleContext {
    /// Get or create an identified (opaque) struct type
    ///
    /// Mirrors `module.context.get_identified_type(name)`
    fn get_identified_type(&self, name: &str) -> PyStructType {
        let context = unsafe { &*self.context_ptr };
        let struct_type = context.opaque_struct_type(name);
        PyStructType {
            struct_type,
            context_ptr: self.context_ptr,
        }
    }

    /// Create integer type
    fn int_type(&self, bits: u32) -> PyIntType {
        let context = unsafe { &*self.context_ptr };
        let ll_type = LLType::int(context, bits);
        PyIntType {
            ll_type,
            context_ptr: self.context_ptr,
        }
    }

    /// Create void type
    fn void_type(&self) -> PyVoidType {
        PyVoidType {
            context_ptr: self.context_ptr,
        }
    }

    /// Create double (f64) type
    fn double_type(&self) -> PyDoubleType {
        let context = unsafe { &*self.context_ptr };
        let ll_type = LLType::double(context);
        PyDoubleType {
            ll_type,
            context_ptr: self.context_ptr,
        }
    }

    /// Create function type
    fn function_type(
        &self,
        return_type: PyAnyType,
        param_types: Vec<PyAnyType>,
        is_var_arg: Option<bool>,
    ) -> PyFunctionType {
        let context = unsafe { &*self.context_ptr };
        let ret_ty = return_type.to_ll_type(context);
        let param_tys: Vec<_> = param_types
            .into_iter()
            .map(|pt| pt.to_ll_type(context))
            .collect();

        PyFunctionType {
            ret_type: ret_ty,
            param_types: param_tys,
            var_args: is_var_arg.unwrap_or(false),
            context_ptr: self.context_ptr,
        }
    }
}

// ============================================================================
// Type Classes
// ============================================================================

/// Enum to handle any type for function parameters
#[derive(Copy, Clone, FromPyObject)]
pub enum PyAnyType {
    Int(PyIntType),
    Double(PyDoubleType),
    Void(PyVoidType),
    Pointer(PyPointerType),
    Struct(PyStructType),
    Array(PyArrayType),
}

impl PyAnyType {
    fn to_ll_type(self, _context: &Context) -> LLType<'static> {
        match self {
            PyAnyType::Int(t) => t.ll_type,
            PyAnyType::Double(t) => t.ll_type,
            PyAnyType::Void(_) => LLType::Void,
            PyAnyType::Pointer(t) => t.ll_type,
            PyAnyType::Struct(t) => LLType::Struct(t.struct_type),
            PyAnyType::Array(t) => t.ll_type,
        }
    }
}

/// Python wrapper for struct types
#[pyclass(name = "StructType", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyStructType {
    struct_type: inkwell::types::StructType<'static>,
    context_ptr: *mut Context,
}

unsafe impl Send for PyStructType {}
unsafe impl Sync for PyStructType {}

#[pymethods]
impl PyStructType {
    /// Convert to pointer type (mirrors `type.as_pointer()` in llvmlite)
    fn as_pointer(&self) -> PyPointerType {
        let context = unsafe { &*self.context_ptr };
        let ll_type = LLType::Struct(self.struct_type);
        let ptr_type = ll_type.as_pointer(context);
        PyPointerType {
            ll_type: ptr_type,
            context_ptr: self.context_ptr,
        }
    }
}

/// Python wrapper for pointer types
#[pyclass(name = "PointerType", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyPointerType {
    ll_type: LLType<'static>,
    context_ptr: *mut Context,
}

unsafe impl Send for PyPointerType {}
unsafe impl Sync for PyPointerType {}

#[pymethods]
impl PyPointerType {
    fn as_pointer(&self) -> PyPointerType {
        let context = unsafe { &*self.context_ptr };
        let ptr_type = self.ll_type.as_pointer(context);
        PyPointerType {
            ll_type: ptr_type,
            context_ptr: self.context_ptr,
        }
    }
}

/// Python wrapper for integer types
#[pyclass(name = "IntType", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyIntType {
    ll_type: LLType<'static>,
    context_ptr: *mut Context,
}

unsafe impl Send for PyIntType {}
unsafe impl Sync for PyIntType {}

#[pymethods]
impl PyIntType {
    fn as_pointer(&self) -> PyPointerType {
        let context = unsafe { &*self.context_ptr };
        let ptr_type = self.ll_type.as_pointer(context);
        PyPointerType {
            ll_type: ptr_type,
            context_ptr: self.context_ptr,
        }
    }

    fn as_array(&self, count: u32) -> PyArrayType {
        let _context = unsafe { &*self.context_ptr };
        let array_type = LLType::array(self.ll_type, count);
        PyArrayType {
            ll_type: array_type,
            context_ptr: self.context_ptr,
        }
    }
}

/// Python wrapper for float types
#[pyclass(name = "DoubleType", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyDoubleType {
    ll_type: LLType<'static>,
    context_ptr: *mut Context,
}

unsafe impl Send for PyDoubleType {}
unsafe impl Sync for PyDoubleType {}

#[pymethods]
impl PyDoubleType {
    fn as_pointer(&self) -> PyPointerType {
        let context = unsafe { &*self.context_ptr };
        let ptr_type = self.ll_type.as_pointer(context);
        PyPointerType {
            ll_type: ptr_type,
            context_ptr: self.context_ptr,
        }
    }

    fn as_array(&self, count: u32) -> PyArrayType {
        let _context = unsafe { &*self.context_ptr };
        let array_type = LLType::array(self.ll_type, count);
        PyArrayType {
            ll_type: array_type,
            context_ptr: self.context_ptr,
        }
    }
}

/// Python wrapper for array types
#[pyclass(name = "ArrayType", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyArrayType {
    ll_type: LLType<'static>,
    context_ptr: *mut Context,
}

unsafe impl Send for PyArrayType {}
unsafe impl Sync for PyArrayType {}

#[pymethods]
impl PyArrayType {
    #[new]
    fn new(element_type: PyAnyType, count: u32) -> Self {
        // Extract context pointer from element type
        let context_ptr = match &element_type {
            PyAnyType::Int(t) => t.context_ptr,
            PyAnyType::Double(t) => t.context_ptr,
            PyAnyType::Void(t) => t.context_ptr,
            PyAnyType::Pointer(t) => t.context_ptr,
            PyAnyType::Struct(t) => t.context_ptr,
            PyAnyType::Array(t) => t.context_ptr,
        };

        let context = unsafe { &*context_ptr };
        let elem_ty = element_type.to_ll_type(context);
        let ll_type = LLType::array(elem_ty, count);

        Self {
            ll_type,
            context_ptr,
        }
    }

    fn as_pointer(&self) -> PyPointerType {
        let context = unsafe { &*self.context_ptr };
        let ptr_type = self.ll_type.as_pointer(context);
        PyPointerType {
            ll_type: ptr_type,
            context_ptr: self.context_ptr,
        }
    }
}

/// Python wrapper for void type
#[pyclass(name = "VoidType", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyVoidType {
    context_ptr: *mut Context,
}

unsafe impl Send for PyVoidType {}
unsafe impl Sync for PyVoidType {}

// ============================================================================
// IRBuilder - Instruction builder
// ============================================================================

/// Python wrapper for LLVM IR instruction builder
///
/// Provides LLVM IR instruction building (compatible with llvmlite API)
#[pyclass(name = "IRBuilder")]
pub struct PyIRBuilder {
    builder_ptr: *mut LLIRBuilder<'static>,
    context_ptr: *mut Context,
    /// Module ID for comment tracking (module pointer as usize)
    module_id: usize,
}

unsafe impl Send for PyIRBuilder {}
unsafe impl Sync for PyIRBuilder {}

impl Drop for PyIRBuilder {
    fn drop(&mut self) {
        unsafe {
            if !self.builder_ptr.is_null() {
                let _ = Box::from_raw(self.builder_ptr);
            }
            // Don't drop context - it's owned by the module
        }
    }
}

#[pymethods]
impl PyIRBuilder {
    #[new]
    fn new(block: PyBasicBlock) -> Self {
        let context_ptr = block.context_ptr;
        let module_id = block.module_id;
        let context_ref: &'static Context = unsafe { &*context_ptr };

        let builder = Box::new(LLIRBuilder::new(context_ref, block.block));
        let builder_ptr = Box::into_raw(builder);

        Self {
            builder_ptr,
            context_ptr,
            module_id,
        }
    }

    /// Add two values
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn add(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .add(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("add failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Subtract two values
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn sub(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .sub(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("sub failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Multiply two values
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn mul(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .mul(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("mul failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Unsigned division
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn udiv(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .udiv(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("udiv failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// XOR operation
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn xor(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .xor(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("xor failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// AND operation
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn and(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .and(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("and failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// OR operation
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn or(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .or(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("or failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Alias for 'and' method (Python keyword collision workaround)
    /// In llvmlite, 'and_' is an attribute that points to the 'and' method
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn and_(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        self.and(lhs, rhs, name)
    }

    /// Alias for 'or' method (Python keyword collision workaround)
    /// In llvmlite, 'or_' is an attribute that points to the 'or' method
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn or_(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        self.or(lhs, rhs, name)
    }

    /// Logical shift right
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn lshr(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .lshr(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("lshr failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Shift left
    #[pyo3(signature = (lhs, rhs, name=""))]
    fn shl(&mut self, lhs: PyLLValue, rhs: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .shl(lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("shl failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Negate a value
    fn neg(&mut self, value: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .neg(value.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("neg failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Bitwise NOT
    fn not_(&mut self, value: PyLLValue, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .not(value.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("not failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Integer comparison (signed)
    #[pyo3(signature = (cmp_op, lhs, rhs, name=""))]
    fn icmp_signed(
        &mut self,
        cmp_op: &str,
        lhs: PyLLValue,
        rhs: PyLLValue,
        name: &str,
    ) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let result = builder
            .icmp_signed(cmp_op, lhs.value, rhs.value, name)
            .map_err(|e| PyRuntimeError::new_err(format!("icmp_signed failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Call a function
    fn call(
        &mut self,
        function: &PyFunction,
        args: Vec<PyLLValue>,
        name: &str,
    ) -> PyResult<Option<PyLLValue>> {
        let builder = unsafe { &*self.builder_ptr };
        let arg_values: Vec<_> = args.into_iter().map(|v| v.value).collect();
        let result = builder
            .call(function.function, &arg_values, name)
            .map_err(|e| PyRuntimeError::new_err(format!("call failed: {e}")))?;
        Ok(result.map(|value| PyLLValue {
            value,
            context_ptr: self.context_ptr,
        }))
    }

    /// Return void
    fn ret_void(&mut self) -> PyResult<()> {
        let builder = unsafe { &mut *self.builder_ptr };
        builder
            .ret_void()
            .map_err(|e| PyRuntimeError::new_err(format!("ret_void failed: {e}")))?;
        Ok(())
    }

    /// Get element pointer (GEP)
    fn gep(&mut self, ptr: PyLLValue, indices: Vec<PyLLValue>, name: &str) -> PyResult<PyLLValue> {
        let builder = unsafe { &mut *self.builder_ptr };
        let index_values: Vec<_> = indices.into_iter().map(|v| v.value).collect();
        let result = builder
            .gep(ptr.value, &index_values, name)
            .map_err(|e| PyRuntimeError::new_err(format!("gep failed: {e}")))?;
        Ok(PyLLValue {
            value: result,
            context_ptr: self.context_ptr,
        })
    }

    /// Position builder at end of block
    fn position_at_end(&mut self, block: PyBasicBlock) {
        let builder = unsafe { &mut *self.builder_ptr };
        builder.position_at_end(block.block);
    }

    /// Add a comment to the IR
    ///
    /// Mirrors builder.comment(text) in llvmlite
    fn comment(&mut self, text: &str) {
        let builder = unsafe { &*self.builder_ptr };

        // Get current block to determine where to insert comment
        if let Some(current_block) = builder.get().get_insert_block() {
            let block_name = current_block
                .get_name()
                .to_str()
                .unwrap_or("entry")
                .to_string();

            // Count instructions in the current block to determine insertion index
            let instruction_count = current_block.get_instructions().count();

            // Add comment to global storage using the module_id from the builder
            let comment = TrackedComment {
                block_name,
                instruction_index: instruction_count,
                text: text.to_string(),
            };

            if let Ok(mut comments) = global_comments().lock() {
                comments
                    .entry(self.module_id)
                    .or_insert_with(Vec::new)
                    .push(comment);
            }
        }
    }

    /// Create an if-then context manager
    ///
    /// Usage: with `builder.if_then(condition)`:
    #[pyo3(signature = (cond, likely=None))]
    #[allow(unused_variables)]
    fn if_then(
        &mut self,
        _py: Python,
        cond: PyLLValue,
        likely: Option<bool>,
    ) -> PyResult<PyIfThen> {
        let context = unsafe { &*self.context_ptr };
        let builder = unsafe { &*self.builder_ptr };

        // Get the current function from the builder's insert block
        let current_block = builder
            .get()
            .get_insert_block()
            .ok_or_else(|| PyRuntimeError::new_err("Builder not positioned in any block"))?;
        let function = current_block
            .get_parent()
            .ok_or_else(|| PyRuntimeError::new_err("Current block has no parent function"))?;

        let then_block = context.append_basic_block(function, "if.then");
        let merge_block = context.append_basic_block(function, "if.merge");

        // Build conditional branch
        builder
            .cbranch(cond.value, then_block, merge_block)
            .map_err(|e| PyRuntimeError::new_err(format!("cbranch failed: {e}")))?;

        // Position at then block
        builder.position_at_end(then_block);

        Ok(PyIfThen {
            builder_ptr: self.builder_ptr,
            merge_block,
        })
    }

    /// Create an if-else context manager
    ///
    /// Usage: with `builder.if_else(condition)` as (then, otherwise):
    #[pyo3(signature = (cond, likely=None))]
    #[allow(unused_variables)]
    fn if_else(
        &mut self,
        py: Python,
        cond: PyLLValue,
        likely: Option<bool>,
    ) -> PyResult<Py<PyIfElse>> {
        let context = unsafe { &*self.context_ptr };
        let builder = unsafe { &*self.builder_ptr };

        // Get the current function from the builder's insert block
        let current_block = builder
            .get()
            .get_insert_block()
            .ok_or_else(|| PyRuntimeError::new_err("Builder not positioned in any block"))?;
        let function = current_block
            .get_parent()
            .ok_or_else(|| PyRuntimeError::new_err("Current block has no parent function"))?;

        let then_block = context.append_basic_block(function, "if.then");
        let else_block = context.append_basic_block(function, "if.else");
        let merge_block = context.append_basic_block(function, "if.merge");

        // Build conditional branch
        builder
            .cbranch(cond.value, then_block, else_block)
            .map_err(|e| PyRuntimeError::new_err(format!("cbranch failed: {e}")))?;

        // Create the if-else context manager
        let if_else = PyIfElse {
            builder_ptr: self.builder_ptr,
            then_block,
            else_block,
            merge_block,
            then_branch: None,
            else_branch: None,
        };

        Py::new(py, if_else)
    }
}

// ============================================================================
// Context managers for control flow
// ============================================================================

/// Context manager for if-then blocks
#[pyclass(name = "IfThen")]
pub struct PyIfThen {
    builder_ptr: *mut LLIRBuilder<'static>,
    merge_block: inkwell::basic_block::BasicBlock<'static>,
}

unsafe impl Send for PyIfThen {}
unsafe impl Sync for PyIfThen {}

#[pymethods]
impl PyIfThen {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        // Already positioned at then block in if_then() method
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_value: Option<&Bound<'_, pyo3::types::PyAny>>,
        _traceback: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        // Branch to merge block and position builder there
        let builder = unsafe { &*self.builder_ptr };
        builder
            .branch(self.merge_block)
            .map_err(|e| PyRuntimeError::new_err(format!("branch failed: {e}")))?;
        builder.position_at_end(self.merge_block);
        Ok(false) // Don't suppress exceptions
    }
}

/// Context manager for individual branches in if-else
#[pyclass(name = "IfBranch")]
pub struct PyIfBranch {
    builder_ptr: *mut LLIRBuilder<'static>,
    block: inkwell::basic_block::BasicBlock<'static>,
    merge_block: inkwell::basic_block::BasicBlock<'static>,
}

unsafe impl Send for PyIfBranch {}
unsafe impl Sync for PyIfBranch {}

#[pymethods]
impl PyIfBranch {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        let builder = unsafe { &*slf.builder_ptr };
        builder.position_at_end(slf.block);
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_value: Option<&Bound<'_, pyo3::types::PyAny>>,
        _traceback: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        // Branch to merge block
        let builder = unsafe { &*self.builder_ptr };
        builder
            .branch(self.merge_block)
            .map_err(|e| PyRuntimeError::new_err(format!("branch failed: {e}")))?;
        Ok(false) // Don't suppress exceptions
    }
}

/// Context manager for if-else blocks
#[pyclass(name = "IfElse")]
pub struct PyIfElse {
    builder_ptr: *mut LLIRBuilder<'static>,
    then_block: inkwell::basic_block::BasicBlock<'static>,
    else_block: inkwell::basic_block::BasicBlock<'static>,
    merge_block: inkwell::basic_block::BasicBlock<'static>,
    then_branch: Option<Py<PyIfBranch>>,
    else_branch: Option<Py<PyIfBranch>>,
}

unsafe impl Send for PyIfElse {}
unsafe impl Sync for PyIfElse {}

#[pymethods]
impl PyIfElse {
    fn __enter__<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Py<PyIfBranch>, Py<PyIfBranch>)> {
        // Create context managers for both branches
        let then_cm = PyIfBranch {
            builder_ptr: slf.builder_ptr,
            block: slf.then_block,
            merge_block: slf.merge_block,
        };

        let else_cm = PyIfBranch {
            builder_ptr: slf.builder_ptr,
            block: slf.else_block,
            merge_block: slf.merge_block,
        };

        let then_py = Py::new(py, then_cm)?;
        let else_py = Py::new(py, else_cm)?;

        slf.then_branch = Some(then_py.clone_ref(py));
        slf.else_branch = Some(else_py.clone_ref(py));

        Ok((then_py, else_py))
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_value: Option<&Bound<'_, pyo3::types::PyAny>>,
        _traceback: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> bool {
        // Position builder at merge block
        let builder = unsafe { &*self.builder_ptr };
        builder.position_at_end(self.merge_block);
        false // Don't suppress exceptions
    }
}

// ============================================================================
// Function and related types
// ============================================================================

/// Python wrapper for LLVM function
#[pyclass(name = "Function", from_py_object)]
#[derive(Clone)]
pub struct PyFunction {
    function: inkwell::values::FunctionValue<'static>, // Use inkwell type directly since it's Copy
    context_ptr: *mut Context,
    /// Module ID for comment tracking
    module_id: usize,
}

unsafe impl Send for PyFunction {}
unsafe impl Sync for PyFunction {}

#[pymethods]
impl PyFunction {
    /// Append a basic block to this function
    fn append_basic_block(&self, name: &str) -> PyBasicBlock {
        let context = unsafe { &*self.context_ptr };
        // Use Context::append_basic_block(function, name)
        let block = context.append_basic_block(self.function, name);
        PyBasicBlock {
            block,
            context_ptr: self.context_ptr,
            module_id: self.module_id,
        }
    }

    /// Get function arguments
    #[getter]
    fn args(&self) -> Vec<PyLLValue> {
        // Get function parameters and wrap in PyLLValue
        self.function
            .get_param_iter()
            .map(|param| {
                // Convert BasicValueEnum to LLValue - only supporting types in LLValue enum
                let value = match param {
                    inkwell::values::BasicValueEnum::IntValue(v) => LLValue::Int(v),
                    inkwell::values::BasicValueEnum::PointerValue(v) => LLValue::Pointer(v),
                    inkwell::values::BasicValueEnum::ArrayValue(v) => LLValue::Array(v),
                    _ => panic!("Unsupported parameter type (float values not in LLValue enum)"),
                };
                PyLLValue {
                    value,
                    context_ptr: self.context_ptr,
                }
            })
            .collect()
    }
}

/// Python wrapper for basic block
#[pyclass(name = "BasicBlock", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyBasicBlock {
    block: inkwell::basic_block::BasicBlock<'static>,
    context_ptr: *mut Context,
    /// Module ID for comment tracking
    module_id: usize,
}

unsafe impl Send for PyBasicBlock {}
unsafe impl Sync for PyBasicBlock {}

/// Python wrapper for function type
#[pyclass(name = "FunctionType", from_py_object)]
#[derive(Clone)]
pub struct PyFunctionType {
    ret_type: LLType<'static>,
    param_types: Vec<LLType<'static>>,
    var_args: bool,
    #[allow(dead_code)]
    context_ptr: *mut Context,
}

unsafe impl Send for PyFunctionType {}
unsafe impl Sync for PyFunctionType {}

#[pymethods]
impl PyFunctionType {
    /// Create a new function type
    ///
    /// Mirrors `ir.FunctionType(return_type`, `param_types`, `var_args=False`)
    #[new]
    #[pyo3(signature = (return_type, param_types, var_args=false))]
    fn new(return_type: PyAnyType, param_types: Vec<PyAnyType>, var_args: bool) -> Self {
        // Extract context from one of the types
        let context_ptr = match &return_type {
            PyAnyType::Int(t) => t.context_ptr,
            PyAnyType::Double(t) => t.context_ptr,
            PyAnyType::Void(t) => t.context_ptr,
            PyAnyType::Pointer(t) => t.context_ptr,
            PyAnyType::Struct(t) => t.context_ptr,
            PyAnyType::Array(t) => t.context_ptr,
        };

        let context = unsafe { &*context_ptr };
        let ret_ty = return_type.to_ll_type(context);
        let param_tys: Vec<_> = param_types
            .into_iter()
            .map(|pt| pt.to_ll_type(context))
            .collect();

        Self {
            ret_type: ret_ty,
            param_types: param_tys,
            var_args,
            context_ptr,
        }
    }
}

/// Python wrapper for LLVM value
#[pyclass(name = "Value", from_py_object)]
#[derive(Copy, Clone)]
pub struct PyLLValue {
    value: LLValue<'static>,
    #[allow(dead_code)]
    context_ptr: *mut Context,
}

unsafe impl Send for PyLLValue {}
unsafe impl Sync for PyLLValue {}

#[pymethods]
impl PyLLValue {
    /// Convert integer value to pointer (inttoptr instruction)
    ///
    /// Mirrors llvmlite's `value.inttoptr(ptr_type)`
    fn inttoptr(&self, ptr_type: PyPointerType) -> PyResult<Self> {
        // Verify source is an integer
        let LLValue::Int(int_val) = &self.value else {
            return Err(PyRuntimeError::new_err("inttoptr requires integer value"));
        };

        // Get the pointer type from PyPointerType
        let LLType::Pointer(target_ptr_type) = ptr_type.ll_type else {
            return Err(PyRuntimeError::new_err("Target must be a pointer type"));
        };

        // Create the inttoptr constant
        let ptr_val = int_val.const_to_pointer(target_ptr_type);

        Ok(Self {
            value: LLValue::Pointer(ptr_val),
            context_ptr: self.context_ptr,
        })
    }
}

// ============================================================================
// GlobalVariable - Global variable support
// ============================================================================

/// Python wrapper for LLVM global variables
///
/// Provides global variable management (compatible with llvmlite API)
#[pyclass(name = "GlobalVariable")]
pub struct PyGlobalVariable {
    global: inkwell::values::GlobalValue<'static>,
    context_ptr: *mut Context,
}

unsafe impl Send for PyGlobalVariable {}
unsafe impl Sync for PyGlobalVariable {}

#[pymethods]
impl PyGlobalVariable {
    /// Create a new global variable
    ///
    /// Mirrors ir.GlobalVariable(module, type, name)
    #[new]
    fn new(module: &mut PyLLVMModule, ty: PyAnyType, name: &str) -> Self {
        let module_ref = unsafe { &mut *module.module_ptr };
        let context = unsafe { &*module.context_ptr };
        let ll_type = ty.to_ll_type(context);
        let global = module_ref.add_global(name, ll_type, None);
        Self {
            global,
            context_ptr: module.context_ptr,
        }
    }

    /// Set the initializer for this global variable
    #[setter]
    fn initializer(&mut self, value: &PyLLValue) {
        match &value.value {
            LLValue::Int(v) => self.global.set_initializer(v),
            LLValue::Float(v) => self.global.set_initializer(v),
            LLValue::Pointer(v) => self.global.set_initializer(v),
            LLValue::Array(v) => self.global.set_initializer(v),
        }
    }

    /// Set whether this global is a constant
    #[setter]
    fn global_constant(&mut self, is_const: bool) {
        self.global.set_constant(is_const);
    }

    /// Set the linkage type
    #[setter]
    fn linkage(&mut self, linkage: &str) {
        use inkwell::module::Linkage;
        let linkage_type = match linkage {
            "private" => Linkage::Private,
            "internal" => Linkage::Internal,
            "weak" => Linkage::WeakAny,
            "common" => Linkage::Common,
            _ => Linkage::External, // default (including "external")
        };
        self.global.set_linkage(linkage_type);
    }

    /// Get element pointer (GEP) from this global
    ///
    /// Mirrors global.gep(indices) in llvmlite
    fn gep(&self, indices: Vec<PyLLValue>) -> PyResult<PyLLValue> {
        // Convert PyLLValue indices to inkwell IntValues
        let int_indices: Result<Vec<_>, _> = indices
            .into_iter()
            .map(|v| match v.value {
                LLValue::Int(i) => Ok(i),
                _ => Err(PyRuntimeError::new_err("GEP indices must be integers")),
            })
            .collect();
        let int_indices = int_indices?;

        // Use const_gep for global variables
        let gep_val = unsafe { self.global.as_pointer_value().const_gep(&int_indices) };

        Ok(PyLLValue {
            value: LLValue::Pointer(gep_val),
            context_ptr: self.context_ptr,
        })
    }

    /// Get the pointer value of this global
    fn as_pointer_value(&self) -> PyLLValue {
        PyLLValue {
            value: LLValue::Pointer(self.global.as_pointer_value()),
            context_ptr: self.context_ptr,
        }
    }
}

// ============================================================================
// Constant - Constant value creation
// ============================================================================

/// Create constant value (mirrors llvmlite's ir.Constant(type, value))
///
/// This is the main entry point for creating constants, matching llvmlite's API:
/// ```python
/// ir.Constant(ir.IntType(32), 5)
/// ir.Constant(ir.ArrayType(ir.IntType(8), 10), b"hello")
/// ```
#[pyfunction]
#[allow(non_snake_case)]
fn Constant(_py: Python, ty: PyAnyType, value: &Bound<'_, PyAny>) -> PyResult<PyLLValue> {
    // Check type isn't void (llvmlite doesn't allow void constants)
    if matches!(ty, PyAnyType::Void(_)) {
        return Err(PyRuntimeError::new_err("Cannot create void constant"));
    }

    // Handle different type/value combinations
    match &ty {
        PyAnyType::Int(int_ty) => {
            // Integer constant - extract value as i64
            // Also handle Python bool (True/False) which are int subclasses
            let int_value = if let Ok(val) = value.extract::<bool>() {
                // Python bool: True -> 1, False -> 0
                i64::from(val)
            } else if let Ok(val) = value.extract::<i64>() {
                val
            } else if let Ok(val) = value.extract::<u64>() {
                // Allow wrapping cast - this is intentional for large unsigned values
                // The value is later handled correctly via unsigned_abs()
                #[allow(clippy::cast_possible_wrap)]
                {
                    val as i64
                }
            } else {
                return Err(PyRuntimeError::new_err(
                    "Constant value must be integer or boolean for IntType",
                ));
            };

            // Create integer constant
            let LLType::Int(int_type) = int_ty.ll_type else {
                return Err(PyRuntimeError::new_err("Expected integer type"));
            };
            let signed = int_value < 0;
            let const_val = LLConstant::int(int_type, int_value.unsigned_abs(), signed);
            Ok(PyLLValue {
                value: const_val,
                context_ptr: int_ty.context_ptr,
            })
        }

        PyAnyType::Array(array_ty) => {
            // Array constant - value should be bytes (most common case for gen_qir.py)
            if let Ok(bytes) = value.extract::<Vec<u8>>() {
                // Byte array
                let context = unsafe { &*array_ty.context_ptr };
                let const_val = LLConstant::array_from_bytes(context, &bytes);
                Ok(PyLLValue {
                    value: const_val,
                    context_ptr: array_ty.context_ptr,
                })
            } else {
                Err(PyRuntimeError::new_err(
                    "Constant value must be bytes for ArrayType (other array types not yet implemented)",
                ))
            }
        }

        PyAnyType::Double(double_ty) => {
            // Float/double constant
            let float_value = value.extract::<f64>().map_err(|_| {
                PyRuntimeError::new_err("Constant value must be float for DoubleType")
            })?;

            let ll_type = double_ty.ll_type;
            let const_val = match ll_type {
                LLType::Float(f) => {
                    // Use inkwell's const_float method directly
                    LLValue::Float(f.const_float(float_value))
                }
                _ => return Err(PyRuntimeError::new_err("Expected float type")),
            };

            Ok(PyLLValue {
                value: const_val,
                context_ptr: double_ty.context_ptr,
            })
        }

        _ => Err(PyRuntimeError::new_err(format!(
            "Constant creation not yet implemented for type: {:?}",
            std::any::type_name_of_val(&ty)
        ))),
    }
}

// ============================================================================
// Type creation functions (module level)
// ============================================================================

// Global context for standalone type creation (mimics llvmlite behavior)
struct GlobalContextPtr(*mut Context);
unsafe impl Send for GlobalContextPtr {}
unsafe impl Sync for GlobalContextPtr {}

static GLOBAL_CONTEXT: OnceLock<GlobalContextPtr> = OnceLock::new();

fn get_global_context() -> *mut Context {
    GLOBAL_CONTEXT
        .get_or_init(|| {
            let context = Box::new(Context::create());
            GlobalContextPtr(Box::into_raw(context))
        })
        .0
}

/// Create integer type (mirrors ir.IntType(bits))
#[pyfunction]
#[allow(non_snake_case)]
fn IntType(_py: Python, bits: u32) -> PyIntType {
    let context_ptr = get_global_context();
    let context = unsafe { &*context_ptr };
    let ll_type = LLType::int(context, bits);
    PyIntType {
        ll_type,
        context_ptr,
    }
}

/// Create void type (mirrors `ir.VoidType()`)
#[pyfunction]
#[allow(non_snake_case)]
fn VoidType(_py: Python) -> PyVoidType {
    let context_ptr = get_global_context();
    PyVoidType { context_ptr }
}

/// Create double type (mirrors `ir.DoubleType()`)
#[pyfunction]
#[allow(non_snake_case)]
fn DoubleType(_py: Python) -> PyDoubleType {
    let context_ptr = get_global_context();
    let context = unsafe { &*context_ptr };
    let ll_type = LLType::double(context);
    PyDoubleType {
        ll_type,
        context_ptr,
    }
}

/// Create a function (mirrors ir.Function(module, `func_type`, name=...))
#[pyfunction]
#[pyo3(signature = (module, func_type, name))]
#[allow(non_snake_case)]
fn Function(module: &mut PyLLVMModule, func_type: &PyFunctionType, name: &str) -> PyFunction {
    // This is just an alias for module.add_function()
    module.add_function(name, func_type)
}

// ============================================================================
// Register with Python
// ============================================================================

/// Register LLVM IR module with Python (compatible with llvmlite API)
pub fn register_llvm_module(parent: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Create an 'ir' submodule compatible with Python's llvmlite.ir
    let ir_module = pyo3::types::PyModule::new(parent.py(), "ir")?;

    // Register main module and context classes
    ir_module.add_class::<PyLLVMModule>()?;
    ir_module.add_class::<PyModuleContext>()?;

    // Register type classes
    ir_module.add_class::<PyStructType>()?;
    ir_module.add_class::<PyPointerType>()?;
    ir_module.add_class::<PyIntType>()?;
    ir_module.add_class::<PyDoubleType>()?;
    ir_module.add_class::<PyArrayType>()?;
    ir_module.add_class::<PyVoidType>()?;

    // Register function and builder classes
    ir_module.add_class::<PyIRBuilder>()?;
    ir_module.add_class::<PyFunction>()?;
    ir_module.add_class::<PyBasicBlock>()?;
    ir_module.add_class::<PyFunctionType>()?;
    ir_module.add_class::<PyLLValue>()?;
    ir_module.add_class::<PyGlobalVariable>()?;

    // Register context manager classes for control flow
    ir_module.add_class::<PyIfThen>()?;
    ir_module.add_class::<PyIfElse>()?;
    ir_module.add_class::<PyIfBranch>()?;

    // Register type and value creation functions
    ir_module.add_function(wrap_pyfunction!(IntType, &ir_module)?)?;
    ir_module.add_function(wrap_pyfunction!(VoidType, &ir_module)?)?;
    ir_module.add_function(wrap_pyfunction!(DoubleType, &ir_module)?)?;
    ir_module.add_function(wrap_pyfunction!(Function, &ir_module)?)?;
    ir_module.add_function(wrap_pyfunction!(Constant, &ir_module)?)?;

    parent.add_submodule(&ir_module)?;
    Ok(())
}

// ============================================================================
// llvmlite.binding module - for bitcode generation
// ============================================================================

/// `ValueRef` for type hints (matches llvmlite.binding.ValueRef)
#[pyclass(name = "ValueRef")]
pub struct PyValueRef;

#[pymethods]
impl PyValueRef {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Module reference returned by `parse_assembly`
#[pyclass(name = "ModuleRef")]
pub struct PyModuleRef {
    llvm_ir: String,
}

#[pymethods]
impl PyModuleRef {
    /// Convert LLVM IR text to bitcode
    fn as_bitcode(&self) -> PyResult<Vec<u8>> {
        use std::io::Write;

        // Create a temporary context and module to parse the IR
        let context = inkwell::context::Context::create();

        // Write IR to a temporary file (inkwell parses files more reliably)
        let mut temp_file = tempfile::NamedTempFile::new().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create temp file: {e}"))
        })?;

        temp_file.write_all(self.llvm_ir.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write temp file: {e}"))
        })?;

        temp_file.flush().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to flush temp file: {e}"))
        })?;

        // Get the path before the file is closed
        let temp_path = temp_file.path().to_path_buf();

        // Parse the LLVM IR from file
        let module = inkwell::module::Module::parse_bitcode_from_path(&temp_path, &context)
            .or_else(|_| {
                // If bitcode parsing fails, try IR parsing
                let memory_buffer = inkwell::memory_buffer::MemoryBuffer::create_from_file(
                    &temp_path,
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to read temp file: {e}"
                    ))
                })?;

                context.create_module_from_ir(memory_buffer).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to parse LLVM IR: {e}"
                    ))
                })
            })?;

        // Write module to bitcode
        let bitcode_buffer = module.write_bitcode_to_memory();
        Ok(bitcode_buffer.as_slice().to_vec())
    }
}

/// Parse LLVM assembly text into a module
#[pyfunction]
fn parse_assembly(llvm_ir: &str) -> PyModuleRef {
    PyModuleRef {
        llvm_ir: llvm_ir.to_string(),
    }
}

/// Shutdown LLVM (no-op for compatibility)
#[pyfunction]
fn shutdown() {
    // In llvmlite, this shuts down LLVM global state
    // For our Rust implementation, we don't need to do anything
    // as Rust's RAII handles cleanup automatically
}

/// Register the binding module (mimics llvmlite.binding)
pub fn register_binding_module(parent: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    let binding_module = pyo3::types::PyModule::new(parent.py(), "binding")?;

    // Register classes
    binding_module.add_class::<PyValueRef>()?;
    binding_module.add_class::<PyModuleRef>()?;

    // Register functions
    binding_module.add_function(wrap_pyfunction!(parse_assembly, &binding_module)?)?;
    binding_module.add_function(wrap_pyfunction!(shutdown, &binding_module)?)?;

    parent.add_submodule(&binding_module)?;
    Ok(())
}
