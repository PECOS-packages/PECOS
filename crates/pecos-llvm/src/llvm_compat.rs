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

//! LLVM IR generation API using inkwell
//!
//! Rust types for LLVM IR generation, designed to be compatible
//! with Python's llvmlite API. We use inkwell (Rust LLVM bindings) to generate proper
//! LLVM IR and expose it through a Python-friendly interface.
//!
//! # Clippy Configuration
//!
//! This module is an internal compatibility layer with clear, self-documenting
//! function signatures. We suppress pedantic warnings about missing error/panic
//! documentation as the errors/panics are obvious from the function signatures.
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
//! Key design: Focused on quantum IR generation needs, providing a clean API for
//! LLVM module creation, type management, and IR building.

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{
    ArrayType, BasicType, BasicTypeEnum, FloatType, IntType, PointerType, StructType,
};
use inkwell::values::{
    ArrayValue, BasicValueEnum, FloatValue, FunctionValue, GlobalValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, IntPredicate};
use pecos_core::prelude::PecosError;

pub type LLResult<T> = Result<T, PecosError>;

// ============================================================================
// Context wrapper
// ============================================================================

/// Wrapper around inkwell's Context that can be used with RefCell/Rc
///
/// llvmlite has implicit context management through Module.context
/// We use Rc<`RefCell`<>> pattern for shared ownership in Python bindings
pub struct LLContext {
    context: Context,
}

impl LLContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            context: Context::create(),
        }
    }

    #[must_use]
    pub fn get(&self) -> &Context {
        &self.context
    }
}

impl Default for LLContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Module wrapper
// ============================================================================

/// Wrapper around inkwell's Module that mirrors llvmlite's ir.Module
pub struct LLModule<'ctx> {
    module: Module<'ctx>,
    context: &'ctx Context,
}

impl std::fmt::Display for LLModule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Call .to_string() on LLVMString to match the original inherent method behavior
        write!(f, "{}", self.module.print_to_string().to_string())
    }
}

impl<'ctx> LLModule<'ctx> {
    #[must_use]
    pub fn new(context: &'ctx Context, name: &str) -> Self {
        Self {
            module: context.create_module(name),
            context,
        }
    }

    pub fn get(&self) -> &Module<'ctx> {
        &self.module
    }

    pub fn get_mut(&mut self) -> &mut Module<'ctx> {
        &mut self.module
    }

    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    // Note: `to_string()` is provided automatically by the Display trait implementation above

    /// Get the LLVM bitcode as bytes
    pub fn to_bitcode(&self) -> Vec<u8> {
        self.module.write_bitcode_to_memory().as_slice().to_vec()
    }

    /// Get an identified (opaque) type by name, creating it if it doesn't exist
    /// This mirrors llvmlite's `module.context.get_identified_type(name)`
    pub fn get_identified_type(&self, name: &str) -> StructType<'ctx> {
        self.context.opaque_struct_type(name)
    }

    /// Add a global variable (mirrors llvmlite's global variable creation)
    pub fn add_global(
        &mut self,
        name: &str,
        ty: LLType<'ctx>,
        initializer: Option<LLValue<'ctx>>,
    ) -> GlobalValue<'ctx> {
        let global = match ty {
            LLType::Array(t) => self.module.add_global(t, None, name),
            LLType::Int(t) => self.module.add_global(t, None, name),
            LLType::Float(t) => self.module.add_global(t, None, name),
            LLType::Pointer(t) => self.module.add_global(t, None, name),
            LLType::Struct(t) => self.module.add_global(t, None, name),
            LLType::Void => panic!("Cannot create global variable of void type"),
        };

        if let Some(init_val) = initializer {
            match init_val {
                LLValue::Int(v) => global.set_initializer(&v),
                LLValue::Float(v) => global.set_initializer(&v),
                LLValue::Pointer(v) => global.set_initializer(&v),
                LLValue::Array(v) => global.set_initializer(&v),
            }
        }

        global
    }

    /// Add a function declaration (mirrors llvmlite's ir.Function)
    pub fn add_function(&mut self, name: &str, fn_type: LLFunctionType<'ctx>) -> LLFunction<'ctx> {
        let function = self.module.add_function(name, fn_type.get(), None);
        LLFunction { function }
    }
}

// ============================================================================
// Type wrappers
// ============================================================================

/// Wrapper for LLVM function types (mirrors llvmlite's ir.FunctionType)
#[derive(Copy, Clone)]
pub struct LLFunctionType<'ctx> {
    fn_type: inkwell::types::FunctionType<'ctx>,
}

impl<'ctx> LLFunctionType<'ctx> {
    #[must_use]
    pub fn new(return_type: LLType<'ctx>, param_types: &[LLType<'ctx>], var_args: bool) -> Self {
        let params: Vec<_> = param_types
            .iter()
            .filter_map(|t| t.to_basic_metadata_type().map(std::convert::Into::into))
            .collect();

        let fn_type = match return_type {
            LLType::Void => {
                // For void return, we need to get the context from somewhere
                // We'll need to pass context or extract it from one of the param types
                panic!("Use new_with_context for void return types")
            }
            LLType::Int(t) => t.fn_type(&params, var_args),
            LLType::Float(t) => t.fn_type(&params, var_args),
            LLType::Pointer(t) => t.fn_type(&params, var_args),
            LLType::Struct(t) => t.fn_type(&params, var_args),
            LLType::Array(t) => t.fn_type(&params, var_args),
        };

        Self { fn_type }
    }

    #[must_use]
    pub fn new_with_context(
        context: &'ctx Context,
        return_type: LLType<'ctx>,
        param_types: &[LLType<'ctx>],
        var_args: bool,
    ) -> Self {
        let params: Vec<_> = param_types
            .iter()
            .filter_map(|t| t.to_basic_metadata_type().map(std::convert::Into::into))
            .collect();

        let fn_type = match return_type {
            LLType::Void => context.void_type().fn_type(&params, var_args),
            LLType::Int(t) => t.fn_type(&params, var_args),
            LLType::Float(t) => t.fn_type(&params, var_args),
            LLType::Pointer(t) => t.fn_type(&params, var_args),
            LLType::Struct(t) => t.fn_type(&params, var_args),
            LLType::Array(t) => t.fn_type(&params, var_args),
        };

        Self { fn_type }
    }

    #[must_use]
    pub fn get(&self) -> inkwell::types::FunctionType<'ctx> {
        self.fn_type
    }
}

/// Wrapper for LLVM types that mirrors llvmlite's type hierarchy
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LLType<'ctx> {
    Void,
    Int(IntType<'ctx>),
    Float(FloatType<'ctx>),
    Pointer(PointerType<'ctx>),
    Struct(StructType<'ctx>),
    Array(ArrayType<'ctx>),
}

// inkwell 0.8.0 only derives `Hash` for `IntType`; the other type wrappers
// are `Eq` (LLVM type-ref pointer equality) but not `Hash`. Hash the same
// `LLVMTypeRef` pointer so `Hash` stays consistent with that `Eq`.
impl std::hash::Hash for LLType<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use inkwell::types::AsTypeRef;
        match self {
            LLType::Void => 0u8.hash(state),
            LLType::Int(t) => {
                1u8.hash(state);
                (t.as_type_ref() as usize).hash(state);
            }
            LLType::Float(t) => {
                2u8.hash(state);
                (t.as_type_ref() as usize).hash(state);
            }
            LLType::Pointer(t) => {
                3u8.hash(state);
                (t.as_type_ref() as usize).hash(state);
            }
            LLType::Struct(t) => {
                4u8.hash(state);
                (t.as_type_ref() as usize).hash(state);
            }
            LLType::Array(t) => {
                5u8.hash(state);
                (t.as_type_ref() as usize).hash(state);
            }
        }
    }
}

impl<'ctx> LLType<'ctx> {
    /// Create void type
    #[must_use]
    pub fn void(context: &'ctx Context) -> Self {
        let _ = context.void_type();
        LLType::Void
    }

    /// Create integer type
    #[must_use]
    pub fn int(context: &'ctx Context, bits: u32) -> Self {
        match bits {
            // Use custom_width_int_type(1) instead of bool_type() to match llvmlite
            // llvmlite renders i1 constants as "i1 1" and "i1 0", not "i1 true" and "i1 false"
            1 => LLType::Int(context.custom_width_int_type(1)),
            8 => LLType::Int(context.i8_type()),
            16 => LLType::Int(context.i16_type()),
            32 => LLType::Int(context.i32_type()),
            64 => LLType::Int(context.i64_type()),
            128 => LLType::Int(context.i128_type()),
            _ => LLType::Int(context.custom_width_int_type(bits)),
        }
    }

    /// Create double (f64) type
    #[must_use]
    pub fn double(context: &'ctx Context) -> Self {
        LLType::Float(context.f64_type())
    }

    /// Create array type (mirrors llvmlite's ir.ArrayType)
    #[must_use]
    pub fn array(element_type: LLType<'ctx>, count: u32) -> Self {
        match element_type {
            LLType::Int(t) => LLType::Array(t.array_type(count)),
            LLType::Float(t) => LLType::Array(t.array_type(count)),
            LLType::Pointer(t) => LLType::Array(t.array_type(count)),
            LLType::Struct(t) => LLType::Array(t.array_type(count)),
            LLType::Array(t) => LLType::Array(t.array_type(count)),
            LLType::Void => panic!("Cannot create array of void type"),
        }
    }

    /// Convert to pointer type (mirrors llvmlite's `as_pointer()`)
    #[must_use]
    pub fn as_pointer(&self, context: &'ctx Context) -> LLType<'ctx> {
        match self {
            LLType::Void => {
                // Void pointers are represented as i8*
                LLType::Pointer(context.i8_type().ptr_type(AddressSpace::default()))
            }
            LLType::Int(t) => LLType::Pointer(t.ptr_type(AddressSpace::default())),
            LLType::Float(t) => LLType::Pointer(t.ptr_type(AddressSpace::default())),
            LLType::Pointer(t) => LLType::Pointer(*t), // Already a pointer
            LLType::Struct(t) => LLType::Pointer(t.ptr_type(AddressSpace::default())),
            LLType::Array(t) => LLType::Pointer(t.ptr_type(AddressSpace::default())),
        }
    }

    /// Get the underlying inkwell type for function signatures
    #[must_use]
    pub fn to_basic_metadata_type(&self) -> Option<BasicTypeEnum<'ctx>> {
        match self {
            LLType::Void => None,
            LLType::Int(t) => Some((*t).into()),
            LLType::Float(t) => Some((*t).into()),
            LLType::Pointer(t) => Some((*t).into()),
            LLType::Struct(t) => Some((*t).into()),
            LLType::Array(t) => Some((*t).into()),
        }
    }

    /// Get int type (panics if not an int)
    #[must_use]
    pub fn as_int_type(&self) -> IntType<'ctx> {
        match self {
            LLType::Int(t) => *t,
            _ => panic!("Expected int type"),
        }
    }

    /// Get pointer type (panics if not a pointer)
    #[must_use]
    pub fn as_pointer_type(&self) -> PointerType<'ctx> {
        match self {
            LLType::Pointer(t) => *t,
            _ => panic!("Expected pointer type"),
        }
    }

    /// Get struct type (panics if not a struct)
    #[must_use]
    pub fn as_struct_type(&self) -> StructType<'ctx> {
        match self {
            LLType::Struct(t) => *t,
            _ => panic!("Expected struct type"),
        }
    }
}

// ============================================================================
// Value wrappers
// ============================================================================

/// Wrapper for LLVM values that mirrors llvmlite's value types
#[derive(Clone, Copy)]
pub enum LLValue<'ctx> {
    Int(IntValue<'ctx>),
    Float(FloatValue<'ctx>),
    Pointer(PointerValue<'ctx>),
    Array(ArrayValue<'ctx>),
}

impl<'ctx> LLValue<'ctx> {
    #[must_use]
    pub fn to_basic_value(&self) -> BasicValueEnum<'ctx> {
        match self {
            LLValue::Int(v) => (*v).into(),
            LLValue::Float(v) => (*v).into(),
            LLValue::Pointer(v) => (*v).into(),
            LLValue::Array(v) => (*v).into(),
        }
    }

    #[must_use]
    pub fn as_int_value(&self) -> IntValue<'ctx> {
        match self {
            LLValue::Int(v) => *v,
            _ => panic!("Expected int value"),
        }
    }

    #[must_use]
    pub fn as_float_value(&self) -> FloatValue<'ctx> {
        match self {
            LLValue::Float(v) => *v,
            _ => panic!("Expected float value"),
        }
    }

    #[must_use]
    pub fn as_pointer_value(&self) -> PointerValue<'ctx> {
        match self {
            LLValue::Pointer(v) => *v,
            _ => panic!("Expected pointer value"),
        }
    }

    #[must_use]
    pub fn as_array_value(&self) -> ArrayValue<'ctx> {
        match self {
            LLValue::Array(v) => *v,
            _ => panic!("Expected array value"),
        }
    }
}

// ============================================================================
// Function wrapper
// ============================================================================

/// Wrapper around inkwell's `FunctionValue` that mirrors llvmlite's ir.Function
pub struct LLFunction<'ctx> {
    function: FunctionValue<'ctx>,
}

impl<'ctx> LLFunction<'ctx> {
    pub fn new(
        module: &mut LLModule<'ctx>,
        name: &str,
        return_type: LLType<'ctx>,
        arg_types: &[LLType<'ctx>],
    ) -> Self {
        let param_types: Vec<_> = arg_types
            .iter()
            .filter_map(|t| t.to_basic_metadata_type().map(std::convert::Into::into))
            .collect();

        let fn_type = match return_type {
            LLType::Void => module.context().void_type().fn_type(&param_types, false),
            LLType::Int(t) => t.fn_type(&param_types, false),
            LLType::Float(t) => t.fn_type(&param_types, false),
            LLType::Pointer(t) => t.fn_type(&param_types, false),
            LLType::Struct(t) => t.fn_type(&param_types, false),
            LLType::Array(t) => t.fn_type(&param_types, false),
        };

        let function = module.get_mut().add_function(name, fn_type, None);

        Self { function }
    }

    #[must_use]
    pub fn get(&self) -> FunctionValue<'ctx> {
        self.function
    }

    /// Append a basic block to this function (mirrors llvmlite's `func.append_basic_block`)
    #[must_use]
    pub fn append_basic_block(&self, context: &'ctx Context, name: &str) -> BasicBlock<'ctx> {
        context.append_basic_block(self.function, name)
    }
}

// ============================================================================
// IRBuilder wrapper
// ============================================================================

/// Wrapper around inkwell's Builder that mirrors llvmlite's ir.IRBuilder
pub struct LLIRBuilder<'ctx> {
    builder: Builder<'ctx>,
}

impl<'ctx> LLIRBuilder<'ctx> {
    #[must_use]
    pub fn new(context: &'ctx Context, block: BasicBlock<'ctx>) -> Self {
        let builder = context.create_builder();
        builder.position_at_end(block);
        Self { builder }
    }

    pub fn get(&self) -> &Builder<'ctx> {
        &self.builder
    }

    /// Position at end of a basic block
    pub fn position_at_end(&self, block: BasicBlock<'ctx>) {
        self.builder.position_at_end(block);
    }

    // ========================================================================
    // Arithmetic operations (mirror llvmlite IRBuilder methods)
    // ========================================================================

    pub fn add(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_add(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build add: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn sub(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_sub(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build sub: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn mul(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_mul(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build mul: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn udiv(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_unsigned_div(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build udiv: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn xor(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_xor(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build xor: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn and(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_and(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build and: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn or(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_or(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build or: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn lshr(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_right_shift(lhs.as_int_value(), rhs.as_int_value(), false, name)
            .map_err(|e| PecosError::Generic(format!("Failed to build lshr: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn shl(
        &self,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_left_shift(lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build shl: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn neg(&self, value: LLValue<'ctx>, name: &str) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_neg(value.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build neg: {e}")))?;
        Ok(LLValue::Int(result))
    }

    pub fn not(&self, value: LLValue<'ctx>, name: &str) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_not(value.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build not: {e}")))?;
        Ok(LLValue::Int(result))
    }

    // ========================================================================
    // Comparison operations
    // ========================================================================

    pub fn icmp_signed(
        &self,
        op: &str,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let predicate = match op {
            "==" => IntPredicate::EQ,
            "!=" => IntPredicate::NE,
            "<" => IntPredicate::SLT,
            ">" => IntPredicate::SGT,
            "<=" => IntPredicate::SLE,
            ">=" => IntPredicate::SGE,
            _ => {
                return Err(PecosError::Generic(format!(
                    "Unknown comparison operator: {op}"
                )));
            }
        };

        let result = self
            .builder
            .build_int_compare(predicate, lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build icmp: {e}")))?;
        Ok(LLValue::Int(result))
    }

    // ========================================================================
    // Function calls
    // ========================================================================

    pub fn call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[LLValue<'ctx>],
        name: &str,
    ) -> LLResult<Option<LLValue<'ctx>>> {
        let arg_values: Vec<_> = args.iter().map(|v| v.to_basic_value().into()).collect();

        let call_site = self
            .builder
            .build_call(function, &arg_values, name)
            .map_err(|e| PecosError::Generic(format!("Failed to build call: {e}")))?;

        Ok(call_site.try_as_basic_value().basic().map(|v| match v {
            BasicValueEnum::IntValue(i) => LLValue::Int(i),
            BasicValueEnum::PointerValue(p) => LLValue::Pointer(p),
            _ => panic!("Unsupported return value type"),
        }))
    }

    // ========================================================================
    // Control flow
    // ========================================================================

    pub fn ret_void(&self) -> LLResult<()> {
        self.builder
            .build_return(None)
            .map_err(|e| PecosError::Generic(format!("Failed to build ret_void: {e}")))?;
        Ok(())
    }

    /// Conditional branch
    pub fn cbranch(
        &self,
        cond: LLValue<'ctx>,
        then_block: BasicBlock<'ctx>,
        else_block: BasicBlock<'ctx>,
    ) -> LLResult<()> {
        self.builder
            .build_conditional_branch(cond.as_int_value(), then_block, else_block)
            .map_err(|e| PecosError::Generic(format!("Failed to build conditional branch: {e}")))?;
        Ok(())
    }

    /// Unconditional branch
    pub fn branch(&self, block: BasicBlock<'ctx>) -> LLResult<()> {
        self.builder
            .build_unconditional_branch(block)
            .map_err(|e| PecosError::Generic(format!("Failed to build branch: {e}")))?;
        Ok(())
    }

    /// Add a comment (as a no-op in IR)
    pub fn comment(&self, _text: &str) {
        // Comments don't generate LLVM IR, they're just for human readers
        // llvmlite also doesn't actually emit comments to the IR
    }

    // ========================================================================
    // GEP (Get Element Pointer)
    // ========================================================================

    pub fn gep(
        &self,
        ptr: LLValue<'ctx>,
        indices: &[LLValue<'ctx>],
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let idx_values: Vec<_> = indices.iter().map(LLValue::as_int_value).collect();

        unsafe {
            let result = self
                .builder
                .build_gep(ptr.as_pointer_value(), &idx_values, name)
                .map_err(|e| PecosError::Generic(format!("Failed to build gep: {e}")))?;
            Ok(LLValue::Pointer(result))
        }
    }

    // ========================================================================
    // Memory ops + casts (unblocks the standard CReg model)
    // ========================================================================

    /// `alloca <ty>` -- stack slot. Caller positions the builder (B2
    /// places `CReg` buffers in the entry block via `position_at_end`).
    pub fn alloca(&self, ll_type: LLType<'ctx>, name: &str) -> LLResult<LLValue<'ctx>> {
        let basic_ty = ll_type
            .to_basic_metadata_type()
            .ok_or_else(|| PecosError::Generic("Cannot alloca a void type".into()))?;
        let result = self
            .builder
            .build_alloca(basic_ty, name)
            .map_err(|e| PecosError::Generic(format!("Failed to build alloca: {e}")))?;
        Ok(LLValue::Pointer(result))
    }

    /// `load` (LLVM-14 typed pointer: pointee inferred from `ptr`).
    pub fn load(&self, ptr: LLValue<'ctx>, name: &str) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_load(ptr.as_pointer_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build load: {e}")))?;
        Ok(match result {
            BasicValueEnum::IntValue(v) => LLValue::Int(v),
            BasicValueEnum::FloatValue(v) => LLValue::Float(v),
            BasicValueEnum::PointerValue(v) => LLValue::Pointer(v),
            BasicValueEnum::ArrayValue(v) => LLValue::Array(v),
            other => {
                return Err(PecosError::Generic(format!(
                    "load: unsupported loaded value type: {other:?}"
                )));
            }
        })
    }

    /// `store` -- discards inkwell's returned pointer (Python `-> None`).
    pub fn store(&self, ptr: LLValue<'ctx>, value: LLValue<'ctx>) -> LLResult<()> {
        self.builder
            .build_store(ptr.as_pointer_value(), value.to_basic_value())
            .map_err(|e| PecosError::Generic(format!("Failed to build store: {e}")))?;
        Ok(())
    }

    /// `zext` int value to a wider int type.
    pub fn zext(
        &self,
        value: LLValue<'ctx>,
        dest_type: LLType<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_z_extend(value.as_int_value(), dest_type.as_int_type(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build zext: {e}")))?;
        Ok(LLValue::Int(result))
    }

    /// `trunc` int value to a narrower int type.
    pub fn trunc(
        &self,
        value: LLValue<'ctx>,
        dest_type: LLType<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let result = self
            .builder
            .build_int_truncate(value.as_int_value(), dest_type.as_int_type(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build trunc: {e}")))?;
        Ok(LLValue::Int(result))
    }

    /// Unsigned integer comparison (mirrors `icmp_signed` with U-predicates).
    pub fn icmp_unsigned(
        &self,
        op: &str,
        lhs: LLValue<'ctx>,
        rhs: LLValue<'ctx>,
        name: &str,
    ) -> LLResult<LLValue<'ctx>> {
        let predicate = match op {
            "==" => IntPredicate::EQ,
            "!=" => IntPredicate::NE,
            "<" => IntPredicate::ULT,
            ">" => IntPredicate::UGT,
            "<=" => IntPredicate::ULE,
            ">=" => IntPredicate::UGE,
            _ => {
                return Err(PecosError::Generic(format!(
                    "Unknown comparison operator: {op}"
                )));
            }
        };
        let result = self
            .builder
            .build_int_compare(predicate, lhs.as_int_value(), rhs.as_int_value(), name)
            .map_err(|e| PecosError::Generic(format!("Failed to build icmp: {e}")))?;
        Ok(LLValue::Int(result))
    }
}

// ============================================================================
// Constant creation
// ============================================================================

/// Create constant values (mirrors llvmlite's ir.Constant)
pub struct LLConstant;

impl LLConstant {
    #[must_use]
    pub fn int(int_type: IntType<'_>, value: u64, signed: bool) -> LLValue<'_> {
        LLValue::Int(int_type.const_int(value, signed))
    }

    #[must_use]
    pub fn int_from_type(lltype: LLType<'_>, value: u64, signed: bool) -> LLValue<'_> {
        match lltype {
            LLType::Int(t) => LLValue::Int(t.const_int(value, signed)),
            _ => panic!("Expected int type for constant"),
        }
    }

    /// Create constant array from bytes (for string constants)
    #[must_use]
    pub fn array_from_bytes<'ctx>(context: &'ctx Context, bytes: &[u8]) -> LLValue<'ctx> {
        let i8_type = context.i8_type();
        let values: Vec<_> = bytes
            .iter()
            .map(|&b| i8_type.const_int(u64::from(b), false))
            .collect();
        LLValue::Array(i8_type.const_array(&values))
    }

    /// Create constant array from values
    pub fn array<'ctx>(
        element_type: LLType<'ctx>,
        values: &[LLValue<'ctx>],
    ) -> LLResult<LLValue<'ctx>> {
        match element_type {
            LLType::Int(t) => {
                let int_vals: Vec<_> = values.iter().map(LLValue::as_int_value).collect();
                Ok(LLValue::Array(t.const_array(&int_vals)))
            }
            _ => Err(PecosError::Generic(
                "Unsupported array element type for constant".to_string(),
            )),
        }
    }

    /// Zero/`zeroinitializer` constant of `ll_type` (backs
    /// `Constant(ty, None)`; Array -> `zeroinitializer`, Int -> `iN 0`).
    pub fn zero(ll_type: LLType<'_>) -> LLResult<LLValue<'_>> {
        match ll_type {
            LLType::Int(t) => Ok(LLValue::Int(t.const_zero())),
            LLType::Float(t) => Ok(LLValue::Float(t.const_zero())),
            LLType::Pointer(t) => Ok(LLValue::Pointer(t.const_zero())),
            LLType::Array(t) => Ok(LLValue::Array(t.const_zero())),
            LLType::Void | LLType::Struct(_) => Err(PecosError::Generic(
                "Cannot create a zero constant for void/struct type".to_string(),
            )),
        }
    }
}
