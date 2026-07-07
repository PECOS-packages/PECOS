//! Implementation for heap allocation of arrays using the selene heap.
//!
//! Array codegen support compatible with Selene's QIS runtime.

use anyhow::Result;
use tket::hugr::llvm::emit::EmitFuncContext;
use tket::hugr::llvm::extension::collections::array::ArrayCodegen;
use tket::hugr::llvm::inkwell::AddressSpace;
use tket::hugr::llvm::inkwell::values::{IntValue, PointerValue};
use tket::hugr::{HugrView, Node};
use tket_qsystem::llvm::array_utils::HeapArrayLowering;

#[derive(Clone, Debug, Default)]
/// Codegen extension for array operations using the selene heap.
pub struct SeleneHeapArrayCodegen;

impl ArrayCodegen for SeleneHeapArrayCodegen {
    fn emit_allocate_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        size: IntValue<'c>,
    ) -> Result<PointerValue<'c>> {
        let iw_ctx = ctx.typing_session().iw_context();
        let ptr_ty = iw_ctx.ptr_type(AddressSpace::default());
        let malloc_sig = ptr_ty.fn_type(&[iw_ctx.i64_type().into()], false);
        let malloc = ctx.get_extern_func("heap_alloc", malloc_sig)?;
        let res = ctx
            .builder()
            .build_call(malloc, &[size.into()], "")?
            .try_as_basic_value()
            .expect_basic("heap_alloc should return a value");
        Ok(res.into_pointer_value())
    }

    fn emit_free_array<'c, H: HugrView<Node = Node>>(
        &self,
        ctx: &mut EmitFuncContext<'c, '_, H>,
        ptr: PointerValue<'c>,
    ) -> Result<()> {
        let iw_ctx = ctx.typing_session().iw_context();
        let ptr_ty = iw_ctx.ptr_type(AddressSpace::default());
        let ptr = ctx.builder().build_bit_cast(ptr, ptr_ty, "")?;

        let free_sig = iw_ctx.void_type().fn_type(&[ptr_ty.into()], false);
        let free = ctx.get_extern_func("heap_free", free_sig)?;
        ctx.builder().build_call(free, &[ptr.into()], "")?;
        Ok(())
    }
}

impl SeleneHeapArrayCodegen {
    /// [`HeapArrayLowering`] using the selene heap.
    pub const LOWERING: HeapArrayLowering<Self> = HeapArrayLowering::new(SeleneHeapArrayCodegen);
}
