"""Test llvmlite-compatible ir module API."""


def test_import_ir_module() -> None:
    from pecos_rslib_llvm import ir  # noqa: F401


def test_create_module() -> None:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    assert repr(module) == "<LLVM Module>"


def test_module_context_and_types() -> None:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context

    # Create various types
    i32 = ctx.int_type(32)
    i64 = ctx.int_type(64)
    void = ctx.void_type()
    double = ctx.double_type()

    # Verify types can be created without raising
    _ = i32, i64, void, double


def test_create_function() -> None:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context
    i32 = ctx.int_type(32)

    # Create function type
    func_type = ctx.function_type(i32, [i32, i32], False)

    # Add function to module
    module.add_function("add", func_type)


def test_create_basic_block_and_builder() -> None:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context
    i32 = ctx.int_type(32)

    func_type = ctx.function_type(i32, [i32, i32], False)
    add_func = module.add_function("add", func_type)

    # Create basic block
    entry_block = add_func.append_basic_block("entry")

    # Create IRBuilder
    ir.IRBuilder(entry_block)


def test_build_add_instruction() -> None:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context
    i32 = ctx.int_type(32)

    func_type = ctx.function_type(i32, [i32, i32], False)
    add_func = module.add_function("add", func_type)
    entry_block = add_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry_block)

    # Get function arguments
    args = add_func.args
    assert len(args) == 2

    # Build add instruction
    builder.add(args[0], args[1], "sum")


def test_generate_llvm_ir() -> None:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context
    void = ctx.void_type()

    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test", func_type)
    entry_block = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry_block)
    builder.ret_void()

    # Get LLVM IR as string
    llvm_ir = str(module)
    assert isinstance(llvm_ir, str)
    assert len(llvm_ir) > 0
    assert "define void @test()" in llvm_ir
    assert "ret void" in llvm_ir
