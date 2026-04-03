"""Test llvmlite-compatible ir module API."""


def test_import_ir_module() -> None:
    """Test that the ir module can be imported."""
    from pecos_rslib_llvm import ir

    assert ir is not None


def test_create_module() -> None:
    """Test creating an LLVM module."""
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    assert module is not None
    assert repr(module) == "<LLVM Module>"


def test_module_context_and_types() -> None:
    """Test accessing module context and creating types."""
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context

    # Create various types
    i32 = ctx.int_type(32)
    i64 = ctx.int_type(64)
    void = ctx.void_type()
    double = ctx.double_type()

    assert i32 is not None
    assert i64 is not None
    assert void is not None
    assert double is not None


def test_create_function() -> None:
    """Test creating a function."""
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context
    i32 = ctx.int_type(32)

    # Create function type
    func_type = ctx.function_type(i32, [i32, i32], False)
    assert func_type is not None

    # Add function to module
    add_func = module.add_function("add", func_type)
    assert add_func is not None


def test_create_basic_block_and_builder() -> None:
    """Test creating basic blocks and IRBuilder."""
    from pecos_rslib_llvm import ir

    module = ir.Module("test_module")
    ctx = module.context
    i32 = ctx.int_type(32)

    func_type = ctx.function_type(i32, [i32, i32], False)
    add_func = module.add_function("add", func_type)

    # Create basic block
    entry_block = add_func.append_basic_block("entry")
    assert entry_block is not None

    # Create IRBuilder
    builder = ir.IRBuilder(entry_block)
    assert builder is not None


def test_build_add_instruction() -> None:
    """Test building arithmetic instructions."""
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
    result = builder.add(args[0], args[1], "sum")
    assert result is not None


def test_generate_llvm_ir() -> None:
    """Test generating LLVM IR as a string."""
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
