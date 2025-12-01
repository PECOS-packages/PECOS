"""Test llvmlite-compatible control flow features (if_then, if_else)."""

import pytest


@pytest.fixture
def module_with_function():
    """Create a module with a test function."""
    from _pecos_rslib import ir

    module = ir.Module("control_flow_test")
    ctx = module.context
    i32 = ctx.int_type(32)
    void = ctx.void_type()

    func_type = ctx.function_type(void, [i32], False)
    test_func = module.add_function("test_func", func_type)
    entry_block = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry_block)

    return module, test_func, builder, i32


def test_if_then_context_manager(module_with_function):
    """Test if_then context manager."""
    from _pecos_rslib import ir

    module, test_func, builder, i32 = module_with_function

    # Create a condition (arg[0] > 0)
    arg = test_func.args[0]
    zero = ir.Constant(i32, 0)
    cond = builder.icmp_signed(">", arg, zero, "cond")

    # Use if_then context manager
    with builder.if_then(cond):
        builder.comment("Inside if_then block")

    builder.ret_void()

    # Verify the IR contains the expected control flow structure
    llvm_ir = str(module)
    assert "if.then" in llvm_ir  # Then block created
    assert "if.merge" in llvm_ir  # Merge block created
    assert "br i1" in llvm_ir  # Branch instruction


def test_if_else_context_manager(module_with_function):
    """Test if_else context manager."""
    from _pecos_rslib import ir

    module, test_func, builder, i32 = module_with_function

    # Create a condition (arg[0] == 0)
    arg = test_func.args[0]
    zero = ir.Constant(i32, 0)
    cond = builder.icmp_signed("==", arg, zero, "cond")

    # Use if_else context manager
    with builder.if_else(cond) as (then, otherwise):
        with then:
            builder.comment("Inside then branch")
        with otherwise:
            builder.comment("Inside else branch")

    builder.ret_void()

    # Verify the IR contains both branches
    llvm_ir = str(module)
    assert "if.then" in llvm_ir  # Then block created
    assert "if.else" in llvm_ir  # Else block created
    assert "if.merge" in llvm_ir  # Merge block created
    assert "br i1" in llvm_ir  # Branch instruction


def test_nested_if_then(module_with_function):
    """Test nested if_then blocks."""
    from _pecos_rslib import ir

    module, test_func, builder, i32 = module_with_function

    # Create conditions
    arg = test_func.args[0]
    zero = ir.Constant(i32, 0)
    ten = ir.Constant(i32, 10)

    cond1 = builder.icmp_signed(">", arg, zero, "cond1")
    cond2 = builder.icmp_signed("<", arg, ten, "cond2")

    # Use nested if_then
    with builder.if_then(cond1):
        builder.comment("Outer if_then")
        with builder.if_then(cond2):
            builder.comment("Inner if_then")

    builder.ret_void()

    # Verify the IR contains nested control flow structure
    llvm_ir = str(module)
    # Should have multiple if.then blocks for nested structure
    assert llvm_ir.count("if.then") >= 2 or "if.then1" in llvm_ir
    # Should have multiple merge blocks
    assert llvm_ir.count("if.merge") >= 2 or "if.merge2" in llvm_ir


def test_control_flow_generates_valid_ir():
    """Test that control flow generates valid LLVM IR."""
    from _pecos_rslib import ir

    module = ir.Module("test_module")
    ctx = module.context
    i32 = ctx.int_type(32)
    void = ctx.void_type()

    func_type = ctx.function_type(void, [i32], False)
    test_func = module.add_function("test", func_type)
    entry_block = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry_block)

    arg = test_func.args[0]
    zero = ir.Constant(i32, 0)
    cond = builder.icmp_signed(">", arg, zero, "cond")

    with builder.if_else(cond) as (then, otherwise):
        with then:
            # Do nothing, just test structure
            pass
        with otherwise:
            # Do nothing, just test structure
            pass

    builder.ret_void()

    # Get IR and verify it's non-empty and contains expected elements
    llvm_ir = str(module)
    assert len(llvm_ir) > 0
    assert "define void @test(i32" in llvm_ir
    assert "br i1" in llvm_ir
    assert "ret void" in llvm_ir
