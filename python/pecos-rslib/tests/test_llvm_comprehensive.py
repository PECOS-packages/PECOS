"""Tests for llvmlite compatibility covering major features."""

import pytest


@pytest.fixture
def qir_module() -> tuple:
    from pecos_rslib_llvm import ir

    module = ir.Module("qir_test")
    ctx = module.context
    return module, ctx


def test_all_basic_types(qir_module) -> None:
    _, ctx = qir_module

    i1 = ctx.int_type(1)  # Boolean
    i8 = ctx.int_type(8)  # Byte
    i32 = ctx.int_type(32)  # Int
    i64 = ctx.int_type(64)  # Long
    double = ctx.double_type()
    void = ctx.void_type()

    # Verify types can be created without raising
    _ = i1, i8, i32, i64, double, void


def test_pointer_types(qir_module) -> None:
    _, ctx = qir_module

    i8 = ctx.int_type(8)
    qubit_ptr = i8.as_pointer()  # Qubit* (opaque)
    result_ptr = i8.as_pointer()  # Result* (opaque)

    # Verify pointer types can be created without raising
    _ = qubit_ptr, result_ptr


def test_array_types(qir_module) -> None:
    _, ctx = qir_module

    i8 = ctx.int_type(8)
    array_type = i8.as_array(10)

    _ = array_type  # Verify array type can be created without raising


def test_function_creation(qir_module) -> None:
    module, ctx = qir_module

    void = ctx.void_type()
    i8_ptr = ctx.int_type(8).as_pointer()

    # Main function
    main_type = ctx.function_type(void, [], False)
    main_func = module.add_function("main", main_type)

    # Quantum gate function
    gate_type = ctx.function_type(void, [i8_ptr], False)
    h_gate = module.add_function("__quantum__qis__h__body", gate_type)

    # Measurement function
    mz_type = ctx.function_type(i8_ptr, [i8_ptr, i8_ptr], False)
    mz_func = module.add_function("__quantum__qis__mz__body", mz_type)

    # Verify functions can be added without raising
    _ = main_func, h_gate, mz_func


def test_global_variables(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    i8 = ctx.int_type(8)
    array_type = i8.as_array(10)

    # Create global variable
    global_var = ir.GlobalVariable(module, array_type, "global_const")

    # Create initializer (using byte array - our implementation supports bytes for arrays)
    const_array = ir.Constant(array_type, b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")
    global_var.initializer = const_array
    global_var.global_constant = True
    global_var.linkage = "private"

    # Note: initializer is write-only, no getter implemented
    _ = global_var


def test_arithmetic_operations(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    i32 = ctx.int_type(32)
    void = ctx.void_type()

    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test_arith", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    a = ir.Constant(i32, 42)
    b = ir.Constant(i32, 10)

    sum_val = builder.add(a, b, "sum")
    diff_val = builder.sub(sum_val, b, "diff")
    prod_val = builder.mul(diff_val, ir.Constant(i32, 2), "prod")
    div_val = builder.udiv(prod_val, b, "div")

    builder.ret_void()

    # Verify arithmetic results are used (builder produces values)
    _ = sum_val, diff_val, prod_val, div_val


def test_bitwise_operations(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    i64 = ctx.int_type(64)
    void = ctx.void_type()

    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test_bitwise", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    x = ir.Constant(i64, 0xFF)
    y = ir.Constant(i64, 0x0F)

    # Use getattr for Python keywords
    and_val = getattr(builder, "and")(x, y, "and")
    or_val = getattr(builder, "or")(x, y, "or")
    xor_val = builder.xor(x, y, "xor")
    shl_val = builder.shl(x, ir.Constant(i64, 2), "shl")
    lshr_val = builder.lshr(x, ir.Constant(i64, 2), "lshr")
    not_val = builder.not_(x, "not")

    builder.ret_void()

    _ = and_val, or_val, xor_val, shl_val, lshr_val, not_val


def test_comparison_operations(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    i32 = ctx.int_type(32)
    void = ctx.void_type()

    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test_cmp", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    a = ir.Constant(i32, 42)
    b = ir.Constant(i32, 10)

    cmp_eq = builder.icmp_signed("==", a, b, "cmp_eq")
    cmp_ne = builder.icmp_signed("!=", a, b, "cmp_ne")
    cmp_gt = builder.icmp_signed(">", a, b, "cmp_gt")
    cmp_lt = builder.icmp_signed("<", a, b, "cmp_lt")

    builder.ret_void()

    _ = cmp_eq, cmp_ne, cmp_gt, cmp_lt


def test_control_flow(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    i32 = ctx.int_type(32)
    void = ctx.void_type()

    func_type = ctx.function_type(void, [i32], False)
    test_func = module.add_function("test_cf", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    arg = test_func.args[0]
    zero = ir.Constant(i32, 0)

    # Test if_then
    cond1 = builder.icmp_signed(">", arg, zero, "cond1")
    with builder.if_then(cond1):
        builder.comment("if_then block")

    # Test if_else
    cond2 = builder.icmp_signed("==", arg, zero, "cond2")
    with builder.if_else(cond2) as (then, otherwise):
        with then:
            builder.comment("then block")
        with otherwise:
            builder.comment("else block")

    builder.ret_void()

    llvm_ir = str(module)
    # Verify control flow structure is created
    assert "if.then" in llvm_ir
    assert "if.else" in llvm_ir
    assert "if.merge" in llvm_ir
    assert "br i1" in llvm_ir


def test_gep_operations(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    i8 = ctx.int_type(8)
    i32 = ctx.int_type(32)
    array_type = i8.as_array(10)

    # Create global variable
    global_var = ir.GlobalVariable(module, array_type, "test_array")

    # Create function to test GEP
    void = ctx.void_type()
    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test_gep", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    zero = ir.Constant(i32, 0)
    gep_result = global_var.gep([zero, zero])

    builder.ret_void()

    _ = gep_result


def test_comments(qir_module) -> None:
    from pecos_rslib_llvm import ir

    module, ctx = qir_module

    void = ctx.void_type()
    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test_comments", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)

    builder.comment("This is a test comment")
    builder.comment("Multiple comments")
    builder.ret_void()

    llvm_ir = str(module)
    assert "This is a test comment" in llvm_ir
    assert "Multiple comments" in llvm_ir


def test_end_to_end_ir_to_bitcode(qir_module) -> None:
    from pecos_rslib_llvm import binding, ir

    module, ctx = qir_module

    # Create a simple function
    void = ctx.void_type()
    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test", func_type)
    entry = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry)
    builder.ret_void()

    # Get LLVM IR
    llvm_ir = str(module)
    assert len(llvm_ir) > 0

    # Convert to bitcode via binding module
    module_ref = binding.parse_assembly(llvm_ir)
    bitcode = module_ref.as_bitcode()

    assert len(bitcode) > 0
    assert bitcode[:2] == b"BC"  # LLVM bitcode magic bytes

    # Test shutdown
    binding.shutdown()
