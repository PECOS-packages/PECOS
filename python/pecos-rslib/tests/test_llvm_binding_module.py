"""Test llvmlite-compatible binding module API."""

import pytest


@pytest.fixture
def simple_llvm_ir() -> str:
    from pecos_rslib_llvm import ir

    module = ir.Module("test_binding")
    ctx = module.context
    void = ctx.void_type()
    func_type = ctx.function_type(void, [], False)
    test_func = module.add_function("test", func_type)
    entry_block = test_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry_block)
    builder.ret_void()

    return str(module)


def test_import_binding_module() -> None:
    from pecos_rslib_llvm import binding  # noqa: F401


def test_binding_shutdown() -> None:
    from pecos_rslib_llvm import binding

    # Should not raise any errors
    binding.shutdown()


def test_binding_multiple_shutdowns() -> None:
    from pecos_rslib_llvm import binding

    # Multiple calls should be safe
    binding.shutdown()
    binding.shutdown()
    binding.shutdown()


def test_parse_assembly(simple_llvm_ir) -> None:
    from pecos_rslib_llvm import binding

    binding.parse_assembly(simple_llvm_ir)


def test_convert_to_bitcode(simple_llvm_ir) -> None:
    from pecos_rslib_llvm import binding

    module_ref = binding.parse_assembly(simple_llvm_ir)
    bitcode = module_ref.as_bitcode()

    assert isinstance(bitcode, bytes)
    assert len(bitcode) > 0
    # LLVM bitcode should start with 'BC' magic bytes
    assert bitcode[:2] == b"BC"


def test_bitcode_format(simple_llvm_ir) -> None:
    from pecos_rslib_llvm import binding

    module_ref = binding.parse_assembly(simple_llvm_ir)
    bitcode = module_ref.as_bitcode()

    # Verify it's binary data (not text)
    assert isinstance(bitcode, bytes)

    # Bitcode should be reasonably sized
    assert len(bitcode) > 10  # At least some header bytes

    # First two bytes should be 'BC' (0x42 0x43)
    assert bitcode[0] == 0x42  # 'B'
    assert bitcode[1] == 0x43  # 'C'


def test_value_ref() -> None:
    from pecos_rslib_llvm import binding

    binding.ValueRef()


def test_ir_and_binding_integration(simple_llvm_ir) -> None:
    from pecos_rslib_llvm import binding

    # Parse IR
    module_ref = binding.parse_assembly(simple_llvm_ir)

    # Convert to bitcode
    bitcode = module_ref.as_bitcode()

    # Shutdown
    binding.shutdown()

    # Verify bitcode is still valid
    assert len(bitcode) > 0
    assert bitcode[:2] == b"BC"


def test_complex_ir_to_bitcode() -> None:
    from pecos_rslib_llvm import binding, ir

    # Create a more complex module
    module = ir.Module("complex_test")
    ctx = module.context
    i32 = ctx.int_type(32)
    void = ctx.void_type()

    # Add function (using void to match ret_void)
    func_type = ctx.function_type(void, [i32, i32], False)
    add_func = module.add_function("add", func_type)
    entry_block = add_func.append_basic_block("entry")
    builder = ir.IRBuilder(entry_block)

    # Build some instructions
    args = add_func.args
    builder.add(args[0], args[1], "sum")
    builder.ret_void()

    llvm_ir = str(module)

    # Convert to bitcode
    module_ref = binding.parse_assembly(llvm_ir)
    bitcode = module_ref.as_bitcode()

    assert len(bitcode) > 0
    assert bitcode[:2] == b"BC"
