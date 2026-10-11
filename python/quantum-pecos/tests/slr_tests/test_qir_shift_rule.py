"""Shift-count guards in both independent SLR QIR generators."""

import pytest
from pecos.slr import CReg, Main
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import generate
from pecos.slr.gen_codes import QIRGenerator


def compile_program(program, generator):
    """Compile without sharing a mutated SLR program across generators."""
    if generator == "ast":
        return generate(slr_to_ast(program), "qir")
    direct = QIRGenerator(_internal=True)
    direct.generate_block(program)
    return direct.get_output()


@pytest.mark.parametrize("generator", ["ast", "direct"])
@pytest.mark.parametrize("direction", ["left", "right"])
def test_shift_rule_qir_guard(generator, direction):
    """Fix demonstration: each variable shift has an unsigned inclusive guard."""
    value, count, result = CReg("value", 32), CReg("count", 32), CReg("result", 32)
    expr = value << count if direction == "left" else value >> count
    program = Main(value, count, result, value.set(8), count.set(64), result.set(expr))
    ir = compile_program(program, generator)
    assert ir.count("icmp uge") == 1
    assert ir.count("select i1") == 1
    assert ", 64" in next(line for line in ir.splitlines() if "icmp uge" in line)


@pytest.mark.parametrize("generator", ["ast", "direct"])
def test_shift_rule_qir_pack_unchanged(generator):
    """Compatibility pin: register packing and unpacking need no guards."""
    value, result = CReg("value", 32), CReg("result", 32)
    ir = compile_program(Main(value, result, value.set(8), result.set(value)), generator)
    assert "icmp uge" not in ir
    assert "select i1" not in ir


@pytest.mark.parametrize(("generator", "count_value"), [("ast", 64), ("direct", 64), ("ast", 2**63)])
@pytest.mark.parametrize("direction", ["left", "right"])
def test_shift_rule_qir_executed(generator, count_value, direction):
    """Fix demonstration: LLVM executes guarded shifts with full runtime counts.

    The JIT callbacks supply only CReg storage and output collection; LLVM
    executes the generated expressions. This also supports the direct
    generator's legacy CReg runtime calls, which qir-qis cannot lower.
    """
    import ctypes

    from llvmlite import binding

    width = 64 if generator == "ast" else 32
    value, count, result = CReg("value", width), CReg("count", width), CReg("result", width)
    expr = value << count if direction == "left" else value >> count
    program = Main(value, count, result, value.set(8), count.set(count_value), result.set(expr))
    ir = compile_program(program, generator)
    words = []
    outputs = []

    @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
    def create_creg(_width):
        word = ctypes.c_uint64()
        words.append(word)
        return ctypes.addressof(word)

    @ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64))
    def get_int_from_creg(ptr):
        return ptr[0]

    @ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64)
    def set_creg_to_int(ptr, value):
        ptr[0] = value

    @ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p)
    def record_output(value, _tag):
        outputs.append(value)

    for name, callback in [
        ("create_creg", create_creg),
        ("get_int_from_creg", get_int_from_creg),
        ("set_creg_to_int", set_creg_to_int),
        ("__quantum__rt__int_record_output", record_output),
    ]:
        binding.add_symbol(name, ctypes.cast(callback, ctypes.c_void_p).value)
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()
    module = binding.parse_assembly(ir)
    module.verify()
    target = binding.Target.from_default_triple().create_target_machine()
    with binding.create_mcjit_compiler(module, target) as engine:
        engine.finalize_object()
        ctypes.CFUNCTYPE(None)(engine.get_function_address("main"))()
    assert outputs == [8, count_value, 0]
