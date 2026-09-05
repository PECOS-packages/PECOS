#!/usr/bin/env python3
"""Exercise the public Guppy-to-LLVM wrappers and their file outputs."""

import re
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit
from pecos import execute_llvm
from pecos._compilation import GuppyFrontend


@pytest.fixture
def simple_quantum_function() -> object:
    @guppy
    def simple_quantum() -> bool:
        q = qubit()
        h(q)
        return measure(q).read()

    return simple_quantum


@pytest.fixture
def hugr_bytes(simple_quantum_function: object) -> bytes:
    return simple_quantum_function.compile().to_bytes()


def test_compile_hugr_to_qis(hugr_bytes: bytes) -> None:
    """The public wrapper emits an entry point and executable QIS calls."""
    llvm_ir = execute_llvm.compile_module_to_string(hugr_bytes)
    assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
    for operation in ("___qalloc", "___lazy_measure", "___qfree"):
        assert re.search(rf"\bcall\b[^\n]*@{operation}\(", llvm_ir), operation


@pytest.mark.parametrize("input_from_file", [False, True], ids=["bytes", "file"])
def test_compile_to_file(hugr_bytes: bytes, tmp_path: Path, input_from_file: bool) -> None:
    """Both file-writing wrappers write the full compiled IR to the requested path."""
    expected = execute_llvm.compile_module_to_string(hugr_bytes)
    output_path = tmp_path / "compiled program.ll"
    if input_from_file:
        input_path = tmp_path / "input program.hugr"
        input_path.write_bytes(hugr_bytes)
        execute_llvm.compile_hugr_file_to_file(input_path, output_path)
    else:
        execute_llvm.compile_module_to_file(hugr_bytes, output_path)
    assert output_path.read_text() == expected


def test_compile_hugr_file_to_string(hugr_bytes: bytes, tmp_path: Path) -> None:
    input_path = tmp_path / "input.hugr"
    input_path.write_bytes(hugr_bytes)
    assert execute_llvm.compile_hugr_file_to_string(input_path) == execute_llvm.compile_module_to_string(hugr_bytes)


def test_guppy_frontend_integration(simple_quantum_function: object) -> None:
    """The supported Rust frontend produces a real IR file and cleans it up."""
    frontend = GuppyFrontend(use_rust_backend=True)
    try:
        qir_file = frontend.compile_function(simple_quantum_function)
        llvm_ir = qir_file.read_text()
        assert re.search(r"define\b[^\n]*@qmain\(", llvm_ir)
        assert re.search(r"\bcall\b[^\n]*@___lazy_measure\(", llvm_ir)
    finally:
        frontend.cleanup()
    assert not qir_file.exists()
