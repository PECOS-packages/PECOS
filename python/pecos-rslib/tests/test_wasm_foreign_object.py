"""Tests for WasmForeignObject error handling (div-by-zero, infinite loops, etc.)."""

import tempfile
import os

import pytest

from pecos_rslib import WasmError, WasmForeignObject


def _wasm_from_wat(wat_content: str, **kwargs) -> WasmForeignObject:
    """Helper: write WAT to a temp file and create a WasmForeignObject."""
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        path = f.name
    try:
        obj = WasmForeignObject.from_file(path, **kwargs)
    finally:
        os.remove(path)
    return obj


def test_div_by_zero() -> None:
    """Division by zero in WASM should raise WasmError."""
    wat = """
    (module
      (func $init (export "init"))
      (func $div_by_zero (export "div_by_zero") (result i32)
        i32.const 1
        i32.const 0
        i32.div_s
      )
    )
    """
    wasm = _wasm_from_wat(wat)
    wasm.init()

    with pytest.raises(WasmError, match="[Dd]ivision by zero"):
        wasm.exec("div_by_zero", [])


def test_infinite_loop() -> None:
    """An infinite loop should raise WasmError due to timeout."""
    wat = """
    (module
      (func $init (export "init"))
      (func $infinite_loop (export "infinite_loop") (result i32)
        (loop $loop
          br $loop
        )
        i32.const 0
      )
    )
    """
    wasm = _wasm_from_wat(wat, timeout=0.2)
    wasm.init()

    with pytest.raises(WasmError, match="timed out"):
        wasm.exec("infinite_loop", [])


def test_normal_execution_no_error() -> None:
    """Normal WASM execution should not raise any error."""
    wat = """
    (module
      (func $init (export "init"))
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )
    )
    """
    wasm = _wasm_from_wat(wat)
    wasm.init()

    result = wasm.exec("add", [3, 4])
    assert result == 7
