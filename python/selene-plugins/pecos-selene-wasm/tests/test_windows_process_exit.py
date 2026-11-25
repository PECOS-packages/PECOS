"""Windows process exit must not join workers that the OS forcibly terminated."""

import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises Windows ExitProcess and DLL detach")
@pytest.mark.parametrize(("pending", "reset_idle"), [(False, False), (False, True), (True, False)])
def test_exit_process_with_async_pool(pending, reset_idle):
    script = r"""
import ctypes as c
import sys
import time
from pecos_selene_wasm import WasmPlugin

library = c.CDLL(str(WasmPlugin().library_file))
def function(name, arguments, result):
    call = getattr(library, name)
    call.argtypes, call.restype = arguments, result
    return call

register = function("pecos_wasm_v2_register", [c.c_uint64, c.c_char_p, c.c_size_t, c.c_double], c.c_int)
acquire = function("pecos_wasm_v1_acquire", [c.c_uint64], c.c_uint64)
lookup = function("pecos_wasm_v1_lookup",
    [c.c_uint64, c.c_char_p, c.c_size_t, c.c_size_t, c.c_size_t], c.c_uint64)
launch = function("pecos_wasm_v2_call",
    [c.c_uint64, c.c_uint64, c.POINTER(c.c_int64), c.c_size_t, c.c_size_t], c.c_int)
read = function("pecos_wasm_v1_read", [c.c_uint64, c.POINTER(c.c_int64), c.c_size_t], c.c_int)
payload = (b'(module (func (export "work") (param i64) (result i64) '
           b'local.get 0 i32.wrap_i64 if (loop br 0) end i64.const 7))')
assert register(0, payload, len(payload), 30.0) == 0
work = lookup(0, b"work", 4, 1, 1)
assert work != 0
pending = sys.argv[1] == "True"
for index in range(4):
    context = acquire(index)
    argument = c.c_int64(int(pending))
    assert launch(context, work, c.byref(argument), 1, 2) == 0
    if not pending:
        output = c.c_int64()
        assert read(context, c.byref(output), 1) == 0
        assert output.value == 7
if sys.argv[2] == "True":
    reset = function("pecos_wasm_v1_reset", [], c.c_int)
    assert reset() == 0
time.sleep(0.05)
exit_process = c.WinDLL("kernel32").ExitProcess
exit_process.argtypes, exit_process.restype = [c.c_uint], None
exit_process(37)
raise AssertionError("ExitProcess returned")
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(pending), str(reset_idle)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 37, child.stdout + child.stderr
    assert "panicked" not in child.stderr, child.stderr
