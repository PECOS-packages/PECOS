"""A failed compilation must still pin the native runtime's signal handlers."""

import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux /proc maps and task inspection")
def test_failed_compilation_pins_library():
    script = r"""
import _ctypes
import ctypes as c
import os
import threading
import time
import traceback
from pecos_selene_wasm import WasmPlugin

path = str(WasmPlugin().library_file)
library = c.CDLL(path)
def function(name, arguments, result):
    call = getattr(library, name)
    call.argtypes, call.restype = arguments, result
    return call

errors = []
def worker():
    try:
        register = function("pecos_wasm_v1_register", [c.c_uint64, c.c_char_p, c.c_size_t], c.c_int)
        acquire = function("pecos_wasm_v1_acquire", [c.c_uint64], c.c_uint64)
        lookup = function("pecos_wasm_v1_lookup",
            [c.c_uint64, c.c_char_p, c.c_size_t, c.c_size_t, c.c_size_t], c.c_uint64)
        call = function("pecos_wasm_v1_call",
            [c.c_uint64, c.c_uint64, c.POINTER(c.c_int64), c.c_size_t], c.c_int)
        reset = function("pecos_wasm_v1_reset", [], c.c_int)
        error = function("pecos_wasm_v1_error", [], c.POINTER(c.c_ubyte))
        payload = b"invalid wasm bytes"
        assert register(0, payload, len(payload)) == 0
        handle = lookup(0, b"work", 4, 0, 0)
        context = acquire(0)
        assert handle != 0 and context != 0
        assert call(context, handle, None, 0) == -1
        message = error()
        assert b"Failed to compile WebAssembly module" in bytes(message[1:1 + message[0]])
        assert reset() == 0
    except BaseException:
        errors.append(traceback.format_exc())

baseline_threads = len(os.listdir("/proc/self/task"))
thread = threading.Thread(target=worker)
thread.start()
thread.join()
assert not errors, "\n".join(errors)
# Python join can return before the OS finishes native TLS destruction.
deadline = time.monotonic() + 10
while len(os.listdir("/proc/self/task")) != baseline_threads:
    assert time.monotonic() < deadline, "worker OS thread did not exit"
    time.sleep(0.01)
# No native runtime call was made on the main thread, and no ticker was started.
_ctypes.dlclose(library._handle)
with open("/proc/self/maps", encoding="utf-8") as maps:
    assert os.path.realpath(path) in maps.read(), "runtime library was unmapped"
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
