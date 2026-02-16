#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Diagnostic script for multiprocessing spawn hangs on macOS/Windows.

This script tests multiprocessing with spawn context in multiple stages,
each with a timeout, to pinpoint exactly where the hang occurs.

Run with: uv run python scripts/debug_multiprocessing_spawn.py
"""

from __future__ import annotations

import multiprocessing
import pickle
import sys
import time

from pecos_rslib import StateVec

TIMEOUT = 60  # seconds per stage


def _worker_basic(_: object) -> str:
    """Worker that returns a constant -- no imports needed."""
    return "ok"


def _worker_import(_: object) -> str:
    """Worker that imports pecos_rslib."""
    import pecos_rslib  # noqa: F401

    return "import_ok"


def _worker_pickle_statevec(data: bytes) -> str:
    """Worker that unpickles a StateVec."""
    obj = pickle.loads(data)
    return f"unpickled_statevec_qubits={obj.num_qubits}"


def _worker_full_pattern(data: bytes) -> int:
    """Worker replicating the test pattern: unpickle + operate."""
    sim = pickle.loads(data)
    sim.run_1q_gate("H", 0)
    return sim.num_qubits


def _log(msg: str) -> None:
    """Print to stderr (unbuffered) with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _run_stage(
    name: str,
    worker: object,
    args: list,
    ctx: multiprocessing.context.BaseContext,
) -> bool:
    """Run a single diagnostic stage with a timeout.

    Returns True if the stage succeeded, False otherwise.
    """
    _log(f"--- Stage: {name} ---")
    _log(f"  Creating Pool(processes=2) with context={ctx.get_start_method()!r}")

    try:
        with ctx.Pool(processes=2) as pool:
            _log("  Pool created. Submitting work via map_async...")
            async_result = pool.map_async(worker, args)
            _log(f"  Work submitted. Waiting up to {TIMEOUT}s for results...")
            results = async_result.get(timeout=TIMEOUT)
            _log(f"  Results: {results}")
            _log(f"  Stage '{name}' PASSED")
            return True
    except multiprocessing.TimeoutError:
        _log(f"  TIMEOUT after {TIMEOUT}s -- stage '{name}' HUNG")
        return False
    except (OSError, pickle.PickleError, ImportError, RuntimeError) as exc:
        _log(f"  EXCEPTION in stage '{name}': {exc}")
        return False


def _main() -> None:
    """Run all diagnostic stages."""
    _log(f"Platform: {sys.platform}")
    _log(f"Python: {sys.version}")
    _log(f"Executable: {sys.executable}")

    method = "fork" if sys.platform == "linux" else "spawn"
    _log(f"Multiprocessing start method: {method}")
    ctx = multiprocessing.get_context(method)

    # Stage 1: Basic spawn -- no imports in worker
    if not _run_stage("basic_spawn", _worker_basic, [None, None], ctx):
        _log("FAILED at basic spawn -- multiprocessing itself is broken")
        sys.exit(1)

    # Stage 2: Import pecos_rslib in worker
    if not _run_stage("import_pecos_rslib", _worker_import, [None, None], ctx):
        _log("FAILED at import -- pecos_rslib import hangs in spawned child")
        sys.exit(2)

    # Stage 3: Pickle/unpickle StateVec in worker
    _log("Preparing StateVec for stage 3...")
    sim = StateVec(3, seed=42)
    sim.run_1q_gate("H", 0)
    sim_bytes = pickle.dumps(sim)
    _log(f"  Pickled StateVec: {len(sim_bytes)} bytes")

    if not _run_stage(
        "pickle_statevec",
        _worker_pickle_statevec,
        [sim_bytes, sim_bytes],
        ctx,
    ):
        _log("FAILED at pickle -- StateVec unpickling hangs in spawned child")
        sys.exit(3)

    # Stage 4: Full test pattern (unpickle + operate)
    if not _run_stage(
        "full_pattern",
        _worker_full_pattern,
        [sim_bytes, sim_bytes],
        ctx,
    ):
        _log("FAILED at full pattern -- operation on unpickled StateVec hangs")
        sys.exit(4)

    _log("ALL STAGES PASSED")
    sys.exit(0)


if __name__ == "__main__":
    _main()
