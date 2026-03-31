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

"""Integration tests for pickle-based multiprocessing of simulators.

Worker functions are defined in ``pecos._mp_workers`` (an installed module)
rather than in this test file, because ``multiprocessing`` with the ``spawn``
start method (default on macOS/Windows) requires workers to be importable
by the child process.  Test files live outside the installed package and
cannot be imported by spawned children.
"""

import multiprocessing
import pickle

import pytest
from pecos._mp_workers import (
    deserialize_and_call,
    run_callable_worker,
    sim_run_from_bytes,
)
from pecos.engines.hybrid_engine_multiprocessing import worker_wrapper
from pecos_rslib import CoinToss, PauliProp, SparseStab, StateVec

# Use spawn context everywhere. All worker functions live in installed modules
# (pecos._mp_workers, pecos.engines.hybrid_engine_multiprocessing), so they are
# importable by spawned child processes. Using fork in a multi-threaded process
# (e.g. pytest) can deadlock and triggers DeprecationWarning in Python 3.12+.
_MP_CONTEXT = "spawn"
_POOL_TIMEOUT = 60  # seconds -- fail fast instead of hanging CI


def _get_pool_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context(_MP_CONTEXT)


# ---------------------------------------------------------------------------
# Basic pickle round-trip tests via deserialize_and_call
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestMultiprocessingStateVec:
    """Tests for multiprocessing StateVec simulators via pickle."""

    def test_pool_map(self) -> None:
        """Test StateVec serialization works with multiprocessing Pool.map."""
        sim = StateVec(3, seed=42)
        sim.run_1q_gate("H", 0)
        sim_bytes = pickle.dumps(sim)
        args = [(sim_bytes, "run_1q_gate", ("H", 0), "num_qubits", ())] * 2
        ctx = _get_pool_context()
        with ctx.Pool(processes=2) as pool:
            results = pool.map_async(deserialize_and_call, args).get(
                timeout=_POOL_TIMEOUT,
            )
        assert results == [3, 3]


@pytest.mark.timeout(120)
class TestMultiprocessingSparseSim:
    """Tests for multiprocessing SparseStab simulators via pickle."""

    def test_pool_map(self) -> None:
        """Test SparseStab serialization works with multiprocessing Pool.map."""
        sim = SparseStab(4)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        sim_bytes = pickle.dumps(sim)
        args = [(sim_bytes, "run_1q_gate", ("H", 0), "num_qubits", ())] * 2
        ctx = _get_pool_context()
        with ctx.Pool(processes=2) as pool:
            results = pool.map_async(deserialize_and_call, args).get(
                timeout=_POOL_TIMEOUT,
            )
        assert results == [4, 4]


@pytest.mark.timeout(120)
class TestMultiprocessingCoinToss:
    """Tests for multiprocessing CoinToss simulators via pickle."""

    def test_pool_map(self) -> None:
        """Test CoinToss serialization works with multiprocessing Pool.map."""
        sim = CoinToss(5, prob=0.3)
        sim_bytes = pickle.dumps(sim)
        args = [(sim_bytes, "run_measure", (0,), "num_qubits", ())] * 2
        ctx = _get_pool_context()
        with ctx.Pool(processes=2) as pool:
            results = pool.map_async(deserialize_and_call, args).get(
                timeout=_POOL_TIMEOUT,
            )
        assert results == [5, 5]


@pytest.mark.timeout(120)
class TestMultiprocessingPauliProp:
    """Tests for multiprocessing PauliProp simulators via pickle."""

    def test_pool_map(self) -> None:
        """Test PauliProp serialization works with multiprocessing Pool.map."""
        sim = PauliProp(num_qubits=3, track_sign=True)
        sim.track_x([0])
        sim_bytes = pickle.dumps(sim)
        args = [(sim_bytes, "h", ([0],), "weight", ())] * 2
        ctx = _get_pool_context()
        with ctx.Pool(processes=2) as pool:
            results = pool.map_async(deserialize_and_call, args).get(
                timeout=_POOL_TIMEOUT,
            )
        # After H on qubit 0: X->Z, so weight should still be 1
        assert all(r == 1 for r in results)


# ---------------------------------------------------------------------------
# Production-pattern tests: Manager queue + worker_wrapper
#
# These mirror the pattern used in hybrid_engine_multiprocessing.run_multisim:
#   1. Create a Manager().Queue() for inter-process messaging
#   2. Pass (queue, callable, kwargs, index) to worker_wrapper via pool.map
#   3. worker_wrapper redirects stdout/stderr to WriteStream on the queue
#   4. worker_wrapper calls the callable and returns (result_dict, run_info)
#   5. Parent drains the queue and aggregates results
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestWorkerWrapperPattern:
    """Tests that mirror the production worker_wrapper + Manager queue pattern."""

    def test_worker_wrapper_with_statevec(self) -> None:
        """Test the production worker_wrapper pattern with StateVec."""
        sim = StateVec(3, seed=42)
        sim.run_1q_gate("H", 0)
        sim_bytes = pickle.dumps(sim)

        ctx = _get_pool_context()
        manager = ctx.Manager()
        queue = manager.Queue()

        kwargs = {
            "sim_bytes": sim_bytes,
            "method": "run_1q_gate",
            "method_args": ("H", 0),
            "result_attr": "num_qubits",
            "seed": 1,
            "shots": 1,
            "foreign_object": None,
        }
        worker_args = [
            (queue, sim_run_from_bytes, {**kwargs, "seed": 1}, 0),
            (queue, sim_run_from_bytes, {**kwargs, "seed": 2}, 1),
        ]

        with ctx.Pool(processes=2) as pool:
            presults = pool.map_async(worker_wrapper, worker_args).get(
                timeout=_POOL_TIMEOUT,
            )

        for result_dict, run_info in presults:
            assert result_dict == {"measurements": [3]}
            assert "pid" in run_info
            assert "i" in run_info

    def test_worker_wrapper_with_sparsestab(self) -> None:
        """Test the production worker_wrapper pattern with SparseStab."""
        sim = SparseStab(4)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        sim_bytes = pickle.dumps(sim)

        ctx = _get_pool_context()
        manager = ctx.Manager()
        queue = manager.Queue()

        kwargs = {
            "sim_bytes": sim_bytes,
            "method": "run_1q_gate",
            "method_args": ("H", 0),
            "result_attr": "num_qubits",
            "seed": 1,
            "shots": 1,
            "foreign_object": None,
        }
        worker_args = [
            (queue, sim_run_from_bytes, {**kwargs, "seed": 1}, 0),
            (queue, sim_run_from_bytes, {**kwargs, "seed": 2}, 1),
        ]

        with ctx.Pool(processes=2) as pool:
            presults = pool.map_async(worker_wrapper, worker_args).get(
                timeout=_POOL_TIMEOUT,
            )

        for result_dict, run_info in presults:
            assert result_dict == {"measurements": [4]}
            assert "pid" in run_info

    def test_queue_message_passing(self) -> None:
        """Test that stdout/stderr from workers is captured on the queue."""
        sim = StateVec(2, seed=0)
        sim_bytes = pickle.dumps(sim)

        ctx = _get_pool_context()
        manager = ctx.Manager()
        queue = manager.Queue()

        kwargs = {
            "sim_bytes": sim_bytes,
            "method": "run_1q_gate",
            "method_args": ("H", 0),
            "result_attr": "num_qubits",
            "seed": 1,
            "shots": 1,
            "foreign_object": None,
        }
        worker_args = [(queue, sim_run_from_bytes, kwargs, 0)]

        with ctx.Pool(processes=1) as pool:
            pool.map_async(worker_wrapper, worker_args).get(timeout=_POOL_TIMEOUT)

        # The queue may contain stdout/stderr messages captured by WriteStream.
        # We just verify the queue is accessible and drainable (no deadlock).
        messages = []
        while not queue.empty():
            messages.append(queue.get())
        # Messages are (pid, stream_type, data) tuples if any output occurred.
        for msg in messages:
            assert len(msg) == 3


# ---------------------------------------------------------------------------
# Callable-with-kwargs pattern tests (run_callable_worker)
#
# This tests the simpler pattern where a callable + kwargs dict are passed
# to the pool, similar to how run_multisim passes eng.run + kwargs to workers.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
class TestRunCallableWorker:
    """Tests for the callable+kwargs worker pattern used in production."""

    def test_callable_worker_statevec(self) -> None:
        """Test passing a callable + kwargs through the pool."""
        sim = StateVec(3, seed=42)
        sim.run_1q_gate("H", 0)
        sim_bytes = pickle.dumps(sim)

        kwargs = {
            "sim_bytes": sim_bytes,
            "method": "run_1q_gate",
            "method_args": ("H", 0),
            "result_attr": "num_qubits",
        }
        args = [(sim_run_from_bytes, kwargs)] * 2

        ctx = _get_pool_context()
        with ctx.Pool(processes=2) as pool:
            results = pool.map_async(run_callable_worker, args).get(
                timeout=_POOL_TIMEOUT,
            )

        assert results == [{"measurements": [3]}, {"measurements": [3]}]
