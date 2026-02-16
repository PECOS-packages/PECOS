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

"""Integration tests for pickle-based multiprocessing of simulators."""

import multiprocessing
import pickle
import sys

import pytest

from pecos_rslib import CoinToss, PauliProp, SparseSim, StateVec


def _statevec_worker(sim_bytes):
    sim = pickle.loads(sim_bytes)
    sim.run_1q_gate("H", 0)
    return sim.num_qubits


def _sparsesim_worker(sim_bytes):
    sim = pickle.loads(sim_bytes)
    sim.run_1q_gate("H", 0)
    return sim.num_qubits


def _cointoss_worker(sim_bytes):
    sim = pickle.loads(sim_bytes)
    sim.run_measure(0)
    return sim.num_qubits


def _pauliprop_worker(sim_bytes):
    sim = pickle.loads(sim_bytes)
    sim.h(0)
    return sim.weight()


# Use fork context on Linux (fast, avoids spawn serialization issues with test files).
# On macOS/Windows where fork is unavailable or unsafe, use spawn.
_MP_CONTEXT = "fork" if sys.platform == "linux" else "spawn"


class TestMultiprocessingStateVec:
    def test_pool_map(self):
        sim = StateVec(3, seed=42)
        sim.run_1q_gate("H", 0)
        sim_bytes = pickle.dumps(sim)
        ctx = multiprocessing.get_context(_MP_CONTEXT)
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_statevec_worker, [sim_bytes, sim_bytes])
        assert results == [3, 3]


class TestMultiprocessingSparseSim:
    def test_pool_map(self):
        sim = SparseSim(4)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        sim_bytes = pickle.dumps(sim)
        ctx = multiprocessing.get_context(_MP_CONTEXT)
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_sparsesim_worker, [sim_bytes, sim_bytes])
        assert results == [4, 4]


class TestMultiprocessingCoinToss:
    def test_pool_map(self):
        sim = CoinToss(5, prob=0.3)
        sim_bytes = pickle.dumps(sim)
        ctx = multiprocessing.get_context(_MP_CONTEXT)
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_cointoss_worker, [sim_bytes, sim_bytes])
        assert results == [5, 5]


class TestMultiprocessingPauliProp:
    def test_pool_map(self):
        sim = PauliProp(num_qubits=3, track_sign=True)
        sim.add_x(0)
        sim_bytes = pickle.dumps(sim)
        ctx = multiprocessing.get_context(_MP_CONTEXT)
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_pauliprop_worker, [sim_bytes, sim_bytes])
        # After H on qubit 0: X->Z, so weight should still be 1
        assert all(r == 1 for r in results)
