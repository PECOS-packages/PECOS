# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for QuEST seed determinism and randomness."""

import pytest
from pecos.engines.hybrid_engine import HybridEngine
from pecos.error_models.generic_error_model import GenericErrorModel

PHIR_BELL_STATE = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
        {"data": "cvar_define", "data_type": "i64", "variable": "c", "size": 2},
        {"qop": "H", "angles": null, "args": [["q", 0]]},
        {"qop": "CX", "angles": null, "args": [[["q", 0], ["q", 1]]]},
        {"qop": "Measure", "returns": [["c", 0]], "args": [["q", 0]]},
        {"qop": "Measure", "returns": [["c", 1]], "args": [["q", 1]]}
    ]
}"""

ERROR_MODEL = GenericErrorModel(
    error_params={
        "p1": 2e-1,
        "p2": 2e-1,
        "p_meas": 2e-1,
        "p_init": 1e-1,
        "p1_error_model": {"X": 0.25, "Y": 0.25, "Z": 0.25, "L": 0.25},
    },
)


@pytest.mark.parametrize("qsim", ["QuestStateVec", "QuestDensityMatrix"])
def test_measurement_determinism_with_use_seed(qsim: str) -> None:
    """Test that measurements are deterministic when using use_seed()."""
    seed = 42
    shots = 100

    # Run first simulation
    engine1 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    engine1.use_seed(seed)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots)

    # Run second simulation with same seed
    engine2 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    engine2.use_seed(seed)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots)

    # Results should be identical
    assert (
        results1["c"] == results2["c"]
    ), f"{qsim}: Same seed should produce identical measurement results"


@pytest.mark.parametrize("qsim", ["QuestStateVec", "QuestDensityMatrix"])
def test_measurement_determinism_with_seed_parameter(qsim: str) -> None:
    """Test that measurements are deterministic when passing seed to run()."""
    seed = 123
    shots = 100

    # Run first simulation
    engine1 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots, seed=seed)

    # Run second simulation with same seed
    engine2 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots, seed=seed)

    # Results should be identical
    assert (
        results1["c"] == results2["c"]
    ), f"{qsim}: Same seed parameter should produce identical results"


@pytest.mark.parametrize("qsim", ["QuestStateVec", "QuestDensityMatrix"])
def test_different_seeds_produce_different_results(qsim: str) -> None:
    """Test that different seeds produce different measurement outcomes."""
    shots = 100

    # Run with seed 1
    engine1 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    engine1.use_seed(12345)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots)

    # Run with seed 2
    engine2 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    engine2.use_seed(67890)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots)

    # Results should be different (with very high probability)
    assert (
        results1["c"] != results2["c"]
    ), f"{qsim}: Different seeds should produce different results"


@pytest.mark.parametrize("qsim", ["QuestStateVec", "QuestDensityMatrix"])
def test_randomness_without_seed(qsim: str) -> None:
    """Test that measurements show randomness when no seed is set.

    Note: This test has a small probability of false failure if random outcomes
    happen to match by chance.
    """
    shots = 50
    num_trials = 5

    all_results = []
    for _ in range(num_trials):
        engine = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
        # No seed set - should be random
        results = engine.run(PHIR_BELL_STATE, shots=shots)
        all_results.append(results["c"])

    # Check that not all results are identical
    # Probability of all being the same is astronomically small
    all_same = all(results == all_results[0] for results in all_results)
    assert not all_same, f"{qsim}: Unseeded simulations should show randomness"


@pytest.mark.parametrize("qsim", ["QuestStateVec", "QuestDensityMatrix"])
def test_seed_produces_reproducible_error_patterns(qsim: str) -> None:
    """Test that error patterns are reproducible with seeds."""
    seed = 999
    shots = 200

    # Run first simulation
    engine1 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots, seed=seed)

    # Run second simulation with same seed
    engine2 = HybridEngine(qsim=qsim, error_model=ERROR_MODEL)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots, seed=seed)

    # Count outcomes
    count_00_1 = sum(1 for x in results1["c"] if x == "00")
    count_11_1 = sum(1 for x in results1["c"] if x == "11")
    count_00_2 = sum(1 for x in results2["c"] if x == "00")
    count_11_2 = sum(1 for x in results2["c"] if x == "11")

    # Exact counts should match (not just distributions)
    assert (
        count_00_1 == count_00_2
    ), f"{qsim}: Exact outcome counts should match with same seed"
    assert (
        count_11_1 == count_11_2
    ), f"{qsim}: Exact outcome counts should match with same seed"
    assert results1["c"] == results2["c"], f"{qsim}: Full sequences should be identical"
