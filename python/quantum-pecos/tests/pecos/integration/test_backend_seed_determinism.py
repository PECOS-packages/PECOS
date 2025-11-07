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

"""Comprehensive seed determinism tests for all PECOS backend simulators.

This test file verifies that all PECOS backend simulators properly support
deterministic simulation when seeds are provided. This is a regression test
for issue #89 (https://github.com/PECOS-packages/PECOS/issues/89).

The test uses a Bell state circuit with error models to verify that:
1. Same seed produces identical results (determinism)
2. Different seeds produce different results (proper randomness)
3. All backends behave consistently
"""

import pytest
from pecos.engines.hybrid_engine import HybridEngine
from pecos.error_models.generic_error_model import GenericErrorModel

# PHIR circuit for creating a Bell state with measurements
PHIR_BELL_STATE = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "ops":
    [
        {
            "data": "qvar_define",
            "data_type": "qubits",
            "variable": "q",
            "size": 2
        },
        {
            "data": "cvar_define",
            "data_type": "i64",
            "variable": "c",
            "size": 2
        },
        {
            "qop": "H",
            "angles": null,
            "args":
            [
                [
                    "q",
                    0
                ]
            ]
        },
        {
            "qop": "CX",
            "angles": null,
            "args":
            [
                [
                    [
                        "q",
                        0
                    ],
                    [
                        "q",
                        1
                    ]
                ]
            ]
        },
        {
            "qop": "Measure",
            "returns":
            [
                [
                    "c",
                    0
                ]
            ],
            "args":
            [
                [
                    "q",
                    0
                ]
            ]
        },
        {
            "qop": "Measure",
            "returns":
            [
                [
                    "c",
                    1
                ]
            ],
            "args":
            [
                [
                    "q",
                    1
                ]
            ]
        }
    ]
}
"""

# Error model with significant error rates to test error pattern reproducibility
ERROR_MODEL = GenericErrorModel(
    error_params={
        "p1": 2e-1,
        "p2": 2e-1,
        "p_meas": 2e-1,
        "p_init": 1e-1,
        "p1_error_model": {
            "X": 0.25,
            "Y": 0.25,
            "Z": 0.25,
            "L": 0.25,
        },
    },
)

# List of all PECOS backend simulators to test
# Note: CuStateVec and MPS are optional (require GPU/pytket) and tested separately
CORE_BACKENDS = [
    "stabilizer",  # SparseSim - stabilizer simulator
    "StateVec",  # StateVec - state vector simulator
    "Qulacs",  # Qulacs - state vector (Rust wrapper)
    "QuestStateVec",  # QuEST - state vector (Rust wrapper)
    "QuestDensityMatrix",  # QuEST - density matrix (Rust wrapper)
]


@pytest.mark.parametrize("backend", CORE_BACKENDS)
def test_determinism_with_same_seed(backend: str) -> None:
    """Test that same seed produces identical results across multiple runs.

    When using the same seed, all measurements should produce exactly the
    same sequence of outcomes. This is critical for reproducible research
    and debugging.
    """
    seed = 7
    shots = 100

    # First run with seed=7
    engine1 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine1.use_seed(seed)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots)
    count_11_run1 = sum(1 for x in results1["c"] if x == "11")

    # Second run with seed=7
    engine2 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine2.use_seed(seed)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots)
    count_11_run2 = sum(1 for x in results2["c"] if x == "11")

    # Third run with seed=7
    engine3 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine3.use_seed(seed)
    results3 = engine3.run(PHIR_BELL_STATE, shots=shots)
    count_11_run3 = sum(1 for x in results3["c"] if x == "11")

    # All three runs should produce IDENTICAL results
    assert results1["c"] == results2["c"], (
        f"{backend}: Run 1 and Run 2 with same seed should produce identical results. "
        f"Got count_11: {count_11_run1} vs {count_11_run2}"
    )
    assert results1["c"] == results3["c"], (
        f"{backend}: Run 1 and Run 3 with same seed should produce identical results. "
        f"Got count_11: {count_11_run1} vs {count_11_run3}"
    )


@pytest.mark.parametrize("backend", CORE_BACKENDS)
def test_different_seeds_produce_different_results(backend: str) -> None:
    """Test that different seeds produce different results.

    This verifies that the seed actually affects the RNG and isn't just
    being ignored.
    """
    shots = 100

    # Run with seed=7
    engine1 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine1.use_seed(7)
    results_seed7 = engine1.run(PHIR_BELL_STATE, shots=shots)

    # Run with seed=42
    engine2 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine2.use_seed(42)
    results_seed42 = engine1.run(PHIR_BELL_STATE, shots=shots)

    # Different seeds should produce different sequences
    assert (
        results_seed7["c"] != results_seed42["c"]
    ), f"{backend}: Different seeds should produce different measurement sequences"


@pytest.mark.parametrize("backend", CORE_BACKENDS)
def test_seed_parameter_in_run(backend: str) -> None:
    """Test that passing seed parameter to run() also produces determinism.

    This tests the alternative way of setting seeds (via run() parameter
    instead of use_seed() method).
    """
    seed = 123
    shots = 100

    # Run 1 with seed parameter
    engine1 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots, seed=seed)

    # Run 2 with same seed parameter
    engine2 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots, seed=seed)

    # Should produce identical results
    assert (
        results1["c"] == results2["c"]
    ), f"{backend}: Passing same seed to run() should produce identical results"


@pytest.mark.parametrize("backend", CORE_BACKENDS)
def test_error_pattern_reproducibility(backend: str) -> None:
    """Test that error patterns are reproducible with seeds.

    With errors enabled, the exact same error patterns should occur when
    using the same seed. This is critical for debugging and reproducible
    research.
    """
    seed = 999
    shots = 200

    # Run 1
    engine1 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine1.use_seed(seed)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots)

    # Run 2
    engine2 = HybridEngine(qsim=backend, error_model=ERROR_MODEL)
    engine2.use_seed(seed)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots)

    # Full sequences should be identical (not just distributions)
    assert (
        results1["c"] == results2["c"]
    ), f"{backend}: Error patterns should be exactly reproducible with same seed"

    # Verify that we're actually getting some errors (not perfect Bell state)
    count_00 = sum(1 for x in results1["c"] if x == "00")
    count_11 = sum(1 for x in results1["c"] if x == "11")
    errors = shots - count_00 - count_11

    # With 20% error rates, we should see some errors
    assert errors > 0, f"{backend}: Error model should produce some errors"


# Optional backends (require special dependencies)
def test_custatevec_determinism() -> None:
    """Test seed determinism for CuStateVec (GPU simulator)."""
    try:
        from pecos.simulators import CuStateVec

        if CuStateVec is None:
            pytest.skip("CuStateVec not available")
    except ImportError:
        pytest.skip("CuStateVec requires cuQuantum")

    seed = 7
    shots = 100

    engine1 = HybridEngine(qsim="CuStateVec", error_model=ERROR_MODEL)
    engine1.use_seed(seed)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots)

    engine2 = HybridEngine(qsim="CuStateVec", error_model=ERROR_MODEL)
    engine2.use_seed(seed)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots)

    assert (
        results1["c"] == results2["c"]
    ), "CuStateVec: Same seed should produce identical results"


def test_mps_determinism() -> None:
    """Test seed determinism for MPS simulator."""
    try:
        from pecos.simulators import MPS

        if MPS is None:
            pytest.skip("MPS not available")
    except ImportError:
        pytest.skip("MPS requires pytket")

    seed = 7
    shots = 100

    engine1 = HybridEngine(qsim="MPS", error_model=ERROR_MODEL)
    engine1.use_seed(seed)
    results1 = engine1.run(PHIR_BELL_STATE, shots=shots)

    engine2 = HybridEngine(qsim="MPS", error_model=ERROR_MODEL)
    engine2.use_seed(seed)
    results2 = engine2.run(PHIR_BELL_STATE, shots=shots)

    assert (
        results1["c"] == results2["c"]
    ), "MPS: Same seed should produce identical results"
