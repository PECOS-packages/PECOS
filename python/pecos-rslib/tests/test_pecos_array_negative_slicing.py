"""Compare PECOS negative-step slicing with NumPy, including clipped bounds."""

import numpy as np
import pytest
from pecos_rslib import Array


@pytest.mark.parametrize("size", [0, 1, 4, 5])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(slice(None, None, -1), id="reverse"),
        pytest.param(slice(3, None, -1), id="reverse-from-index"),
        pytest.param(slice(-1, None, -1), id="negative-start"),
        *[pytest.param(slice(3, stop, -1), id=f"stop-{stop}") for stop in [-1, -2, -3, -4, -5, -10, -100, -1000]],
        pytest.param(slice(3, 3, -1), id="equal-bounds"),
        pytest.param(slice(None, None, -2), id="step-minus-two"),
    ],
)
def test_negative_step_matches_numpy(size: int, index: slice) -> None:
    reference = np.arange(size, dtype=np.float64)
    actual = Array(reference)[index]
    np.testing.assert_array_equal(np.asarray(actual), reference[index])
