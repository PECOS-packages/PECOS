"""Enforce broad, independent verification of every general-noise channel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import ChannelCoverage

pytestmark = pytest.mark.slow

REQUIRED_CHANNELS = frozenset(
    {
        "combined-channels",
        "gate-leakage",
        "idle-coherent",
        "idle-linear",
        "idle-sine-squared",
        "layered-multiqubit",
        "measurement-crosstalk",
        "measurement-crosstalk-multiqubit",
        "measurement-crosstalk-repeated",
        "preparation",
        "preparation-crosstalk",
        "preparation-leakage",
        "readout",
        "single-qubit-emission",
        "single-qubit-pauli",
        "single-qubit-seepage",
        "two-qubit-angle-scaling",
        "two-qubit-emission",
        "two-qubit-pauli",
        "two-qubit-seepage",
    },
)
MINIMUM_SENSITIVE_CASES = 3
INDEPENDENT_ORACLES = frozenset({"analytic", "basis-state", "qutrit"})


def test_every_noise_channel_has_redundant_independent_evidence(
    noise_channel_coverage: dict[str, ChannelCoverage],
) -> None:
    """Fail when a channel drops below the original harness's three-case bar."""
    missing = REQUIRED_CHANNELS - noise_channel_coverage.keys()
    assert not missing, f"noise channels have no registered semantic evidence: {sorted(missing)}"

    sparse = {
        channel: len(noise_channel_coverage[channel].cases)
        for channel in REQUIRED_CHANNELS
        if len(noise_channel_coverage[channel].cases) < MINIMUM_SENSITIVE_CASES
    }
    assert not sparse, f"noise channels need at least {MINIMUM_SENSITIVE_CASES} sensitive cases: {sparse}"

    non_independent = {
        channel: sorted(noise_channel_coverage[channel].oracles)
        for channel in REQUIRED_CHANNELS
        if noise_channel_coverage[channel].oracles.isdisjoint(INDEPENDENT_ORACLES)
    }
    assert not non_independent, f"noise channels lack an independent oracle: {non_independent}"
