# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Python coverage for paired DUT/reference decoder comparison."""

from __future__ import annotations

import pytest

pytest.importorskip("pecos_rslib")

from pecos_rslib.qec import SampleBatch


def test_sample_batch_compare_decoders_exposes_joint_counts() -> None:
    dem = "error(0.1) D0 L0\n"
    batch = SampleBatch([[0], [1], [0], [1]], [0, 1, 0, 1])

    first = batch.compare_decoders(dem, "pymatching", "pymatching")
    second = batch.compare_decoders(dem, "pymatching", "pymatching")

    assert first.total_shots == 4
    assert first.counts == [[4, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert first.dut_correct_reference_correct == 4
    assert first.dut_only_failures == 0
    assert first.both_failed == 0
    assert first.dut_only_failure_interval[0] >= 0.0
    assert first.dut_only_failure_interval[1] <= 1.0
    assert second.counts == first.counts
