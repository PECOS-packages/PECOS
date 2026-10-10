# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regression for terminal decomposition's observable labels (issue #780)."""

import collections
import re

import pytest
from pecos.qec import DetectorErrorModel, ParsedDem
from pecos.qec.surface.logical_circuit import LogicalCircuitBuilder
from pecos.qec.surface.patch import SurfacePatch
from pecos_rslib.decoders import pymatching
from pecos_rslib.quantum import TickCircuit


def disagreeing_sets(text: str) -> int:
    """Count detector supports with multiple observable masks, including components."""
    by_support = collections.defaultdict(set)
    for line in text.splitlines():
        match = re.match(r"error\(([^)]*)\)\s+(.*)", line.strip())
        if match is None:
            continue
        for part in match[2].split("^"):
            targets = part.split()
            detectors = frozenset(int(t[1:]) for t in targets if t.startswith("D"))
            observables = 0
            for target in targets:
                if target.startswith("L"):
                    observables ^= 1 << int(target[1:])
            if detectors:
                by_support[detectors].add(observables)
    return sum(len(masks) > 1 for masks in by_support.values())


def test_terminal_memory_observables_and_decoding() -> None:
    """Run the issue's d=5, 15-round, fixed-seed decoding comparison."""
    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(5), "p")
    builder.add_memory(["p"], 15, {"p": "Z"})
    probability = 0.004
    dem = DetectorErrorModel.from_circuit(
        builder.to_tick_circuit(),
        p1=probability,
        p2=probability,
        p_meas=probability,
        p_prep=probability,
    )
    errors = {}
    for name, text in (
        ("terminal", dem.to_string_terminal_graphlike_decomposed()),
        ("source", dem.to_string_source_graphlike_decomposed()),
    ):
        assert disagreeing_sets(text) == 0
        batch = ParsedDem.from_string(text).to_dem_sampler().sample_batch(20_000, seed=11)
        errors[name] = batch.decode(text, pymatching(correlated=False)).num_errors
    print(f"issue #780: terminal={errors['terminal']}, source={errors['source']} / 20000")
    # Measured on origin/dev with this d=5, 15-round Z-memory circuit, p=0.004,
    # ParsedDem sampling of 20,000 shots with seed 11, and uncorrelated PyMatching:
    # the terminal projection produced 6,796 errors. Source is a reference only;
    # coordinate-based components need not decode as accurately as physical ones.
    assert errors["terminal"] < 6_796, errors


def test_terminal_standalone_conflict_raises_value_error() -> None:
    """Two measurement faults have one detector but distinct logical labels."""
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().mz([0])
    circuit.tick().mz([1])
    circuit.add_detector(records=[-2, -1])
    circuit.add_observable(records=[-2])
    dem = DetectorErrorModel.from_circuit(circuit, p1=0.0, p2=0.0, p_meas=0.01, p_prep=0.0)
    with pytest.raises(ValueError, match=r"conflicting standalone.*D0"):
        dem.to_string_terminal_graphlike_decomposed()
