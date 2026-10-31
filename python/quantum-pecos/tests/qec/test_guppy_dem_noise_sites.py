# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Runtime-emitted gates retain their independent physical noise locations."""

from __future__ import annotations

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.platform import barrier
from guppylang.std.quantum import measure, qubit, x
from pecos import NoiseParameters
from pecos.qec import DemInferenceSettings, Detector, DetectorErrorModel, Observable, rec
from pecos.tracing import capture_qis_operation_trace, qis_operation_trace_to_tick_circuit


@guppy
def _cancelling_pair() -> None:
    q = qubit()
    x(q)
    # Keep both gates through compiler optimization. The runtime trace, rather
    # than the source program's ideal unitary, defines the physical fault sites.
    barrier(q)
    x(q)
    measured = measure(q).read()
    result("raw measurements", measured)
    result("DETECTOR", measured)
    result("obs", measured)


@pytest.mark.parametrize("runtime_name", ["simple", "soft_rz"])
@pytest.mark.parametrize("inferred", [False, True])
def test_cancelled_clifford_pair_keeps_noise_sites(runtime_name: str, *, inferred: bool) -> None:
    """Two independent X faults flip the record with probability 2*p*(1-p)."""
    plugin = pytest.importorskip(f"selene_{runtime_name}_runtime_plugin")
    runtime = getattr(plugin, "SimpleRuntimePlugin" if runtime_name == "simple" else "SoftRZRuntimePlugin")()
    trace = qis_operation_trace_to_tick_circuit(capture_qis_operation_trace(_cancelling_pair, 1, runtime=runtime))
    assert trace.gate_counts_by_type()["RXY1Q"] == 2
    builder = (
        DetectorErrorModel.builder()
        .with_program(_cancelling_pair)
        .with_qubits(1)
        .with_runtime(runtime)
        .with_seed(7)
        .with_noise(
            NoiseParameters().with_p1(0.1).with_p1_weights({"X": 1.0}).with_p2(0).with_p_meas(0).with_p_prep(0),
        )
    )
    if inferred:
        builder = builder.with_inference(DemInferenceSettings().with_validation_rows(8))
    else:
        builder = builder.with_detectors([Detector(rec[-1])]).with_observables([Observable(rec[-1])])
    build = builder.build()
    errors = [line for line in build.dem.to_string().splitlines() if line.startswith("error(")]
    assert len(errors) == 1, build.dem.to_string()
    probability, targets = errors[0][6:].split(")")
    assert float(probability) == pytest.approx(2 * 0.1 * (1 - 0.1))
    assert targets.split() == ["D0", "L0"]
