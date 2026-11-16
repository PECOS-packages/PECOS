# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface-memory simulation, traced DEMs and Frontier share a runtime contract."""

import pecos
import pecos_rslib as pr
import pytest
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel, surface_memory_dem_spec
from pecos.qec.surface import NoiseParameters
from pecos_rslib.qec import SampleBatch


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize(
    ("module", "runtime_class"),
    [
        ("selene_simple_runtime_plugin", "SimpleRuntimePlugin"),
        ("selene_soft_rz_runtime_plugin", "SoftRZRuntimePlugin"),
    ],
)
def test_public_runtime_surface_memory_decodes_with_frontier(
    basis: str,
    module: str,
    runtime_class: str,
) -> None:
    """Decode live results with the DEM traced through the same public runtime."""
    frontier = pytest.importorskip("pecos_rslib_exp").frontier
    runtime = getattr(pytest.importorskip(module), runtime_class)()
    program = make_surface_code(distance=3, num_rounds=3, basis=basis)
    detectors, observables = surface_memory_dem_spec(3, 3, basis)
    build = (
        DetectorErrorModel.builder()
        .with_program(program)
        .with_qubits(get_num_qubits(3))
        .with_runtime(runtime)
        .with_detectors(detectors)
        .with_observables(observables)
        .with_noise(NoiseParameters(p_meas=0.01))
        .build()
    )
    assert (build.dem.num_detectors, build.dem.num_observables) == (28, 1)
    dem_text = str(build.dem)
    for noisy in (False, True):
        simulation = (
            pecos.sim(program)
            .classical(pecos.selene_engine(runtime))
            .qubits(get_num_qubits(3))
            .quantum(pr.sparse_stab())
            .workers(1)
            .seed(7)
        )
        if noisy:
            simulation = simulation.noise(pr.general_noise().with_p_meas(0.01))
        pairs = build.evaluate_result_columns(simulation.run(8).to_dict())
        if not noisy:
            assert all(not any(syndrome) and logical == 0 for syndrome, logical in pairs)
        batch = SampleBatch([syndrome for syndrome, _ in pairs], [logical for _, logical in pairs])
        decoded = batch.decode(dem_text, frontier(k=64), workers=1)
        assert (decoded.num_scored, decoded.num_undecoded) == (8, 0)
        if not noisy:
            assert decoded.num_errors == 0

    sampled = build.dem.to_sampler().sample_batch(8, seed=11)
    decoded = sampled.decode(dem_text, frontier(k=64), workers=1)
    assert (decoded.num_scored, decoded.num_undecoded) == (8, 0)
