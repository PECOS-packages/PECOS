"""Order-preserving paper variants retain ideal contracts in Guppy and TickCircuit."""

import sys

import pytest
import stim
from pecos import sim, stabilizer
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    lacroix_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.generic import GuppyOrdering, SchedulePolicy
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Native aggregate helper ABI adapter not implemented")


@pytest.mark.parametrize("family", ["488", "swap666", "lacroix", "periodic"])
def test_compact_paper_memories_preserve_ideal_named_parities_and_tick_export(family):
    if family == "488":
        impl = color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ)
        reference = color_memory(impl, 2, "Z")
    elif family == "swap666":
        impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
        reference = color_memory(impl, 3, "Z")
    elif family == "lacroix":
        reference = lacroix_memory("d3_west", 2, "Z")
    else:
        reference = periodic_488_memory(periodic_488_implementation(periodic_488_code(4), gate_set="native-cz"), 2, "Z")
    compact = reference.rescheduled(SchedulePolicy.COMPACT_UNITARIES)
    assert [op for m in compact.moments for op in m] == [op for m in reference.moments for op in m]
    program = compact.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS, allocation_provenance=True)
    columns = sim(program).qubits(compact.num_qubits).quantum(stabilizer()).seed(19).run(8).to_dict()
    for shot in range(8):
        detectors, observables = compact.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert not any(detectors)
        assert not any(observables)
    circuit = stim.Circuit(tick_circuit_to_stim(compact.to_tick_circuit()))
    assert not circuit.compile_detector_sampler(seed=19).sample(64, append_observables=True).any()
    assert compact.measurement_aliases == reference.measurement_aliases
    assert compact.citations == reference.citations
