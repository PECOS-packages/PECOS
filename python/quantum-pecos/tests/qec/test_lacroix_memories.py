"""Independent published-circuit and backend checks for Lacroix memories."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import stim
from pecos.qec.color import lacroix_memory, lacroix_memory_program
from pecos.qec.generic import SchedulePolicy

FIXTURES = Path(__file__).parent / "fixtures/lacroix"
PATCHES = ("d3_west", "d3_east", "d3_north", "d5")


def _wire_histories(circuit, mapping, *, implicit_zero):
    """Compare all per-wire physical order, ignoring idle-only timing slots.

    Consecutive zero resets collapse; authors' initial Stim state is made explicit.
    Measurement ordinals are included in each wire's history.
    """
    histories = {q: [("R", (q,))] if implicit_zero else [] for q in mapping.values()}
    event = 0
    for op in circuit:
        if op.name in {"TICK", "QUBIT_COORDS", "I", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if op.name == "CX":
            assert all(t.is_sweep_bit_target for t in op.targets_copy()[::2])
            continue
        assert op.name in {"R", "H", "CZ", "M"}
        targets = [t.value for t in op.targets_copy()]
        arity = 2 if op.name == "CZ" else 1
        for i in range(0, len(targets), arity):
            group = targets[i : i + arity]
            if any(q not in mapping for q in group):
                assert op.name == "R"
                continue
            qs = tuple(mapping[q] for q in group)
            item = (op.name, qs, event) if op.name == "M" else (op.name, qs)
            event += op.name == "M"
            for q in qs:
                if op.name == "R" and histories[q] and histories[q][-1] == item:
                    continue
                histories[q].append(item)
    return histories


def _parities(circuit):
    count = 0
    detectors = []
    observable = Counter()
    for op in circuit:
        count += len(op.targets_copy()) if op.name == "M" else 0
        if op.name == "DETECTOR":
            detectors.append(tuple(sorted(count + t.value for t in op.targets_copy())))
        elif op.name == "OBSERVABLE_INCLUDE":
            assert op.gate_args_copy() == [0]
            observable.update(count + t.value for t in op.targets_copy())
    return detectors, tuple(sorted(i for i, n in observable.items() if n % 2))


@pytest.mark.parametrize("patch", PATCHES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [3, 5])
def test_published_reference_gates_and_parities(patch, basis, rounds):
    reference = stim.Circuit((FIXTURES / f"{patch}_{basis}_r{rounds:02}.stim").read_text())
    plan = lacroix_memory(patch, rounds, basis)
    source_ids = {i: int(name.split("[")[1].split("]")[0]) for i, name in enumerate(plan.role_names)}
    assert _wire_histories(reference, {q: q for q in source_ids.values()}, implicit_zero=True) == _wire_histories(
        plan.to_stim(),
        source_ids,
        implicit_zero=False,
    )
    dets, obs = _parities(reference)
    assert [p.records for p in plan.detectors] == dets
    assert plan.observables[0].records == obs
    assert plan.num_measurements == reference.num_measurements
    plan.to_stim().detector_error_model()


@pytest.mark.parametrize("patch", PATCHES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [2, 4, 7])
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_round_composition_and_noiseless_detector_contract(patch, basis, rounds, schedule):
    program = lacroix_memory_program(patch, rounds, basis)
    assert len(program.calls) == rounds + 2
    for call in program.calls[1:-1]:
        assert sum(any(g.name == "CZ" for g in m) for m in call.gadget.moments) == 10
    plan = program.compile(schedule)
    assert plan.num_qubits == (37 if patch == "d5" else 13)
    c = plan.to_stim()
    c.detector_error_model()
    assert not c.compile_detector_sampler(seed=5).sample(64, append_observables=True).any()
    definitions = plan.to_tick_circuit().circuit_definitions()
    assert [d["measurements"] for d in definitions["detectors"]] == [list(p.records) for p in plan.detectors]


def test_reference_hashes():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for name, metadata in provenance["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == metadata["sha256"]


@pytest.mark.parametrize("kwargs", [{"rounds": 1}, {"rounds": True}, {"rounds": 2.0}, {"basis": "Y"}, {"patch": "d7"}])
def test_unsupported_reference_contracts(kwargs):
    with pytest.raises(ValueError, match=r"rounds|basis|patch"):
        lacroix_memory(**kwargs)
