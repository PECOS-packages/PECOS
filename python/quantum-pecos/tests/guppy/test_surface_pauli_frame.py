# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Logical frame witnesses ported from the reviewed Guppy prototypes."""

import ast
import math
from collections import Counter
from itertools import product

import pecos_rslib as prs
import pytest
from pecos import sim, stab_vec
from pecos.guppy_gen._module_loader import load_cached_guppy_source
from pecos.guppy_gen.protocol_render import render_surface_protocol_module
from pecos.guppy_gen.surface_teleportation import (
    render_surface_hook_injection_module,
    render_surface_t_teleportation_module,
)
from pecos.qec.surface import SurfacePatch


def _run(patch, body, *, shots=64, seed=3):
    module = load_cached_guppy_source("frame_witness", render_surface_protocol_module(patch) + body)
    engine = prs.qis_engine().selene_runtime().interface(prs.qis_helios_interface())
    return (
        sim(module["experiment"])
        .classical(engine)
        .quantum(stab_vec())
        .qubits(4 * patch.geometry.num_qubits)
        .seed(seed)
        .run(shots)
        .to_dict()
    )


def _round(data, *, swapped=False):
    helper = "swapped_a" if swapped else data
    return f"""
    syn = syndrome_extraction_{helper}({data})
    output("synx_{data}", syn.synx)
    output("synz_{data}", syn.synz)
"""


def _sz(data, frame, anc, tag):
    return f"""
    {anc} = prep_y_basis()
{_round(data)}
    first = syndrome_extraction_anc({anc})
    transversal_cx({data}, {anc})
{_round(data)}
    syn = syndrome_extraction_anc({anc})
    final_anc = measure_z_basis({anc})
    decision = sz_teleportation_decision(first, final_anc)
    output("decision_{tag}", decision)
    frame_sz_teleportation({frame}, decision)
"""


def _parity(rows, support):
    return [sum(int(row[q]) for q in support) % 2 for row in rows]


def _detector_count(results, geom, data, *, basis="x", swapped=False):
    # Guppy records name the CURRENT measured family after H. On a swapped
    # Z readout, these are Z checks on the original X stabilizer supports.
    stabs = geom.x_stabilizers if (basis == "x") != swapped else geom.z_stabilizers
    return sum(
        all(sum(int(final[q]) for q in stab.data_qubits) % 2 == int(syn[-len(stabs) + stab.index]) for stab in stabs)
        for final, syn in zip(results[f"final_{data}"], results[f"syn{basis}_{data}"], strict=True)
    )


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("scenario", ["y", "ss", "ssh", "cx"])
def test_clifford_frame_witness(distance, scenario):
    patch = SurfacePatch.create(distance=distance)
    geom = patch.geometry
    if scenario == "cx":
        body = f"""
@guppy
def experiment() -> None:
    ctrl = prep_x_basis()
    tgt = prep_x_basis()
    fc = PauliFrame(False, False)
    ft = PauliFrame(False, False)
{_round("ctrl")}
{_sz("tgt", "ft", "anc", "a")}
    transversal_cx(ctrl, tgt)
    frame_cx(fc, ft)
{_round("ctrl")}
{_round("tgt")}
    final_ctrl = measure_x_basis(ctrl)
    frame_correct_x_readout(fc, final_ctrl)
    output("final_ctrl", final_ctrl)
    final_tgt = measure_x_basis(tgt)
    frame_correct_x_readout(ft, final_tgt)
    output("final_tgt", final_tgt)
"""
    else:
        body = f"""
@guppy
def experiment() -> None:
    data = prep_x_basis()
    frame = PauliFrame(False, False)
{_sz("data", "frame", "anc", "a")}
"""
        if scenario == "y":
            body += f"""
    frame_s(frame)
    fold = syndrome_extraction_fold_szdg_data(data)
    output("fold_synz", fold.synz)
{_round("data")}
"""
        else:
            body += _sz("data", "frame", "anc2", "b")
        if scenario == "ssh":
            body += f"""
    transversal_h(data)
    frame_h(frame)
{_round("data", swapped=True)}
    final = measure_z_basis(data)
    frame_correct_z_readout_swapped(frame, final)
"""
        else:
            body += """
    final = measure_x_basis(data)
    frame_correct_x_readout(frame, final)
"""
        body += '    output("final_data", final)\n'
    results = _run(patch, body)
    if scenario == "cx":
        ctrl = _parity(results["final_ctrl"], geom.logical_x.data_qubits)
        tgt = _parity(results["final_tgt"], geom.logical_x.data_qubits)
        witness = [a ^ b for a, b in zip(ctrl, tgt, strict=True)]
        detectors = {data: _detector_count(results, geom, data) for data in ("ctrl", "tgt")}
    else:
        witness = _parity(results["final_data"], geom.logical_x.data_qubits)
        if scenario == "y":
            witness = [value ^ (sum(row) % 2) for value, row in zip(witness, results["fold_synz"], strict=True)]
        detectors = {
            "data": _detector_count(
                results,
                geom,
                "data",
                basis="z" if scenario == "ssh" else "x",
                swapped=scenario == "ssh",
            ),
        }
    measurements = {
        "witness": dict(Counter(witness)),
        "detectors": detectors,
        "decisions": dict(Counter(results["decision_a"])),
    }
    print(f"d={distance} {scenario}: {measurements}")
    assert set(results["decision_a"]) == {False, True}, measurements
    assert witness == [int(scenario in ("ss", "ssh"))] * 64, measurements
    assert all(count == 64 for count in detectors.values()), measurements


@pytest.mark.parametrize("distance", [3, 5])
def test_sz_with_incoming_x_frame(distance):
    """S|+i> = |-> and H|-> = |1>; the recorded physical X must not change that."""
    patch = SurfacePatch.create(distance=distance)
    body = f"""
@guppy
def experiment() -> None:
    data = prepare_injected_y()
    apply_logical_x(data)
    frame = PauliFrame(True, False)
{_sz("data", "frame", "anc", "a")}
    transversal_h(data)
    frame_h(frame)
{_round("data", swapped=True)}
    final = measure_z_basis(data)
    frame_correct_z_readout_swapped(frame, final)
    output("final_data", final)
"""
    results = _run(patch, body, seed=7)
    parities = _parity(results["final_data"], patch.geometry.logical_x.data_qubits)
    measurements = {
        "parities": dict(Counter(parities)),
        "detectors": _detector_count(results, patch.geometry, "data", basis="z", swapped=True),
        "decisions": dict(Counter(results["decision_a"])),
    }
    print(f"d={distance} incoming_x_sz: {measurements}")
    assert parities == [1] * 64, measurements
    assert measurements["detectors"] == 64, measurements
    assert set(results["decision_a"]) == {False, True}, measurements


@pytest.mark.parametrize("distance", [3, 5])
def test_cx_with_incoming_x_frame(distance):
    """CX maps physical |10> to |11>; propagated X frames recover intended |00>."""
    patch = SurfacePatch.create(distance=distance)
    body = f"""
@guppy
def experiment() -> None:
    ctrl = prep_z_basis()
    tgt = prep_z_basis()
    apply_logical_x(ctrl)
    fc = PauliFrame(True, False)
    ft = PauliFrame(False, False)
{_round("ctrl")}
{_round("tgt")}
    transversal_cx(ctrl, tgt)
    frame_cx(fc, ft)
{_round("ctrl")}
{_round("tgt")}
    final_ctrl = measure_z_basis(ctrl)
    frame_correct_z_readout(fc, final_ctrl)
    output("final_ctrl", final_ctrl)
    final_tgt = measure_z_basis(tgt)
    frame_correct_z_readout(ft, final_tgt)
    output("final_tgt", final_tgt)
"""
    results = _run(patch, body, seed=7)
    parities = {
        data: _parity(results[f"final_{data}"], patch.geometry.logical_z.data_qubits) for data in ("ctrl", "tgt")
    }
    measurements = {
        "parities": {data: dict(Counter(bits)) for data, bits in parities.items()},
        "detectors": {data: _detector_count(results, patch.geometry, data, basis="z") for data in ("ctrl", "tgt")},
    }
    print(f"d={distance} incoming_x_cx: {measurements}")
    assert parities == {"ctrl": [0] * 64, "tgt": [0] * 64}, measurements
    assert measurements["detectors"] == {"ctrl": 64, "tgt": 64}, measurements


@pytest.mark.parametrize("dagger", [False, True], ids=["t", "tdg"])
def test_framed_t(dagger):
    patch = SurfacePatch.create(distance=3)
    body = f"""
@guppy
def experiment() -> None:
    data = prepare_injected_y()
    apply_logical_x(data)
    frame = PauliFrame(True, False)
    resource = prepare_injected_{"tdg" if dagger else "t"}()
    apply_t_teleportation_framed(data, resource, frame, {dagger})
{_round("data")}
    final = measure_x_basis(data)
    frame_correct_x_readout(frame, final)
    output("final_data", final)
"""
    results = _run(patch, body, shots=512, seed=5)
    values = [1 - 2 * parity for parity in _parity(results["final_data"], patch.geometry.logical_x.data_qubits)]
    mean = sum(values) / len(values)
    expected = (1 if dagger else -1) / math.sqrt(2)
    measurements = {
        "mean": mean,
        "parities": dict(Counter(values)),
        "detectors": _detector_count(results, patch.geometry, "data"),
        "decisions": dict(Counter(results["t_correction"])),
    }
    print(f"dagger={dagger}: {measurements}")
    # Historical prototype errors were 0.016 and 0.031. A 0.12 margin is
    # about 3.8 standard errors at 512 shots, far from the x-blind mean of 0.
    assert mean == pytest.approx(expected, abs=0.12), measurements
    assert measurements["detectors"] == 512, measurements
    assert set(results["t_correction"]) == {False, True}, measurements


def test_framed_t_after_sz_with_random_x_checks():
    """S|+i> = |->, so T gives <X> = -1/sqrt(2) despite random X-check signs."""
    patch = SurfacePatch.create(distance=3)
    body = f"""
@guppy
def experiment() -> None:
    data = prepare_injected_y()
    frame = PauliFrame(False, False)
{_sz("data", "frame", "anc", "a")}
    syn = syndrome_extraction_data(data)
    output("pre_synx", syn.synx)
    output("pre_synz", syn.synz)
    resource = prepare_injected_t()
    apply_t_teleportation_framed(data, resource, frame, False)
{_round("data")}
    final = measure_x_basis(data)
    frame_correct_x_readout(frame, final)
    output("final_data", final)
"""
    results = _run(patch, body, shots=512, seed=7)
    parities = _parity(results["final_data"], patch.geometry.logical_x.data_qubits)
    measurements = {
        "mean": sum(1 - 2 * parity for parity in parities) / len(parities),
        "nonzero_x_checks": sum(any(row) for row in results["pre_synx"]),
        "nonzero_z_checks": sum(any(row) for row in results["pre_synz"]),
        "detectors": _detector_count(results, patch.geometry, "data"),
    }
    print(f"sz_then_t: {measurements}")
    assert measurements["mean"] == pytest.approx(-1 / math.sqrt(2), abs=0.12), measurements
    assert measurements["nonzero_x_checks"] > 0, measurements
    assert measurements["nonzero_z_checks"] == 0, measurements
    assert measurements["detectors"] == 512, measurements


@pytest.mark.parametrize(
    "renderer",
    [render_surface_protocol_module, render_surface_t_teleportation_module, render_surface_hook_injection_module],
)
def test_frame_block_once_and_compiles(renderer):
    source = renderer(SurfacePatch.create(distance=3))
    tree = ast.parse(source)
    assert sum(isinstance(node, ast.ClassDef) and node.name == "PauliFrame" for node in tree.body) == 1
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    helpers = {
        "frame_h",
        "frame_s",
        "frame_cx",
        "frame_sz_teleportation",
        "frame_correct_z_readout",
        "frame_correct_x_readout",
        "frame_correct_z_readout_swapped",
        "frame_correct_x_readout_swapped",
        "apply_t_teleportation_framed",
    }
    module = load_cached_guppy_source("frame_module", source)
    for name in helpers:
        assert functions.count(name) == 1
        assert module[name].compile_function() is not None


def test_frame_rules_on_all_inputs():
    """Nonzero incoming frames exercise both CX directions and S conjugation."""
    body = ["\n@guppy", "def experiment() -> None:"]
    expected = []
    for x, z, tx, tz in product((False, True), repeat=4):
        body.extend(
            [
                f"    frame = PauliFrame({x}, {z})",
                f"    target = PauliFrame({tx}, {tz})",
                "    frame_cx(frame, target)",
                '    output("bits", array(frame.x, frame.z, target.x, target.z))',
                "    frame_h(frame)",
                '    output("bits", array(frame.x, frame.z))',
                "    frame_s(frame)",
                '    output("bits", array(frame.x, frame.z))',
                "    frame_sz_teleportation(frame, True)",
                '    output("bits", array(frame.x, frame.z))',
            ],
        )
        expected.extend([x, z ^ tz, tx ^ x, tz, z ^ tz, x, z ^ tz, x ^ z ^ tz, z ^ tz, not x])
    results = _run(SurfacePatch.create(distance=3), "\n".join(body) + "\n", shots=1)
    assert results["bits"] == [expected]


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize(("basis", "swapped"), product(("x", "z"), (False, True)))
def test_readout_supports(distance, basis, swapped):
    """Both frame values and all four readouts preserve the measured checks."""
    patch = SurfacePatch.create(distance=distance)
    geom = patch.geometry
    correction = geom.logical_x if (basis == "z") != swapped else geom.logical_z
    measured = geom.logical_z if (basis == "z") != swapped else geom.logical_x
    stabs = geom.z_stabilizers if (basis == "z") != swapped else geom.x_stabilizers
    suffix = "_swapped" if swapped else ""
    body = ["\n@guppy", "def experiment() -> None:"]
    for value in (False, True):
        body.extend(
            [
                f"    frame = PauliFrame({value}, {value})",
                f"    bits = array(False for _ in range({geom.num_data}))",
                f"    frame_correct_{basis}_readout{suffix}(frame, bits)",
                f'    output("bits_{value}", bits)',
            ],
        )
    results = _run(patch, "\n".join(body) + "\n", shots=1)
    for value in (False, True):
        bits = results[f"bits_{value}"][0]
        assert bits == [value and q in correction.data_qubits for q in range(geom.num_data)]
        assert sum(bits[q] for q in measured.data_qubits) % 2 == value
        assert all(sum(bits[q] for q in stab.data_qubits) % 2 == 0 for stab in stabs)
