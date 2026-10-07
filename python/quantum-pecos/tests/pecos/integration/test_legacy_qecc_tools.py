"""Legacy QECC analysis must exercise the current instruction and decoder APIs."""

import ast
from importlib import import_module

import pecos as pc
import pytest
from pecos.analysis.fault_tolerance_checks import distance_check, fault_check, t_errors_check
from pecos.analysis.tool_collection import _apply_err as apply_err
from pecos.analysis.tool_collection import _apply_err_spacetime as apply_err_spacetime
from pecos.analysis.tool_collection import form_errors
from pecos.tools.tool_collection import _apply_err as legacy_apply_err
from pecos.tools.tool_collection import _apply_err_spacetime as legacy_apply_err_spacetime


@pytest.mark.parametrize("check", [t_errors_check, fault_check])
@pytest.mark.parametrize(("weight", "expected"), [(1, (True, 1)), (2, (False, 2))])
@pytest.mark.parametrize("physical", [False, True])
def test_correctable_errors(check, weight, expected, physical) -> None:
    """All single Paulis are correctable; some weight-two Paulis defeat d=3."""
    qecc = pc.qeccs.Surface4444(distance=3)
    kwargs = {}
    if physical:
        extraction = qecc.instruction("instr_syn_extract").circuit
        kwargs["syn_extract" if check is t_errors_check else "logical_gate"] = extraction
    assert check(qecc, t_weight=weight, verbose=False, **kwargs) == expected


@pytest.mark.parametrize("mode", [None, "X", "Z", "power"])
def test_distance_check(mode) -> None:
    """The first undetectable nonstabilizer reported has weight three."""
    qecc = pc.qeccs.Surface4444(distance=3)
    result = distance_check(qecc, mode=mode)
    assert isinstance(result, str)
    xs, zs = result.removeprefix("Logical error found: Xs - ").split(" Zs - ")
    xs, zs = set(ast.literal_eval(xs)), set(ast.literal_eval(zs))
    assert len(xs | zs) == 3
    error = pc.circuits.QuantumCircuit(
        [{symbol: locations for symbol, locations in [("X", xs), ("Z", zs)] if locations}],
    )
    state = pc.simulators.SparseStabPy(qecc.num_qudits)
    runner = pc.circuit_runners.Standard(seed=1)
    init = pc.circuits.LogicalCircuit(suppress_warning=True)
    init.append(qecc.gate("ideal init |0>"))
    extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
    extraction.append(qecc.gate("I", num_syn_extract=1))
    runner.run(state, init)
    runner.run(state, error)
    output, _ = runner.run(state, extraction)
    assert not output.simplified(last=True)
    logical_ops = qecc.instruction("instr_syn_extract").final_logical_ops[0]
    assert any(not pc.quantum.commute.qubit_pauli(op, error) for op in logical_ops.values())


@pytest.mark.parametrize("module", ["pecos.analysis.threshold_tools", "pecos.tools.threshold_tools"])
@pytest.mark.parametrize("p", [0.0, 0.5])
@pytest.mark.parametrize("simulator", [None, pc.simulators.SparseStabPy])
def test_codecapacity_logical_rate2(module, p, simulator) -> None:
    """Seeded two-basis sampling has no noiseless failures and detects strong noise."""
    qecc = pc.qeccs.Surface4444(distance=3)
    rate, elapsed = import_module(module).codecapacity_logical_rate2(
        20,
        qecc,
        3,
        pc.noise.DepolarModel(model_level="code_capacity"),
        {"p": p},
        pc.decoders.MWPM2D(qecc),
        seed=1,
        state_sim=simulator,
        verbose=False,
    )
    assert rate == 0.0 if p == 0.0 else 0.0 < rate <= 1.0
    assert elapsed >= 0.0


@pytest.mark.parametrize(
    ("apply_error", "apply_spacetime"),
    [(apply_err, apply_err_spacetime), (legacy_apply_err, legacy_apply_err_spacetime)],
    ids=["analysis", "tools"],
)
@pytest.mark.parametrize("spacetime", [False, True])
@pytest.mark.parametrize("weight", [0, 1, 2])
def test_apply_errors(apply_error, apply_spacetime, spacetime, weight) -> None:
    """Recovery corrects one X and fails on two Xs from a logical string."""
    qecc = pc.qeccs.Surface4444(distance=3)
    runner = pc.circuit_runners.Standard(seed=1)
    state = pc.simulators.SparseStabPy(qecc.num_qudits)
    init = pc.circuits.LogicalCircuit(suppress_warning=True)
    init.append(qecc.gate("ideal init |0>"))
    logical_z = qecc.instruction("instr_init_zero").final_logical_ops[0]["Z"]
    decoder = pc.decoders.MWPM2D(qecc)
    xs = set(sorted(qecc.sides["left"])[:weight])
    if spacetime:
        errors = form_errors([(0, q) for q in xs], [])
        sign = apply_spacetime(state, runner, init, errors, decoder, logical_z, qecc)
        assert errors == form_errors([(0, q) for q in xs], [])
    else:
        extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
        extraction.append(qecc.gate("I", num_syn_extract=1))
        error = pc.circuits.QuantumCircuit([{"X": xs}])
        sign = apply_error(state, runner, init, extraction, error, decoder, logical_z)
    assert sign == (weight == 2)
