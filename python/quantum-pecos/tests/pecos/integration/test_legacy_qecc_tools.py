"""Legacy QECC analysis must exercise the current instruction and decoder APIs."""

import ast

import pecos as pc
import pytest
from pecos.analysis import pseudo_threshold_tools, threshold_tools, tool_collection
from pecos.analysis.fault_tolerance_checks import distance_check, fault_check, t_errors_check
from pecos.analysis.tool_collection import _apply_err as apply_err
from pecos.analysis.tool_collection import _apply_err_spacetime as apply_err_spacetime
from pecos.analysis.tool_collection import form_errors


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
    with pytest.warns(DeprecationWarning, match="does not establish circuit-level fault tolerance") as caught:
        assert check(qecc, t_weight=weight, verbose=False, **kwargs) == expected
    assert check.__name__ in str(caught[0].message)
    assert "Fault Tolerance Analysis" in str(caught[0].message)
    assert "docs/user-guide/fault-tolerance.md" in str(caught[0].message)


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


@pytest.mark.parametrize("p", [0.0, 0.5])
@pytest.mark.parametrize("simulator", [None, pc.simulators.SparseStabPy])
def test_codecapacity_logical_rate2(p, simulator) -> None:
    """Seeded two-basis sampling has no noiseless failures and detects strong noise."""
    qecc = pc.qeccs.Surface4444(distance=3)
    rate, elapsed = threshold_tools.codecapacity_logical_rate2(
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


@pytest.mark.parametrize("spacetime", [False, True])
@pytest.mark.parametrize("weight", [0, 1, 2])
def test_apply_errors(spacetime, weight) -> None:
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
        sign = apply_err_spacetime(state, runner, init, errors, decoder, logical_z, qecc)
        assert errors == form_errors([(0, q) for q in xs], [])
    else:
        extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
        extraction.append(qecc.gate("I", num_syn_extract=1))
        error = pc.circuits.QuantumCircuit([{"X": xs}])
        sign = apply_err(state, runner, init, extraction, error, decoder, logical_z)
    assert sign == (weight == 2)


class _NoZRecovery:
    """Keep MWPM's X component while discarding every Z correction."""

    def __init__(self, qecc):
        self.decoder = pc.decoders.MWPM2D(qecc)

    def decode(self, measurements):
        recovery = self.decoder.decode(measurements)
        xs = set()
        for symbol, locations, _ in recovery.items():
            if symbol in {"X", "Y"}:
                xs ^= locations
        return pc.circuits.QuantumCircuit([{"X": xs}])


def test_fault_tolerance_checks_data_both_bases(monkeypatch) -> None:
    """A decoder omitting Z corrections must fail before reaching circuit faults."""
    qecc = pc.qeccs.Surface4444(distance=3)

    def unexpected_spacetime(*_args):
        pytest.fail("Data errors passed despite missing Z corrections")

    monkeypatch.setattr(tool_collection, "_apply_err_spacetime", unexpected_spacetime)
    with (
        pytest.warns(DeprecationWarning, match="does not establish circuit-level fault tolerance") as caught,
        pytest.raises(Exception, match="Decoder failed to correct error:") as exc,
    ):
        tool_collection.fault_tolerance_check(qecc, _NoZRecovery(qecc))
    # The first Z error along logical X must be reported as a data-qubit circuit.
    qudit = next(q for q in qecc.data_qudit_set if q in qecc.sides["left"])
    error = pc.circuits.QuantumCircuit([{"X": set(), "Z": {qudit}}])
    assert "fault_tolerance_check" in str(caught[0].message)
    assert "Fault Tolerance Analysis" in str(caught[0].message)
    assert "docs/user-guide/fault-tolerance.md" in str(caught[0].message)
    assert str(exc.value) == f"Decoder failed to correct error: {error}"


def test_fault_tolerance_checks_spacetime_both_bases(monkeypatch) -> None:
    """A circuit Z fault is tested in the plus state and reported individually."""
    qecc = pc.qeccs.Surface4444(distance=3)
    qudit = min(qecc.sides["left"])
    error = {0: {"Z": {qudit}}}

    def errors(qubits, **_kwargs):
        # Skip data faults to isolate the circuit-fault phase.
        if qubits == qecc.data_qudit_set:
            return iter(())
        return iter([(set(), {(0, qudit)})])

    monkeypatch.setattr(tool_collection, "gen_pauli_errors", errors)
    with (
        pytest.warns(DeprecationWarning, match="does not establish circuit-level fault tolerance") as caught,
        pytest.raises(Exception, match="Decoder failed to correct error:") as exc,
    ):
        tool_collection.fault_tolerance_check(qecc, _NoZRecovery(qecc))
    assert "fault_tolerance_check" in str(caught[0].message)
    assert "Fault Tolerance Analysis" in str(caught[0].message)
    assert "docs/user-guide/fault-tolerance.md" in str(caught[0].message)
    assert str(exc.value) == f"Decoder failed to correct error: {error}"


def test_threshold_code_capacity_both_bases() -> None:
    """The wrapper can call the two-basis sampler with explicit depolarizing noise."""
    result = threshold_tools.threshold_code_capacity(
        pc.qeccs.Surface4444,
        pc.noise.DepolarModel(model_level="code_capacity"),
        pc.decoders.MWPM2D,
        [0.0, 0.2],
        [3],
        5,
        basis="both",
        circuit_runner=pc.circuit_runners.Standard(seed=1),
    )
    assert result["distances"] == [3]
    assert list(result["ps_physical"]) == [0.0, 0.2]
    rates = list(result["p_logical"])
    assert rates[0] == 0.0
    assert len(rates) == 2
    assert 0.0 <= rates[1] <= 1.0


def test_threshold_code_capacity_modes() -> None:
    """Every supported mode and basis samples a rate, including observed lifetimes."""
    for mode, bases in [(1, [None, "zero", "plus", "both"]), (2, [None, "zero", "plus"])]:
        for basis in bases:
            result = threshold_tools.threshold_code_capacity(
                pc.qeccs.Surface4444,
                pc.noise.DepolarModel(model_level="code_capacity"),
                pc.decoders.MWPM2D,
                [0.4, 0.5],
                [3],
                5,
                mode=mode,
                basis=basis,
                circuit_runner=pc.circuit_runners.Standard(seed=1),
            )
            rates = list(result["p_logical"])
            assert len(rates) == 2
            assert all(0.0 < rate <= 1.0 for rate in rates)
            if mode == 2:
                # Five short sampled lifetimes cannot equal the unused ten-million-round cap.
                assert all(rate >= 1.0 / 100 for rate in rates)
    with pytest.raises(ValueError, match="Mode 2 requires basis"):
        threshold_tools.threshold_code_capacity(
            pc.qeccs.Surface4444,
            pc.noise.DepolarModel(model_level="code_capacity"),
            pc.decoders.MWPM2D,
            [0.4],
            [3],
            5,
            mode=2,
            basis="both",
        )


def test_pseudo_threshold_two_basis_sampler() -> None:
    """The other rate2 caller must also omit the single-basis selection argument."""
    result = pseudo_threshold_tools.pseudo_threshold_code_capacity(
        [0.0, 0.2],
        3,
        5,
        error_gen=pc.noise.DepolarModel(model_level="code_capacity"),
        mode=2,
        basis="both",
        verbose=False,
        circuit_runner=pc.circuit_runners.Standard(seed=1),
    )
    rates = list(result["plog"])
    assert len(rates) == 2
    assert rates[0] == 0.0
    assert 0.0 <= rates[1] <= 1.0
