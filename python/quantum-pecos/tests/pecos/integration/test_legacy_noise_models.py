"""Legacy Pauli models work through the circuit runner and threshold entry points."""

import pecos as pc
import pytest
from pecos.analysis.pseudo_threshold_tools import pseudo_threshold_code_capacity
from pecos.analysis.threshold_tools import threshold_code_capacity, threshold_code_capacity_calc


@pytest.mark.parametrize(
    ("model_class", "single_paulis", "pair_paulis"),
    [
        (pc.noise.XModel, ("X", "X"), (("I", "X"), ("X", "X"))),
        (pc.noise.ZModel, ("Z", "Z"), (("I", "Z"), ("Z", "Z"))),
        (pc.noise.XZModel, ("X", "Z"), (("I", "X"), ("Z", "Z"))),
        (pc.noise.DepolarModel, ("X", "Z"), (("I", "X"), ("Z", "Z"))),
    ],
)
@pytest.mark.parametrize("level", ["code_capacity", "phenomenological", "circuit"])
@pytest.mark.parametrize("sample_index", [0, -1], ids=["first-pauli", "last-pauli"])
def test_legacy_models_run_at_supported_levels(
    model_class,
    single_paulis,
    pair_paulis,
    level,
    sample_index,
    monkeypatch,
) -> None:
    """Every sampled error has the specified tick, slot, Pauli and target."""

    def choose(population, size):
        assert size == 1
        if isinstance(population, int):
            return pc.array([0 if sample_index == 0 else population - 1])
        return pc.array([population[sample_index]])

    # Directed draws cover both a two-qubit error with an identity component
    # and an error on both qubits, without depending on random-stream ordering.
    monkeypatch.setattr(pc.random, "choice", choose)
    model = model_class(model_level=level)
    qecc = pc.qeccs.Surface4444(distance=3)
    state = pc.simulators.SparseStabPy(qecc.num_qudits)
    runner = pc.circuit_runners.Standard(seed=1)
    init = pc.circuits.LogicalCircuit(suppress_warning=True)
    init.append(qecc.gate("ideal init |0>"))
    extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
    extraction.append(qecc.gate("I", num_syn_extract=1))
    expected = set()
    single_pauli = single_paulis[sample_index]
    pair = pair_paulis[sample_index]
    for tick, time, params in extraction.iter_ticks():
        if level in {"code_capacity", "phenomenological"} and time[-1] == 0:
            expected.update((time, "after", single_pauli, q) for q in params["data_qudit_set"])
        for gate, locations, _ in tick.items():
            if gate.startswith("measure") and level != "code_capacity":
                expected.update((time, "before", single_pauli, q) for q in locations)
            elif level == "circuit":
                if gate == "CNOT":
                    expected.update(
                        (time, "after", pauli, q)
                        for location in locations
                        for pauli, q in zip(pair, location, strict=True)
                        if pauli != "I"
                    )
                else:
                    assert gate.startswith("init") or gate == "H"
                    expected.update((time, "after", single_pauli, q) for q in locations)

    runner.run(state, init)
    measurements, errors = runner.run(state, extraction, error_gen=model, error_params={"p": 1.0})
    actual = {
        (time, slot, symbol, q)
        for time, tick in errors.items()
        for slot, circuit in tick.items()
        for symbol, locations, _ in circuit.items()
        for q in locations
    }
    assert actual == expected
    assert all(symbol != "I" for _, _, symbol, _ in actual)
    if level == "code_capacity":
        runner.run(state, pc.decoders.MWPM2D(qecc).decode(measurements))


def test_threshold_code_capacity_default_xmodel() -> None:
    """Passing no noise model uses the wrapper's default XModel."""
    result = threshold_code_capacity(
        pc.qeccs.Surface4444,
        None,
        pc.decoders.MWPM2D,
        [0.0, 1.0],
        [3],
        3,
        circuit_runner=pc.circuit_runners.Standard(seed=1),
    )
    assert result["distances"] == [3]
    assert list(result["ps_physical"]) == [0.0, 1.0]
    assert list(result["p_logical"]) == [0.0, 1.0]


def test_pseudo_threshold_code_capacity_default_xmodel() -> None:
    """The pseudo-threshold entry point shares the working default model."""
    result = pseudo_threshold_code_capacity(
        [0.0, 1.0],
        3,
        3,
        verbose=False,
        circuit_runner=pc.circuit_runners.Standard(seed=1),
    )
    assert list(result["plog"]) == [0.0, 1.0]


def test_threshold_code_capacity_calc_default_xmodel() -> None:
    """The fitting wrapper delivers default-model samples to its supplied fitter."""

    def fit(plist, dlist, plog, _func, _p0):
        assert list(plist) == [0.0, 1.0]
        assert list(dlist) == [3, 3]
        assert list(plog) == [0.0, 1.0]
        return pc.array([1.0]), pc.array([0.0])

    result = threshold_code_capacity_calc(
        [0.0, 1.0],
        [3],
        3,
        threshold_fit=fit,
        verbose=False,
        circuit_runner=pc.circuit_runners.Standard(seed=1),
    )
    assert list(result["plog"]) == [0.0, 1.0]
    assert list(result["opt"]) == [1.0]
    assert list(result["std"]) == [0.0]
