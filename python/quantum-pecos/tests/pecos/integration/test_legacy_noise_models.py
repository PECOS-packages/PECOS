"""Legacy Pauli models work through the circuit runner and threshold entry points."""

import pecos as pc
import pytest
from pecos.analysis.pseudo_threshold_tools import pseudo_threshold_code_capacity
from pecos.analysis.threshold_tools import threshold_code_capacity, threshold_code_capacity_calc


@pytest.mark.parametrize("model_class", [pc.noise.XModel, pc.noise.ZModel, pc.noise.XZModel, pc.noise.DepolarModel])
@pytest.mark.parametrize("level", ["code_capacity", "phenomenological", "circuit"])
def test_legacy_models_run_at_supported_levels(model_class, level) -> None:
    """Construction and sampling must reach every supported model level."""
    model = model_class(model_level=level)
    qecc = pc.qeccs.Surface4444(distance=3)
    state = pc.simulators.SparseStabPy(qecc.num_qudits)
    runner = pc.circuit_runners.Standard(seed=1)
    init = pc.circuits.LogicalCircuit(suppress_warning=True)
    init.append(qecc.gate("ideal init |0>"))
    extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
    extraction.append(qecc.gate("I", num_syn_extract=1))
    runner.run(state, init)
    measurements, errors = runner.run(state, extraction, error_gen=model, error_params={"p": 1.0})
    assert errors
    symbols = {symbol for tick in errors.values() for circuit in tick.values() for symbol, _, _ in circuit.items()}
    locations = {
        qudit
        for tick in errors.values()
        for circuit in tick.values()
        for symbol, qudits, _ in circuit.items()
        if symbol != "I"
        for qudit in qudits
    }
    if level == "code_capacity":
        if model_class is pc.noise.DepolarModel:
            assert symbols
            assert symbols <= {"X", "Y", "Z"}
        else:
            expected = {pc.noise.XModel: {"X"}, pc.noise.ZModel: {"Z"}, pc.noise.XZModel: {"X", "Z"}}
            assert symbols == expected[model_class]
        assert locations == qecc.data_qudit_set
        runner.run(state, pc.decoders.MWPM2D(qecc).decode(measurements))
    else:
        # ZModel historically includes ZX among its two-qubit errors.
        allowed = {"I", "X", "Y", "Z"} if model_class is pc.noise.DepolarModel else {"I", "X", "Z"}
        assert symbols <= allowed
        if level == "phenomenological":
            assert qecc.data_qudit_set <= locations


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
