"""Lowered programs retain native builders and specific QIS refusals."""

import pecos as pc
import pecos_rslib
import pytest
from guppylang import guppy
from guppylang.std.quantum import measure, qubit


@guppy
def measure_qubit() -> bool:
    return measure(qubit()).read()


@pytest.fixture(params=["guppy", "hugr"])
def program(request):
    if request.param == "guppy":
        return pc.Guppy(measure_qubit)
    return pc.Hugr(measure_qubit.compile().to_bytes())


@pytest.fixture(params=["sim", "engine"])
def builder(request, program):
    if request.param == "sim":
        return pc.sim(program)
    return pc.selene_engine().program(program).to_sim()


def test_foreign_object_refusal_names_qis_wiring_issue(builder) -> None:
    """Fluent configuration keeps the native type and the issue #854 refusal."""
    assert isinstance(builder, pecos_rslib.SimBuilder)
    configured = builder.qubits(1).seed(42)
    assert isinstance(configured, pecos_rslib.SimBuilder)
    message = r"only supported for QASM programs; WASM foreign objects on the QIS route are not wired yet .*#854"
    with pytest.raises(TypeError, match=message):
        configured.foreign_object(object())
    with pytest.raises(TypeError, match=message):
        configured.foreign_object(foreign_obj=object())


def test_lowered_program_refuses_neo_stack(builder) -> None:
    """The native QIS builder rejects neo and names the supported stack."""
    assert isinstance(builder, pecos_rslib.SimBuilder)
    with pytest.raises(ValueError, match=r"Only QASM programs are routed to the neo stack.*engines stack"):
        builder.stack("neo")
    assert isinstance(builder.stack("engines"), pecos_rslib.SimBuilder)
