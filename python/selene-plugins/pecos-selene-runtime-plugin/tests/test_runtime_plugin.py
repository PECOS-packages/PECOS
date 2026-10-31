"""Exercise the packaged Runtime through the real Selene host."""

import os
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit, x
from pecos_selene_runtime_plugin import PecosRuntimePlugin
from selene_sim import Stim, build


@guppy
def feedback() -> None:
    q = qubit()
    x(q)
    m = measure(q)
    q2 = qubit()
    if m.read():
        x(q2)
    result("answer", measure(q2).read())


def plugin(**kwargs: int) -> PecosRuntimePlugin:
    override = os.environ.get("PECOS_RUNTIME_TEST_LIBRARY")
    return PecosRuntimePlugin(library_path=Path(override) if override else None, **kwargs)


def test_feedback_and_shot_reset() -> None:
    runner = build(feedback.compile())
    shots = [
        dict(shot)
        for shot in runner.run_shots(
            Stim(),
            runtime=plugin(max_batch_size=2, duration_ns_rxy=17, duration_ns_measure=23, duration_ns_reset=11),
            n_qubits=2,
            n_shots=4,
        )
    ]
    assert len(shots) == 4
    assert [shot["answer"] for shot in shots] == [True] * 4


@pytest.mark.parametrize("kwargs", [{"max_batch_size": 0}, {"duration_ns_rxy": -1}, {"max_live_results": True}])
def test_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        plugin(**kwargs)


def test_shared_qis_feedback_and_unread_shutdown_tail() -> None:
    source = Path(__file__).resolve().parents[4] / "crates/pecos-selene-runtime/tests/fixtures/feedback.ll"
    runner = build(source)
    shots = [dict(shot) for shot in runner.run_shots(Stim(), runtime=plugin(max_batch_size=2), n_qubits=2, n_shots=3)]
    assert len(shots) == 3
