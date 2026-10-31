"""The standard PECOS runtime plugin through both public Python hosts."""

import os
from dataclasses import replace
from pathlib import Path

import pecos
import pecos_rslib as pr
import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit, x
from pecos_selene_runtime_plugin import PecosRuntimePlugin
from selene_sim import Stim, build


@guppy
def feedback() -> None:
    a = qubit()
    x(a)
    first = measure(a)
    b = qubit()
    if first.read():
        x(b)
    result("answer", measure(b).read())


@pytest.fixture(params=[1, 2])
def runtime(request: pytest.FixtureRequest) -> PecosRuntimePlugin:
    """Use the packaged plugin, optionally selecting a local development build."""
    override = os.environ.get("PECOS_RUNTIME_TEST_LIBRARY")
    return PecosRuntimePlugin(
        max_batch_size=request.param,
        library_path=Path(override) if override else None,
    )


def test_guppy_feedback_matches_standalone_selene(runtime: PecosRuntimePlugin) -> None:
    """One plugin object and one Guppy program work in both hosts."""
    runner = build(feedback.compile())
    selene_shots = [dict(shot) for shot in runner.run_shots(Stim(), runtime=runtime, n_qubits=2, n_shots=4)]
    pecos_shots = (
        pecos.sim(pecos.Guppy(feedback))
        .classical(pecos.selene_engine(runtime=runtime))
        .quantum(pecos.stabilizer())
        .qubits(2)
        .workers(2)
        .seed(42)
        .run(4)
        .to_dict()
    )
    assert [shot["answer"] for shot in selene_shots] == [True] * 4
    assert pecos_shots["answer"] == [True] * 4


class PassThrough:
    """Accept the runtime's ordinary gates on the scheduled event route."""

    def validate(self, batch: object) -> None:
        assert all(not isinstance(op, tuple) for op in batch.operations)

    def translate(self, batch: object) -> list:
        return batch.operations


@pytest.mark.parametrize("transport", ["flat", "scheduled", "events"])
def test_shared_qis_feedback_and_repeated_runs(runtime: PecosRuntimePlugin, transport: str) -> None:
    """Lazy futures, cached reads and unread tail survive workers and shot resets."""
    source = Path(__file__).resolve().parents[4] / "crates/pecos-selene-runtime/tests/fixtures/feedback.ll"
    classical = pecos.selene_engine(runtime=runtime).interface(pr.qis_helios_interface())
    profile = None
    if transport == "scheduled":
        classical.scheduled_batches()
        profile = pr.scheduled_idle_z(2)
    elif transport == "events":
        classical.scheduled_event_batches()
        profile = pr.scheduled_event_idle_z(2, lambda _: PassThrough())
    simulation = (
        pecos.sim(pecos.Qis(source.read_text()))
        .classical(classical)
        .quantum(pr.state_vector())
        .qubits(2)
        .workers(2)
        .seed(42)
    )
    if profile is not None:
        simulation = simulation.noise(profile)
    built = simulation.build()
    for shots in (3, 5):
        results = built.run(shots).to_dict()
        for index, expected in enumerate([1, 0, 1, 0]):
            assert results[f"measurement_{index}"] == [expected] * shots


@pytest.mark.parametrize("transport", ["flat", "scheduled", "events"])
@pytest.mark.parametrize("faulted", [False, True])
def test_leakage_value_drives_reset_with_real_noise(
    runtime: PecosRuntimePlugin,
    transport: str,
    faulted: bool,
) -> None:
    """Outcome 2 survives forcing/cached reads and controls reset under actual noise."""
    source = Path(__file__).resolve().parents[4] / "crates/pecos-selene-runtime/tests/fixtures/leakage_feedback.ll"
    # An intentionally long synthetic duration makes idle leakage deterministic:
    # linear rate 1/s times a 1 s gap gives probability 1. RZZ acts only on B/C,
    # which stay in |00>; A would remain |1> in the absence of leakage.
    timed = replace(runtime, duration_ns_rxy=17, duration_ns_rzz=1_000_000_000 if faulted else 0)
    classical = pecos.selene_engine(runtime=timed)
    if transport == "flat":
        profile = (
            pr.general_noise()
            .with_p1(float(faulted))
            .with_p1_emission_ratio(1.0)
            .with_p1_emission_model({"L": 1.0})
            .with_leakage_scale(1.0)
        )
    else:
        profile = pr.scheduled_idle_noise(3, linear=1.0, linear_model={"L": 1.0})
        if transport == "events":
            classical.scheduled_event_batches()
            profile = pr.scheduled_event_idle_noise(profile, lambda _: PassThrough())
        else:
            classical.scheduled_batches()
    built = (
        pecos.sim(pecos.Qis(source.read_text()))
        .classical(classical)
        .quantum(pr.state_vector())
        .noise(profile)
        .qubits(3)
        .workers(2)
        .seed(42)
        .build()
    )
    for shots in (3, 5):
        results = built.run(shots).to_dict()
        assert results["measurement_0"] == [2 if faulted else 1] * shots
        assert results["measurement_1"] == [0 if faulted else 1] * shots
        assert results["measurement_2"] == [0 if faulted else 1] * shots
