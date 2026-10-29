"""Observed-data decoding affects subsequent live QIS execution."""

from dataclasses import replace
from pathlib import Path

import pytest
from pecos import Qis, selene_engine, sim
from pecos.qec import RuntimeDecoder, RuntimeDecodeRequest, RuntimeDecodeWindow
from pecos_rslib import DagCircuit
from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

PROGRAM = (Path(__file__).resolve().parents[4] / "crates/pecos-qis/tests/fixtures/runtime_decode.ll").read_text()


def window() -> RuntimeDecodeWindow:
    """Known one-bit oracle: each modeled error flips D0 and L0 together."""
    dag = DagCircuit()
    dag.pz([0])
    dag.mz([0])
    model = (
        DemBuilder(DagFaultAnalyzer(dag).build_influence_map())
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-1]}]')
        .with_observables_json('[{"id":0,"records":[-1]}]')
        .build()
    )
    return RuntimeDecodeWindow(
        window_id=7,
        model=model,
        decoder="pecos_uf",
        detector_ids=("parity",),
        output_ids=("frame_x",),
        detector_order=(0,),
        output_map=((0,),),
        fault_ids=("readout",),
        independent=True,
        commit=("parity",),
    )


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("route", [(), (0,)])
def test_sim_live_frame_loop_and_shot_reuse(runtime, workers, route):
    definition = replace(window(), output_map=(route,))
    config = RuntimeDecoder({7: definition})
    runner = (
        sim(Qis(PROGRAM))
        .classical(selene_engine(runtime).runtime_decoder(config))
        .qubits(1)
        .workers(workers)
        .seed(5)
        .build()
    )
    expected = [[0 if route else 1] * 3] * 4
    assert runner.run(4).to_dict()["after"] == expected
    assert runner.run(4).to_dict()["after"] == expected


def test_provider_gets_only_public_identity_and_is_fresh_per_shot():
    definition = window()
    sessions = []

    def factory():
        requests = []
        sessions.append(requests)

        def provide(request):
            assert isinstance(request, RuntimeDecodeRequest)
            assert vars(request).keys() == {"window_id", "instance_id", "context"}
            requests.append(request)
            return definition

        return provide

    result = (
        sim(Qis(PROGRAM))
        .classical(selene_engine().runtime_decoder(RuntimeDecoder(provider_factory=factory)))
        .qubits(1)
        .workers(2)
        .run(4)
    )
    assert result.to_dict()["after"] == [[0, 0, 0]] * 4
    assert sessions == [[RuntimeDecodeRequest(7, i) for i in range(3)]] * 4


def test_provider_failure_stops_program_and_next_run_gets_fresh_state():
    definition = window()
    fail = [True]
    seen = []

    def factory():
        def provide(request):
            seen.append(request.instance_id)
            if request.instance_id == 1 and fail[0]:
                fail[0] = False
                msg = "synthetic provider failure"
                raise ValueError(msg)
            return definition

        return provide

    runner = (
        sim(Qis(PROGRAM))
        .classical(
            selene_engine().runtime_decoder(RuntimeDecoder(provider_factory=factory)),
        )
        .qubits(1)
        .build()
    )
    with pytest.raises(Exception, match="synthetic provider failure"):
        runner.run(1)
    assert seen == [0, 1]
    assert runner.run(1).to_dict()["after"] == [[0, 0, 0]]
    assert seen == [0, 1, 0, 1, 2]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"independent": False}, "independent closed"),
        ({"buffer": ("future",)}, "no buffer"),
        ({"boundary_state": ("leaked",)}, "boundary state"),
        ({"commit": ()}, "partial commits"),
        ({"detector_order": (1,)}, "complete permutation"),
        ({"detector_ids": ("",)}, "unique nonempty"),
        ({"fault_ids": ("readout", "readout")}, "unique nonempty"),
        ({"fault_ids": ()}, "explicitly declare"),
        ({"output_ids": ()}, "every model column"),
        ({"output_map": ()}, "every output"),
        ({"output_map": ((64,),)}, "frame bits"),
    ],
)
def test_rejects_unsupported_or_inconsistent_contracts(updates, message):
    with pytest.raises(ValueError, match=message):
        replace(window(), **updates)


def test_checked_request_identity_bits_and_failure_latching():
    config = RuntimeDecoder({7: window()})
    for args in [(7, 0, [2]), (7, 0, []), (7, 1, [1]), (8, 0, [1])]:
        session = config()
        with pytest.raises((ValueError, KeyError)):
            session.decode(*args)
        with pytest.raises(RuntimeError, match="session failed"):
            session.decode(7, 0, [1])
    wrong = RuntimeDecoder(provider_factory=lambda: lambda _: replace(window(), window_id=8))()
    with pytest.raises(ValueError, match="different window identity"):
        wrong.decode(7, 0, [1])


def test_output_routing_uses_gf2_fanout_and_explicit_projection():
    definition = replace(window(), output_map=((0, 2, 2, 63),))
    session = RuntimeDecoder({7: definition})()
    assert session.decode(7, 0, [1]) == 1 | (1 << 63)
    assert session.decode(7, 1, [0]) == 0


def test_detector_permutation_changes_live_feedback_at_equal_dimensions():
    dag = DagCircuit()
    dag.pz([0, 1])
    dag.mz([0])
    dag.mz([1])
    model = (
        DemBuilder(DagFaultAnalyzer(dag).build_influence_map())
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
        .with_observables_json('[{"id":0,"records":[-2]}]')
        .build()
    )
    source = PROGRAM.replace(
        "%data = alloca i8",
        "%data = alloca [2 x i8]\n    store [2 x i8] zeroinitializer, ptr %data",
    ).replace("ptr %data, i64 1)", "ptr %data, i64 2)")
    for order, expected in [((0, 1), 0), ((1, 0), 1)]:
        definition = replace(
            window(),
            model=model,
            detector_ids=("first", "second"),
            commit=("first", "second"),
            detector_order=order,
        )
        result = (
            sim(Qis(source))
            .classical(
                selene_engine().runtime_decoder(RuntimeDecoder({7: definition})),
            )
            .qubits(1)
            .run(1)
        )
        assert result.to_dict()["after"] == [[expected] * 3]


def test_supplied_model_can_come_from_existing_cached_slice_composer():
    from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
    from pecos_rslib.qec import DemSliceRoundSchedule, DetectorErrorModel

    builder = LogicalCircuitBuilder()
    builder.add_patch(SurfacePatch.create(3), "memory")
    builder.add_memory("memory", 3, "Z")
    dag = builder.to_dag_circuit()
    model = DetectorErrorModel.from_circuit(dag, p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001)
    schedule = model.round_schedule(DagFaultAnalyzer(dag).build_influence_map(), dag)
    cached = [(schedule.cached_slice(i), i) for i in range(4)]
    composed = DemSliceRoundSchedule.from_cached_slices(model, cached, [0], []).compose(
        start_round=0,
        commit_rounds=4,
        buffer_rounds=0,
        forward_boundary="hard",
    )
    ids = tuple(f"D{i}" for i in range(composed.num_detectors))
    definition = replace(
        window(),
        model=composed,
        detector_ids=ids,
        commit=ids,
        detector_order=tuple(range(len(ids))),
        fault_ids=("init", "bulk", "pre_terminal", "terminal"),
    )
    session = RuntimeDecoder({7: definition})()
    assert session.decode(7, 0, [0] * len(ids)) == 0
    # Existing internal windowed decoder also consumes the same structured model.
    definition = replace(definition, decoder="windowed:step=2,buffer=2,inner=pecos_uf")
    assert RuntimeDecoder({7: definition})().decode(7, 0, [0] * len(ids)) == 0


def test_tracked_pauli_outputs_are_rejected():
    from pecos_rslib import PauliString
    from pecos_rslib.qec import DetectorErrorModel

    dag = DagCircuit()
    dag.pz([0])
    dag.tracked_pauli(PauliString.from_str("Z"), label="tracked_z")
    dag.mz([0])
    model = DetectorErrorModel.from_circuit(dag, p1=0.1)
    assert model.num_tracked_paulis == 1
    with pytest.raises(ValueError, match="tracked-Pauli"):
        replace(window(), model=model, detector_ids=(), output_ids=(), detector_order=(), output_map=(), commit=())


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("ptr %data, i64 1)", "ptr %data, i64 1048577)", "detector limit"),
        ("ptr %data, i64 1)", "ptr null, i64 1)", "null"),
        ("store i8 %bit,", "store i8 2,", "zero or one"),
        ("i64 %instance, ptr", "i64 1, ptr", "instance sequence"),
    ],
)
def test_invalid_live_abi_request_fails_before_provider(original, replacement, message):
    starts = []

    def factory():
        starts.append(True)
        return lambda _request: window()

    with pytest.raises(Exception, match=message):
        (
            sim(Qis(PROGRAM.replace(original, replacement)))
            .classical(selene_engine().runtime_decoder(RuntimeDecoder(provider_factory=factory)))
            .qubits(1)
            .run(1)
        )
    assert not starts


@pytest.mark.parametrize("second_flip", [False, True])
@pytest.mark.parametrize("route", [(), (0,)])
def test_nonzero_composed_slices_correct_live_parity(second_flip, route):
    """Each readout fault flips its own D and L0, so L0 = D0 XOR D1."""
    from pecos_rslib.qec import DemSliceRoundSchedule

    dag = DagCircuit()
    for _ in range(2):
        dag.pz([0])
        dag.mz([0])
    for i, node in enumerate(dag.nodes()):
        dag.set_gate_attr(node, "dem_slice_round", i // 2)
    influence = DagFaultAnalyzer(dag).build_influence_map()
    model = (
        DemBuilder(influence)
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-2],"coords":[0,0,0]},{"id":1,"records":[-1],"coords":[0,0,1]}]')
        .with_observables_json('[{"id":0,"records":[-2,-1]}]')
        .build()
    )
    schedule = model.round_schedule(influence, dag)
    composed = DemSliceRoundSchedule.from_cached_slices(
        model,
        [(schedule.cached_slice(i), i) for i in range(2)],
        [0],
        [],
    ).compose(start_round=0, commit_rounds=2, buffer_rounds=0, forward_boundary="hard")
    definition = replace(
        window(),
        model=composed,
        detector_ids=("first", "second"),
        detector_order=(0, 1),
        commit=("first", "second"),
        output_map=(route,),
        fault_ids=("first_readout", "second_readout"),
    )
    # Deterministic X probes have the modeled readout's detector/output action.
    # Expected answers come from parity algebra, not another decoder invocation.
    fixture = Path(__file__).resolve().parents[4] / "crates/pecos-qis/tests/fixtures/runtime_decode_composed.ll"
    source = fixture.read_text().replace(
        "@second_flip = constant i1 false",
        f"@second_flip = constant i1 {str(second_flip).lower()}",
    )
    runner = (
        sim(Qis(source))
        .classical(selene_engine().runtime_decoder(RuntimeDecoder({7: definition})))
        .qubits(2)
        .workers(2)
        .build()
    )
    expected = 0 if route else int(not second_flip)
    for _ in range(2):
        result = runner.run(4).to_dict()
        assert result["observed"] == [[1, int(second_flip)] * 3] * 4
        assert result["after"] == [[expected] * 3] * 4


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("route", [(), (0,)])
def test_internal_noise_measurements_preserve_decode_feedback_and_reuse(flip, route):
    """Real noise-controller MZ continuations stay separate from program reads."""
    from pecos_rslib import GeneralNoiseModelBuilder, StateTransitionChannel

    # Each pre-X hook internally measures, then preserves or flips that state.
    # Identity: internal first result=0, program syndrome=1; correction gives 0.
    # Flip: the hook's X and program X cancel, so the syndrome is 0.
    channel = StateTransitionChannel(1.0, {"0->1" if flip else "0->0": 1.0, "1->0" if flip else "1->1": 1.0})
    noise = GeneralNoiseModelBuilder().add_p1_transition_channel_before_gate(channel)
    # The supplied one-bit model tests feedback, not a Pauli representation of
    # the state-transition channel's physical noise distribution.
    checked = RuntimeDecoder({7: replace(window(), output_map=(route,))})
    sessions = []
    fail = [True]

    class Session:
        def __init__(self):
            self.inner = checked()
            self.seen = []
            sessions.append(self.seen)

        def decode(self, window_id, instance_id, syndrome):
            assert syndrome == bytes([int(not flip)])
            self.seen.append(instance_id)
            if instance_id == 1 and fail[0]:
                fail[0] = False
                msg = "decoder failure after internal noise measurements"
                raise ValueError(msg)
            return self.inner.decode(window_id, instance_id, syndrome)

    runner = (
        sim(Qis(PROGRAM))
        .classical(selene_engine().runtime_decoder(Session))
        .qubits(1)
        .noise(noise)
        .workers(1)
        .seed(13)
        .build()
    )
    with pytest.raises(RuntimeError, match="decoder failure after internal noise measurements"):
        runner.run(1)
    assert sessions == [[0, 1]]
    expected = int(not flip and not route)
    for _ in range(2):
        result = runner.run_with_workers(4, 2).to_dict()
        assert result == {"after": [[expected] * 3] * 4}
    assert sessions[1:] == [[0, 1, 2]] * 8
