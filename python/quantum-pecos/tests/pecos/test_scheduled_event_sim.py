"""Python v4 factories through the production QIS/native/Monte Carlo route."""

import math
from pathlib import Path

import pytest
from test_scheduled_sim_integration import RAMSEY, build_timed_runtime


@pytest.fixture
def event_runtime(tmp_path: Path) -> Path:
    return build_timed_runtime(tmp_path, 0, events=True)[0]


def simulation(library, factory, program=RAMSEY):
    import pecos
    import pecos_rslib as pr
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    classical = (
        pr.qis_engine()
        .selene_runtime_plugin(
            str(library),
            SimpleRuntimePlugin().get_init_args(),
            custom_event_policy="reject_unhandled",
        )
        .scheduled_event_batches()
        .interface(pr.qis_helios_interface())
    )
    return (
        pecos.sim(pecos.Qis(program) if isinstance(program, str) else program)
        .classical(classical)
        .qubits(1)
        .quantum(pr.state_vector())
        .noise(pr.scheduled_event_idle_z(1, factory))
        .seed(42)
        .workers(2)
    )


class PhaseAdapter:
    """The invented event inserts RZ(pi), flipping the Ramsey readout."""

    def __init__(self):
        self.next_batch = 0

    def validate(self, batch):
        for op in batch.operations:
            if isinstance(op, tuple) and op != (4242, b"\x01"):
                message = "unknown synthetic event"
                raise ValueError(message)

    def translate(self, batch):
        from pecos_rslib.quantum import Gate, GateType

        assert batch.batch_index == self.next_batch
        self.next_batch += 1
        gates = [Gate(GateType.RZ, [math.pi], [0]) if isinstance(op, tuple) else op for op in batch.operations]
        for position, native_id, source_id in batch.measurements:
            assert not isinstance(batch.operations[position], tuple)
            assert native_id >= 0
            assert source_id >= 0
        return gates


def test_event_translation_and_shot_factories(event_runtime):
    contexts = []

    def factory(context):
        contexts.append(context)
        return PhaseAdapter()

    result = simulation(event_runtime, factory).run(6).to_dict()
    assert result["measurement_0"] == [1] * 6
    assert len(contexts) == len(set(contexts)) == 6
    assert {c[1] for c in contexts} == {0, 1}


def test_built_simulation_repeated_runs(event_runtime):
    contexts = []

    def factory(context):
        contexts.append(context)
        return PhaseAdapter()

    built = simulation(event_runtime, factory).build()
    assert built.run(2).to_dict()["measurement_0"] == [1, 1]
    assert built.run_with_workers(4, 2).to_dict()["measurement_0"] == [1] * 4
    assert len(contexts) == len(set(contexts)) == 6


@pytest.mark.parametrize("stage", ["factory", "validate", "translate"])
def test_callback_errors_are_reported(event_runtime, stage):
    class Failing(PhaseAdapter):
        def validate(self, batch):
            if stage == "validate":
                message = "intentional validation failure"
                raise ValueError(message)
            super().validate(batch)

        def translate(self, batch):
            if stage == "translate":
                message = "intentional translation failure"
                raise ValueError(message)
            return super().translate(batch)

    def factory(_context):
        if stage == "factory":
            message = "intentional factory failure"
            raise ValueError(message)
        return Failing()

    with pytest.raises(RuntimeError, match=f"scheduled adapter {stage}:"):
        simulation(event_runtime, factory).run(1)


@pytest.mark.parametrize("output", ["iterator", "wrong_type", "too_many", "remove_measurement"])
def test_output_contract_rejects(event_runtime, output):
    class Invalid(PhaseAdapter):
        def translate(self, batch):
            gates = super().translate(batch)
            if output == "iterator":
                return iter(gates)
            if output == "wrong_type":
                return [None]
            if output == "too_many" and gates:
                return [gates[0]] * 4097
            if output == "remove_measurement" and batch.measurements:
                return []
            return gates

    with pytest.raises(RuntimeError, match=r"adapter|measurement"):
        simulation(event_runtime, lambda _: Invalid()).run(1)


def test_validate_requires_none(event_runtime):
    class Invalid(PhaseAdapter):
        def validate(self, _batch):
            return False

    with pytest.raises(RuntimeError, match="validate must return None or raise"):
        simulation(event_runtime, lambda _: Invalid()).run(1)


def test_factory_configuration():
    import pecos_rslib as pr

    with pytest.raises(TypeError, match="must be callable"):
        pr.scheduled_event_idle_z(1, None)
    with pytest.raises(ValueError, match="capacity"):
        pr.scheduled_event_idle_z(17, lambda _: PhaseAdapter())


def test_reentrant_built_run_rejects_instead_of_deadlocking(event_runtime):
    holder = {}

    def factory(_context):
        with pytest.raises(RuntimeError, match="busy"):
            holder["built"].run(1)
        return PhaseAdapter()

    holder["built"] = simulation(event_runtime, factory).build()
    assert holder["built"].run(1).to_dict()["measurement_0"] == [1]


def test_feedback_through_production_result_mapping(event_runtime):
    program = RAMSEY.replace(
        "declare void @setup",
        "declare i1 @___read_future_bool(i64)\ndeclare void @setup",
    ).replace(
        "  call void @___qfree(i64 %q)",
        """  %value = call i1 @___read_future_bool(i64 %r)
  br i1 %value, label %flip, label %done
flip:
  call void @___rxy(i64 %q, double 0x400921FB54442D18, double 0.0)
  br label %done
done:
  %second = call i64 @___lazy_measure(i64 %q)
  call void @___qfree(i64 %q)""",
    )
    result = simulation(event_runtime, lambda _: PhaseAdapter(), program).run(4).to_dict()
    assert result["measurement_0"] == [1] * 4
    assert result["measurement_1"] == [0] * 4


def test_failed_run_does_not_reuse_adapter_state(event_runtime):
    fail = [True]

    def factory(_context):
        if fail[0]:
            message = "first run fails"
            raise ValueError(message)
        return PhaseAdapter()

    built = simulation(event_runtime, factory).build()
    with pytest.raises(RuntimeError, match="first run fails"):
        built.run(1)
    fail[0] = False
    assert built.run(2).to_dict()["measurement_0"] == [1, 1]


def test_actual_guppy_program(event_runtime):
    from guppylang import guppy
    from guppylang.std.builtins import result
    from guppylang.std.quantum import measure, qubit, x

    @guppy
    def prepare_one() -> None:
        q = qubit()
        x(q)
        result("outcome", measure(q).read())

    values = simulation(event_runtime, lambda _: PhaseAdapter(), prepare_one).run(4).to_dict()
    assert values["outcome"] == [1] * 4


def test_batch_snapshot_is_read_only_and_lists_are_copies(event_runtime):
    class Inspect(PhaseAdapter):
        def validate(self, batch):
            with pytest.raises(AttributeError):
                batch.start_nanos = 123
            operations = batch.operations
            original_length = len(operations)
            operations.clear()
            assert len(batch.operations) == original_length
            super().validate(batch)

    assert simulation(event_runtime, lambda _: Inspect()).run(1).to_dict()["measurement_0"] == [1]


@pytest.mark.parametrize(("coherent", "expected"), [(0.0, 1), (math.pi, 0)])
def test_event_normalization_preserves_nonzero_idle(tmp_path, coherent, expected):
    import pecos_rslib as pr

    library = build_timed_runtime(tmp_path, 1_000_000_000, events=True)[0]
    result = (
        simulation(library, lambda _: PhaseAdapter())
        .noise(pr.scheduled_event_idle_z(1, lambda _: PhaseAdapter(), coherent=coherent))
        .run(3)
        .to_dict()
    )
    assert result["measurement_0"] == [expected] * 3
