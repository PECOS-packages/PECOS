"""Python v4 factories through the production QIS/native/Monte Carlo route."""

import math
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def event_runtime(tmp_path: Path, scheduled_support):
    return scheduled_support.build_runtime(tmp_path, 0, events=True)[0], scheduled_support.ramsey


def simulation(runtime, factory, program=None):
    library, default_program = runtime
    program = default_program if program is None else program
    import pecos
    import pecos_rslib as pr
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    classical = (
        pr.qis_engine()
        .selene_runtime_plugin(
            str(library),
            SimpleRuntimePlugin().get_init_args(),
            custom_event_policy="capture",
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


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("iterator", "scheduled adapter output:"),
        ("wrong_type", "scheduled adapter output gate:"),
        ("too_many", "scheduled adapter expansion limit"),
        ("remove_measurement", "adapter changed measurement order, kind or targets"),
        ("unsupported_gate", "scheduled adapter output gate:.*unsupported scheduled gate"),
    ],
)
def test_output_contract_rejects(event_runtime, output, message):
    class Invalid(PhaseAdapter):
        def translate(self, batch):
            gates = super().translate(batch)
            if output == "iterator":
                return iter(gates)
            if output == "unsupported_gate":
                from pecos_rslib.quantum import Gate, GateType

                return [Gate(GateType.H, qubits=[0])]
            if output == "wrong_type":
                return [None]
            if output == "too_many" and gates:
                return [gates[0]] * 4097
            if output == "remove_measurement" and batch.measurements:
                return []
            return gates

    with pytest.raises(RuntimeError, match=message):
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


def run_isolated(runtime, body):
    """A parent-process timeout kills even a child blocked while holding the GIL."""
    library, program = runtime
    prefix = f"""
import sys, runpy
sys.path = {sys.path!r}
from pathlib import Path
import pytest
support = runpy.run_path({str(Path(__file__).resolve())!r})
simulation = support['simulation']
PhaseAdapter = support['PhaseAdapter']
runtime = (Path({str(library)!r}), {program!r})
"""
    completed = subprocess.run(
        [sys.executable, "-c", prefix + body],
        check=False,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert (
        completed.returncode == 0
    ), f"Isolated test exited with {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


@pytest.mark.parametrize("method", ["run", "run_with_workers", "reset"])
@pytest.mark.parametrize("stage", ["factory", "validate", "translate"])
def test_callback_reentry_rejects_without_deadlock(event_runtime, method, stage):
    run_isolated(
        event_runtime,
        f"""
holder = {{}}
def reenter():
    args = {{'run': (1,), 'run_with_workers': (1, 1), 'reset': ()}}[{method!r}]
    with pytest.raises(RuntimeError, match='scheduled adapter callback'):
        getattr(holder['built'], {method!r})(*args)
class Adapter(PhaseAdapter):
    def validate(self, batch):
        if {stage!r} == 'validate': reenter()
        return super().validate(batch)
    def translate(self, batch):
        if {stage!r} == 'translate': reenter()
        return super().translate(batch)
def factory(_):
    if {stage!r} == 'factory': reenter()
    return Adapter()
holder['built'] = simulation(runtime, factory).build()
assert holder['built'].run(1).to_dict()['measurement_0'] == [1]
holder.clear()
""",
    )


def test_concurrent_reset_waits_without_holding_gil(event_runtime):
    run_isolated(
        event_runtime,
        """
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Timer
entered, release, reset_started = Event(), Event(), Event()
def factory(_):
    entered.set()
    assert release.wait(10)
    return PhaseAdapter()
built = simulation(runtime, factory).build()
def reset():
    reset_started.set()
    return built.reset()
with ThreadPoolExecutor(max_workers=2) as pool:
    running = pool.submit(built.run, 1)
    assert entered.wait(5)
    timer = Timer(0.2, release.set)
    timer.start()
    resetting = pool.submit(reset)
    assert reset_started.wait(5)
    assert running.result(10).to_dict()['measurement_0'] == [1]
    assert resetting.result(10) is built
    timer.join()
assert built.run(2).to_dict()['measurement_0'] == [1, 1]
""",
    )


def test_plain_built_runs_still_serialize(event_runtime):
    run_isolated(
        event_runtime,
        """
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pecos
import pecos_rslib as pr
from selene_simple_runtime_plugin import SimpleRuntimePlugin
plugin = SimpleRuntimePlugin()
classical = pr.qis_engine().selene_runtime_plugin(str(plugin.library_file), plugin.get_init_args())
classical = classical.interface(pr.qis_helios_interface())
built = pecos.sim(pecos.Qis(runtime[1])).classical(classical).qubits(1).quantum(pr.state_vector()).workers(1).build()
barrier = Barrier(4)
def run(i):
    barrier.wait()
    result = built.run(100) if i % 2 else built.run_with_workers(100, 1)
    return result.to_dict()['measurement_0']
with ThreadPoolExecutor(max_workers=4) as pool:
    values = list(pool.map(run, range(4)))
assert values == [[0] * 100] * 4
""",
    )


def test_feedback_through_production_result_mapping(event_runtime):
    program = (
        event_runtime[1]
        .replace(
            "declare void @setup",
            "declare i1 @___read_future_bool(i64)\ndeclare void @setup",
        )
        .replace(
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
    )
    result = simulation(event_runtime, lambda _: PhaseAdapter(), program).run(4).to_dict()
    assert result["measurement_0"] == [1] * 4
    assert result["measurement_1"] == [0] * 4


def test_failed_run_does_not_reuse_adapter_state(event_runtime):
    fail = [True]
    instances = []

    class FailingAfterProgress(PhaseAdapter):
        def translate(self, batch):
            gates = super().translate(batch)
            if fail[0]:
                message = "translation failed after state advanced"
                raise ValueError(message)
            return gates

    def factory(_context):
        obj = FailingAfterProgress()
        instances.append(obj)
        return obj

    built = simulation(event_runtime, factory).build()
    with pytest.raises(RuntimeError, match="translation failed after state advanced"):
        built.run_with_workers(1, 1)
    assert instances[0].next_batch > 0
    fail[0] = False
    assert built.run_with_workers(2, 1).to_dict()["measurement_0"] == [1, 1]
    assert len(instances) == 3


@pytest.mark.parametrize(("apply_phase", "expected"), [(False, 0), (True, 1)])
def test_actual_guppy_program(event_runtime, apply_phase, expected):
    from guppylang import guppy
    from guppylang.std.angles import angle
    from guppylang.std.builtins import owned, result
    from guppylang.std.quantum import measure, qubit, ry

    @guppy.declare
    def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...

    @guppy
    def ramsey() -> None:
        q = qubit()
        ry(q, angle(0.5))
        # Preserve both pulses across compiler optimization.
        q = pecos_qis_runtime_barrier_qubit_hugr(q)
        ry(q, angle(-0.5))
        result("outcome", measure(q).read())

    contexts = []

    class GuppyPhaseAdapter(PhaseAdapter):
        def translate(self, batch):
            gates = super().translate(batch)
            if not apply_phase:
                gates = [gate for op, gate in zip(batch.operations, gates, strict=True) if not isinstance(op, tuple)]
            return gates

    def factory(context):
        contexts.append(context)
        return GuppyPhaseAdapter()

    values = simulation(event_runtime, factory, ramsey).run(4).to_dict()
    assert len(contexts) == len(set(contexts)) == 4
    assert values["outcome"] == [expected] * 4


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
def test_event_normalization_preserves_nonzero_idle(tmp_path, coherent, expected, scheduled_support):
    import pecos_rslib as pr

    library = scheduled_support.build_runtime(tmp_path, 1_000_000_000, events=True)[0]
    result = (
        simulation((library, scheduled_support.ramsey), lambda _: PhaseAdapter())
        .noise(pr.scheduled_event_idle_z(1, lambda _: PhaseAdapter(), coherent=coherent))
        .run(3)
        .to_dict()
    )
    assert result["measurement_0"] == [expected] * 3


@pytest.mark.parametrize("builder_name", ["qasm_engine", "phir_json_engine", "phir_engine"])
def test_other_engines_reject_event_factory(builder_name):
    import pecos_rslib as pr

    builder = getattr(pr, builder_name)().to_sim()
    with pytest.raises(TypeError, match="requires QIS/HUGR"):
        builder.noise(pr.scheduled_event_idle_z(1, lambda _: PhaseAdapter()))


@pytest.mark.parametrize("noise_first", [False, True])
def test_lowered_guppy_accepts_event_noise_and_rejects_neo_in_either_order(noise_first):
    import pecos
    import pecos_rslib as pr
    from guppylang import guppy

    @guppy
    def empty() -> None:
        pass

    builder = pecos.sim(empty)
    profile = pr.scheduled_event_idle_z(1, lambda _: PhaseAdapter())
    if noise_first:
        builder = builder.noise(profile)
    with pytest.raises(ValueError, match="Only QASM programs are routed to the neo stack"):
        builder.stack("neo")
    if not noise_first:
        builder = builder.noise(profile)
    builder.stack("engines")


def test_trace_capture_rejects_event_factory(event_runtime):
    with pytest.raises(TypeError, match="without operation tracing"):
        simulation(event_runtime, lambda _: PhaseAdapter()).capture_operation_trace()
