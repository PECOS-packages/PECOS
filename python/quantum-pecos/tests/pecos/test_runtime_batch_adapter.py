"""The same public runtime adapter drives execution, tracing, and idle DEMs."""

import math

import pecos
import pecos_rslib as pr
import pytest
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.qsystem.helios import phased_x
from guppylang.std.quantum import measure, qubit
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import NoiseParameters
from selene_simple_runtime_plugin import SimpleRuntimePlugin


@guppy
def ramsey_program() -> bool:
    q = qubit()
    # Native gates preserve the two scheduling sites through Guppy compilation.
    phased_x(q, pi / 2, pi / 2)
    phased_x(q, -pi / 2, pi / 2)
    return measure(q).read()


class PhaseAdapter:
    """Interpret an invented physical Z event; reject every other custom event."""

    def __init__(self):
        self.next_batch = 0

    def normalize(self, batch):
        assert batch["batch_index"] == self.next_batch
        self.next_batch += 1
        output = []
        for op in batch["operations"]:
            if "Custom" in op:
                if op != {"Custom": {"tag": 4242, "data": [1]}}:
                    message = "unknown synthetic physical event"
                    raise ValueError(message)
                output.append({"Rz": {"qubit_id": 0, "theta": math.pi}})
            else:
                output.append(op)
        return output


@pytest.fixture
def runtime(tmp_path, scheduled_support):
    library, _ = scheduled_support.build_runtime(tmp_path, 100, events=True)

    class Plugin:
        library_file = library
        batch_adapter_factory = PhaseAdapter

        def get_init_args(self):
            return SimpleRuntimePlugin().get_init_args()

    return Plugin(), pecos.Qis(scheduled_support.ramsey)


def simulation(plugin, program):
    return (
        pecos.sim(program)
        .classical(pecos.selene_engine(plugin))
        .qubits(1)
        .quantum(pr.state_vector())
        .seed(42)
        .workers(2)
    )


def test_physical_event_execution_and_trace_agree(runtime):
    plugin, program = runtime
    created = []

    def factory():
        adapter = PhaseAdapter()
        created.append(adapter)
        return adapter

    plugin.batch_adapter_factory = factory
    assert simulation(plugin, program).run(6).to_dict()["measurement_0"] == [1] * 6
    assert len(created) == 6
    trace = pecos.capture_qis_operation_trace(program, 1, runtime=plugin)
    gates = [g for chunk in trace for g in chunk["lowered_quantum_ops"]]
    assert [g["params"] for g in gates if g["gate_type"] == "Idle"] == [[100e-9]]
    assert any(g["gate_type"] == "RZ" and g["angles"] == [math.pi] for g in gates)
    replay = pecos.qis_operation_trace_to_tick_circuit(trace)
    assert [g.params for _, g in replay.gate_batches() if g.gate_type.name == "Idle"] == [[100.0]]
    assert len(created) == 7


def test_existing_idle_noise_uses_the_runtime_duration(runtime):
    plugin, program = runtime
    # One guaranteed Z idle fault cancels the event's Z on this Ramsey experiment.
    noise = pr.general_noise().with_p_idle_linear(1e7, {"Z": 1.0})
    assert simulation(plugin, program).noise(noise).run(4).to_dict()["measurement_0"] == [0] * 4

    circuit = pecos.trace_program_to_tick_circuit(program, 1, runtime=plugin)
    circuit.lower_clifford_rotations()
    circuit.set_meta("detectors", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "1")
    converted = NoiseParameters(
        p_idle_linear=2e6,
        p_idle_linear_model={"Z": 1.0},
        p_idle_sin_squared=1e6,
        p_idle_sin_squared_model={"Z": 1.0},
    ).for_runtime_idle_time_units()
    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0,
        p2=0,
        p_meas=0,
        p_prep=0,
        p_idle_z_linear_rate=converted.idle_memory_rates[2],
        p_idle_z_quadratic_sine_rate=converted.idle_memory_rates[8],
    )
    expected = 0.2 + math.sin(0.1) ** 2 - 2 * 0.2 * math.sin(0.1) ** 2
    error = next(line for line in str(dem).splitlines() if line.startswith("error("))
    assert float(error.split("(")[1].split(")")[0]) == pytest.approx(expected, abs=1e-6)


def test_public_dem_builder_reuses_simulator_idle_parameters(runtime):
    plugin, _ = runtime
    physical_noise = NoiseParameters(
        p1=0,
        p2=0,
        p_meas=0,
        p_prep=0,
        p_idle_linear=1e7,
        p_idle_linear_model={"Z": 1.0},
    )
    simulator_noise = pr.general_noise().with_p_idle_linear(
        physical_noise.effective_p_idle_z_linear_rate,
        {"Z": 1.0},
    )
    # A guaranteed idle Z cancels the physical event Z in this Ramsey experiment.
    assert simulation(plugin, ramsey_program).run(4).to_dict()["measurement_0"] == [1] * 4
    assert simulation(plugin, ramsey_program).noise(simulator_noise).run(4).to_dict()["measurement_0"] == [0] * 4

    build = (
        DetectorErrorModel.builder()
        .with_program(ramsey_program)
        .with_qubits(1)
        .with_runtime(plugin)
        .with_detectors_json('[{"id":0,"records":[-1]}]')
        .with_noise(physical_noise.for_runtime_idle_time_units())
        .build()
    )
    # No artificial idle insertion: the runtime's 100 ns gap is the sole site.
    assert [g.params for _, g in build.circuit.gate_batches() if g.gate_type.name == "Idle"] == [[100.0]]
    errors = [line for line in str(build.dem).splitlines() if line.startswith("error(")]
    assert len(errors) == 1
    assert float(errors[0].split("(")[1].split(")")[0]) == 1.0
    assert errors[0].endswith("D0")


def test_explicit_factory_and_strict_default(runtime):
    plugin, program = runtime
    plugin.batch_adapter_factory = None
    with pytest.raises(RuntimeError, match="unsupported runtime custom event tag 4242"):
        pecos.capture_qis_operation_trace(program, 1, runtime=plugin)
    trace = pecos.capture_qis_operation_trace(program, 1, runtime=plugin, batch_adapter_factory=PhaseAdapter)
    assert trace[-1]["stage"] == "trace_complete"


def test_capture_only_cannot_certify_a_dem_trace(runtime):
    plugin, program = runtime
    plugin.batch_adapter_factory = None
    trace = (
        pecos.sim(program)
        .classical(pecos.selene_engine(plugin, custom_event_policy="capture"))
        .qubits(1)
        .quantum(pr.state_vector())
        .capture_operation_trace(1)
    )
    assert trace[-1]["runtime_events_complete"] is False
    with pytest.raises(ValueError, match="unhandled runtime custom events"):
        pecos.qis_operation_trace_to_tick_circuit(trace)


@pytest.mark.parametrize(
    "corruption",
    ["custom", "missing_measurement", "wrong_result", "bad_qubit", "nan", "extra_field"],
)
def test_invalid_adapter_output_rejects(runtime, corruption):
    plugin, program = runtime

    class Invalid(PhaseAdapter):
        def normalize(self, batch):
            output = super().normalize(batch)
            if corruption == "custom":
                return batch["operations"]
            if corruption == "missing_measurement":
                return [op for op in output if "Measure" not in op]
            if corruption == "wrong_result":
                for op in output:
                    if "Measure" in op:
                        op["Measure"]["result_id"] += 1
            if corruption == "bad_qubit":
                output.append({"Reset": {"qubit_id": 1000}})
            if corruption == "nan":
                output.append({"Rz": {"qubit_id": 0, "theta": math.nan}})
            if corruption == "extra_field":
                output.append({"Reset": {"qubit_id": 0, "physical_effect": "unknown"}})
            return output

    plugin.batch_adapter_factory = Invalid
    with pytest.raises(RuntimeError, match="runtime batch adapter"):
        pecos.capture_qis_operation_trace(program, 1, runtime=plugin)


@pytest.mark.parametrize("runtime_name", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_standard_runtimes_need_no_event_adapter(runtime_name, scheduled_support):
    trace = pecos.capture_qis_operation_trace(pecos.Qis(scheduled_support.ramsey), 1, runtime=runtime_name)
    circuit = pecos.qis_operation_trace_to_tick_circuit(trace)
    assert circuit.num_measurements() == 1
