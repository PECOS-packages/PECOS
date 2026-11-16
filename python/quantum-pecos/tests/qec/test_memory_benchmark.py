"""Independent frozen statistics and analytic boundary cases for memory reports."""

import json
from pathlib import Path

import pytest
from pecos.qec.memory_benchmark import MemoryCounts, equal_weight_memory_estimate

DATA = json.loads((Path(__file__).parent / "fixtures/si1000/benchmark_statistics.json").read_text())
INSTANCES = {row["instance_id"]: row for row in DATA["instances"]}


@pytest.mark.parametrize(
    ("outcome", "budget", "status"),
    [("timeout", 100, "native_sampling_timeout"), ("timeout", 5, "budget_exhausted"),
     ("error", 100, "native_sampling_error"), ("success", 100, "complete")],
)
@pytest.mark.parametrize("threads", [1, 2])
def test_native_worker_deadline_preserves_only_completed_shots(tmp_path, monkeypatch, outcome, budget, status, threads):
    import hashlib
    import importlib.util
    import subprocess
    import sys

    path = Path(__file__).resolve().parents[4] / "examples/python_examples/benchmark_si1000_memories.py"
    spec = importlib.util.spec_from_file_location("worker_benchmark_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binary = tmp_path / "decoder"
    binary.write_bytes(b"mock decoder: never executed")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "source_commit": module.SOURCE_COMMIT, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }))
    case = "planar_488_d3_r3_X_phase0_p0p001"
    output = tmp_path / "results"
    clock, calls = [0.], []

    def run(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        if "--native-worker" in command:
            request = json.loads(Path(command[-1]).read_text())
            Path(request["samples_path"]).write_text("0\n")  # Deliberately incomplete on failure.
            if outcome == "timeout":
                clock[0] += kwargs["timeout"]
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if outcome == "error":
                raise subprocess.CalledProcessError(2, command)
            Path(request["samples_path"]).write_text("0\n" * 4)
            Path(request["receipt_path"]).write_text(json.dumps({"ideal_reference": [[], []]}))
        else:
            stats = Path(command[command.index("--stats-out") + 1])
            stats.write_text(json.dumps({
                "num_shots": 4, "num_errors": 0, "num_low_confidence": 0, "num_det_orders": 63,
                "num_threads": int(command[command.index("--threads") + 1]),
                "det_beam": 20, "beam_climbing": True, "no_revisit_dets": True, "pqlimit": 1000000,
                "det_order_seed": 518278944, "sparsify_errors": False,
            }))
        clock[0] += 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--tesseract", str(binary), "--decoder-provenance", str(provenance), "--case", case,
        "--sampling-backend", "pecos", "--shots", "4", "--timeout", "10",
        "--wall-time-budget", str(budget), "--output", str(output), "--threads", str(threads),
    ])
    if outcome == "error":
        with pytest.raises(subprocess.CalledProcessError):
            module.main()
    else:
        module.main()
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == status
    assert report["decoder_threads"] == threads
    assert calls[1][1] == min(10, budget - 1)
    assert report["results"][0]["counts"]["shots"] == 4
    if outcome == "success":
        assert len(report["results"]) == len(calls) - 1 == 2
        assert report["pending_runs"] == []
        native_command = calls[-1][0]
        assert "--sample-num-shots" not in native_command
        assert "--sample-seed" not in native_command
        assert "--in-includes-appended-observables" in native_command
    else:
        assert len(report["results"]) == 1
        assert len(calls) == 2  # No decoding of partial native samples.
        assert report["pending_runs"] == [case + "/pecos"]


@pytest.mark.parametrize("point", DATA["points"], ids=lambda p: f"{p['family']}-d{p['distance']}-p{p['p']}")
def test_all_frozen_published_estimates_and_intervals(point):
    ids = [entry["instance_id"] for entry in point["instance_weights"]]
    assert len(ids) == len(set(ids))
    rows = [INSTANCES[key] for key in ids]
    expected = {(basis, phase) for basis in "XZ" for phase in ((0, 1) if point["family"] == "planar_488" else (0,))}
    assert {(row["basis"], row["start_phase"]) for row in rows} == expected
    assert all(entry["weight"] == 1 / len(ids) for entry in point["instance_weights"])
    result = equal_weight_memory_estimate(
        [MemoryCounts(row["shots"], row["block_errors"], row.get("low_confidence", 0)) for row in rows],
        num_logicals=point["k"],
        rounds=point["rounds"],
    )
    assert result.shots == point["shots"]
    assert result.failures == point["block_errors"]
    assert result.block_error_rate == pytest.approx(point["mean_block_error_rate"], rel=2e-12, abs=1e-18)
    assert result.rate_per_round_per_logical == pytest.approx(
        point["logical_error_rate_per_round_per_logical"], rel=2e-12, abs=1e-18,
    )
    assert result.interval == pytest.approx(point["interval95"], rel=2e-12, abs=1e-18)


def test_strata_are_not_shot_weighted_and_interval_endpoints_are_transformed():
    result = equal_weight_memory_estimate([MemoryCounts(10, 5), MemoryCounts(1000, 0)], num_logicals=2, rounds=3)
    assert result.block_error_rate == 0.25
    assert result.rate_per_round_per_logical == pytest.approx(1 - 0.75 ** (1 / 6))
    assert result.interval[0] < result.rate_per_round_per_logical < result.interval[1]


def test_zero_and_all_failures_have_one_sided_exact_bounds():
    zero = equal_weight_memory_estimate([MemoryCounts(100, 0)], num_logicals=1, rounds=1)
    full = equal_weight_memory_estimate([MemoryCounts(100, 100)], num_logicals=1, rounds=1)
    assert zero.rate_per_round_per_logical == zero.interval[0] == 0
    assert zero.interval[1] == pytest.approx(1 - 0.025**0.01)
    assert full.rate_per_round_per_logical == full.interval[1] == 1
    assert full.interval[0] == pytest.approx(0.025**0.01)


@pytest.mark.parametrize(
    ("shots", "failures", "unresolved"), [(0, 0, 0), (1, 2, 0), (1, -1, 0), (1, 0, 2), (True, 0, 0), (10, 1.5, 0)],
)
def test_invalid_counts(shots, failures, unresolved):
    with pytest.raises(ValueError, match="Counts"):
        MemoryCounts(shots, failures, unresolved)


def test_unresolved_decoder_outcomes_block_estimates():
    with pytest.raises(ValueError, match="Unresolved"):
        equal_weight_memory_estimate([MemoryCounts(100, 0, 1)], num_logicals=1, rounds=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_logicals": 0, "rounds": 1},
        {"num_logicals": 1, "rounds": True},
        {"num_logicals": 1, "rounds": 1, "confidence": 1.0},
        {"num_logicals": 1, "rounds": 1, "confidence": float("nan")},
    ],
)
def test_invalid_estimate_dimensions(kwargs):
    with pytest.raises(ValueError, match="Estimate"):
        equal_weight_memory_estimate([MemoryCounts(10, 1)], **kwargs)


def test_benchmark_only_groups_complete_basis_phase_sets():
    import importlib.util

    path = Path(__file__).resolve().parents[4] / "examples/python_examples/benchmark_si1000_memories.py"
    spec = importlib.util.spec_from_file_location("memory_benchmark_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"case": f"planar_488_d3_r3_{basis}_phase{phase}_p0p001", "variant": "pecos",
         "counts": {"shots": 100 * (index + 1), "failures": index, "unresolved": 0}}
        for index, (basis, phase) in enumerate((("X", 0), ("X", 1), ("Z", 0), ("Z", 1)))
    ]
    partial = module.group_estimates(rows[:-1])
    assert partial["complete_groups"] == []
    assert partial["incomplete_groups"][0]["missing"] == [("Z", 1)]
    complete = module.group_estimates(rows)
    assert complete["incomplete_groups"] == []
    estimate = complete["complete_groups"][0]["estimate"]
    assert estimate["strata"] == 4
    assert estimate["block_error_rate"] == pytest.approx((0 + 1/200 + 2/300 + 3/400) / 4)


@pytest.mark.parametrize("p", [0., .2])
def test_native_benchmark_samples_are_deviations_not_authored_signs(tmp_path, p):
    import importlib.util

    from pecos.qec.generic import SI1000, CircuitOp, MeasurementParity, ScheduledCircuit

    path = Path(__file__).resolve().parents[4] / "examples/python_examples/benchmark_si1000_memories.py"
    spec = importlib.util.spec_from_file_location("native_benchmark_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = ScheduledCircuit(1, ((CircuitOp("R", (0,)),), (CircuitOp("X", (0,)),), (CircuitOp("M", (0,)),)),
                            (MeasurementParity((0,), "detector"),),
                            (MeasurementParity((0,), "observable", True),))
    noisy = SI1000(p).apply(plan)
    destination = tmp_path / "samples.01"
    reference = module.write_native_samples(noisy, noisy.to_tick_circuit(), destination, shots=32, seed=101)
    assert reference == ((True,), (False,))
    rows = destination.read_text().splitlines()
    assert len(rows) == 32
    assert set(rows) <= {"00", "11"}
    assert ("11" in rows) == (p > 0)
