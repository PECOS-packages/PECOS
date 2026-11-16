"""Common-sample timing comparisons must not report speedups from censored runs."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def profiler():
    path = Path(__file__).resolve().parents[4] / "examples/python_examples/profile_si1000_decoder.py"
    spec = importlib.util.spec_from_file_location("decoder_profile_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    return [
        {
            "threads": worker,
            "repetition": repeat,
            "shots": 8,
            "status": "complete",
            "wall_seconds": elapsed,
            "predictions_sha256": "identical",
        }
        for worker, repeat, elapsed in ((1, 0, 4.0), (4, 0, 2.0), (4, 1, 3.0), (1, 1, 6.0))
    ]


def test_speedups_use_median_wall_time_on_common_samples(profiler):
    summary = profiler.scaling_summary(_rows(), [1, 4], 2)
    assert summary["comparison_complete"]
    assert summary["predictions_identical"]
    assert summary["speedups"][1] == {
        "threads": 4,
        "median_wall_seconds": 2.5,
        "speedup_over_serial": 2.0,
        "shots_per_wall_second": 3.2,
    }


@pytest.mark.parametrize("failure", ["missing", "timeout", "predictions", "duplicate"])
def test_censoring_disagreement_and_duplicate_runs_block_speedups(profiler, failure):
    rows = _rows()
    if failure == "missing":
        rows.pop()
    elif failure == "timeout":
        rows[-1]["status"] = "timeout"
        del rows[-1]["predictions_sha256"]
    elif failure == "predictions":
        rows[-1]["predictions_sha256"] = "different"
    else:
        rows[-1] = rows[0].copy()
    summary = profiler.scaling_summary(rows, [1, 4], 2)
    assert not summary["comparison_complete"]
    assert summary["speedups"] == []


@pytest.mark.parametrize("outcome", [
    "success", "sampling_timeout", "decoder_timeout", "bad_predictions", "bad_stats",
    "missing_stats", "missing_predictions",
])
def test_profile_preserves_inputs_and_enforces_completion(tmp_path, monkeypatch, profiler, outcome):
    import stim

    benchmark = tmp_path / "benchmark"
    source = benchmark / "case" / "pecos"
    source.mkdir(parents=True)
    circuit = source / "circuit.stim"
    circuit.write_text("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    dem = source / "decoder.dem"
    dem.write_text(str(stim.Circuit(circuit.read_text()).detector_error_model()))
    binary = tmp_path / "decoder"
    binary.write_bytes(b"mock; never executed")
    command = [
        str(binary),
        "--num-det-orders",
        "21",
        "--det-order-seed",
        "518278944",
        "--beam",
        "20",
        "--pqlimit",
        "1000000",
        "--beam-climbing",
        "--no-revisit-dets",
        "--det-order-bfs",
        "--det-order-coordinate",
        "--det-order-index",
    ]
    (source / "command.json").write_text(json.dumps(command))
    report = {
        "decoder": {
            "source_commit": profiler.SOURCE_COMMIT,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "results": [
            {
                "case": "case",
                "variant": "pecos",
                "circuit_sha256": hashlib.sha256(circuit.read_bytes()).hexdigest(),
                "dem_sha256": hashlib.sha256(dem.read_bytes()).hexdigest(),
            },
        ],
    }
    (benchmark / "results.json").write_text(json.dumps(report))
    original = {p: p.read_bytes() for p in benchmark.rglob("*") if p.is_file()}
    clock, calls = [0.0], []

    def run(argv, **kwargs):
        calls.append((argv, kwargs["timeout"]))
        if "--sample-worker" in argv:
            if outcome == "sampling_timeout":
                clock[0] += kwargs["timeout"]
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            profiler._sample_worker(Path(argv[-1]))  # noqa: SLF001 - exercise the isolated worker
        else:
            if outcome == "decoder_timeout":
                clock[0] += kwargs["timeout"]
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            stats_path = Path(argv[argv.index("--stats-out") + 1])
            if outcome == "missing_stats":
                return subprocess.CompletedProcess(argv, 0)
            worker = int(argv[argv.index("--threads") + 1])
            stats_path.write_text(
                json.dumps(
                    {
                        "num_shots": 3,
                        "num_threads": worker,
                        "num_low_confidence": 0,
                        "num_det_orders": 63,
                        "det_beam": 20,
                        "det_order_seed": 518278944,
                        "pqlimit": 1000000,
                        "beam_climbing": True,
                        "no_revisit_dets": True,
                        "sparsify_errors": outcome == "bad_stats",
                        "total_time_seconds": 6.0,
                    },
                ),
            )
            if outcome != "missing_predictions":
                text = "00\n" * 3 if outcome == "bad_predictions" else "0\n" * 3
                Path(argv[argv.index("--out") + 1]).write_text(text)
        clock[0] += 1
        return subprocess.CompletedProcess(argv, 0)

    output = tmp_path / "profile"
    monkeypatch.setattr(profiler.subprocess, "run", run)
    monkeypatch.setattr(profiler.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--benchmark",
            str(benchmark),
            "--case",
            "case",
            "--output",
            str(output),
            "--shots",
            "3",
            "--threads",
            "1",
            "4",
            "--repetitions",
            "2",
            "--timeout",
            "5",
            "--wall-time-budget",
            "7",
        ],
    )
    if outcome in {"bad_stats", "bad_predictions"}:
        with pytest.raises(ValueError, match=r"Incomplete|Decoder prediction"):
            profiler.main()
    elif outcome in {"missing_stats", "missing_predictions"}:
        with pytest.raises(FileNotFoundError):
            profiler.main()
    else:
        profiler.main()
    saved = json.loads((output / "profile.json").read_text())
    assert all(p.read_bytes() == data for p, data in original.items())
    if outcome == "success":
        assert saved["status"] == "complete"
        assert [row["threads"] for row in saved["runs"]] == [1, 4, 4, 1]
        assert saved["comparison_complete"]
        assert saved["predictions_identical"]
        # The same exact sample file is fed to every invocation.
        assert len({argv[argv.index("--in") + 1] for argv, _ in calls[1:]}) == 1
    else:
        expected = {
            "sampling_timeout": "sampling_timeout",
            "decoder_timeout": "budget_exhausted",
            "bad_predictions": "invalid_predictions",
            "bad_stats": "invalid_decoder_result",
            "missing_stats": "invalid_decoder_result",
            "missing_predictions": "invalid_predictions",
        }
        assert saved["status"] == expected[outcome]
        assert not saved["comparison_complete"]
        assert saved["speedups"] == []
        if outcome == "decoder_timeout":
            assert [timeout for _, timeout in calls] == [5, 5, 1]
            assert len(saved["runs"]) == 2
            assert all(row["status"] == "timeout" for row in saved["runs"])
