"""Check that the public handoff harness cannot turn incomplete evidence into parity."""

import copy
import importlib.util
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4] / "examples/guppy_wasm"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("parity_harness_tests", ROOT / "parity/parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def report(harness):
    configuration = harness.settings(shots=2, repeats=2)
    cases = []
    for case in harness.manifest()["cases"]:
        observation = (
            {"error": case["expected_error"]}
            if "expected_error" in case
            else {"shots": [dict(case["expected"]) for _ in range(2)]}
        )
        cases.append(
            {
                "id": case["id"],
                "status": "completed",
                "prepare_seconds": 1.0,
                "runs": [
                    {"seconds": seconds, "observation": copy.deepcopy(observation)} for seconds in (5.0, 2.0, 4.0)
                ],
            },
        )
    return {
        "schema_version": 1,
        "suite_id": harness.manifest()["suite_id"],
        "files": harness.fingerprints(),
        "settings": configuration,
        "host_tag": "machine-a",
        "stack_tag": "test-only",
        "adapter_kind": "external",
        "cases": cases,
    }


def test_comparison_checks_oracle_and_separates_cold_run(harness, report):
    candidate = copy.deepcopy(report)
    for record in candidate["cases"]:
        for run in record["runs"]:
            run["seconds"] *= 2
    result = harness.compare(report, candidate)
    assert result["all_workloads_conform"]
    assert not result["full_stack_parity_demonstrated"]
    assert result["cases"][0]["candidate_over_reference_median"] == 2
    # Even identical wrong answers from both runners must fail the oracle.
    report["cases"][0]["runs"][0]["observation"]["shots"][0]["sum"] = 99
    assert not harness.compare(report, report)["all_workloads_conform"]


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing", "Missing case"),
        ("duplicate", "duplicate case"),
        ("hash", "Report inputs"),
        ("settings", "Execution settings"),
        ("missing-run", "Missing run"),
        ("nan", "Invalid run timing"),
        ("extra", "Invalid report fields"),
        ("empty", "Missing case"),
    ],
)
def test_invalid_evidence_is_rejected(harness, report, fault, message):
    candidate = copy.deepcopy(report)
    if fault == "missing":
        candidate["cases"].pop()
    elif fault == "duplicate":
        candidate["cases"].append(candidate["cases"][0])
    elif fault == "hash":
        candidate["files"]["arithmetic.py"] = "0" * 64
    elif fault == "settings":
        candidate["settings"]["seed"] += 1
    elif fault == "missing-run":
        candidate["cases"][0]["runs"].pop()
    elif fault == "nan":
        candidate["cases"][0]["runs"][0]["seconds"] = float("nan")
    elif fault == "extra":
        candidate["private_details"] = "must not be accepted"
    else:
        candidate["cases"] = []
    with pytest.raises(ValueError, match=message):
        harness.compare(report, candidate)


def test_different_hosts_do_not_produce_performance_claims(harness, report):
    candidate = copy.deepcopy(report)
    candidate["host_tag"] = "another-machine"
    result = harness.compare(report, candidate)
    assert result["all_workloads_conform"]
    assert not result["same_host_tag"]
    assert all(case["candidate_over_reference_median"] is None for case in result["cases"])


@pytest.mark.parametrize("error", ["execution_error", "invalid_observation"])
def test_wrong_failures_are_not_expected_traps(harness, report, error):
    record = next(case for case in report["cases"] if case["id"] == "wasm-unreachable")
    record["runs"][0]["observation"] = {"error": error}
    assert not harness.compare(report, report)["all_workloads_conform"]


def test_prepare_failure_is_not_a_successful_trap(harness, report):
    record = next(case for case in report["cases"] if case["id"] == "wasm-unreachable")
    record.update(status="prepare_error", prepare_seconds=None, runs=[])
    assert not harness.compare(report, report)["all_workloads_conform"]


def test_safe_observations_drop_text_and_snapshot_mutable_results(harness):
    case = harness.manifest()["cases"][0]
    private_text = "private location, implementation or error text"
    assert harness.safe_observation({"shots": [{"sum": private_text}]}, case, 1) == {"error": "invalid_observation"}
    assert harness.safe_observation({"error": private_text}, case, 1) == {"error": "invalid_observation"}
    source = {"shots": [{"sum": 5}]}
    snapshot = harness.safe_observation(source, case, 1)
    source["shots"][0]["sum"] = 99
    assert snapshot == {"shots": [{"sum": 5}]}


def test_bundle_is_allowlisted_and_reproducible(harness, tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in harness.PAYLOAD:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((harness.ROOT / name).read_bytes())
    (source / "private_adapter.py").write_text("private implementation")
    (source / "private_report.json").write_text('{"private": true}')
    monkeypatch.setattr(harness, "ROOT", source)
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    harness.bundle(a)
    harness.bundle(b)
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as archive:
        assert set(archive.namelist()) == {"guppy_wasm/" + name for name in harness.PAYLOAD} | {
            "guppy_wasm/SHA256.json",
        }
        assert json.loads(archive.read("guppy_wasm/SHA256.json")) == harness.fingerprints()


@pytest.mark.parametrize("contents", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_json_is_rejected(harness, tmp_path, contents):
    path = tmp_path / "report.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=r"Duplicate JSON key|Nonfinite JSON number"):
        harness.read_json(path)


@pytest.mark.parametrize("failure", [RuntimeError, SystemExit])
def test_runner_distinguishes_preparation_and_execution_failures(harness, monkeypatch, failure):
    case = {
        "id": "trap",
        "source": "stateful.py",
        "module": "counter.wat",
        "function": "trapped",
        "expected_error": "wasm_trap",
    }
    monkeypatch.setattr(harness, "manifest", lambda: {"suite_id": "test", "cases": [case]})

    # Do not import Guppy: exercise the runner protocol independently of compilers.
    class Source:
        WASM_PATH = harness.ROOT / "counter.wat"
        trapped = None

    monkeypatch.setattr(harness, "load_module", lambda *_: Source)

    @contextmanager
    def fail_prepare(*_):
        message = "private details"
        raise failure(message)
        yield

    @contextmanager
    def fail_run(*_):
        def run():
            message = "private details"
            raise failure(message)

        yield run

    for adapter, status in [(fail_prepare, "prepare_error"), (fail_run, "completed")]:
        output = harness.run_suite(
            adapter,
            harness.settings(),
            stack_tag="test",
            host_tag="host",
            adapter_kind="external",
        )
        assert output["cases"][0]["status"] == status
        assert not harness.compare(output, output)["all_workloads_conform"]
        assert "private details" not in json.dumps(output)


def test_integer_state_cannot_be_replaced_with_booleans(harness, report):
    record = next(case for case in report["cases"] if case["id"] == "state-context-isolation")
    record["runs"][0]["observation"]["shots"][0]["independent"] = True
    assert not harness.compare(report, report)["all_workloads_conform"]
    assert harness.scalar_matches(1, True)
    assert not harness.scalar_matches(True, 1)
    assert not harness.scalar_matches(1.0, True)


@pytest.mark.parametrize("directory", [False, True])
def test_bundle_rejects_leaf_and_ancestor_symlinks(harness, tmp_path, monkeypatch, directory):
    source = tmp_path / "source"
    for name in harness.PAYLOAD:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((harness.ROOT / name).read_bytes())
    target = source / "parity" if directory else source / "parity/adapter_template.py"
    outside = tmp_path / "outside"
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=directory)
    monkeypatch.setattr(harness, "ROOT", source)
    archive = tmp_path / "must-not-exist.zip"
    with pytest.raises(ValueError, match="symlinks"):
        harness.bundle(archive)
    assert not archive.exists()


def test_public_adapter_rejects_duplicate_records_and_false_traps(harness):
    adapters = harness.load_module(harness.ROOT / "parity/public_runners.py", "test_public_parity_adapters")
    with pytest.raises(ValueError, match="Duplicate result label"):
        adapters.records_to_shot([("sum", 4), ("sum", 5)])
    assert adapters.records_to_shot([("sum", 5)]) == {"sum": 5}
    for message in ("WASM service unreachable", "Simulation failed: WASM service unreachable"):
        error = RuntimeError(message)
        with pytest.raises(RuntimeError, match="service unreachable"):
            adapters.pecos_failure(error)
    # A real-looking trap in stderr must not convert an unrelated panic to success.
    error = RuntimeError(adapters.TRAP_MESSAGE)
    with pytest.raises(RuntimeError, match="WebAssembly function"):
        adapters.normalized_failure(error, "unrelated panic", 2001)
    with pytest.raises(RuntimeError, match="WebAssembly function"):
        adapters.normalized_failure(error, adapters.TRAP_MESSAGE, 9999)
    assert adapters.normalized_failure(error, adapters.TRAP_MESSAGE, 2001) == {"error": "wasm_trap"}
    diagnostic = "Simulation failed: QIS program panic: code=2001, message=" + adapters.TRAP_MESSAGE
    assert adapters.pecos_failure(RuntimeError(diagnostic)) == {"error": "wasm_trap"}


def test_loader_executes_current_source_despite_stale_bytecode(harness, tmp_path):
    import os
    import py_compile

    path = tmp_path / "source.py"
    path.write_text("value = 4\n")
    before = path.stat()
    py_compile.compile(str(path), doraise=True)
    path.write_text("value = 5\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert harness.load_module(path, "parity_stale_bytecode_probe").value == 5


def test_cli_invalid_evidence_has_distinct_exit_status(harness, tmp_path):
    import subprocess
    import sys

    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(harness.ROOT / "parity/parity.py"),
            "compare",
            str(tmp_path / "missing.json"),
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert not output.exists()


def test_selene_parity_cli_executes_all_public_workloads(harness, tmp_path):
    import subprocess
    import sys

    output = tmp_path / "selene.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(harness.ROOT / "parity/parity.py"),
            "run",
            "--backend",
            "selene",
            "--output",
            str(output),
            "--stack-tag",
            "ci-selene",
            "--host-tag",
            "ci",
            "--shots",
            "2",
            "--repeats",
            "1",
            "--workers",
            "2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = harness.read_json(output)
    assert {"i32-boundaries", "i32-state-feedback"} <= {case["id"] for case in report["cases"]}
    assert len(report["cases"]) == 21
    assert harness.compare(report, report)["all_workloads_conform"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("adapter_kind",), []),
        (("cases", 0, "id"), []),
        (("cases", 0, "status"), []),
        (("cases", 0, "runs", 0, "observation"), {"error": []}),
        (("cases", 0, "runs", 0, "seconds"), 10**1000),
    ],
)
def test_malformed_field_types_are_rejected(harness, report, path, value):
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=r"schema|Invalid|Unexpected|Unknown"):
        harness.validate_report(report)


def test_worker_count_cannot_exceed_shots(harness):
    with pytest.raises(ValueError, match="Workers cannot exceed shots"):
        harness.settings(shots=1, workers=8)


@pytest.mark.parametrize("source", ["raise SystemExit()", "prepare = None", "value = 1"])
def test_cli_rejects_unloadable_adapter(harness, tmp_path, source):
    import subprocess
    import sys

    adapter = tmp_path / "adapter.py"
    adapter.write_text(source)
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(harness.ROOT / "parity/parity.py"),
            "run",
            "--adapter",
            str(adapter),
            "--output",
            str(output),
            "--stack-tag",
            "test",
            "--host-tag",
            "test",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert not output.exists()
    assert "Traceback" not in completed.stderr


def test_public_adapter_symlink_is_rejected_before_execution(harness, tmp_path, monkeypatch):
    import sys

    source = tmp_path / "source"
    (source / "parity").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("raise KeyboardInterrupt()")
    (source / "parity/public_runners.py").symlink_to(outside)
    monkeypatch.setattr(harness, "ROOT", source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity.py",
            "run",
            "--backend",
            "selene",
            "--output",
            str(tmp_path / "report.json"),
            "--stack-tag",
            "test",
            "--host-tag",
            "test",
        ],
    )
    with pytest.raises(ValueError, match="Cannot load"):
        harness.main()


def test_unrepresentable_ratio_is_omitted(harness, report):
    candidate = copy.deepcopy(report)
    for run in report["cases"][0]["runs"]:
        run["seconds"] = 1e-300
    for run in candidate["cases"][0]["runs"]:
        run["seconds"] = 1e300
    assert harness.compare(report, candidate)["cases"][0]["candidate_over_reference_median"] is None


def test_overflowed_median_does_not_produce_zero_ratio(harness, report):
    candidate = copy.deepcopy(report)
    for run in report["cases"][0]["runs"]:
        run["seconds"] = 1e308
    for run in candidate["cases"][0]["runs"]:
        run["seconds"] = 1e307
    assert harness.compare(report, candidate)["cases"][0]["candidate_over_reference_median"] is None
