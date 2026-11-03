"""Diagnostic evidence for frontend admission, without enabling a Guppy API."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[4] / "scripts/diagnostics/runtime_decoder_guppy.py"


def probe(*arguments):
    # Preserve this test process's resolved packages, including isolated native
    # builds. Do not execute editable .pth files again in the child interpreter.
    launch = (
        f"import sys, runpy; sys.path = {sys.path!r}; "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    process = subprocess.run(
        [sys.executable, "-S", "-c", launch, str(PROBE), *arguments],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode in (0, 2), process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert process.returncode == (0 if report["outcome"] == "completed" else 2), report
    return report


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
@pytest.mark.parametrize(("case", "rounds"), [("single", 1), ("shared-measurement", 1), ("helper-loop", 3)])
@pytest.mark.parametrize("correct", [False, True])
def test_real_guppy_feedback_and_worker_shot_reuse(runtime, case, rounds, correct):
    report = probe("--case", case, "--runtime", runtime, *([] if correct else ["--no-correction"]))
    assert report["outcome"] == "completed", report
    assert report["sessions"] == [list(range(rounds))] * 4
    expected = [int(not correct)] * 2 if rounds == 1 else [[int(not correct)] * rounds] * 2
    assert report["results"] == [expected] * 2
    assert all(len(report[key]) == 64 for key in ("hugr_sha256", "compiled_sha256", "linked_sha256"))


def test_real_guppy_provider_failure_recovers_on_reused_runner():
    report = probe("--inject-failure")
    assert report["outcome"] == "completed", report
    assert "synthetic probe provider failure" in report["injected_failure"]
    assert report["sessions"] == [[0]] * 5
    assert report["results"] == [[0, 0]] * 2


def test_guppy_control_capture_still_rejects_before_provider():
    report = probe("--control")
    assert report["outcome"] == "rejected", report
    assert "compiler control provenance unsupported" in report["error"]
    assert report["sessions"] == report["results"] == []


@pytest.mark.parametrize("runtime", ["selene_simple_runtime", "selene_soft_rz_runtime"])
def test_branch_loop_cannot_silently_return_a_wrong_success(runtime):
    report = probe("--case", "branch-loop", "--runtime", runtime)
    if report["outcome"] == "completed":
        # A compiler fix may make the probe work. Require the actual contract,
        # rather than locking the test to a particular bad compilation.
        assert report["sessions"] == [[0, 1, 2]] * 4
        assert report["results"] == [[[0, 0, 0]] * 2] * 2
    else:
        # On the documented affected stack, the loop repeats instance zero.
        # Only that precise fail-closed outcome is accepted, not arbitrary
        # compiler/linker/provider errors or a subprocess timeout.
        assert "invalid decoder instance sequence: expected 1, received 0" in report["error"], report
        # Two shots are dispatched concurrently; one or both may have started
        # when the run fails. Neither may reach a second provider invocation.
        assert 1 <= len(report["sessions"]) <= 2
        assert all(session == [0] for session in report["sessions"])
        assert report["results"] == []
