"""Keep the compiler-only counter reproducer and its negative control usable."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[4] / "scripts/diagnostics/guppy_external_counter.py"


@pytest.mark.parametrize("case", ["branch", "plain"])
def test_external_counter_pass_isolation(case, tmp_path):
    launch = (
        f"import sys, runpy; sys.path = {sys.path!r}; "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    process = subprocess.run(
        [sys.executable, "-S", "-c", launch, str(PROBE), "--case", case, "--artifacts", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode in (0, 2), process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    stages = report["stages"]
    kinds = {name: stage["kind"] for name, stage in stages.items()}
    retained = dict.fromkeys(
        ("minimal", "default", "normalize_no_fold", "normalize_fold", "qsystem_no_fold", "qsystem_fold"),
        "input_retained",
    )
    if case == "branch" and report["outcome"] == "constant_zero":
        # Recognize only the demonstrated pass split, not arbitrary compiler
        # failures. This is evidence capture, not a passing compiler test.
        expected = retained | dict.fromkeys(("default", "normalize_fold", "qsystem_fold"), "constant_zero")
        assert process.returncode == 2
        assert kinds == expected, report
    else:
        # A future compiler fix can remove this specific symptom. The live
        # admission test still checks actual execution and instance ordering.
        assert process.returncode == 0
        assert report["outcome"] == "input_retained"
        assert kinds == retained, report
    for name, stage in stages.items():
        assert hashlib.sha256((tmp_path / f"{name}.hugr").read_bytes()).hexdigest() == stage["sha256"]
        graph = json.loads((tmp_path / f"{name}.json").read_text())
        nodes = graph["modules"][0]["nodes"]
        assert nodes[stage["call_node"]]["op"] == "Call"
        assert nodes[stage["source_node"]]["op"] == stage["source_op"]
    for level, digest in report["llvm_sha256"].items():
        assert hashlib.sha256((tmp_path / f"minimal-O{level}.ll").read_bytes()).hexdigest() == digest
