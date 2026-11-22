"""Run, bundle and compare public Guppy/WASM workloads without private dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import linecache
import math
import re
import statistics
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD = (
    "arithmetic.py",
    "arithmetic.wat",
    "stateful.py",
    "counter.wat",
    "floating_point.py",
    "floating_point.wat",
    "indexed_exports.py",
    "indexed_exports.wat",
    "integer32.py",
    "integer32.wat",
    "parity/throughput.py",
    "parity/workloads.json",
    "parity/parity.py",
    "parity/public_runners.py",
    "parity/adapter_template.py",
    "parity/README.md",
)
ERRORS = {"wasm_trap"}


def require(condition: bool, message: str) -> None:
    """Reject malformed evidence rather than treating it as a failed workload."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    """Read strict JSON, rejecting duplicate keys and nonfinite constants."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_value):
        message = "Nonfinite JSON number"
        raise ValueError(message)

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid_constant)


def write_json(path: Path, value: dict) -> None:
    """Write a machine-readable artifact; refuse nonfinite measurements."""
    require(path.resolve() not in {(ROOT / name).resolve() for name in PAYLOAD}, "Output would overwrite a suite file")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(payload)


def manifest() -> dict:
    """Load the public workload oracle independently of any runner report."""
    value = read_json(public_path("parity/workloads.json"))
    require(
        isinstance(value, dict) and type(value.get("schema_version")) is int and value["schema_version"] == 1,
        "Unknown workload schema",
    )
    cases = value.get("cases")
    require(isinstance(cases, list) and bool(cases), "Empty workload suite")
    seen = set()
    for case in cases:
        require(isinstance(case, dict), "Invalid workload")
        required = {"id", "source", "module", "function"}
        require(
            set(case) - {"wasm_int_width"} in (required | {"expected"}, required | {"expected_error"}),
            "Invalid workload fields",
        )
        width = case.get("wasm_int_width", 64)
        require(type(width) is int and width in (32, 64), "Invalid workload WASM integer width")
        require(isinstance(case["id"], str) and re.fullmatch(r"[a-z0-9-]+", case["id"]), "Invalid workload ID")
        require(case["id"] not in seen, "Duplicate workload ID")
        seen.add(case["id"])
        require(case["source"] in PAYLOAD and case["source"].endswith(".py"), "Invalid workload source")
        require(case["module"] in PAYLOAD and case["module"].endswith(".wat"), "Invalid workload module")
        require(isinstance(case["function"], str) and case["function"].isidentifier(), "Invalid workload function")
        if "expected_error" in case:
            require(case["expected_error"] in ERRORS, "Unknown expected failure")
        else:
            require(isinstance(case["expected"], dict) and bool(case["expected"]), "Empty result oracle")
            for key, expected in case["expected"].items():
                require(isinstance(key, str) and key.isidentifier(), "Invalid result label")
                require(finite_scalar(expected), "Invalid scalar oracle")
    return value


def public_path(name: str) -> Path:
    """Reject leaf or directory symlinks and paths outside the public payload."""
    require(name in PAYLOAD, "File is not in the public payload")
    parts = Path(name).parts
    require(
        not any(ROOT.joinpath(*parts[:n]).is_symlink() for n in range(1, len(parts) + 1)),
        "Bundle payload cannot contain symlinks",
    )
    path = ROOT / name
    require(path.resolve().is_relative_to(ROOT.resolve()), "Payload path escapes bundle root")
    return path


def fingerprints() -> dict[str, str]:
    """Hash the explicit public bundle, never arbitrary neighboring files."""
    return {name: hashlib.sha256(public_path(name).read_bytes()).hexdigest() for name in PAYLOAD}


def load_module(path: Path, name: str):
    """Load an explicitly selected source or local adapter."""
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load Python module")
    module = importlib.util.module_from_spec(spec)
    # Execute the hashed source, not timestamp-validated bytecode from a prior run.
    linecache.cache.pop(str(path), None)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102 - Explicitly selected source.
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def settings(shots=8, repeats=3, workers=1, seed=42) -> dict:
    """Specify identical shot and simulation settings for every adapter."""
    value = {
        "shots": shots,
        "repeats": repeats,
        "workers": workers,
        "seed": seed,
        "qubits": 1,
        "platform": "helios",
        "simulator_family": "state-vector",
    }
    validate_settings(value)
    return value


def validate_settings(value: dict) -> None:
    """Reject underspecified or invalid execution settings."""
    require(isinstance(value, dict), "Invalid settings")
    require(
        set(value) == {"shots", "repeats", "workers", "seed", "qubits", "platform", "simulator_family"},
        "Invalid settings fields",
    )
    for key in ("shots", "repeats", "workers"):
        require(type(value[key]) is int and value[key] > 0, "Counts must be positive integers")
    require(value["workers"] <= value["shots"], "Workers cannot exceed shots")
    require(type(value["seed"]) is int and 0 <= value["seed"] < 2**32, "Invalid seed")
    require(
        type(value["qubits"]) is int
        and value["qubits"] == 1
        and value["platform"] == "helios"
        and value["simulator_family"] == "state-vector",
        "Unsupported execution settings",
    )


def finite_scalar(value) -> bool:
    """Accept finite numeric scalars without overflowing integer conversion."""
    try:
        return type(value) in (int, float, bool) and math.isfinite(value)
    except OverflowError:
        return False


def scalar_matches(actual, expected) -> bool:
    """Match known exact scalar answers, permitting boolean measurement encoding."""
    if type(expected) is bool and type(actual) is int:
        return actual in (0, 1) and bool(actual) == expected
    return type(actual) is type(expected) and actual == expected


def safe_observation(value, case: dict, shots: int) -> dict:
    """Whitelist public numeric observations; never export arbitrary adapter text."""
    if not isinstance(value, dict):
        return {"error": "invalid_observation"}
    if set(value) == {"error"} and isinstance(value["error"], str) and value["error"] in ERRORS:
        return {"error": value["error"]}
    if set(value) != {"shots"} or not isinstance(value["shots"], list) or len(value["shots"]) != shots:
        return {"error": "invalid_observation"}
    expected = case.get("expected", {})
    for shot in value["shots"]:
        if not isinstance(shot, dict) or set(shot) != set(expected):
            return {"error": "invalid_observation"}
        for actual in shot.values():
            if not finite_scalar(actual):
                return {"error": "invalid_observation"}
    return {"shots": [dict(shot) for shot in value["shots"]]}


def observation_passes(observation: dict, case: dict, shots: int) -> bool:
    """Check the independent oracle, not a runner's claimed pass status."""
    if "expected_error" in case:
        return observation == {"error": case["expected_error"]}
    if set(observation) != {"shots"} or len(observation["shots"]) != shots:
        return False
    return all(
        set(shot) == set(case["expected"])
        and all(scalar_matches(shot[key], expected) for key, expected in case["expected"].items())
        for shot in observation["shots"]
    )


def run_suite(prepare, configuration: dict, *, stack_tag: str, host_tag: str, adapter_kind: str) -> dict:
    """Execute all cases, recording preparation and each complete run separately."""
    validate_settings(configuration)
    for tag in (stack_tag, host_tag):
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tag)), "Use a short non-sensitive tag")
    suite = manifest()
    hashes = fingerprints()
    report = {
        "schema_version": 1,
        "suite_id": suite["suite_id"],
        "files": hashes,
        "settings": configuration,
        "stack_tag": stack_tag,
        "host_tag": host_tag,
        "adapter_kind": adapter_kind,
        "cases": [],
    }
    for case in suite["cases"]:
        record = {"id": case["id"], "status": "prepare_error", "prepare_seconds": None, "runs": []}
        with TemporaryDirectory(prefix="guppy-wasm-parity-") as build_dir:
            entered = False
            start = time.perf_counter()
            try:
                source = load_module(public_path(case["source"]), "parity_" + case["id"].replace("-", "_"))
                program = getattr(source, case["function"])
                modules = {str(source.WASM_PATH): public_path(case["module"]).read_bytes()}
                case_settings = dict(configuration, wasm_int_width=case.get("wasm_int_width", 64))
                with prepare(program, modules, case_settings, Path(build_dir)) as run:
                    record["prepare_seconds"] = time.perf_counter() - start
                    entered = True
                    for _ in range(configuration["repeats"] + 1):
                        start = time.perf_counter()
                        try:
                            observation = safe_observation(run(), case, configuration["shots"])
                        # Do not export exception text or classify arbitrary failures as traps.
                        except (Exception, SystemExit):
                            observation = {"error": "execution_error"}
                        record["runs"].append({"seconds": time.perf_counter() - start, "observation": observation})
                    record["status"] = "completed"
            except (Exception, SystemExit):
                record["status"] = "cleanup_error" if entered else "prepare_error"
        report["cases"].append(record)
    require(hashes == fingerprints(), "Suite files changed during execution")
    return report


def duration(value) -> bool:
    """Timings are finite nonnegative numbers, never boolean flags."""
    try:
        return type(value) in (float, int) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def validate_report(report: dict) -> dict[str, dict]:
    """Require complete, current and well-formed evidence before comparison."""
    suite = manifest()
    fields = {"schema_version", "suite_id", "files", "settings", "stack_tag", "host_tag", "adapter_kind", "cases"}
    require(isinstance(report, dict) and set(report) == fields, "Invalid report fields")
    require(
        type(report["schema_version"]) is int
        and report["schema_version"] == 1
        and report["suite_id"] == suite["suite_id"],
        "Report schema or suite mismatch",
    )
    require(report["files"] == fingerprints(), "Report inputs differ from this bundle")
    validate_settings(report["settings"])
    require(
        isinstance(report["adapter_kind"], str) and report["adapter_kind"] in {"pecos", "selene", "external"},
        "Invalid adapter kind",
    )
    for key in ("stack_tag", "host_tag"):
        require(
            isinstance(report[key], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", report[key]),
            "Invalid report tag",
        )
    require(isinstance(report["cases"], list), "Invalid report cases")
    expected = {case["id"]: case for case in suite["cases"]}
    records = {}
    for record in report["cases"]:
        require(
            isinstance(record, dict) and set(record) == {"id", "status", "prepare_seconds", "runs"},
            "Invalid case record",
        )
        case_id = record["id"]
        require(
            isinstance(case_id, str) and case_id in expected and case_id not in records,
            "Unexpected or duplicate case",
        )
        records[case_id] = record
        require(
            isinstance(record["status"], str) and record["status"] in {"completed", "prepare_error", "cleanup_error"},
            "Unknown case status",
        )
        require(record["prepare_seconds"] is None or duration(record["prepare_seconds"]), "Invalid preparation time")
        require(isinstance(record["runs"], list), "Invalid run records")
        if record["status"] == "completed":
            require(duration(record["prepare_seconds"]), "Missing preparation time")
            require(len(record["runs"]) == report["settings"]["repeats"] + 1, "Missing run observations")
        for run in record["runs"]:
            require(
                isinstance(run, dict) and set(run) == {"seconds", "observation"} and duration(run["seconds"]),
                "Invalid run timing",
            )
            observation = run["observation"]
            require(isinstance(observation, dict), "Invalid observation")
            if set(observation) == {"error"}:
                require(
                    isinstance(observation["error"], str)
                    and observation["error"] in ERRORS | {"invalid_observation", "execution_error"},
                    "Unknown failure",
                )
            else:
                require(
                    safe_observation(observation, expected[case_id], report["settings"]["shots"]) == observation,
                    "Invalid observation payload",
                )
    require(set(records) == set(expected), "Missing case records")
    return records


def compare(reference: dict, candidate: dict) -> dict:
    """Report scoped conformance and raw performance ratios, never full parity."""
    left, right = validate_report(reference), validate_report(candidate)
    require(reference["settings"] == candidate["settings"], "Execution settings differ")
    same_host = reference["host_tag"] == candidate["host_tag"]
    rows = []
    for case in manifest()["cases"]:
        a, b = left[case["id"]], right[case["id"]]

        def passed(record, expected_case=case):
            return record["status"] == "completed" and all(
                observation_passes(run["observation"], expected_case, reference["settings"]["shots"])
                for run in record["runs"]
            )

        row = {"id": case["id"], "reference_passed": passed(a), "candidate_passed": passed(b)}
        row["candidate_over_reference_median"] = None
        if same_host and passed(a) and passed(b) and "expected_error" not in case:
            baseline = statistics.median(run["seconds"] for run in a["runs"][1:])
            measured = statistics.median(run["seconds"] for run in b["runs"][1:])
            if baseline > 0 and math.isfinite(baseline) and math.isfinite(measured):
                ratio = measured / baseline
                if math.isfinite(ratio):
                    row["candidate_over_reference_median"] = ratio
        rows.append(row)
    return {
        "schema_version": 1,
        "suite_id": reference["suite_id"],
        "reference": reference["stack_tag"],
        "candidate": candidate["stack_tag"],
        "same_host_tag": same_host,
        "all_workloads_conform": all(row["reference_passed"] and row["candidate_passed"] for row in rows),
        "full_stack_parity_demonstrated": False,
        "cases": rows,
    }


def bundle(destination: Path) -> None:
    """Create a deterministic archive containing only allowlisted public files."""
    manifest()
    hashes = fingerprints()
    require(
        destination.resolve() not in {(ROOT / name).resolve() for name in PAYLOAD},
        "Output would overwrite a suite file",
    )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in PAYLOAD:
            content = public_path(name).read_bytes()
            require(hashlib.sha256(content).hexdigest() == hashes[name], "Files changed while bundling")
            info = zipfile.ZipInfo("guppy_wasm/" + name)
            info.create_system = 3
            archive.writestr(info, content)
        info = zipfile.ZipInfo("guppy_wasm/SHA256.json")
        info.create_system = 3
        archive.writestr(info, json.dumps(hashes, indent=2, sort_keys=True) + "\n")


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        message = "must be positive"
        raise argparse.ArgumentTypeError(message)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    package = sub.add_parser("bundle")
    package.add_argument("output", type=Path)
    run = sub.add_parser("run")
    choice = run.add_mutually_exclusive_group(required=True)
    choice.add_argument("--backend", choices=("pecos", "selene"))
    choice.add_argument("--adapter", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--stack-tag", required=True)
    run.add_argument("--host-tag", required=True)
    run.add_argument("--shots", type=positive, default=8)
    run.add_argument("--repeats", type=positive, default=3)
    run.add_argument("--workers", type=positive, default=1)
    comparison = sub.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bundle":
        bundle(args.output)
        return 0
    if args.command == "compare":
        result = compare(read_json(args.reference), read_json(args.candidate))
        write_json(args.output, result)
        return 0 if result["all_workloads_conform"] else 1
    configuration = settings(args.shots, args.repeats, args.workers)
    try:
        if args.adapter:
            prepare = load_module(args.adapter, "local_parity_adapter").prepare
        else:
            adapters = load_module(public_path("parity/public_runners.py"), "public_parity_adapters")
            prepare = getattr(adapters, args.backend)
        require(callable(prepare), "Adapter must provide a callable")
    except (Exception, SystemExit):
        message = "Cannot load the selected adapter"
        raise ValueError(message) from None
    report = run_suite(
        prepare,
        configuration,
        stack_tag=args.stack_tag,
        host_tag=args.host_tag,
        adapter_kind=args.backend or "external",
    )
    write_json(args.output, report)
    result = compare(report, report)
    print(json.dumps({"cases": len(report["cases"]), "all_workloads_conform": result["all_workloads_conform"]}))
    return 0 if result["all_workloads_conform"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError):
        print(
            "Invalid inputs, incompatible evidence, or an output I/O error; no valid result produced.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
