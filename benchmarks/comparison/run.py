"""Explicitly invoked four-route benchmarks; standard library only, POSIX hosts."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
STAGES = ("prepare", "execute_materialize", "persist", "teardown")
CPU = ("self_user", "self_system", "children_user", "children_system")


def require(ok: bool, message: str) -> None:
    """Reject an invalid configuration or evidence condition."""
    if not ok:
        raise ValueError(message)


def sha(path: Path | str) -> str:
    """Hash one named input or artifact."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path | str, value: Any) -> None:  # noqa: ANN401 - JSON serialization boundary
    """Save finite JSON evidence."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path: Path | str) -> Any:  # noqa: ANN401 - JSON serialization boundary
    """Read a JSON evidence document."""
    return json.loads(path.read_text())


def inputs(config: dict[str, Any]) -> None:
    """Verify every named input before execution."""
    require(bool(config["inputs"]), "named input hashes required")
    for name, digest in config["inputs"].items():
        require(Path(name).is_absolute() and sha(name) == digest, "input changed: " + name)


def tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the budget and return a balanced four-route schedule."""
    require(config["schema"] == 1, "unsupported schema")
    routes = config["routes"]
    require(len(routes) == 4, "first milestone supports exactly four routes")
    names = [r["name"] for r in routes]
    require(len(set(names)) == 4 and all(re.fullmatch(r"[a-z][a-z0-9_-]*", n) for n in names), "invalid route names")
    for key in ("cycles", "shots", "warmup_shots", "timeout_seconds"):
        require(type(config[key]) is int and config[key] > 0, "invalid " + key)
    for route in routes:
        require(route["argv"] and all(type(s) is str for s in route["argv"]), "argv must be a string list")
        require(Path(route["argv"][0]).is_absolute(), "absolute executable required")
        require(all(type(k) is str and type(v) is str for k, v in route["env"].items()), "invalid environment")
    orders = []
    for cycle in range(config["cycles"]):
        orders.extend(ORDERS if cycle % 2 == 0 else reversed(ORDERS))
    return [
        {"block": b, "route": names[r], "index": b * 4 + p}
        for b, order in enumerate(orders)
        for p, r in enumerate(order)
    ]


def launch_spec(config: dict[str, Any], task: dict[str, Any], folder: Path) -> tuple[list[str], dict[str, str]]:
    """Render argv and the explicit environment without a shell."""
    route = next(r for r in config["routes"] if r["name"] == task["route"])
    values = {"output": str(folder), "route": task["route"], "shots": config["shots"], "warmup": config["warmup_shots"]}
    return ([s.format(**values) for s in route["argv"]], {k: v.format(**values) for k, v in route["env"].items()})


def interval(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Validate raw cumulative usage and compute a phase duration."""
    for point in (a, b):
        require(set(point) == {"wall_ns", *CPU}, "invalid usage point fields")
        require(type(point["wall_ns"]) is int, "wall clock must be integer nanoseconds")
        require(
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in point.values()),
            "invalid usage point",
        )
    wall = (b["wall_ns"] - a["wall_ns"]) / 1e9
    parts = [b[k] - a[k] for k in CPU]
    require(wall > 0 and all(v >= 0 for v in parts), "nonmonotonic usage")
    return {"wall_seconds": wall, "cpu_seconds": sum(parts)}


def report(config: dict[str, Any], task: dict[str, Any], folder: Path) -> dict[str, float]:
    """Validate worker evidence and compute primary API durations."""
    r = read(folder / "report.json")
    require(type(r["shots"]) is int and type(r["warmup_shots"]) is int, "integer worker budget required")
    require(
        r["route"] == task["route"] and r["shots"] == config["shots"] and r["warmup_shots"] == config["warmup_shots"],
        "worker budget/route differs",
    )
    require(
        r["correctness_passed"] is True and r["prepared_program_reused"] is True,
        "worker did not attest correctness/reuse",
    )
    require(r["evidence"] and all(type(x) is str for x in r["evidence"]), "missing worker evidence")
    for name in r["evidence"]:
        p = folder / name
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts and p.is_file() and not p.is_symlink(),
            "invalid evidence path",
        )
    metrics = {}
    for phase in ("warmup", "timed"):
        points = r[phase]
        require(len(points) == 5, "five phase boundaries required")
        metrics[phase] = {stage: interval(points[i], points[i + 1]) for i, stage in enumerate(STAGES)}
    interval(r["warmup"][-1], r["timed"][0])
    m = metrics["timed"]
    return {key: sum(m[s][key] for s in STAGES[:2]) for key in ("wall_seconds", "cpu_seconds")}


def summary(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Recompute all route statistics from saved raw points."""
    series = {r["name"]: {"wall_seconds": [], "cpu_seconds": []} for r in config["routes"]}
    for task in tasks(config):
        folder = root / str(task["index"])
        proc = read(folder / "process.json")
        require(
            proc["returncode"] == 0 and proc["cleanup_confirmed"] is True and proc["timed_out"] is False,
            "incomplete worker",
        )
        argv, env = launch_spec(config, task, folder)
        require(read(folder / "command.json") == {"argv": argv, "env": env}, "saved command changed")
        for k, v in report(config, task, folder).items():
            require(v > 0, "positive API duration required for ratios")
            series[task["route"]][k].append(v)
    ratios = {
        a + "/" + b: {k: [x / y for x, y in zip(series[a][k], series[b][k], strict=True)] for k in series[a]}
        for a in series
        for b in series
        if a != b
    }
    return {
        "series": series,
        "medians": {r: {k: statistics.median(v) for k, v in row.items()} for r, row in series.items()},
        "block_ratios": ratios,
        "median_block_ratios": {r: {k: statistics.median(v) for k, v in row.items()} for r, row in ratios.items()},
    }


def alive(group: int) -> bool:
    """Check whether a process group still exists."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    else:
        return True


def execute(argv: list[str], env: dict[str, str], folder: Path, timeout: float) -> None:
    """Run one isolated worker and retain its cleanup receipt."""
    record = {"returncode": None, "timed_out": False, "cleanup_confirmed": False}
    proc = None
    try:
        with (folder / "worker.log").open("x") as log:
            proc = subprocess.Popen(  # noqa: S603 - explicitly trusted argv, no shell
                argv,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["process_group"] = proc.pid
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                raise
            require(not alive(proc.pid), "worker left live descendants")
            require(proc.returncode == 0, "worker failed")
    finally:
        try:
            if proc is not None:
                if alive(proc.pid):
                    with suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                deadline = time.monotonic() + 5
                while alive(proc.pid) and time.monotonic() < deadline:
                    time.sleep(0.02)
                record.update(returncode=proc.returncode, cleanup_confirmed=not alive(proc.pid))
        finally:
            write(folder / "process.json", record)


def inventory(root: Path) -> dict[str, str]:
    """Hash output files while rejecting symlinks."""
    files = {}
    for p in root.rglob("*"):
        require(not p.is_symlink(), "symlink output not admitted")
        if p.is_file() and p != root / "manifest.json":
            files[str(p.relative_to(root))] = sha(p)
    return files


def verify_run(root: Path) -> None:
    """Verify saved artifacts, inputs and all summary arithmetic."""
    root = root.resolve()
    require(not (root / "failure.json").exists(), "failed run")
    require(read(root / "manifest.json") == inventory(root), "manifest differs")
    frozen = read(root / "freeze.json")
    require(sha(__file__) == frozen["runner_sha256"], "runner changed")
    config = frozen["config"]
    inputs(config)
    require(read(root / "summary.json") == summary(config, root), "summary differs")
    print("Saved benchmark verified")


def _terminate(signum: int, _frame: object) -> None:
    raise RuntimeError("benchmark interrupted by signal " + str(signum))


def main() -> None:
    """Run an explicitly requested benchmark action."""
    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(os.name == "posix", "POSIX process-group support required")
    if args.action == "verify":
        verify_run(args.output.resolve())
        return
    config = read(args.config.resolve())
    schedule = tasks(config)
    inputs(config)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "workers": len(schedule),
                    "executions": len(schedule) * (config["shots"] + config["warmup_shots"]),
                    "tasks": schedule,
                },
                indent=2,
            ),
        )
        return
    root = args.output.resolve()
    for task in schedule:
        launch_spec(config, task, root / str(task["index"]))
    root.mkdir(parents=True, exist_ok=False)
    write(
        root / "freeze.json",
        {"config": config, "runner_sha256": sha(__file__), "machine": platform.platform(), "python": sys.version},
    )
    try:
        for task in schedule:
            folder = root / str(task["index"])
            folder.mkdir()
            inputs(config)
            argv, env = launch_spec(config, task, folder)
            write(folder / "command.json", {"argv": argv, "env": env})
            write(folder / "load-before.json", list(os.getloadavg()))
            execute(argv, env, folder, config["timeout_seconds"])
            report(config, task, folder)
            write(folder / "load-after.json", list(os.getloadavg()))
            print(task["index"] + 1, "/", len(schedule), task["route"], flush=True)
        inputs(config)
        write(root / "summary.json", summary(config, root))
    except BaseException as exc:
        write(root / "failure.json", {"error": repr(exc)})
        raise
    finally:
        write(root / "manifest.json", inventory(root))
    verify_run(root)


if __name__ == "__main__":
    main()
