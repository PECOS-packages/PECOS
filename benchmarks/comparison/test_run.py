"""Fast synthetic harness tests; no simulator or performance thresholds."""

# ruff: noqa: PT009, PT027, S603 - standard-library tests execute only their synthetic fixture

import importlib.util
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("comparison_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

FAKE = """import json,sys
from pathlib import Path
p=Path(sys.argv[1]);(p/'output.json').write_text('[]')
points=[dict(wall_ns=1000000000*(i+1),self_user=float(i+1),self_system=0.,
             children_user=0.,children_system=0.) for i in range(5)]
timed=[dict(p,wall_ns=p["wall_ns"]+6000000000,self_user=p["self_user"]+6) for p in points]
(p/'report.json').write_text(json.dumps(dict(route=sys.argv[2],shots=int(sys.argv[3]),warmup_shots=1,
    correctness_passed=True,prepared_program_reused=True,evidence=['output.json'],warmup=points,timed=timed)))
"""


class ComparisonTests(unittest.TestCase):
    """Exercise scheduling, evidence and process failure handling."""

    def config(self, root: Path) -> dict:
        """Build an isolated synthetic configuration."""
        worker = root / "worker.py"
        worker.write_text(FAKE)
        return {
            "schema": 1,
            "cycles": 1,
            "shots": 2,
            "warmup_shots": 1,
            "timeout_seconds": 10,
            "inputs": {str(worker): runner.sha(worker)},
            "routes": [
                {"name": n, "env": {}, "argv": [sys.executable, str(worker), "{output}", "{route}", "{shots}"]}
                for n in ("one", "two", "three", "four")
            ],
        }

    def test_balanced_schedule(self) -> None:
        """Balanced schedule."""
        with tempfile.TemporaryDirectory() as d:
            c = self.config(Path(d))
            ts = runner.tasks(c)
            self.assertEqual(len(ts), 16)
            for n in ("one", "two", "three", "four"):
                self.assertEqual(sorted(t["index"] % 4 for t in ts if t["route"] == n), [0, 1, 2, 3])
            c["shots"] = True
            with self.assertRaises(ValueError):
                runner.tasks(c)

    def test_invalid_points_and_child_cpu(self) -> None:
        """Invalid points and child cpu."""
        a = {"wall_ns": 1, "self_user": 0.0, "self_system": 0.0, "children_user": 0.0, "children_system": 0.0}
        b = dict(a, wall_ns=1000000001, children_user=2.0)
        self.assertEqual(runner.interval(a, b), {"wall_seconds": 1.0, "cpu_seconds": 2.0})
        for value in (-1, float("nan")):
            with self.assertRaises(ValueError):
                runner.interval(a, dict(b, self_user=value))

    def test_complete_run_verify_tamper_and_no_overwrite(self) -> None:
        """Complete run verify tamper and no overwrite."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = self.config(root)
            cfg = root / "config.json"
            runner.write(cfg, c)
            out = root / "results"
            command = [sys.executable, str(Path(runner.__file__)), "run", "--config", str(cfg), "--output", str(out)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            runner.verify_run(out)
            self.assertEqual(runner.read(out / "summary.json")["median_block_ratios"]["one/two"]["wall_seconds"], 1.0)
            self.assertNotEqual(
                subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode,
                0,
            )
            (out / "0/output.json").write_text("[1]")
            with self.assertRaises(ValueError):
                runner.verify_run(out)

    def test_worker_failure_stops_schedule(self) -> None:
        """Worker failure stops schedule."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = self.config(root)
            c["routes"][0]["argv"] = [sys.executable, "-c", "raise SystemExit(2)"]
            cfg = root / "config.json"
            runner.write(cfg, c)
            out = root / "results"
            result = subprocess.run(
                [sys.executable, runner.__file__, "run", "--config", str(cfg), "--output", str(out)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((out / "failure.json").exists())
            self.assertFalse((out / "1").exists())
            with self.assertRaises(ValueError):
                runner.verify_run(out)

    def test_correctness_acknowledgement_required(self) -> None:
        """Correctness acknowledgement required."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = self.config(root)
            runner.write(
                root / "report.json",
                {
                    "route": "one",
                    "shots": 2,
                    "warmup_shots": 1,
                    "correctness_passed": False,
                    "prepared_program_reused": True,
                },
            )
            with self.assertRaises(ValueError):
                runner.report(c, {"route": "one"}, root)

    def test_timeout_cleanup(self) -> None:
        """Timeout kills the isolated worker and records cleanup."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.execute([sys.executable, "-c", "import time; time.sleep(10)"], {}, root, 0.05)
            record = runner.read(root / "process.json")
            self.assertTrue(record["timed_out"])
            self.assertTrue(record["cleanup_confirmed"])
            self.assertFalse(runner.alive(record["process_group"]))

    def test_sigterm_cleanup(self) -> None:
        """Terminating the launcher cleans its detached worker."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = self.config(root)
            script = root / "worker.py"
            script.write_text(
                "import sys,time,os; from pathlib import Path; "
                "Path(sys.argv[1], 'pid').write_text(str(os.getpid())); time.sleep(20)",
            )
            config["inputs"][str(script)] = runner.sha(script)
            cfg = root / "config.json"
            runner.write(cfg, config)
            out = root / "results"
            proc = subprocess.Popen(
                [sys.executable, runner.__file__, "run", "--config", str(cfg), "--output", str(out)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 10
                while not (out / "0/pid").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue((out / "0/pid").exists())
                proc.send_signal(signal.SIGTERM)
                self.assertNotEqual(proc.wait(timeout=10), 0)
                record = runner.read(out / "0/process.json")
                self.assertTrue(record["cleanup_confirmed"])
                self.assertFalse(runner.alive(record["process_group"]))
                self.assertTrue((out / "failure.json").exists())
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
