"""Compare the installed Base QIS plugin's real output route with PECOS."""

import json
import subprocess
import sys
from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit


@guppy
def numeric_outputs() -> None:
    result("m", measure(qubit()).read())
    result("i", 7)
    result("u", nat(11))
    result("f", 2.5)


def run_probe() -> dict:
    """Use each runtime's supported setup, with its own owned execution context."""
    from pecos import Guppy, selene_engine, sim
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from selene_base_qis_plugin import BaseQISInterface
    from selene_helios_qis_plugin import HeliosInterface
    from selene_sim import build
    from selene_sim.backends import IdealErrorModel, Quest, SimpleRuntime

    # The installed launcher links through its shim to the actual Base QIS
    # plugin. Do not dlopen it and assume RTLD_GLOBAL overrides its imports:
    # Mach-O binds these to libselene, and Windows uses explicit DLL imports.
    interface = HeliosInterface()
    base = interface.dependencies[0].dependencies[0]
    assert isinstance(base, BaseQISInterface)
    assert base.library_file.is_file()
    instance = build(compile_guppy_to_hugr(numeric_outputs), interface=interface)
    try:
        reference = [
            dict(shot)
            for shot in instance.run_shots(
                simulator=Quest(random_seed=42),
                n_qubits=1,
                runtime=SimpleRuntime(random_seed=42),
                error_model=IdealErrorModel(),
                n_shots=2,
                random_seed=42,
            )
        ]
    finally:
        instance.delete_files()
    actual = sim(Guppy(numeric_outputs)).classical(selene_engine()).qubits(1).seed(42).run(2).to_dict()
    return {"reference": reference, "pecos": actual}


def test_installed_plugin_outputs_match_pecos_in_subprocess() -> None:
    """A native crash must fail this test without terminating the pytest process."""
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert process.returncode == 0, (
        f"installed-plugin comparison exited with {process.returncode}\n"
        f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
    )
    data = json.loads(process.stdout)
    assert data["reference"] == [{"m": False, "i": 7, "u": 11, "f": 2.5}] * 2
    assert data["pecos"] == {"m": [0, 0], "i": [7, 7], "u": [11, 11], "f": [2.5, 2.5]}


if __name__ == "__main__":
    print(json.dumps(run_probe()))
