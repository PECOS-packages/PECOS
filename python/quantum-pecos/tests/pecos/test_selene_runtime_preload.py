"""The Selene runtime preload must not hide a runtime's unresolved imports."""

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_preloaded_runtime_with_missing_import_fails_at_load(tmp_path: Path) -> None:
    """A lazily preloaded runtime would make the later eager plugin load a no-op."""
    if platform.system() == "Windows":
        pytest.skip("The Selene runtime preload is POSIX-only")
    assert shutil.which("cc") is not None, "This test requires a C compiler"
    source = tmp_path / "missing_import.c"
    source.write_text(
        "extern void pecos_test_missing_import(void);\nvoid pecos_test_export(void) { pecos_test_missing_import(); }\n",
    )
    library = tmp_path / ("missing_import.dylib" if platform.system() == "Darwin" else "missing_import.so")
    args = [shutil.which("cc"), "-shared", "-fPIC", str(source), "-o", str(library)]
    # Keep the fixture lazily bound so only the loader's flags decide when the import fails.
    args.append("-Wl,-undefined,dynamic_lookup" if platform.system() == "Darwin" else "-Wl,-z,lazy")
    subprocess.run(args, check=True, capture_output=True, text=True)

    script = textwrap.dedent(
        '''
        import sys

        import pecos
        import pecos_rslib as pr

        library = sys.argv[1]
        program = pecos.Qis("""
            define i64 @qmain(i64 %0) #0 {
                %qubit = call i64 @__quantum__rt__qubit_allocate()
                call void @__quantum__qis__h__body(i64 %qubit)
                ret i64 0
            }
            declare i64 @__quantum__rt__qubit_allocate()
            declare void @__quantum__qis__h__body(i64)
            attributes #0 = { "EntryPoint" }
        """)
        classical = pr.qis_engine().selene_runtime_plugin(library, []).interface(pr.qis_helios_interface())
        try:
            pecos.sim(program).classical(classical).qubits(1).seed(1).run(1)
        except RuntimeError as error:
            assert "pecos_test_missing_import" in str(error), str(error)
            print("rejected", flush=True)
        else:
            raise AssertionError("a runtime with an unresolved import must fail to load")
        ''',
    )
    env = {**os.environ, "PECOS_SELENE_PRELOAD": str(library), "PECOS_CACHE_DIR": str(tmp_path / "program-cache")}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(library)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "rejected" in completed.stdout
