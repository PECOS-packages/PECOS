"""The Selene runtime preload must not hide a runtime's unresolved imports."""

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _build_lazy_library(tmp_path: Path, name: str, source: str) -> Path:
    """Compile a shared library that the platform binds lazily unless asked otherwise."""
    if platform.system() == "Windows":
        pytest.skip("The Selene runtime preload is POSIX-only")
    assert shutil.which("cc") is not None, "This test requires a C compiler"
    source_path = tmp_path / f"{name}.c"
    source_path.write_text(source)
    library = tmp_path / (f"{name}.dylib" if platform.system() == "Darwin" else f"{name}.so")
    args = [shutil.which("cc"), "-shared", "-fPIC", str(source_path), "-o", str(library)]
    # Keep the fixture lazily bound so only the loader's flags decide when an import fails.
    args.append("-Wl,-undefined,dynamic_lookup" if platform.system() == "Darwin" else "-Wl,-z,lazy")
    subprocess.run(args, check=True, capture_output=True, text=True)
    return library


def _run_with_preload(tmp_path: Path, library: Path, script: str) -> str:
    env = {
        **os.environ,
        "PECOS_SELENE_PRELOAD": str(library),
        "PECOS_CACHE_DIR": str(tmp_path / "program-cache"),
    }
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(library)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_preload_runs_for_an_explicit_runtime(tmp_path: Path) -> None:
    """Positive control: PECOS_SELENE_PRELOAD is read and its library is loaded."""
    library = _build_lazy_library(tmp_path, "valid_runtime", "void pecos_test_export(void) {}\n")
    script = """
        import ctypes
        import os
        import sys
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            import pecos_rslib
        # RTLD_NOLOAD only returns a handle when the library is already loaded.
        ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        print("preloaded", flush=True)
    """
    assert "preloaded" in _run_with_preload(tmp_path, library, script)


def test_preloaded_runtime_with_missing_import_fails_at_load(tmp_path: Path) -> None:
    """A lazily preloaded runtime would make the later eager plugin load a no-op."""
    library = _build_lazy_library(
        tmp_path,
        "missing_import",
        "extern void pecos_test_missing_import(void);\nvoid pecos_test_export(void) { pecos_test_missing_import(); }\n",
    )
    script = '''
        import sys
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import pecos
            import pecos_rslib as pr
        messages = [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)]
        assert any("pecos_test_missing_import" in message for message in messages), messages

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
    '''
    assert "rejected" in _run_with_preload(tmp_path, library, script)
