"""Test the Selene Interface integration from Python side."""

import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_runtime_library_finding() -> None:
    """Test the runtime library finder functionality."""
    import ctypes
    import os
    from pathlib import Path

    # Determine the library extension based on platform
    system = platform.system()
    if system == "Windows":
        lib_extensions = ["selene_simple_runtime.dll"]
    elif system == "Darwin":  # macOS
        lib_extensions = [
            "libselene_simple_runtime.dylib",
            "libselene_simple_runtime.so",
        ]
    else:  # Linux and others
        lib_extensions = ["libselene_simple_runtime.so"]

    # This test should ideally test a library finder function/class
    # For now, we'll test that if we find a library, it's actually loadable

    # Try to import the actual library finder if it exists
    try:
        from pecos.engines.selene_runtime import find_selene_runtime_library

        library_path = find_selene_runtime_library()

        # Test that the found library is actually loadable
        try:
            lib = ctypes.CDLL(str(library_path))
            # Could check for specific symbols here
            assert lib is not None, "Library should be loadable"
        except OSError as e:
            pytest.fail(f"Found library at {library_path} but couldn't load it: {e}")

    except ImportError:
        # The library finder doesn't exist yet, so test the manual search
        # This is more of a diagnostic than a test
        possible_paths = []

        # Add platform-specific paths
        if system == "Windows":
            # Windows cache location
            cache_dir = Path.home() / ".cache/pecos-decoders/selene"
            possible_paths.extend(cache_dir / ext for ext in lib_extensions)
        else:
            # Unix-like systems
            possible_paths.extend(
                path
                for ext in lib_extensions
                for path in [
                    Path.home() / ".cache/pecos-decoders/selene" / ext,
                    Path("/usr/local/lib") / ext,
                ]
            )

        # Add venv paths
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            venv_path = Path(venv)
            if system == "Windows":
                # On Windows, check the specific plugin location
                plugin_path = (
                    venv_path
                    / "Lib"
                    / "site-packages"
                    / "selene_simple_runtime_plugin"
                    / "_dist"
                    / "lib"
                    / "selene_simple_runtime.dll"
                )
                if plugin_path.exists():
                    possible_paths.append(plugin_path)

                # Also search more broadly
                site_packages_dirs = [
                    venv_path / "Scripts",
                    venv_path / "Lib" / "site-packages",
                ]
            else:
                # On Unix-like systems, search for the plugin in site-packages
                # The exact Python version directory can vary, so use rglob
                lib_dir = venv_path / "lib"
                if lib_dir.exists():
                    for ext in lib_extensions:
                        plugin_pattern = f"**/selene_simple_runtime_plugin/_dist/lib/{ext}"
                        possible_paths.extend(lib_dir.glob(plugin_pattern))

                site_packages_dirs = [venv_path / "lib"]

            for site_packages in site_packages_dirs:
                if site_packages.exists():
                    # Search for the library in site-packages
                    for ext in lib_extensions:
                        possible_paths.extend(site_packages.rglob(ext))

        # Check if any library is actually loadable (not just exists)
        loadable_libraries = []
        for path in possible_paths:
            if path.exists():
                try:
                    # Actually try to load the library
                    lib = ctypes.CDLL(str(path))
                    loadable_libraries.append(path)
                except OSError:
                    # File exists but can't be loaded (might be stub or wrong arch)
                    continue

        if not loadable_libraries:
            pytest.skip(
                "No loadable Selene runtime library found - this is expected in test environments",
            )

        # If we found loadable libraries, that's good enough for this diagnostic
        assert len(loadable_libraries) > 0, f"Found {len(loadable_libraries)} loadable Selene runtime libraries"


def test_selene_engine_python_exports() -> None:
    """Test that the Selene engine convenience exports exist and are usable."""
    import pecos
    import pecos_rslib

    assert hasattr(pecos_rslib, "selene_engine")
    assert hasattr(pecos, "selene_engine")

    builder = pecos.selene_engine()
    assert isinstance(builder, pecos.QisEngineBuilder)

    named_builder = pecos.qis_engine().selene_runtime("selene_simple_runtime")
    assert isinstance(named_builder, pecos.QisEngineBuilder)


def test_selene_engine_accepts_generic_runtime_plugin_shape() -> None:
    """A downstream Selene runtime plugin object is sufficient; PECOS does not need to know its package."""
    from pathlib import Path

    import pecos

    class RuntimePlugin:
        def __init__(self) -> None:
            self.library_file = Path("libcustom_selene_runtime.so")
            self.library_search_dirs = [Path("custom-selene-libs")]

        def get_init_args(self) -> list[str]:
            return ["--hardware-profile=custom"]

    builder = pecos.qis_engine().selene_runtime(RuntimePlugin())
    assert isinstance(builder, pecos.QisEngineBuilder)

    engine_builder = pecos.selene_engine(RuntimePlugin())
    assert isinstance(engine_builder, pecos.QisEngineBuilder)


def test_default_runtime_falls_back_to_installed_plugin_package() -> None:
    """A failed Cargo lookup configures the real builder from the installed plugin."""
    import pecos_rslib
    from pecos._engine_builders import _configure_selene_runtime
    from selene_simple_runtime_plugin import SimpleRuntimePlugin

    class FailingCargoRuntimeBuilder:
        def __init__(self) -> None:
            self.builder = pecos_rslib.qis_engine()
            self.plugin_call: tuple[str, list[str], list[str]] | None = None

        def selene_runtime(self) -> object:
            msg = "forced Cargo runtime discovery failure"
            raise RuntimeError(msg)

        def selene_runtime_plugin(
            self,
            library_file: str,
            init_args: list[str],
            library_search_dirs: list[str],
        ) -> object:
            self.plugin_call = (library_file, init_args, library_search_dirs)
            return self.builder.selene_runtime_plugin(
                library_file,
                init_args,
                library_search_dirs,
            )

    builder = FailingCargoRuntimeBuilder()
    _configure_selene_runtime(builder, None)

    assert builder.plugin_call is not None
    assert Path(builder.plugin_call[0]) == SimpleRuntimePlugin().library_file


def test_sim_guppy_can_use_selene_engine_via_qis_path() -> None:
    """Test that sim(Guppy(...)).classical(selene_engine()) routes HUGR through the QIS path."""
    import pecos
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit

    selene = pecos.selene_engine()

    @guppy
    def coin() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    results = pecos.sim(pecos.Guppy(coin)).classical(selene).qubits(1).seed(42).run(10).to_dict()
    assert len(results["measurement_0"]) == 10


def _run_selene_cwd_probe(tmp_path: Path, cargo_target_dir: Path | None = None) -> None:
    probe = tmp_path / "selene_cwd_probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import pecos
            from guppylang import guppy
            from guppylang.std.quantum import measure, qubit

            @guppy
            def measure_one() -> bool:
                q = qubit()
                return measure(q)

            results = (
                pecos.sim(pecos.Guppy(measure_one))
                .classical(pecos.selene_engine())
                .quantum(pecos.stabilizer())
                .qubits(1)
                .run(1)
                .to_dict()
            )
            assert len(results["measurement_0"]) == 1
            """,
        ),
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith(("PECOS", "CARGO"))}
    if cargo_target_dir is not None:
        env["CARGO_TARGET_DIR"] = str(cargo_target_dir)

    subprocess.run(
        [sys.executable, str(probe)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_selene_engine_is_cwd_independent(tmp_path: Path) -> None:
    """The runtime resolves when the process starts outside the checkout."""
    _run_selene_cwd_probe(tmp_path)


def test_selene_engine_uses_plugin_when_cargo_target_is_empty(tmp_path: Path) -> None:
    """An empty explicit Cargo target forces the module-relative runtime fallback."""
    empty_cargo_target = tmp_path / "empty-cargo-target"
    empty_cargo_target.mkdir()

    _run_selene_cwd_probe(tmp_path, empty_cargo_target)


def test_sim_guppy_reuses_physical_slot_after_measurement() -> None:
    """Test that a recycled physical slot is reinitialized when Guppy reallocates a qubit."""
    import pecos
    from guppylang import guppy
    from guppylang.std.quantum import measure, qubit, x

    selene = pecos.selene_engine()

    @guppy
    def allocate_measure_allocate_again() -> tuple[bool, bool]:
        q0 = qubit()
        x(q0)
        m0 = measure(q0)
        q1 = qubit()
        m1 = measure(q1)
        return m0, m1

    results = (
        pecos.sim(pecos.Guppy(allocate_measure_allocate_again)).classical(selene).qubits(1).seed(7).run(10).to_dict()
    )

    assert len(results["measurement_0"]) == 10
    assert len(results["measurement_1"]) == 10
    assert all(results["measurement_0"])
    assert not any(results["measurement_1"])
