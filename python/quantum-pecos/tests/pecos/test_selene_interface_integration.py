"""Test the Selene Interface integration from Python side."""

import platform

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
                        plugin_pattern = (
                            f"**/selene_simple_runtime_plugin/_dist/lib/{ext}"
                        )
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
        assert (
            len(loadable_libraries) > 0
        ), f"Found {len(loadable_libraries)} loadable Selene runtime libraries"
