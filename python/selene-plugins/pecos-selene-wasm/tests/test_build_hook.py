"""Exercise source rebuilds and Cargo artifact discovery with real native libraries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location("wasm_hatch_build", Path(__file__).parents[1] / "hatch_build.py")
assert _SPEC is not None
assert _SPEC.loader is not None
hook_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hook_module)


def native_value(library: Path) -> int:
    """Use a fresh process so the OS loader cannot reuse or lock a previous DLL."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes, sys; print(ctypes.CDLL(sys.argv[1]).build_probe())",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout)


@pytest.mark.parametrize("target_location", ["default", "environment", "config"])
def test_source_rebuild_uses_current_cargo_artifact(tmp_path: Path, monkeypatch, target_location: str) -> None:
    """Reinstalling must rebuild changed Rust dependencies, wherever Cargo puts them."""
    workspace = tmp_path / "workspace"
    package = workspace / "python/selene-plugins/pecos-selene-wasm"
    runtime = workspace / "runtime"
    (package / "src").mkdir(parents=True)
    (runtime / "src").mkdir(parents=True)
    (workspace / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["python/selene-plugins/pecos-selene-wasm", "runtime"]\nresolver = "2"\n',
    )
    (package / "Cargo.toml").write_text(
        '[package]\nname = "pecos-selene-wasm"\nversion = "0.1.0"\nedition = "2021"\n'
        '[lib]\ncrate-type = ["cdylib"]\n'
        '[dependencies]\nreview-runtime = { path = "../../../runtime" }\n',
    )
    (package / "src/lib.rs").write_text(
        '#[no_mangle]\npub extern "C" fn build_probe() -> i32 { review_runtime::value() }\n',
    )
    (runtime / "Cargo.toml").write_text('[package]\nname = "review-runtime"\nversion = "0.1.0"\nedition = "2021"\n')
    source = runtime / "src/lib.rs"
    source.write_text("pub fn value() -> i32 { 1 }\n")

    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    if target_location != "default":
        target = tmp_path / "custom-cargo-target"
        if target_location == "environment":
            monkeypatch.setenv("CARGO_TARGET_DIR", str(target))
        else:
            (workspace / ".cargo").mkdir()
            (workspace / ".cargo/config.toml").write_text(f"[build]\ntarget-dir = {json.dumps(str(target))}\n")
    subprocess.run(["cargo", "generate-lockfile", "--offline"], cwd=workspace, check=True, capture_output=True)

    hook = hook_module.PecosSeleneWasmBuildHook(
        str(package),
        {},
        None,
        None,
        str(package / "dist"),
        "wheel",
        app=SimpleNamespace(display_info=lambda _: None, display_error=lambda _: None),
    )

    def build() -> Path:
        build_data = {"artifacts": []}
        hook.initialize("standard", build_data)
        assert build_data["pure_python"] is False
        assert len(build_data["artifacts"]) == 1
        return package / build_data["artifacts"][0]

    assert native_value(build()) == 1
    # A fresh Cargo cache hit must still report the library's actual location.
    assert native_value(build()) == 1
    source.write_text("pub fn value() -> i32 { 2026 }\n")
    assert native_value(build()) == 2026
    if target_location != "default":
        assert not (workspace / "target").exists()

    # A failed rebuild must not silently fall back to the previously staged DLL.
    source.write_text("this is not valid Rust\n")
    with pytest.raises(RuntimeError, match="Cargo build failed"):
        build()
