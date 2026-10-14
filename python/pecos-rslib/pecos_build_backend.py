"""Maturin backend with the dynamically loaded QIS libraries included in wheels.

These libraries are not link-time dependencies of the Python extension. Cargo's
artifact messages identify the exact files built for the requested target and
profile, including hash-suffixed Selene plugins. Do not glob a shared target
directory: it can contain incompatible artifacts from earlier builds.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

import maturin

build_sdist = maturin.build_sdist
get_requires_for_build_editable = maturin.get_requires_for_build_editable
get_requires_for_build_sdist = maturin.get_requires_for_build_sdist
get_requires_for_build_wheel = maturin.get_requires_for_build_wheel
prepare_metadata_for_build_editable = maturin.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = maturin.prepare_metadata_for_build_wheel

_ROOT = Path(__file__).resolve().parent
_LIBRARIES = frozenset({"pecos_qis_ffi", "selene_simple_runtime", "selene_soft_rz_runtime"})
_PACKAGE_LIB_DIR = "pecos_rslib/_native"


def _cargo_options(config_settings: dict | None, *, editable: bool) -> list[str]:
    """Match the target/profile options used by the delegated Maturin build."""
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text())["tool"]["maturin"]
    settings = config_settings or {}
    build_args = settings.get(
        "maturin.build-args",
        settings.get("build-args", os.environ.get("MATURIN_PEP517_ARGS", "")),
    )
    args = shlex.split(build_args) if isinstance(build_args, str) else list(build_args)
    parser = argparse.ArgumentParser(add_help=False)
    for option in ("target", "target-dir", "profile", "jobs"):
        parser.add_argument(f"--{option}")
    parser.add_argument("-j", dest="jobs")
    parser.add_argument("-r", "--release", action="store_true")
    parser.add_argument("--config", action="append", default=[])
    for option in ("locked", "frozen", "offline"):
        parser.add_argument(f"--{option}", action="store_true")
    parsed, _ = parser.parse_known_args(args)
    profile = (
        config.get("editable-profile", config.get("profile", "release"))
        if editable
        else config.get("profile", "release")
    )
    options = ["--profile", parsed.profile or ("release" if parsed.release else profile)]
    for option in ("target", "target-dir", "jobs"):
        if value := getattr(parsed, option.replace("-", "_")):
            options.extend([f"--{option}", value])
    for value in [*config.get("config", []), *parsed.config]:
        options.extend(["--config", value])
    for option in ("locked", "frozen", "offline"):
        if getattr(parsed, option) or config.get(option, False):
            options.append(f"--{option}")
    return options


def _runtime_artifacts(config_settings: dict | None, *, editable: bool) -> list[Path]:
    """Build the runtime dependencies and select their Cargo-reported artifacts."""
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(_ROOT / "Cargo.toml"),
        "--package",
        "pecos-qis",
        "--no-default-features",
        "--features",
        "selene",
        "--message-format=json-render-diagnostics",
        *_cargo_options(config_settings, editable=editable),
    ]
    result = subprocess.run(  # noqa: S603 - build frontend options
        command, stdout=subprocess.PIPE, text=True, check=True
    )
    artifacts: dict[str, Path] = {}
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") != "compiler-artifact":
            continue
        name = message["target"]["name"]
        if name not in _LIBRARIES:
            continue
        for filename in message["filenames"]:
            path = Path(filename)
            if path.suffix in {".so", ".dylib", ".dll"}:
                if name in artifacts and artifacts[name] != path:
                    msg = f"Ambiguous runtime artifacts for {name}: {artifacts[name]} and {path}"
                    raise RuntimeError(msg)
                artifacts[name] = path
    missing = _LIBRARIES - artifacts.keys()
    if missing:
        msg = f"Cargo did not produce required QIS runtime libraries: {sorted(missing)}"
        raise RuntimeError(msg)
    files = list(artifacts.values())
    ffi = artifacts["pecos_qis_ffi"]
    if ffi.suffix == ".dll":
        import_lib = ffi.with_suffix(".dll.lib")
        if not import_lib.is_file():
            msg = f"Cargo did not produce the QIS runtime import library: {import_lib}"
            raise RuntimeError(msg)
        files.append(import_lib)
    return files


def _bundle_runtime(wheel: Path, artifacts: list[Path]) -> None:
    """Add native artifacts and regenerate the wheel's integrity RECORD."""
    with tempfile.TemporaryDirectory() as directory:
        replacement = Path(directory) / wheel.name
        with (
            zipfile.ZipFile(wheel) as source,
            zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as dest,
        ):
            records = []

            def write(name: str, data: bytes, info: zipfile.ZipInfo | None = None) -> None:
                dest.writestr(info or name, data)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                records.append((name, f"sha256={digest}", len(data)))

            record_name = next(name for name in source.namelist() if name.endswith(".dist-info/RECORD"))
            for info in source.infolist():
                if info.filename == record_name or info.filename.startswith(f"{_PACKAGE_LIB_DIR}/"):
                    continue
                write(info.filename, source.read(info), info)
            for artifact in artifacts:
                write(f"{_PACKAGE_LIB_DIR}/{artifact.name}", artifact.read_bytes())
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows([*records, (record_name, "", "")])
            dest.writestr(record_name, output.getvalue())
        shutil.copyfile(replacement, wheel)


def build_wheel(
    wheel_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None
) -> str:
    """Build an extension wheel with its complete QIS runtime payload."""
    artifacts = _runtime_artifacts(config_settings, editable=False)
    filename = maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
    _bundle_runtime(Path(wheel_directory) / filename, artifacts)
    return filename


def build_editable(
    wheel_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None
) -> str:
    """Make the same runtime payload available to an editable installation."""
    artifacts = _runtime_artifacts(config_settings, editable=True)
    filename = maturin.build_editable(wheel_directory, config_settings, metadata_directory)
    lib_dir = _ROOT / _PACKAGE_LIB_DIR
    lib_dir.mkdir(parents=True, exist_ok=True)
    for old in lib_dir.iterdir():
        if old.is_file():
            old.unlink()
    for artifact in artifacts:
        shutil.copy2(artifact, lib_dir / artifact.name)
    return filename
