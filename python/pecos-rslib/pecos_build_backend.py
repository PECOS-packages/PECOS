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
from contextlib import contextmanager
from pathlib import Path

import maturin

build_sdist = maturin.build_sdist
get_requires_for_build_editable = maturin.get_requires_for_build_editable
get_requires_for_build_sdist = maturin.get_requires_for_build_sdist
get_requires_for_build_wheel = maturin.get_requires_for_build_wheel
prepare_metadata_for_build_editable = maturin.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = maturin.prepare_metadata_for_build_wheel

_ROOT = Path(__file__).resolve().parent
_PACKAGE_LIB_DIR = "pecos_rslib/_native"
_STAGING_TEMP_PREFIX = ".pecos-runtime-"


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
    options = [
        "--profile",
        parsed.profile or ("release" if parsed.release else profile),
    ]
    for option in ("target", "target-dir", "jobs"):
        if value := getattr(parsed, option.replace("-", "_")):
            options.extend([f"--{option}", value])
    for value in [*config.get("config", []), *parsed.config]:
        options.extend(["--config", value])
    for option in ("locked", "frozen", "offline"):
        if getattr(parsed, option) or config.get(option, False):
            options.append(f"--{option}")
    return options


def _cargo_output(command: list[str]) -> str:
    """Run a Cargo command assembled from build frontend options."""
    result = subprocess.run(  # noqa: S603 - build frontend options
        command, stdout=subprocess.PIPE, text=True, check=True
    )
    return result.stdout


def _runtime_libraries(options: list[str]) -> frozenset[str]:
    """Read the library contract through Cargo, including relocated workspaces."""
    command = [
        "cargo",
        "metadata",
        "--no-deps",
        "--format-version",
        "1",
        "--manifest-path",
        str(_ROOT / "Cargo.toml"),
        *(option for option in options if option in {"--offline", "--frozen", "--locked"}),
    ]
    for package in json.loads(_cargo_output(command))["packages"]:
        if package["name"] == "pecos-qis":
            libraries = ((package.get("metadata") or {}).get("pecos") or {}).get("runtime-libraries")
            if not isinstance(libraries, list) or not libraries:
                msg = (
                    "crates/pecos-qis/Cargo.toml must declare "
                    "[package.metadata.pecos] runtime-libraries as a non-empty list of library names"
                )
                raise RuntimeError(msg)
            if "pecos_qis_ffi" not in libraries:
                msg = "pecos-qis [package.metadata.pecos] runtime-libraries must include pecos_qis_ffi"
                raise RuntimeError(msg)
            return frozenset(libraries)
    msg = f"{shlex.join(command)} did not report pecos-qis; run from the PECOS workspace"
    raise RuntimeError(msg)


def _runtime_artifacts(config_settings: dict | None, *, editable: bool) -> dict[str, Path]:
    """Build the runtime dependencies and select their Cargo-reported artifacts."""
    options = _cargo_options(config_settings, editable=editable)
    libraries = _runtime_libraries(options)
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
        *options,
    ]
    artifacts: dict[str, Path] = {}
    import_libraries: set[Path] = set()
    for line in _cargo_output(command).splitlines():
        message = json.loads(line)
        if message.get("reason") != "compiler-artifact":
            continue
        name = message["target"]["name"]
        if name not in libraries:
            continue
        for filename in message["filenames"]:
            path = Path(filename)
            if path.suffix in {".so", ".dylib", ".dll"}:
                if name == "pecos_qis_ffi" and path.suffix == ".dll":
                    imports = [Path(file) for file in message["filenames"] if file.endswith(".dll.lib")]
                    if path.name != "pecos_qis_ffi.dll" or any(
                        item.name != "pecos_qis_ffi.dll.lib" for item in imports
                    ):
                        msg = (
                            f"Cannot stage pecos_qis_ffi DLL {path} with import libraries {imports} "
                            f"from {shlex.join(command)}: the import library names the DLL it was linked with, "
                            "so the pair cannot be renamed; rebuild with canonical pecos_qis_ffi.dll "
                            "and pecos_qis_ffi.dll.lib names"
                        )
                        raise RuntimeError(msg)
                    if len(imports) != 1:
                        msg = (
                            f"Expected one pecos_qis_ffi import library in Cargo filenames, "
                            f"found {len(imports)} ({imports}) from {shlex.join(command)}; "
                            "PECOS stages the MSVC .dll.lib import library; "
                            f"Cargo reported {message['filenames']}; rebuild for an MSVC Windows target"
                        )
                        raise RuntimeError(msg)
                    import_libraries.update(imports)
                if name in artifacts and artifacts[name] != path:
                    msg = f"Ambiguous runtime artifacts for {name}: {artifacts[name]} and {path} from {shlex.join(command)}"
                    raise RuntimeError(msg)
                artifacts[name] = path
    missing = libraries - artifacts.keys()
    if missing:
        msg = f"Cargo did not produce required QIS runtime libraries: {sorted(missing)} from {shlex.join(command)}"
        raise RuntimeError(msg)
    files = {f"{'' if path.suffix == '.dll' else 'lib'}{name}{path.suffix}": path for name, path in artifacts.items()}
    ffi = artifacts["pecos_qis_ffi"]
    if ffi.suffix == ".dll":
        if len(import_libraries) != 1:
            msg = (
                f"Expected one pecos_qis_ffi import library, found {len(import_libraries)} "
                f"({import_libraries}) from {shlex.join(command)}; rebuild the Windows QIS FFI library"
            )
            raise RuntimeError(msg)
        files["pecos_qis_ffi.dll.lib"] = next(iter(import_libraries))
    return files


def _bundle_runtime(wheel: Path, artifacts: dict[str, Path]) -> None:
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
            for name, artifact in artifacts.items():
                write(f"{_PACKAGE_LIB_DIR}/{name}", artifact.read_bytes())
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows([*records, (record_name, "", "")])
            dest.writestr(record_name, output.getvalue())
        shutil.copyfile(replacement, wheel)


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build an extension wheel with its complete QIS runtime payload."""
    artifacts = _runtime_artifacts(config_settings, editable=False)
    filename = maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
    _bundle_runtime(Path(wheel_directory) / filename, artifacts)
    return filename


def build_editable(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Make the same runtime payload available to an editable installation."""
    artifacts = _runtime_artifacts(config_settings, editable=True)
    filename = maturin.build_editable(wheel_directory, config_settings, metadata_directory)
    _stage_runtime(artifacts, _ROOT / _PACKAGE_LIB_DIR)
    return filename


@contextmanager
def _staging_io(
    operation: str,
    *,
    remedy: str = "check the path and permissions, then retry runtime staging",
):
    """Give one filesystem step its operation, paths and retry guidance."""
    try:
        yield
    except OSError as error:
        msg = f"Cannot {operation}: {error}; {remedy}"
        raise RuntimeError(msg) from error


def _stage_runtime(artifacts: dict[str, Path], lib_dir: Path) -> None:
    """Replace complete files before removing obsolete editable runtime libraries."""
    with _staging_io(f"create directory {lib_dir}"):
        lib_dir.mkdir(parents=True, exist_ok=True)
    _replace_runtime(artifacts, lib_dir)


def _replace_runtime(artifacts: dict[str, Path], lib_dir: Path) -> None:
    for name, artifact in artifacts.items():
        with (
            _staging_io(f"create temporary file in {lib_dir}"),
            tempfile.NamedTemporaryFile(dir=lib_dir, prefix=_STAGING_TEMP_PREFIX, delete=False) as temporary,
        ):
            temporary_path = Path(temporary.name)
        try:
            with _staging_io(f"copy {artifact} to {temporary_path}"):
                shutil.copy2(artifact, temporary_path)
            with _staging_io(
                f"rename {temporary_path} to {lib_dir / name}",
                remedy=(
                    "close processes that have the library loaded "
                    "(for example a Python session using pecos_rslib), then retry runtime staging"
                    if os.name == "nt"
                    else "check the path and permissions, then retry runtime staging"
                ),
            ):
                os.replace(temporary_path, lib_dir / name)
        finally:
            with _staging_io(f"remove {temporary_path}"):
                if temporary_path.exists():
                    temporary_path.unlink()
    names = set(artifacts)
    with _staging_io(f"list {lib_dir}"):
        entries = list(lib_dir.iterdir())
    for old in entries:
        if old.name in names or old.name.startswith(_STAGING_TEMP_PREFIX):
            continue
        with _staging_io(f"inspect {old}"):
            removable = old.is_file() or old.is_symlink()
        if removable:
            with _staging_io(f"remove {old}"):
                old.unlink()
