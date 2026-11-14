"""Wheel packaging must use Cargo's selected artifacts and valid integrity records."""

from __future__ import annotations

import base64
import csv
import importlib.util
import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pecos_build_backend", Path(__file__).parents[1] / "pecos_build_backend.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
backend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backend)


_LIBRARIES = frozenset({"pecos_qis_ffi", "selene_simple_runtime", "selene_soft_rz_runtime"})


def _cargo_output(command: list[str], messages: list[dict]) -> SimpleNamespace:
    if command[1] == "metadata":
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "packages": [
                        {
                            "name": "pecos-qis",
                            "metadata": {"pecos": {"runtime-libraries": sorted(_LIBRARIES)}},
                        }
                    ]
                }
            )
        )
    return SimpleNamespace(stdout="\n".join(map(json.dumps, messages)))


def test_bundle_replaces_stale_libraries_and_updates_record(tmp_path: Path) -> None:
    """Only the current Cargo payload survives, and every wheel hash verifies."""
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pecos_rslib/__init__.py", "# package\n")
        archive.writestr("pecos_rslib/_native/stale.so", b"old")
        archive.writestr("test.dist-info/RECORD", "obsolete")
    library = tmp_path / "runtime.so"
    library.write_bytes(b"current runtime")

    backend._bundle_runtime(wheel, {library.name: library})

    with zipfile.ZipFile(wheel) as archive:
        assert "pecos_rslib/_native/stale.so" not in archive.namelist()
        assert archive.read("pecos_rslib/_native/runtime.so") == b"current runtime"
        rows = list(csv.reader(io.StringIO(archive.read("test.dist-info/RECORD").decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name.endswith("/RECORD"):
                assert (digest, size) == ("", "")
                continue
            data = archive.read(name)
            assert digest == "sha256=" + base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()
            assert int(size) == len(data)


def test_runtime_artifacts_use_cargo_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hash-suffixed artifacts are selected without guessing from directory contents."""
    messages = []
    expected = {}
    for name in sorted(_LIBRARIES):
        path = tmp_path / f"lib{name}-current.so"
        expected[f"lib{name}.so"] = path
        (tmp_path / f"lib{name}-stale.so").write_bytes(b"stale")
        messages.append(
            {
                "reason": "compiler-artifact",
                "target": {"name": name},
                "filenames": [str(path)],
            }
        )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda command, **kwargs: _cargo_output(command, messages),
    )
    assert backend._runtime_artifacts({"build-args": "--profile dev"}, editable=False) == expected


def test_runtime_artifacts_reject_incomplete_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful Cargo command without runtime outputs cannot produce a wheel."""
    monkeypatch.setattr(backend.subprocess, "run", lambda command, **kwargs: _cargo_output(command, []))
    with pytest.raises(RuntimeError, match="Cargo did not produce required QIS runtime libraries"):
        backend._runtime_artifacts(None, editable=False)


def test_windows_bundle_requires_matching_import_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """QIS compilation on Windows needs the import library beside the runtime DLL."""
    messages = [
        {
            "reason": "compiler-artifact",
            "target": {"name": name},
            "filenames": [str(tmp_path / f"{name}.dll")],
        }
        for name in sorted(_LIBRARIES)
    ]
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda command, **kwargs: _cargo_output(command, messages),
    )
    with pytest.raises(RuntimeError, match="pecos_qis_ffi import library"):
        backend._runtime_artifacts(None, editable=False)
    import_lib = tmp_path / "pecos_qis_ffi.dll.lib"
    import_lib.write_bytes(b"import library")
    with pytest.raises(RuntimeError, match="pecos_qis_ffi import library"):
        backend._runtime_artifacts(None, editable=False)
    message = next(message for message in messages if message["target"]["name"] == "pecos_qis_ffi")
    message["filenames"].append(str(import_lib))
    assert import_lib in backend._runtime_artifacts(None, editable=False).values()
    message["filenames"].append(str(tmp_path / "other" / "pecos_qis_ffi.dll.lib"))
    with pytest.raises(RuntimeError, match="pecos_qis_ffi import library"):
        backend._runtime_artifacts(None, editable=False)


@pytest.mark.parametrize("key", ["build-args", "maturin.build-args"])
def test_build_options_preserve_target_and_profile(key: str) -> None:
    """Native artifacts follow the frontend's cross-target and custom profile."""
    options = backend._cargo_options(
        {key: "--target aarch64-apple-darwin --target-dir custom-target --profile native --locked"},
        editable=False,
    )
    assert options == [
        "--profile",
        "native",
        "--target",
        "aarch64-apple-darwin",
        "--target-dir",
        "custom-target",
        "--locked",
    ]


@pytest.mark.parametrize("flags", ["--locked", "--offline --frozen"])
def test_metadata_uses_manifest_and_cargo_policy(flags: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata is queried through Cargo with the frontend's resolution policy."""
    commands = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return _cargo_output(command, [])

    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend._runtime_libraries(flags.split()) == _LIBRARIES
    assert commands == [
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(backend._ROOT / "Cargo.toml"),
            *flags.split(),
        ]
    ]


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"pecos": {}},
        {"pecos": {"runtime-libraries": []}},
        {"pecos": {"runtime-libraries": ["selene_simple_runtime"]}},
    ],
)
def test_metadata_without_runtime_libraries_names_the_manifest_key(
    metadata: dict | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or empty library declaration is a named error, not a KeyError."""
    output = json.dumps({"packages": [{"name": "pecos-qis", "metadata": metadata}]})
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(stdout=output),
    )
    with pytest.raises(RuntimeError, match=r"\[package\.metadata\.pecos\] runtime-libraries"):
        backend._runtime_libraries([])


def test_editable_staging_preserves_loaded_files_and_delays_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed copy leaves old files intact; renaming publishes only complete files."""
    source = tmp_path / "runtime.dll"
    source.write_bytes(b"new")
    directory = tmp_path / "_native"
    directory.mkdir()
    staged = directory / source.name
    staged.write_bytes(b"old")
    stale = directory / "stale.dll.lib"
    stale.write_bytes(b"stale")
    replace = backend.os.replace
    replacements = []

    def checked_replace(temporary: Path, destination: Path) -> None:
        assert temporary.parent == directory
        assert temporary.name.startswith(".pecos-runtime-")
        assert temporary.read_bytes() == b"new"
        assert destination.read_bytes() == b"old"
        replacements.append(destination)
        replace(temporary, destination)

    monkeypatch.setattr(backend.os, "replace", checked_replace)
    backend._stage_runtime({source.name: source}, directory)
    assert replacements == [staged]
    assert staged.read_bytes() == b"new"
    assert not stale.exists()
    stale.write_bytes(b"stale")
    source.unlink()
    with pytest.raises(RuntimeError, match="Cannot copy"):
        backend._stage_runtime({source.name: source}, directory)
    assert staged.read_bytes() == b"new"
    assert stale.read_bytes() == b"stale"


def test_staging_rename_failure_explains_loaded_library_remedy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blocked replacement preserves the old payload and explains how to retry."""
    source = tmp_path / "runtime.dll"
    source.write_bytes(b"new")
    directory = tmp_path / "_native"
    directory.mkdir()
    target = directory / source.name
    target.write_bytes(b"old")
    stale = directory / "stale.dll"
    stale.write_bytes(b"stale")

    def blocked_replace(source: Path, destination: Path) -> None:
        raise PermissionError(destination)

    monkeypatch.setattr(backend.os, "replace", blocked_replace)
    with pytest.raises(RuntimeError, match="Cannot rename") as error:
        backend._stage_runtime({source.name: source}, directory)
    remedy = "close processes" if backend.os.name == "nt" else "check the path and permissions"
    for text in [str(target), remedy, "retry"]:
        assert text in str(error.value)
    assert target.read_bytes() == b"old"
    assert stale.read_bytes() == b"stale"
    assert set(directory.iterdir()) == {target, stale}


@pytest.mark.parametrize("extension", ["so", "dylib", "dll"])
def test_canonical_runtime_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extension: str) -> None:
    """Both build entry points publish exact names and remove old hashed payloads."""
    deps = tmp_path / "deps"
    deps.mkdir()
    messages = []
    expected = {}
    hashed_names = []
    for name in sorted(_LIBRARIES):
        prefix = "" if extension == "dll" else "lib"
        canonical = f"{prefix}{name}.{extension}"
        source = deps / (
            canonical
            if name == "pecos_qis_ffi" and extension == "dll"
            else f"{prefix}{name}-580b7c95dd39b09a.{extension}"
        )
        source.write_bytes(name.encode())
        filenames = [str(source)]
        expected[canonical] = name.encode()
        hashed_names.append(source.name)
        if name == "pecos_qis_ffi" and extension == "dll":
            import_lib = source.with_suffix(".dll.lib")
            import_lib.write_bytes(b"import")
            filenames.append(str(import_lib))
            expected[f"{name}.dll.lib"] = b"import"
            hashed_names.append(import_lib.name)
        messages.append({"reason": "compiler-artifact", "target": {"name": name}, "filenames": filenames})
    monkeypatch.setattr(backend.subprocess, "run", lambda command, **kwargs: _cargo_output(command, messages))
    monkeypatch.setattr(backend, "_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool.maturin]\n")
    directory = tmp_path / "pecos_rslib" / "_native"
    directory.mkdir(parents=True)
    for name in hashed_names:
        (directory / name).write_bytes(b"obsolete")
    monkeypatch.setattr(backend.maturin, "build_editable", lambda *args: "editable.whl")
    assert backend.build_editable(str(tmp_path)) == "editable.whl"
    assert {path.name for path in directory.iterdir()} == set(expected)
    for name, data in expected.items():
        assert (directory / name).is_file()
        assert (directory / name).read_bytes() == data
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("test.dist-info/RECORD", "obsolete")
        for name in hashed_names:
            archive.writestr(f"pecos_rslib/_native/{name}", b"obsolete")
    monkeypatch.setattr(backend.maturin, "build_wheel", lambda *args: wheel.name)
    assert backend.build_wheel(str(tmp_path)) == wheel.name
    with zipfile.ZipFile(wheel) as archive:
        rows = list(csv.reader(io.StringIO(archive.read("test.dist-info/RECORD").decode())))
        assert set(archive.namelist()) == {f"pecos_rslib/_native/{name}" for name in expected} | {
            "test.dist-info/RECORD"
        }
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name.endswith("/RECORD"):
                assert (digest, size) == ("", "")
                continue
            data = archive.read(name)
            assert data == expected[Path(name).name]
            assert digest == "sha256=" + base64.urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode()
            assert int(size) == len(data)


@pytest.mark.parametrize(
    ("dll", "import_name"),
    [
        ("pecos_qis_ffi-hash.dll", "pecos_qis_ffi.dll.lib"),
        ("pecos_qis_ffi.dll", "pecos_qis_ffi-hash.dll.lib"),
        ("pecos_qis_ffi-hash.dll", "pecos_qis_ffi-hash.dll.lib"),
    ],
)
def test_windows_ffi_pair_cannot_be_renamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dll: str, import_name: str
) -> None:
    """Import libraries embed their linked DLL name, so FFI pairs must retain it."""
    filenames = [str(tmp_path / dll), str(tmp_path / import_name)]
    messages = [{"reason": "compiler-artifact", "target": {"name": "pecos_qis_ffi"}, "filenames": filenames}]
    messages.extend(
        {"reason": "compiler-artifact", "target": {"name": name}, "filenames": [str(tmp_path / f"{name}.dll")]}
        for name in sorted(_LIBRARIES - {"pecos_qis_ffi"})
    )
    monkeypatch.setattr(backend.subprocess, "run", lambda command, **kwargs: _cargo_output(command, messages))
    with pytest.raises(RuntimeError) as error:
        backend._runtime_artifacts(None, editable=False)
    for text in [
        *filenames,
        "import library names the DLL it was linked with",
        "pair cannot be renamed",
        "cargo build",
    ]:
        assert text in str(error.value)


def test_windows_gnu_import_error_lists_reported_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported GNU import libraries produce a diagnostic naming the MSVC contract."""
    filenames = [str(tmp_path / "pecos_qis_ffi.dll"), str(tmp_path / "libpecos_qis_ffi.dll.a")]
    messages = [{"reason": "compiler-artifact", "target": {"name": "pecos_qis_ffi"}, "filenames": filenames}]
    monkeypatch.setattr(backend.subprocess, "run", lambda command, **kwargs: _cargo_output(command, messages))
    with pytest.raises(RuntimeError) as error:
        backend._runtime_artifacts(None, editable=False)
    for text in [*filenames, "MSVC .dll.lib", "cargo build"]:
        assert text in str(error.value)


def test_staging_preserves_other_stagers_temporary_files(tmp_path: Path) -> None:
    """Cleanup leaves another stager's unpublished temporary file intact."""
    source = tmp_path / "runtime.dll"
    source.write_bytes(b"runtime")
    destination = tmp_path / "_native"
    destination.mkdir()
    in_flight = destination / ".pecos-runtime-other-stager"
    in_flight.write_bytes(b"in flight")
    backend._stage_runtime({source.name: source}, destination)
    assert in_flight.read_bytes() == b"in flight"
    assert (destination / source.name).read_bytes() == b"runtime"
