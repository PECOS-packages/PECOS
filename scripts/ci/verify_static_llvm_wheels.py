"""Verify that release LLVM wheels are isolated from process-global LLVM state."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

_LLVM_SHARED_NAME = re.compile(r"^libLLVM(?:-[0-9]+)?\.so(?:\.[0-9.]+)?$")
_NEEDED_ENTRY = re.compile(r"\(NEEDED\).*Shared library: \[([^]]+)]")


class WheelVerificationError(RuntimeError):
    """Raised when a wheel violates the LLVM isolation contract."""


def require_tool(name: str) -> str:
    """Return an absolute tool path or fail with a clear message."""
    path = shutil.which(name)
    if path is None:
        message = f"required tool is not available on PATH: {name}"
        raise WheelVerificationError(message)
    return path


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and include captured output in any failure."""
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        message = (
            f"command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        raise WheelVerificationError(message)
    return result


def extract_wheel(wheel: Path, destination: Path) -> list[str]:
    """Extract a wheel while rejecting paths outside the destination."""
    destination_root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                message = f"{wheel}: unsafe wheel member path: {member.filename}"
                raise WheelVerificationError(message)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return [member.filename for member in members]


def find_extension(root: Path, module_name: str) -> Path:
    """Find the single native extension for a Python module in an extracted wheel."""
    candidates = [
        path
        for path in root.rglob("*.so")
        if path.name == f"{module_name}.so" or path.name.startswith(f"{module_name}.")
    ]
    if len(candidates) != 1:
        rendered = ", ".join(str(path.relative_to(root)) for path in candidates) or "none"
        message = f"expected one {module_name} extension, found {len(candidates)}: {rendered}"
        raise WheelVerificationError(message)
    return candidates[0]


def needed_libraries(readelf: str, extension: Path) -> list[str]:
    """Return the ELF dynamic dependencies of an extension."""
    output = run_command([readelf, "-d", str(extension)]).stdout
    return [match.group(1) for line in output.splitlines() if (match := _NEEDED_ENTRY.search(line))]


def verify_no_llvm_needed(readelf: str, wheel: Path, extension: Path) -> None:
    """Assert that an extension has no dynamic dependency on libLLVM."""
    needed = needed_libraries(readelf, extension)
    llvm_needed = [name for name in needed if name.startswith("libLLVM")]
    if llvm_needed:
        message = f"{wheel.name}:{extension.name} has libLLVM NEEDED entries: {llvm_needed}"
        raise WheelVerificationError(message)
    print(f"Checked {wheel.name}:{extension.name}: no libLLVM NEEDED entry")


def verify_no_bundled_llvm(wheel: Path, members: list[str]) -> None:
    """Assert that the wheel does not contain a bundled libLLVM file."""
    bundled = [name for name in members if PurePosixPath(name).name.startswith("libLLVM")]
    if bundled:
        message = f"{wheel.name} contains bundled libLLVM files: {bundled}"
        raise WheelVerificationError(message)
    print(f"Checked {wheel.name}: no bundled libLLVM file")


def verify_exported_symbols(nm: str, wheel: Path, extension: Path) -> None:
    """Assert that the LLVM extension exports only its Python initializer."""
    output = run_command([nm, "-D", "--defined-only", str(extension)]).stdout
    symbols = [line.split()[-1] for line in output.splitlines() if line.split()]
    expected = ["PyInit_pecos_rslib_llvm"]
    if symbols != expected:
        message = f"{wheel.name}:{extension.name} defined dynamic symbols are {symbols}, expected {expected}"
        raise WheelVerificationError(message)
    print(f"Checked {wheel.name}:{extension.name}: sole defined dynamic symbol is {expected[0]}")


def shared_llvm_from_prefix(prefix: Path) -> tuple[Path, Path]:
    """Locate the managed monolithic shared LLVM library in an installed prefix."""
    lib_dir = prefix / "lib"
    candidates = {
        path.resolve() for path in lib_dir.iterdir() if path.is_file() and _LLVM_SHARED_NAME.fullmatch(path.name)
    }
    if not candidates:
        message = f"no managed shared libLLVM found in {lib_dir}"
        raise WheelVerificationError(message)
    return max(candidates, key=lambda path: path.stat().st_size), lib_dir


def shared_llvm_from_archive(archive_path: Path, destination: Path) -> tuple[Path, Path]:
    """Extract the managed monolithic shared LLVM library from a release archive."""
    with tarfile.open(archive_path) as archive:
        candidates = [
            member
            for member in archive.getmembers()
            if member.isfile() and _LLVM_SHARED_NAME.fullmatch(PurePosixPath(member.name).name)
        ]
        if not candidates:
            message = f"no managed shared libLLVM found in {archive_path}"
            raise WheelVerificationError(message)
        member = max(candidates, key=lambda item: item.size)
        source = archive.extractfile(member)
        if source is None:
            message = f"could not read managed shared libLLVM member {member.name} from {archive_path}"
            raise WheelVerificationError(message)
        destination.mkdir(parents=True, exist_ok=True)
        llvm_library = destination / PurePosixPath(member.name).name
        with source, llvm_library.open("wb") as output:
            shutil.copyfileobj(source, output)
    return llvm_library, destination


def verify_clash_falsifier(
    patchelf: str,
    llvm_wheel: Path,
    managed_llvm: Path,
    managed_lib_dir: Path,
    work_dir: Path,
) -> None:
    """Load a distinct-SONAME LLVM globally before importing the static wheel."""
    clash_library = work_dir / "libLLVM-pecos-clash-falsifier.so"
    clash_soname = clash_library.name
    shutil.copy2(managed_llvm, clash_library)
    run_command([patchelf, "--set-soname", clash_soname, str(clash_library)])
    actual_soname = run_command([patchelf, "--print-soname", str(clash_library)]).stdout.strip()
    if actual_soname != clash_soname:
        message = f"failed to assign distinct LLVM SONAME: expected {clash_soname}, found {actual_soname}"
        raise WheelVerificationError(message)

    venv_dir = work_dir / "venv"
    run_command([sys.executable, "-m", "venv", str(venv_dir)])
    venv_python = venv_dir / "bin" / "python"
    run_command(
        [str(venv_python), "-m", "pip", "install", "--no-deps", "--no-index", str(llvm_wheel.resolve())],
    )

    # The reference libLLVM must load on this host before it can serve as the
    # clash neighbor; if it cannot (e.g. libstdc++ too old for a
    # manylinux-built library), that is an environment problem, not a wheel
    # defect, and must be reported as such.
    test_program = (
        "import ctypes, sys\n"
        "try:\n"
        "    ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)\n"
        "except OSError as error:\n"
        "    print(f'could not load the reference LLVM (environment issue, "
        "not a wheel defect): {error}', file=sys.stderr)\n"
        "    raise SystemExit(86)\n"
        "import pecos_rslib_llvm\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    current_library_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        f"{managed_lib_dir}:{current_library_path}" if current_library_path else str(managed_lib_dir)
    )
    run_command([str(venv_python), "-c", test_program, str(clash_library)], env=env)
    print(
        f"Checked {llvm_wheel.name}: import succeeds after RTLD_GLOBAL load of "
        f"{managed_llvm.name} copied with SONAME {clash_soname}",
    )


def verify(args: argparse.Namespace) -> None:
    """Run all Linux wheel isolation assertions."""
    base_wheel = args.base_wheel.resolve()
    llvm_wheel = args.llvm_wheel.resolve()
    for wheel in (base_wheel, llvm_wheel):
        if not wheel.is_file() or wheel.suffix != ".whl":
            message = f"wheel does not exist: {wheel}"
            raise WheelVerificationError(message)

    readelf = require_tool("readelf")
    nm = require_tool("nm")
    patchelf = require_tool("patchelf")

    with tempfile.TemporaryDirectory(prefix="pecos-static-llvm-wheel-") as temporary:
        work_dir = Path(temporary)
        base_root = work_dir / "base-wheel"
        llvm_root = work_dir / "llvm-wheel"
        extract_wheel(base_wheel, base_root)
        llvm_members = extract_wheel(llvm_wheel, llvm_root)
        base_extension = find_extension(base_root, "pecos_rslib")
        llvm_extension = find_extension(llvm_root, "pecos_rslib_llvm")

        verify_no_llvm_needed(readelf, base_wheel, base_extension)
        verify_no_llvm_needed(readelf, llvm_wheel, llvm_extension)
        verify_no_bundled_llvm(llvm_wheel, llvm_members)
        verify_exported_symbols(nm, llvm_wheel, llvm_extension)

        if args.llvm_prefix is not None:
            managed_llvm, managed_lib_dir = shared_llvm_from_prefix(args.llvm_prefix.resolve())
        else:
            managed_llvm, managed_lib_dir = shared_llvm_from_archive(
                args.llvm_archive.resolve(),
                work_dir / "managed-llvm",
            )
        verify_clash_falsifier(
            patchelf,
            llvm_wheel,
            managed_llvm,
            managed_lib_dir,
            work_dir,
        )


def main() -> None:
    """Parse arguments and report a concise CI failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-wheel", required=True, type=Path)
    parser.add_argument("--llvm-wheel", required=True, type=Path)
    llvm_source = parser.add_mutually_exclusive_group(required=True)
    llvm_source.add_argument("--llvm-prefix", type=Path)
    llvm_source.add_argument("--llvm-archive", type=Path)
    args = parser.parse_args()

    try:
        verify(args)
    except (OSError, tarfile.TarError, WheelVerificationError, zipfile.BadZipFile) as error:
        message = f"Static LLVM wheel verification failed: {error}"
        raise SystemExit(message) from error


if __name__ == "__main__":
    main()
