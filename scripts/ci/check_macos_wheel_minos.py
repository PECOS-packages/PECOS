"""Validate the deployment target embedded in every Mach-O wheel payload."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


class WheelValidationError(ValueError):
    """Raised when a wheel does not satisfy the macOS deployment policy."""


def version_tuple(version: str) -> tuple[int, int, int]:
    """Convert a dotted numeric macOS version into a comparable tuple."""
    try:
        parts = [int(part) for part in version.split(".")]
    except ValueError as error:
        message = f"invalid macOS version {version!r}"
        raise WheelValidationError(message) from error

    if not 1 <= len(parts) <= 3:
        message = f"invalid macOS version {version!r}"
        raise WheelValidationError(message)
    padded = [*parts, 0, 0]
    return padded[0], padded[1], padded[2]


def minimum_versions(binary: Path) -> list[str]:
    """Return all minimum macOS versions advertised by a Mach-O binary."""
    result = subprocess.run(
        ["/usr/bin/otool", "-l", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )

    versions: list[str] = []
    load_command: str | None = None
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "cmd":
            load_command = fields[1]
        elif (load_command == "LC_BUILD_VERSION" and len(fields) >= 2 and fields[0] == "minos") or (
            load_command == "LC_VERSION_MIN_MACOSX" and len(fields) >= 2 and fields[0] == "version"
        ):
            versions.append(fields[1])
            load_command = None

    return versions


def wheel_macho_versions(wheel: Path) -> list[tuple[str, str]]:
    """Inspect every extension and dynamic library inside a macOS wheel."""
    versions: list[tuple[str, str]] = []
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as tmp:
        temporary_binary = Path(tmp) / "payload"
        for member in archive.infolist():
            if not member.filename.endswith((".so", ".dylib")):
                continue

            with archive.open(member) as source, temporary_binary.open("wb") as target:
                shutil.copyfileobj(source, target)

            binary_versions = minimum_versions(temporary_binary)
            if not binary_versions:
                message = f"{wheel.name}:{member.filename}: no macOS minimum version found"
                raise WheelValidationError(message)
            versions.extend((member.filename, version) for version in binary_versions)

    if not versions:
        message = f"{wheel.name}: no Mach-O payloads found"
        raise WheelValidationError(message)
    return versions


def main() -> None:
    """Validate Mach-O minimum versions in every supplied wheel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-version", required=True)
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()

    maximum = version_tuple(args.maximum_version)
    failures: list[str] = []
    inspected = 0

    for wheel_name in args.wheels:
        wheel = Path(wheel_name)
        if not wheel.is_file() or wheel.suffix != ".whl":
            failures.append(f"wheel does not exist: {wheel}")
            continue

        try:
            payload_versions = wheel_macho_versions(wheel)
        except (
            OSError,
            subprocess.CalledProcessError,
            WheelValidationError,
            zipfile.BadZipFile,
        ) as error:
            failures.append(str(error))
            continue

        for payload, version in payload_versions:
            inspected += 1
            if version_tuple(version) > maximum:
                failures.append(
                    f"{wheel.name}:{payload}: minimum macOS {version} exceeds {args.maximum_version}",
                )

    if failures:
        raise SystemExit("Mach-O deployment validation failed:\n- " + "\n- ".join(failures))

    print(
        f"Validated {inspected} Mach-O payload(s) for macOS <= {args.maximum_version}",
    )


if __name__ == "__main__":
    main()
