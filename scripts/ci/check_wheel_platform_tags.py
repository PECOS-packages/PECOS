"""Fail CI when a built wheel does not advertise the intended platform."""

from __future__ import annotations

import argparse
from pathlib import Path


def platform_tags(wheel: Path) -> set[str] | None:
    """Return the dot-separated platform tags from a wheel filename."""
    filename_parts = wheel.name.removesuffix(".whl").rsplit("-", maxsplit=3)
    if len(filename_parts) != 4:
        return None

    return set(filename_parts[-1].split("."))


def main() -> None:
    """Validate every supplied wheel against the expected platform tag."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-platform-tag", required=True)
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()

    failures: list[str] = []
    for wheel_name in args.wheels:
        wheel = Path(wheel_name)
        if not wheel.is_file() or wheel.suffix != ".whl":
            failures.append(f"wheel does not exist: {wheel}")
            continue

        actual_tags = platform_tags(wheel)
        if actual_tags is None:
            failures.append(f"invalid wheel filename: {wheel.name}")
            continue

        if args.expected_platform_tag not in actual_tags:
            failures.append(
                f"{wheel.name}: expected {args.expected_platform_tag!r}, found {sorted(actual_tags)!r}",
            )

    if failures:
        raise SystemExit("Wheel platform validation failed:\n- " + "\n- ".join(failures))

    print(
        f"Validated {len(args.wheels)} wheel(s) for {args.expected_platform_tag!r}",
    )


if __name__ == "__main__":
    main()
