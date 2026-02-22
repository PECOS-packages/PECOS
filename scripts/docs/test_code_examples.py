#!/usr/bin/env python3

# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Test script for validating code examples in PECOS documentation.

This script extracts code blocks from Markdown files and tests them
to ensure they run correctly. It supports both Python and Rust code examples.
"""

from __future__ import annotations

import argparse
import functools
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Directory containing the Markdown files to test
DOCS_DIR = Path("docs")


def find_markdown_files() -> list[Path]:
    """Find all Markdown files in the documentation directory."""
    return list(DOCS_DIR.rglob("*.md"))


@functools.lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """Check if CUDA is available for running GPU examples.

    Uses the same pattern as the Justfile: `pecos cuda check -q` for toolkit,
    plus cupy availability check for Python CUDA packages.
    Result is cached after first call.
    """
    # Check for CUDA toolkit using pecos CLI (same as Justfile pattern)
    cargo_path = shutil.which("cargo")
    if cargo_path is None:
        return False

    try:
        result = subprocess.run(
            [
                cargo_path,
                "run",
                "-p",
                "pecos",
                "--features",
                "cli",
                "--",
                "cuda",
                "check",
                "-q",
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    # Check for cupy Python package (needed for Python CUDA examples)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import cupy; print(cupy.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or "True" not in result.stdout:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False

    return True


def _dedent_code(code: str) -> str:
    """Remove common leading indentation from code."""
    lines = code.strip("\n").split("\n")
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent != float("inf"):
        dedented_lines = []
        for line in lines:
            if line.strip():
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append(line)
        return "\n".join(dedented_lines)
    return code.strip()


def extract_code_blocks(
    file_path: str | Path,
    language: str = "python",
) -> list[tuple[str, bool, str | None]]:
    """Extract code blocks of a specific language from a Markdown file.

    Returns:
        List of tuples (code_block, should_skip, expected_error) where:
        - should_skip is True if the block has a skip marker
        - expected_error is a regex pattern if the block should fail with that error

    Skip markers supported:
        - <!--skip--> or <!--skip: reason--> before the code fence
        - <!--pytest.mark.skip--> before the code fence
        - <!--skip-if-no-cuda--> skips only if CUDA is not available
        - ```python,skip or ```rust,ignore suffix on the fence

    Expected error markers:
        - <!--expect-error: pattern--> before the code fence
        - The pattern is matched against stderr output

    Hidden blocks (```hidden-python```) are accumulated as preamble and
    prepended to subsequent visible blocks for testing. Use <!--preamble-reset-->
    to clear accumulated preamble.
    """
    with Path(file_path).open(encoding="utf-8") as f:
        content = f.read()

    # Pattern to find code blocks with optional skip/reset/expect-error marker before them
    # Captures: (marker_comment, block_type, lang_suffix, code)
    marker_pattern = r"(<!--\s*(?:skip[^>]*|expect-error[^>]*|pytest\.mark\.skip(?:if)?[^>]*|preamble-reset)\s*-->\s*)?"
    fence_pattern = rf"```(hidden-)?{language}(,(?:skip|ignore|no_run))?(.*?)```"
    full_pattern = marker_pattern + fence_pattern

    matches = re.findall(full_pattern, content, re.DOTALL)

    blocks = []
    preamble = []

    for marker_comment, is_hidden, lang_suffix, code in matches:
        # Check for preamble reset
        if marker_comment and "preamble-reset" in marker_comment:
            preamble = []
            continue

        # Handle conditional CUDA skip
        skip_if_no_cuda = marker_comment and "skip-if-no-cuda" in marker_comment.lower()

        should_skip = lang_suffix in (
            ",skip",
            ",ignore",
            ",no_run",
        )

        if skip_if_no_cuda:
            # Only skip if CUDA is not available
            should_skip = not is_cuda_available()
        elif marker_comment and "skip" in marker_comment.lower():
            # Regular skip marker (but not skip-if-no-cuda which contains "skip")
            should_skip = True

        # Check for expected error pattern
        expected_error = None
        if marker_comment and "expect-error" in marker_comment.lower():
            # Extract pattern from <!--expect-error: pattern-->
            match = re.search(r"expect-error:\s*(.+?)\s*-->", marker_comment)
            if match:
                expected_error = match.group(1).strip()
            should_skip = False  # Don't skip, we want to run and check error

        cleaned_code = _dedent_code(code)

        if is_hidden:
            # Hidden blocks accumulate as preamble (not tested individually)
            preamble.append(cleaned_code)
        else:
            # Visible blocks get preamble prepended
            full_code = "\n\n".join(preamble) + "\n\n" + cleaned_code if preamble else cleaned_code
            blocks.append((full_code, should_skip, expected_error))

    return blocks


def _uses_guppy_decorator(code_block: str) -> bool:
    """Check if the code uses @guppy decorator (requires file-based execution)."""
    return "@guppy" in code_block


def run_python_block(
    code_block: str,
    block_number: int,
    file_path: str | Path,
    expected_error: str | None = None,
) -> bool | None:
    """Test a Python code block by executing it and checking for errors.

    If expected_error is provided, the test passes if the code fails with
    an error message matching the pattern (regex).
    """
    if expected_error:
        print(
            f"Testing Python block #{block_number} from {file_path} (expecting error)...",
        )
    else:
        print(f"Testing Python block #{block_number} from {file_path}...")

    # Get the Python executable path
    python_executable = sys.executable
    if not Path(python_executable).exists():
        print(f"FAIL: Python executable not found at {python_executable}")
        return False

    try:
        # Guppy code needs to be in a file for inspect.getsourcelines() to work
        if _uses_guppy_decorator(code_block):
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code_block)
                temp_path = f.name
            try:
                result = subprocess.run(
                    [python_executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=60,  # Guppy compilation can be slow
                    check=False,
                    shell=False,
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            # Execute the code block directly
            result = subprocess.run(
                [python_executable, "-c", code_block],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                shell=False,
            )

        # Handle expected error case
        if expected_error:
            if result.returncode == 0:
                print(
                    f"FAIL: Python block #{block_number} from {file_path} was expected to fail but succeeded",
                )
                return False
            # Check if error matches expected pattern
            if re.search(expected_error, result.stderr):
                print(
                    f"PASS: Python block #{block_number} from {file_path} (failed as expected with matching error)",
                )
                return True
            print(
                f"FAIL: Python block #{block_number} from {file_path} "
                f"failed but error didn't match pattern '{expected_error}':",
            )
            print(result.stderr[:500])  # Truncate long errors
            return False

        # Normal case: expect success
        if result.returncode != 0:
            print(f"FAIL: Error in Python block #{block_number} from {file_path}:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        if expected_error and re.search(expected_error, "TimeoutExpired"):
            print(
                f"PASS: Python block #{block_number} from {file_path} (timed out as expected)",
            )
            return True
        print(f"FAIL: Timeout in Python block #{block_number} from {file_path}")
        return False
    except OSError as e:
        print(
            f"FAIL: OS error testing Python block #{block_number} from {file_path}: {e}",
        )
        return False
    except subprocess.SubprocessError as e:
        print(
            f"FAIL: Subprocess error testing Python block #{block_number} from {file_path}: {e}",
        )
        return False
    else:
        print(f"PASS: Python block #{block_number} from {file_path}")
        return True


def run_rust_block(
    code_block: str,
    block_number: int,
    file_path: str | Path,
) -> bool | None:
    """Test a Rust code block by compiling and running it."""
    print(f"Testing Rust block #{block_number} from {file_path}...")

    # Create a temporary directory for the Rust project
    with tempfile.TemporaryDirectory() as tmpdir:
        # If the code doesn't contain a main function, add one
        if "fn main" not in code_block:
            code_block = f"fn main() {{\n{code_block}\n}}"

        # Write the code to a temporary file
        temp_file = Path(tmpdir) / "main.rs"
        with temp_file.open("w", encoding="utf-8") as f:
            f.write(code_block)

        result = False
        error_msg = None

        try:
            # Find rustc executable
            rustc_path = shutil.which("rustc")
            if not rustc_path:
                error_msg = f"FAIL: rustc not found in PATH for Rust block #{block_number} from {file_path}"
            else:
                # Compile and run the Rust code
                compile_result = subprocess.run(
                    [rustc_path, str(temp_file), "-o", str(Path(tmpdir) / "rust_test")],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    shell=False,
                )

                if compile_result.returncode != 0:
                    error_msg = (
                        f"FAIL: Compilation error in Rust block #{block_number} "
                        f"from {file_path}:\n{compile_result.stderr}"
                    )
                else:
                    # Run the compiled program
                    run_result = subprocess.run(
                        [str(Path(tmpdir) / "rust_test")],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                        shell=False,
                    )

                    if run_result.returncode != 0:
                        error_msg = (
                            f"FAIL: Runtime error in Rust block #{block_number} from {file_path}:\n"
                            f"{run_result.stderr}"
                        )
                    else:
                        print(f"PASS: Rust block #{block_number} from {file_path}")
                        result = True
        except subprocess.TimeoutExpired:
            error_msg = f"FAIL: Timeout in Rust block #{block_number} from {file_path}"
        except OSError as e:
            error_msg = f"FAIL: OS error testing Rust block #{block_number} from {file_path}: {e}"
        except subprocess.SubprocessError as e:
            error_msg = f"FAIL: Subprocess error testing Rust block #{block_number} from {file_path}: {e}"

        if error_msg:
            print(error_msg)

        return result


def main() -> None:
    """Main function to test all code examples in documentation."""
    parser = argparse.ArgumentParser(description="Test code examples in documentation")
    parser.add_argument(
        "--skip-rust",
        action="store_true",
        default=True,  # Skip Rust by default - needs crate context for compilation
        help="Skip Rust code blocks (default: True, Rust snippets need crate context)",
    )
    parser.add_argument(
        "--test-rust",
        action="store_true",
        help="Test Rust code blocks (overrides --skip-rust)",
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Only test Python code blocks",
    )
    args = parser.parse_args()

    skip_rust = args.skip_rust and not args.test_rust

    print("Testing PECOS documentation code examples...")
    print(
        "Skip markers: <!--skip-->, <!--skip-if-no-cuda-->, <!--pytest.mark.skip-->, ```python,skip, ```rust,ignore",
    )
    print(
        "Expected error: <!--expect-error: pattern--> to test error cases",
    )
    print(
        "Hidden preamble: ```hidden-python``` blocks are prepended to subsequent blocks",
    )
    cuda_available = is_cuda_available()
    print(f"CUDA available: {cuda_available}")
    if skip_rust:
        print("Note: Rust blocks skipped by default (use --test-rust to enable)")

    markdown_files = find_markdown_files()
    print(f"Found {len(markdown_files)} Markdown files to test")

    python_results = []
    rust_results = []
    python_skipped = 0
    rust_skipped = 0

    # Test Python code blocks
    for file_path in markdown_files:
        python_blocks = extract_code_blocks(file_path, "python")
        for i, (block, should_skip, expected_error) in enumerate(python_blocks, 1):
            if should_skip:
                print(f"SKIP: Python block #{i} from {file_path}")
                python_skipped += 1
                continue
            result = run_python_block(block, i, file_path, expected_error)
            python_results.append((file_path, i, result))

    # Test Rust code blocks
    if not args.python_only:
        for file_path in markdown_files:
            rust_blocks = extract_code_blocks(file_path, "rust")
            for i, (block, should_skip, _expected_error) in enumerate(rust_blocks, 1):
                if should_skip or skip_rust:
                    rust_skipped += 1
                    if not skip_rust:  # Only print if individually skipped
                        print(f"SKIP: Rust block #{i} from {file_path}")
                    continue
                result = run_rust_block(block, i, file_path)
                rust_results.append((file_path, i, result))

    # Print summary
    python_passed = sum(1 for _, _, result in python_results if result)
    python_total = len(python_results)
    rust_passed = sum(1 for _, _, result in rust_results if result)
    rust_total = len(rust_results)

    print("\n===== SUMMARY =====")
    python_success_rate = f"{python_passed / python_total * 100:.1f}%" if python_total > 0 else "N/A"
    print(
        f"Python: {python_passed}/{python_total} blocks passed, "
        f"{python_skipped} skipped ({python_success_rate} success rate)",
    )
    rust_success_rate = f"{rust_passed / rust_total * 100:.1f}%" if rust_total > 0 else "N/A"
    print(
        f"Rust: {rust_passed}/{rust_total} blocks passed, "
        f"{rust_skipped} skipped ({rust_success_rate} success rate)",
    )

    # Print failed tests
    if python_passed < python_total or rust_passed < rust_total:
        print("\nFailed tests:")

        for file_path, block_num, result in python_results:
            if not result:
                print(f"- Python block #{block_num} in {file_path}")

        for file_path, block_num, result in rust_results:
            if not result:
                print(f"- Rust block #{block_num} in {file_path}")

    # Return non-zero exit code if any tests failed
    if python_passed < python_total or rust_passed < rust_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
