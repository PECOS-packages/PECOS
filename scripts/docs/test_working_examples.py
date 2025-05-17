#!/usr/bin/env python3
"""
Test script for validating the working examples in PECOS documentation.

This script focuses on testing code examples that are known to work,
making it useful for CI testing or demonstrating the testing framework.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Files to test (relative to the docs directory)
TEST_FILES = [
    "user-guide/resources/working-examples.md",
    "user-guide/resources/code-testing-examples.md",
    "development/code-examples.md",
]

# Base directory for markdown files
DOCS_DIR = Path("docs")


def extract_code_blocks(file_path, language="python"):
    """Extract code blocks of a specific language from a Markdown file."""
    with Path(file_path).open(encoding="utf-8") as f:
        content = f.read()

    # Find all code blocks with the specified language
    # Exclude blocks with "untested" marker
    pattern = (
        rf"```(?:{language}|exec-{language}|hidden-{language})(?!.*?untested)(.*?)```"
    )
    blocks = re.findall(pattern, content, re.DOTALL)

    # Clean up the blocks (remove leading/trailing whitespace)
    blocks = [block.strip() for block in blocks]

    return blocks


def test_python_block(code_block, block_number, file_path):
    """Test a Python code block by executing it and checking for errors."""
    print(f"Testing Python block #{block_number} from {file_path}...")

    try:
        # Execute the code block and capture output
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code_block],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode != 0:
            print(f"FAIL: Error in Python block #{block_number} from {file_path}:")
            print(result.stderr)
            return False
        else:
            print(f"PASS: Python block #{block_number} from {file_path}")
            return True
    except subprocess.TimeoutExpired:
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


def test_rust_block(code_block, block_number, file_path):
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

        try:
            # Find rustc executable
            rustc_path = shutil.which("rustc")
            if not rustc_path:
                print(
                    f"FAIL: rustc not found in PATH for Rust block #{block_number} from {file_path}",
                )
                return False

            # Compile and run the Rust code
            compile_result = subprocess.run(  # noqa: S603
                [rustc_path, str(temp_file), "-o", str(Path(tmpdir) / "rust_test")],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if compile_result.returncode != 0:
                print(
                    f"FAIL: Compilation error in Rust block #{block_number} from {file_path}:",
                )
                print(compile_result.stderr)
                return False

            # Run the compiled program
            run_result = subprocess.run(  # noqa: S603
                [str(Path(tmpdir) / "rust_test")],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if run_result.returncode != 0:
                print(
                    f"FAIL: Runtime error in Rust block #{block_number} from {file_path}:",
                )
                print(run_result.stderr)
                return False
            else:
                print(f"PASS: Rust block #{block_number} from {file_path}")
                return True
        except subprocess.TimeoutExpired:
            print(f"FAIL: Timeout in Rust block #{block_number} from {file_path}")
            return False
        except OSError as e:
            print(
                f"FAIL: OS error testing Rust block #{block_number} from {file_path}: {e}",
            )
            return False
        except subprocess.SubprocessError as e:
            print(
                f"FAIL: Subprocess error testing Rust block #{block_number} from {file_path}: {e}",
            )
            return False


def main():
    """Main function to test working examples in the documentation."""
    print("Testing PECOS documentation working examples...")

    python_results = []
    rust_results = []

    # Test specified files
    for rel_path in TEST_FILES:
        file_path = DOCS_DIR / rel_path
        if not file_path.exists():
            print(f"Warning: File {file_path} not found, skipping...")
            continue

        print(f"\nTesting file: {file_path}")

        # Test Python code blocks
        python_blocks = extract_code_blocks(file_path, "python")
        for i, block in enumerate(python_blocks, 1):
            result = test_python_block(block, i, file_path)
            python_results.append((file_path, i, result))

        # Test Rust code blocks
        rust_blocks = extract_code_blocks(file_path, "rust")
        for i, block in enumerate(rust_blocks, 1):
            result = test_rust_block(block, i, file_path)
            rust_results.append((file_path, i, result))

    # Print summary
    python_passed = sum(1 for _, _, result in python_results if result)
    python_total = len(python_results)
    rust_passed = sum(1 for _, _, result in rust_results if result)
    rust_total = len(rust_results)

    print("\n===== SUMMARY =====")
    python_success_rate = (
        f"{python_passed/python_total*100:.1f}%" if python_total > 0 else "N/A"
    )
    print(
        f"Python: {python_passed}/{python_total} blocks passed ({python_success_rate} success rate)",
    )
    rust_success_rate = (
        f"{rust_passed/rust_total*100:.1f}%" if rust_total > 0 else "N/A"
    )
    print(
        f"Rust: {rust_passed}/{rust_total} blocks passed ({rust_success_rate} success rate)",
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
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()
