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

"""Generate pytest test files from documentation code examples.

This script extracts code blocks from Markdown files and generates pytest-compatible
test files. This allows running doc tests with standard pytest commands:

    pytest python/quantum-pecos/tests/docs/ -v

Supported markers in markdown:
    <!--skip--> or <!--skip: reason-->     - Skip this block
    <!--skip-if-no-cuda-->                 - Skip if CUDA+cupy not available
    <!--skip-if-no-cuda-rust-->            - Skip if CUDA Rust bindings not available
    <!--expect-error: pattern-->           - Expect error matching regex pattern
    <!--expect-output: text-->             - Expect stdout to contain text
    <!--test-name: my_test-->              - Name the test function
    <!--mark.slow-->                       - Add @pytest.mark.slow
    <!--continuation-->                    - Continue from previous block's state
    <!--setup-->                           - Module-level setup code
    <!--teardown-->                        - Module-level teardown code
    <!--preamble-reset-->                  - Clear accumulated preamble
    ```hidden-python```                    - Hidden preamble (prepended to following blocks)
    ```python,skip``` or ```python,notest``` - Skip this block

For Rust code blocks:
    ```rust``` or ```rust,ignore```        - Rust code (ignore = skip)
    <!--cargo-deps: serde, tokio-->        - Cargo dependencies needed
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

# Directory containing the Markdown files
DOCS_DIR = Path("docs")

# Output directory for generated tests
OUTPUT_DIR = Path("python/quantum-pecos/tests/docs/generated")


@dataclass
class CodeBlock:
    """Represents a code block extracted from markdown."""

    code: str
    language: str
    block_number: int
    source_file: Path
    line_number: int = 0

    # Markers
    skip: bool = False
    skip_reason: str | None = None
    skip_if_no_cuda: bool = False
    skip_if_no_cuda_rust: bool = False
    expect_error: str | None = None
    expect_output: str | None = None
    test_name: str | None = None
    marks: list[str] = field(default_factory=list)
    is_continuation: bool = False
    is_setup: bool = False
    is_teardown: bool = False
    is_hidden: bool = False

    # Preamble accumulated from hidden blocks
    preamble: str = ""

    # Rust-specific
    cargo_deps: list[str] = field(default_factory=list)

    # Test data files to copy (from docs/assets/test-data/)
    test_data: list[str] = field(default_factory=list)


def _sanitize_name(name: str) -> str:
    """Convert a string to a valid Python identifier (lowercase)."""
    # Lowercase for valid module names
    name = name.lower()
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Remove leading digits
    name = re.sub(r"^[0-9]+", "", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name or "unnamed"


def _dedent_code(code: str) -> str:
    """Remove common leading indentation from code."""
    lines = code.strip("\n").split("\n")
    min_indent = float("inf")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent != float("inf") and min_indent > 0:
        dedented_lines = []
        for line in lines:
            if line.strip():
                dedented_lines.append(line[int(min_indent) :])
            else:
                dedented_lines.append("")
        return "\n".join(dedented_lines)
    return code.strip()


def _uses_guppy_decorator(code: str) -> bool:
    """Check if code uses @guppy decorator (requires file-based execution)."""
    return "@guppy" in code


def _rust_is_incomplete(code: str) -> bool:
    """Check if Rust code is incomplete and cannot be compiled.

    Returns True for code that:
    - Is a trait/struct definition without a complete program
    - Is a code snippet that can't compile standalone
    - Contains ellipsis (abbreviated code)
    """
    # Check for incomplete code patterns:
    # - Trait definitions without implementations (conceptual code)
    if re.search(r"pub\s+trait\s+\w+", code):
        return True
    # - Struct definitions without main (likely illustrative)
    if re.search(r"pub\s+struct\s+\w+", code) and "fn main()" not in code:
        return True
    # - Enum definitions without main
    if re.search(r"pub\s+enum\s+\w+", code) and "fn main()" not in code:
        return True
    # - impl blocks without main
    if re.search(r"impl\s+\w+", code) and "fn main()" not in code:
        return True
    # - Code with ellipsis (abbreviated) -- but not in comments
    for line in code.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # Check for ... in non-comment code (e.g. "todo!()" is fine, "..." is abbreviated)
        code_part = stripped.split("//")[0]  # strip inline comments
        if "..." in code_part:
            return True
    # - No main function, but check if it's a wrappable snippet
    if "fn main()" not in code and "fn main ()" not in code:
        # If it has use statements with actual code, it can be wrapped
        if re.search(r"^use\s+", code, re.MULTILINE):
            # Check if there's actual executable code (not just imports/comments)
            lines = code.strip().split("\n")
            has_executable = False
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("use ") and not stripped.startswith("//"):
                    has_executable = True
                    break
            if has_executable:
                return False  # Can be wrapped
        return True

    return False


def _rust_needs_cargo(code: str) -> bool:
    """Check if Rust code requires cargo (uses external crates)."""
    # Check for use of project crates (pecos or any pecos_* crate)
    if re.search(r"use\s+pecos(_\w+)?::", code):
        return True
    # Check for use of other common external crates
    if re.search(r"use\s+(serde|tokio|anyhow|thiserror|hugr|tket2)::", code):
        return True
    # Check for extern crate declarations
    return bool(re.search(r"extern\s+crate\s+\w+", code))


def _parse_marker_comment(comment: str) -> dict:
    """Parse a marker comment and return extracted attributes."""
    result = {
        "skip": False,
        "skip_reason": None,
        "skip_if_no_cuda": False,
        "skip_if_no_cuda_rust": False,
        "expect_error": None,
        "expect_output": None,
        "test_name": None,
        "marks": [],
        "is_continuation": False,
        "is_setup": False,
        "is_teardown": False,
        "preamble_reset": False,
        "cargo_deps": [],
        "test_data": [],
    }

    if not comment:
        return result

    comment_lower = comment.lower()

    # Check for preamble reset
    if "preamble-reset" in comment_lower:
        result["preamble_reset"] = True
        return result

    # Check for skip-if-no-cuda-rust (must check before skip-if-no-cuda)
    if "skip-if-no-cuda-rust" in comment_lower:
        result["skip_if_no_cuda_rust"] = True

    # Check for skip-if-no-cuda (must check before generic skip)
    elif "skip-if-no-cuda" in comment_lower:
        result["skip_if_no_cuda"] = True

    # Check for regular skip
    elif "skip" in comment_lower:
        result["skip"] = True
        # Extract reason if present: <!--skip: reason-->
        match = re.search(r"skip:\s*(.+?)\s*-->", comment, re.IGNORECASE)
        if match:
            result["skip_reason"] = match.group(1).strip()

    # Check for expect-error
    if "expect-error" in comment_lower:
        match = re.search(r"expect-error:\s*(.+?)\s*-->", comment, re.IGNORECASE)
        if match:
            result["expect_error"] = match.group(1).strip()
            result["skip"] = False  # Don't skip, we want to test the error

    # Check for expect-output
    if "expect-output" in comment_lower:
        match = re.search(r"expect-output:\s*(.+?)\s*-->", comment, re.IGNORECASE)
        if match:
            result["expect_output"] = match.group(1).strip()

    # Check for test-name
    match = re.search(r"test-name:\s*(\w+)", comment, re.IGNORECASE)
    if match:
        result["test_name"] = match.group(1)

    # Check for pytest marks: <!--mark.slow--> or <!--mark.gpu-->
    for mark_match in re.finditer(r"mark\.(\w+)", comment, re.IGNORECASE):
        result["marks"].append(mark_match.group(1))

    # Check for continuation
    if "continuation" in comment_lower:
        result["is_continuation"] = True

    # Check for setup/teardown
    if "<!--setup-->" in comment_lower or "phmdoctest-setup" in comment_lower:
        result["is_setup"] = True
    if "<!--teardown-->" in comment_lower or "phmdoctest-teardown" in comment_lower:
        result["is_teardown"] = True

    # Check for cargo dependencies (Rust-specific)
    if "cargo-deps" in comment_lower:
        match = re.search(r"cargo-deps:\s*(.+?)\s*-->", comment, re.IGNORECASE)
        if match:
            deps = [d.strip() for d in match.group(1).split(",")]
            result["cargo_deps"] = deps

    # Check for test data files to copy
    if "test-data" in comment_lower:
        match = re.search(r"test-data:\s*(.+?)\s*-->", comment, re.IGNORECASE)
        if match:
            files = [f.strip() for f in match.group(1).split(",")]
            result["test_data"] = files

    return result


def _count_line_number(content: str, position: int) -> int:
    """Count the line number (1-indexed) for a given character position."""
    return content[:position].count("\n") + 1


def extract_code_blocks(file_path: Path, language: str = "python") -> list[CodeBlock]:
    """Extract code blocks of a specific language from a Markdown file."""
    with file_path.open(encoding="utf-8") as f:
        content = f.read()

    # Check for document-level skip marker at the start of the file
    # Format: <!--skip: reason--> or <!--skip--> at the beginning (after optional heading)
    doc_skip = False
    doc_skip_reason = None
    doc_skip_match = re.search(
        r"^(?:#[^\n]*\n\n?)?<!--\s*skip(?::\s*([^>]+))?\s*-->",
        content,
        re.IGNORECASE,
    )
    if doc_skip_match:
        doc_skip = True
        if doc_skip_match.group(1):
            doc_skip_reason = doc_skip_match.group(1).strip()

    # Pattern to find code blocks with optional marker comment before them
    # Captures: (marker_comment, hidden_prefix, lang_suffix, code)
    marker_pattern = r"(<!--[^>]*-->\s*)?"
    fence_pattern = rf"```(hidden-)?{language}(,(?:skip|ignore|no_run|notest))?\n(.*?)```"
    full_pattern = marker_pattern + fence_pattern

    blocks = []
    preamble_parts: list[str] = []
    setup_code = ""
    block_number = 0

    for match in re.finditer(full_pattern, content, re.DOTALL):
        marker_comment = match.group(1) or ""
        is_hidden = bool(match.group(2))
        lang_suffix = match.group(3) or ""
        code = match.group(4)

        # Calculate line number where the code block starts
        line_number = _count_line_number(content, match.start())

        # Parse the marker comment
        attrs = _parse_marker_comment(marker_comment)

        # Handle preamble reset
        if attrs["preamble_reset"]:
            preamble_parts = []
            continue

        # Handle fence-level skip markers
        if lang_suffix in (",skip", ",ignore", ",no_run", ",notest"):
            attrs["skip"] = True

        cleaned_code = _dedent_code(code)

        # Handle setup/teardown
        if attrs["is_setup"]:
            setup_code = cleaned_code
            continue
        if attrs["is_teardown"]:
            continue

        # Handle hidden blocks (accumulate as preamble)
        if is_hidden:
            # For Rust: if new preamble has fn main(), replace all previous preambles
            # This allows different sections to have different preambles
            if language == "rust" and "fn main()" in cleaned_code:
                preamble_parts = [cleaned_code]
            else:
                preamble_parts.append(cleaned_code)
            continue

        # Regular visible block
        block_number += 1

        # Build full code with preamble
        if preamble_parts:
            preamble = "\n\n".join(preamble_parts)
            # Check for placeholder pattern: // CODE or /* CODE */
            if "// CODE" in preamble:
                full_code = preamble.replace("// CODE", cleaned_code)
            elif "/* CODE */" in preamble:
                full_code = preamble.replace("/* CODE */", cleaned_code)
            else:
                # Default: append code after preamble
                full_code = preamble + "\n\n" + cleaned_code
        else:
            full_code = cleaned_code

        # Add setup code if present
        if setup_code:
            full_code = setup_code + "\n\n" + full_code

        # Apply document-level skip if present (block-level skip takes precedence)
        block_skip = attrs["skip"] or doc_skip
        block_skip_reason = attrs["skip_reason"] or doc_skip_reason

        block = CodeBlock(
            code=full_code,
            language=language,
            block_number=block_number,
            source_file=file_path,
            line_number=line_number,
            skip=block_skip,
            skip_reason=block_skip_reason,
            skip_if_no_cuda=attrs["skip_if_no_cuda"],
            skip_if_no_cuda_rust=attrs["skip_if_no_cuda_rust"],
            expect_error=attrs["expect_error"],
            expect_output=attrs["expect_output"],
            test_name=attrs["test_name"],
            marks=attrs["marks"],
            is_continuation=attrs["is_continuation"],
            preamble="\n\n".join(preamble_parts) if preamble_parts else "",
            cargo_deps=attrs["cargo_deps"],
            test_data=attrs["test_data"],
        )
        blocks.append(block)

    return blocks


def generate_test_function(block: CodeBlock, file_stem: str) -> str:
    """Generate a pytest test function for a code block."""
    # Determine test function name - include language for uniqueness
    if block.test_name:
        func_name = f"test_{_sanitize_name(block.test_name)}"
    elif block.language == "rust":
        func_name = f"test_{file_stem}_rust_{block.block_number}"
    else:
        func_name = f"test_{file_stem}_block_{block.block_number}"

    lines = []

    # Add pytest markers
    if block.skip:
        reason = block.skip_reason or "Marked as skip in documentation"
        lines.append(f'@pytest.mark.skip(reason="{reason}")')
    elif block.language == "rust" and _rust_is_incomplete(block.code):
        lines.append(
            '@pytest.mark.skip(reason="Rust code is incomplete (no main function or is a code snippet)")',
        )
    elif block.skip_if_no_cuda:
        lines.append(
            '@pytest.mark.skipif(not cuda_available(), reason="CUDA (cupy) not available")',
        )
    elif block.skip_if_no_cuda_rust:
        lines.append(
            '@pytest.mark.skipif(not cuda_rust_available(), reason="CUDA Rust bindings not available")',
        )

    lines.extend(f"@pytest.mark.{mark}" for mark in block.marks)

    # Function signature with return type annotation
    lines.append(f"def {func_name}() -> None:")

    # Docstring with source file and line number for easy navigation
    lines.append(f'    """Test from {block.source_file}:{block.line_number}."""')

    # Generate function body based on test type and language
    if block.language == "rust":
        if block.expect_error:
            lines.extend(_generate_rust_expect_error_body(block))
        else:
            lines.extend(_generate_rust_exec_body(block))
    elif block.expect_error:
        lines.extend(_generate_expect_error_body(block))
    elif block.expect_output:
        lines.extend(_generate_expect_output_body(block))
    elif _uses_guppy_decorator(block.code):
        lines.extend(_generate_guppy_body(block))
    else:
        lines.extend(_generate_exec_body(block))

    return "\n".join(lines)


def _generate_exec_body(block: CodeBlock) -> list[str]:
    """Generate test body that uses exec() for regular Python code."""
    # Escape the code for embedding in a string
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    # Strip trailing whitespace from each line to avoid W293
    code_lines = [line.rstrip() for line in escaped_code.split("\n")]

    return [
        '    code = """',
        *code_lines,
        '"""',
        "    exec(code, {})  # noqa: S102",
    ]


def _generate_guppy_body(block: CodeBlock) -> list[str]:
    """Generate test body for Guppy code (needs file-based execution)."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    return [
        "    import subprocess",
        "    import sys",
        "    import tempfile",
        "    from pathlib import Path",
        "",
        '    code = """',
        *[line.rstrip() for line in escaped_code.split("\n")],
        '"""',
        "",
        "    # Guppy needs file-based execution for inspect.getsourcelines()",
        "    # Run in temp directory to avoid polluting project root with generated files",
        "    with tempfile.TemporaryDirectory() as tmpdir:",
        "        temp_path = Path(tmpdir) / 'test_code.py'",
        "        temp_path.write_text(code)",
        "",
        "        result = subprocess.run(",
        "            [sys.executable, str(temp_path)],",
        "            capture_output=True,",
        "            text=True,",
        "            timeout=60,",
        "            check=False,",
        "            cwd=tmpdir,",
        "        )",
        "        if result.returncode != 0:",
        '            pytest.fail(f"Guppy code failed:\\n{result.stderr}")',
    ]


def _generate_expect_error_body(block: CodeBlock) -> list[str]:
    """Generate test body that expects an error matching a pattern."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    # Don't double-escape backslashes in patterns - they're already regex escapes
    escaped_pattern = block.expect_error.replace('"', '\\"') if block.expect_error else ""

    if _uses_guppy_decorator(block.code):
        # Guppy code needs subprocess execution
        # Run in temp directory to avoid polluting project root with generated files
        lines = [
            "    import subprocess",
            "    import sys",
            "    import tempfile",
            "    import re",
            "    from pathlib import Path",
            "",
            '    code = """',
            *[line.rstrip() for line in escaped_code.split("\n")],
            '"""',
            f'    expected_pattern = r"{escaped_pattern}"',
            "",
            "    with tempfile.TemporaryDirectory() as tmpdir:",
            "        temp_path = Path(tmpdir) / 'test_code.py'",
            "        temp_path.write_text(code)",
            "",
            "        result = subprocess.run(",
            "            [sys.executable, str(temp_path)],",
            "            capture_output=True,",
            "            text=True,",
            "            timeout=60,",
            "            check=False,",
            "            cwd=tmpdir,",
            "        )",
            "        assert result.returncode != 0, 'Expected code to fail but it succeeded'",
            "        assert re.search(expected_pattern, result.stderr), \\",
            '            f"Error did not match pattern {expected_pattern!r}:\\n{result.stderr}"',
        ]
    else:
        # Regular code can use subprocess with -c
        lines = [
            "    import subprocess",
            "    import sys",
            "    import re",
            "",
            '    code = """',
            *[line.rstrip() for line in escaped_code.split("\n")],
            '"""',
            f'    expected_pattern = r"{escaped_pattern}"',
            "",
            "    result = subprocess.run(",
            '        [sys.executable, "-c", code],',
            "        capture_output=True,",
            "        text=True,",
            "        timeout=30,",
            "        check=False,",
            "    )",
            "    assert result.returncode != 0, 'Expected code to fail but it succeeded'",
            "    assert re.search(expected_pattern, result.stderr), \\",
            '        f"Error did not match pattern {expected_pattern!r}:\\n{result.stderr}"',
        ]
    return lines


def _generate_rust_exec_body(block: CodeBlock) -> list[str]:
    """Generate test body for simple Rust code (compile with rustc).

    Rust tests that need cargo dependencies are filtered out before reaching
    this function -- they are tested via the unified Rust crate instead.
    """
    return _generate_rust_rustc_body(block)


def _generate_rust_rustc_body(block: CodeBlock) -> list[str]:
    """Generate test body for simple Rust code (compile with rustc)."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    code_lines = [line.rstrip() for line in escaped_code.split("\n")]

    # Check if code has a main function, if not wrap it
    has_main = "fn main()" in block.code

    lines = [
        "    import subprocess",
        "    import tempfile",
        "    import os",
        "    from pathlib import Path",
        "",
        '    code = """',
        *code_lines,
        '"""',
        "",
    ]

    if not has_main:
        lines.extend(
            [
                "    # Wrap code in main function if not present",
                '    if "fn main()" not in code:',
                '        code = f"fn main() {{\\n{code}\\n}}"',
                "",
            ],
        )

    lines.extend(
        [
            "    # Create temp directory for Rust compilation",
            "    with tempfile.TemporaryDirectory() as tmpdir:",
            '        src_path = Path(tmpdir) / "test.rs"',
            '        bin_path = Path(tmpdir) / "test"',
            "        src_path.write_text(code)",
            "",
            "        # Compile with rustc",
            "        compile_result = subprocess.run(",
            '            ["rustc", str(src_path), "-o", str(bin_path)],',
            "            capture_output=True,",
            "            text=True,",
            "            timeout=60,",
            "            check=False,",
            "        )",
            "        if compile_result.returncode != 0:",
            '            pytest.fail(f"Rust compilation failed:\\n{compile_result.stderr}")',
            "",
            "        # Run the compiled binary",
            "        run_result = subprocess.run(",
            "            [str(bin_path)],",
            "            capture_output=True,",
            "            text=True,",
            "            timeout=30,",
            "            check=False,",
            "        )",
            "        if run_result.returncode != 0:",
            '            pytest.fail(f"Rust execution failed:\\n{run_result.stderr}")',
        ],
    )

    return lines


# NOTE: Rust tests that need cargo dependencies are tested via the unified
# Rust crate at python/quantum-pecos/tests/docs/rust_crate/ instead of
# creating temporary Cargo projects per test. Run with:
#   cargo test --manifest-path python/quantum-pecos/tests/docs/rust_crate/Cargo.toml


def _generate_rust_expect_error_body(block: CodeBlock) -> list[str]:
    """Generate test body for Rust code that expects a compilation error."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    code_lines = [line.rstrip() for line in escaped_code.split("\n")]
    escaped_pattern = block.expect_error.replace('"', '\\"') if block.expect_error else ""

    return [
        "    import subprocess",
        "    import tempfile",
        "    import re",
        "    from pathlib import Path",
        "",
        '    code = """',
        *code_lines,
        '"""',
        f'    expected_pattern = r"{escaped_pattern}"',
        "",
        "    # Create temp directory for Rust compilation",
        "    with tempfile.TemporaryDirectory() as tmpdir:",
        '        src_path = Path(tmpdir) / "test.rs"',
        '        bin_path = Path(tmpdir) / "test"',
        "        src_path.write_text(code)",
        "",
        "        # Compile with rustc (expect failure)",
        "        compile_result = subprocess.run(",
        '            ["rustc", str(src_path), "-o", str(bin_path)],',
        "            capture_output=True,",
        "            text=True,",
        "            timeout=60,",
        "            check=False,",
        "        )",
        '        assert compile_result.returncode != 0, "Expected Rust compilation to fail but it succeeded"',
        "        assert re.search(expected_pattern, compile_result.stderr), \\",
        '            f"Error did not match pattern {expected_pattern!r}:\\n{compile_result.stderr}"',
    ]


def _generate_expect_output_body(block: CodeBlock) -> list[str]:
    """Generate test body that checks stdout contains expected text."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    escaped_output = block.expect_output.replace('"', '\\"') if block.expect_output else ""

    if _uses_guppy_decorator(block.code):
        # Guppy code needs file-based execution
        # Run in temp directory to avoid polluting project root with generated files
        lines = [
            "    import subprocess",
            "    import sys",
            "    import tempfile",
            "    from pathlib import Path",
            "",
            '    code = """',
            *[line.rstrip() for line in escaped_code.split("\n")],
            '"""',
            f'    expected_output = "{escaped_output}"',
            "",
            "    with tempfile.TemporaryDirectory() as tmpdir:",
            "        temp_path = Path(tmpdir) / 'test_code.py'",
            "        temp_path.write_text(code)",
            "",
            "        result = subprocess.run(",
            "            [sys.executable, str(temp_path)],",
            "            capture_output=True,",
            "            text=True,",
            "            timeout=60,",
            "            check=False,",
            "            cwd=tmpdir,",
            "        )",
            "        if result.returncode != 0:",
            '            pytest.fail(f"Code failed:\\n{result.stderr}")',
            "        assert expected_output in result.stdout, \\",
            '            f"Expected output containing {expected_output!r}, got:\\n{result.stdout}"',
        ]
    else:
        # Regular code can use subprocess with -c
        lines = [
            "    import subprocess",
            "    import sys",
            "",
            '    code = """',
            *[line.rstrip() for line in escaped_code.split("\n")],
            '"""',
            f'    expected_output = "{escaped_output}"',
            "",
            "    result = subprocess.run(",
            '        [sys.executable, "-c", code],',
            "        capture_output=True,",
            "        text=True,",
            "        timeout=30,",
            "        check=False,",
            "    )",
            "    if result.returncode != 0:",
            '        pytest.fail(f"Code failed:\\n{result.stderr}")',
            "    assert expected_output in result.stdout, \\",
            '        f"Expected output containing {expected_output!r}, got:\\n{result.stdout}"',
        ]
    return lines


def generate_test_file(file_path: Path, blocks: list[CodeBlock]) -> str:
    """Generate a complete pytest test file for a markdown file."""
    file_stem = _sanitize_name(file_path.stem)

    # Check if any block needs CUDA checking
    needs_cuda_check = any(b.skip_if_no_cuda for b in blocks)
    needs_cuda_rust_check = any(b.skip_if_no_cuda_rust for b in blocks)

    lines = [
        f'"""Auto-generated tests from {file_path}. DO NOT EDIT."""',
        "",
        "",
        "import pytest",
        "",
    ]

    # Only include cuda_available if needed
    if needs_cuda_check:
        lines.extend(
            [
                "",
                "",
                "def _check_cuda() -> bool:",
                '    """Return True if CUDA toolkit and cupy are available."""',
                "    import subprocess",
                "    import sys",
                "",
                "    try:",
                "        result = subprocess.run(",
                '            ["cargo", "run", "-p", "pecos", "--features", "cli",',
                '             "--", "cuda", "check", "-q"],',
                "            capture_output=True, timeout=30, check=False,",
                "        )",
                "        if result.returncode != 0:",
                "            return False",
                "        result = subprocess.run(",
                '            [sys.executable, "-c",',
                '             "import cupy; print(cupy.cuda.is_available())"],',
                "            capture_output=True, text=True, timeout=10, check=False,",
                "        )",
                '        return result.returncode == 0 and "True" in result.stdout',
                "    except (FileNotFoundError, subprocess.TimeoutExpired):",
                "        return False",
                "",
                "",
                "_CUDA_RESULT: bool | None = None",
                "",
                "",
                "def cuda_available() -> bool:",
                '    """Return cached CUDA availability."""',
                "    global _CUDA_RESULT  # noqa: PLW0603",
                "    if _CUDA_RESULT is None:",
                "        _CUDA_RESULT = _check_cuda()",
                "    return _CUDA_RESULT",
                "",
            ],
        )

    # Only include cuda_rust_available if needed
    if needs_cuda_rust_check:
        lines.extend(
            [
                "",
                "",
                "def _check_cuda_rust() -> bool:",
                '    """Return True if CUDA Rust bindings (pecos_rslib_cuda) are available."""',
                "    try:",
                "        from pecos_rslib_cuda import is_cuquantum_available",
                "        return is_cuquantum_available()",
                "    except ImportError:",
                "        return False",
                "",
                "",
                "_CUDA_RUST_RESULT: bool | None = None",
                "",
                "",
                "def cuda_rust_available() -> bool:",
                '    """Return cached CUDA Rust bindings availability."""',
                "    global _CUDA_RUST_RESULT  # noqa: PLW0603",
                "    if _CUDA_RUST_RESULT is None:",
                "        _CUDA_RUST_RESULT = _check_cuda_rust()",
                "    return _CUDA_RUST_RESULT",
                "",
            ],
        )

    lines.append("")

    # Generate test functions
    for block in blocks:
        lines.append(generate_test_function(block, file_stem))
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def generate_conftest() -> str:
    """Generate the conftest.py file with shared fixtures."""
    return '''\
"""Pytest configuration for documentation tests.

This module provides fixtures and utilities for testing documentation code examples.
"""

import subprocess
import sys

import pytest

# Cache CUDA availability
_CUDA_AVAILABLE: bool | None = None


def _check_cuda_available() -> bool:
    """Check if CUDA is available for running GPU examples.

    Uses the same pattern as the Justfile: `pecos cuda check -q` for toolkit,
    plus cupy availability check for Python CUDA packages.
    """
    # Check for CUDA toolkit using pecos CLI (same as Justfile pattern)
    try:
        result = subprocess.run(
            ["cargo", "run", "-p", "pecos", "--features", "cli", "--", "cuda", "check", "-q"],
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


def cuda_available() -> bool:
    """Return cached CUDA availability status."""
    global _CUDA_AVAILABLE  # noqa: PLW0603
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = _check_cuda_available()
    return _CUDA_AVAILABLE


@pytest.fixture(scope="session")
def cuda_check() -> bool:
    """Fixture that returns CUDA availability."""
    return cuda_available()


@pytest.fixture(autouse=True)
def restore_cwd():  # noqa: ANN201
    """Restore the current working directory after each test.

    Some tests (e.g., WASM examples) change the working directory,
    which can interfere with other tests that rely on path resolution.
    """
    from pathlib import Path

    original_cwd = Path.cwd()
    yield
    import os

    os.chdir(original_cwd)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Print CUDA status at collection time."""
    cuda = cuda_available()
    print(f"\\nCUDA available: {cuda}")
'''


def _strip_use_lines(code: str) -> tuple[list[str], str]:
    """Remove all use/extern crate lines from Rust code regardless of indentation.

    Returns (ordered list of unique use statements, code with uses removed).
    Multi-line use blocks (use foo::{bar, baz};) are handled.
    """
    lines = code.split("\n")
    seen: set[str] = set()
    uses: list[str] = []
    remaining: list[str] = []
    in_multiline_use = False
    multiline_buf: list[str] = []

    for line in lines:
        stripped = line.strip()
        if in_multiline_use:
            multiline_buf.append(stripped)
            if stripped.endswith("};"):
                stmt = "\n".join(multiline_buf)
                if stmt not in seen:
                    seen.add(stmt)
                    uses.append(stmt)
                in_multiline_use = False
                multiline_buf = []
            continue
        if stripped.startswith(("use ", "extern crate ")):
            if stripped.startswith("use ") and "{" in stripped and "};" not in stripped:
                in_multiline_use = True
                multiline_buf = [stripped]
            else:
                if stripped not in seen:
                    seen.add(stripped)
                    uses.append(stripped)
        else:
            remaining.append(line)

    # Collapse runs of 3+ blank lines into 2
    cleaned: list[str] = []
    blank_run = 0
    for line in remaining:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)

    return _deduplicate_uses(uses), "\n".join(cleaned)


def _deduplicate_uses(uses: list[str]) -> list[str]:
    """Merge and deduplicate Rust use statements.

    Handles:
    - Exact duplicates
    - Individual imports subsumed by braced imports from same path
      (e.g. `use foo::Bar;` redundant when `use foo::{Bar, Baz};` exists)
    - Merging braced imports from the same path
    - Glob imports subsume all imports from same path
    """
    # Parse into categories
    braced: dict[str, set[str]] = {}  # path -> set of names
    glob_paths: set[str] = set()
    individual: list[tuple[str, str]] = []  # (path, name)
    other: list[str] = []

    for u in uses:
        # Glob: use path::*;
        m = re.match(r"use\s+(.+?)::\*;", u)
        if m:
            glob_paths.add(m.group(1))
            continue
        # Braced: use path::{A, B};
        m = re.match(r"use\s+(.+?)::\{(.+?)};", u)
        if m:
            path = m.group(1)
            items = {item.strip() for item in m.group(2).split(",")}
            braced.setdefault(path, set()).update(items)
            continue
        # Individual: use path::Name;
        m = re.match(r"use\s+(.+?)::(\w+);", u)
        if m:
            individual.append((m.group(1), m.group(2)))
            continue
        other.append(u)

    result: list[str] = []

    # Glob imports
    result.extend(f"use {path}::*;" for path in sorted(glob_paths))

    # Fold individual imports into braced groups if path already has braced
    for path, name in individual:
        if path in glob_paths:
            continue  # glob covers everything
        braced.setdefault(path, set()).add(name)

    # Emit braced (or single) imports
    for path in sorted(braced):
        if path in glob_paths:
            continue
        names = sorted(braced[path])
        if len(names) == 1:
            result.append(f"use {path}::{names[0]};")
        else:
            result.append(f"use {path}::{{{', '.join(names)}}};")

    result.extend(other)
    return result


def _rust_wrap_as_test(code: str, test_name: str) -> str:
    """Wrap a Rust snippet as a #[test] function instead of fn main().

    Use statements stay at module level, everything else goes inside the test function.
    If code already has fn main(), replace it with the test function.
    """
    # If code has fn main(), replace it
    if "fn main()" in code or "fn main ()" in code:
        code = code.replace(
            "fn main() -> Result<(), Box<dyn std::error::Error>>",
            f"fn {test_name}() -> Result<(), Box<dyn std::error::Error>>",
        )
        code = re.sub(r"fn main\s*\(\s*\)", f"fn {test_name}()", code, count=1)
        # Insert #[test] before the fn line
        lines = code.split("\n")
        result = []
        for line in lines:
            if line.strip().startswith(f"fn {test_name}"):
                result.append("#[test]")
            result.append(line)
        return "\n".join(result)

    # Split into module-level (use/extern/comments) and body
    lines = code.strip().split("\n")
    module_level = []
    main_body = []
    in_multiline_use = False

    for line in lines:
        stripped = line.strip()
        if in_multiline_use:
            module_level.append(line)
            in_multiline_use = not stripped.endswith("};")
            continue
        if stripped.startswith(("use ", "extern crate ")) or (stripped == "" and not main_body):
            module_level.append(line)
            if stripped.startswith("use ") and "{" in stripped and "};" not in stripped:
                in_multiline_use = True
        elif stripped.startswith("//") and not main_body:
            module_level.append(line)
        else:
            main_body.append(line)

    uses_question_mark = "?" in "\n".join(main_body)

    result_lines = module_level.copy()
    result_lines.append("")
    result_lines.append("#[test]")
    if uses_question_mark:
        result_lines.append(f"fn {test_name}() -> Result<(), Box<dyn std::error::Error>> {{")
    else:
        result_lines.append(f"fn {test_name}() {{")
    for line in main_body:
        result_lines.append("    " + line if line.strip() else line)
    if uses_question_mark:
        result_lines.append("    Ok(())")
    result_lines.append("}")
    return "\n".join(result_lines)


def _generate_unified_rust_crate(markdown_files: list[Path], docs_dir: Path, crate_dir: Path) -> None:
    """Generate a unified Rust test crate with all doc examples as #[test] functions.

    This compiles dependencies once instead of per-test, making Rust doc tests ~100x faster.
    """
    tests_dir = crate_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale auto-generated test files before regenerating
    for old_file in tests_dir.glob("*.rs"):
        if old_file.name != "README.rs":
            old_file.unlink()

    total_tests = 0

    for md_file in sorted(markdown_files):
        rust_blocks = extract_code_blocks(md_file, "rust")
        if not rust_blocks:
            continue

        # Filter to blocks that need cargo and are not skipped/incomplete
        testable = []
        for i, block in enumerate(rust_blocks, 1):
            if block.skip or _rust_is_incomplete(block.code):
                continue
            if not _rust_needs_cargo(block.code):
                continue
            testable.append((i, block))

        if not testable:
            continue

        # Generate test file name from doc path
        relative = md_file.relative_to(docs_dir)
        test_module = _sanitize_name(str(relative.with_suffix("")).replace("/", "_"))
        test_file = tests_dir / f"{test_module}.rs"

        test_content = f"//! Auto-generated Rust tests from {relative}\n"
        test_content += "//! DO NOT EDIT - Generated by scripts/docs/generate_doc_tests.py\n"
        test_content += (
            "#![allow(unused_imports, unused_variables, unused_mut, unused_assignments, dead_code, non_snake_case)]\n"
        )

        for i, block in testable:
            test_name = f"test_{test_module}_rust_{i}"
            wrapped = _rust_wrap_as_test(block.code, test_name)
            # Move all use statements inside the test function to avoid
            # cross-test conflicts (e.g. prelude::* vs specific imports).
            uses, stripped = _strip_use_lines(wrapped)
            lines = stripped.split("\n")
            result: list[str] = []
            inserted = False
            for line in lines:
                result.append(line)
                # Insert uses right after the fn signature line
                if not inserted and line.strip().startswith(f"fn {test_name}"):
                    result.extend(f"    {u}" for u in uses)
                    inserted = True
            test_content += "\n" + "\n".join(result) + "\n"
            total_tests += 1

        # Strip trailing whitespace from each line
        cleaned_lines = [line.rstrip() for line in test_content.split("\n")]
        test_file.write_text("\n".join(cleaned_lines))

    print(f"Generated unified Rust test crate: {total_tests} tests in {crate_dir}")


def main() -> None:
    """Generate pytest test files from documentation."""
    parser = argparse.ArgumentParser(
        description="Generate pytest tests from documentation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for generated tests (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help=f"Documentation directory (default: {DOCS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    args = parser.parse_args()

    # Find all markdown files
    markdown_files = list(args.docs_dir.rglob("*.md"))
    print(f"Found {len(markdown_files)} Markdown files")

    # Create output directory
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate conftest.py
        conftest_path = args.output_dir.parent / "conftest.py"
        conftest_content = generate_conftest()
        conftest_path.write_text(conftest_content)
        print(f"Generated {conftest_path}")

        # Create __init__.py files
        for init_dir in [args.output_dir.parent, args.output_dir]:
            init_file = init_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Auto-generated doc test package."""\n')

    total_python_blocks = 0
    total_rust_blocks = 0
    total_skipped = 0
    files_generated = 0

    for md_file in markdown_files:
        # Extract Python and Rust blocks
        python_blocks = extract_code_blocks(md_file, "python")
        rust_blocks = extract_code_blocks(md_file, "rust")

        # Combine all blocks
        all_blocks = python_blocks + rust_blocks
        if not all_blocks:
            continue

        total_python_blocks += len(python_blocks)
        total_rust_blocks += len(rust_blocks)

        # Filter out Rust-cargo blocks (tested via unified crate)
        pytest_blocks = [b for b in all_blocks if not (b.language == "rust" and _rust_needs_cargo(b.code))]
        if not pytest_blocks:
            continue

        # Count skipped blocks
        total_skipped += sum(
            1
            for b in pytest_blocks
            if b.skip
            or b.skip_if_no_cuda
            or b.skip_if_no_cuda_rust
            or (b.language == "rust" and _rust_is_incomplete(b.code))
        )

        # Generate test file
        test_content = generate_test_file(md_file, pytest_blocks)

        # Create output path preserving directory structure
        relative_path = md_file.relative_to(args.docs_dir)
        test_file_name = f"test_{_sanitize_name(relative_path.stem)}.py"
        output_subdir = args.output_dir / relative_path.parent
        output_path = output_subdir / test_file_name

        if args.dry_run:
            print(
                f"Would generate: {output_path} ({len(python_blocks)} Python, {len(rust_blocks)} Rust blocks)",
            )
        else:
            output_subdir.mkdir(parents=True, exist_ok=True)
            # Ensure __init__.py exists in subdirectories
            init_file = output_subdir / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Auto-generated doc test package."""\n')
            output_path.write_text(test_content)
            files_generated += 1
            print(
                f"Generated: {output_path} ({len(python_blocks)} Python, {len(rust_blocks)} Rust blocks)",
            )

    # Generate unified Rust test crate
    if not args.dry_run:
        rust_crate_dir = args.output_dir.parent / "rust_crate"
        if rust_crate_dir.exists():
            _generate_unified_rust_crate(markdown_files, args.docs_dir, rust_crate_dir)

    print("\nSummary:")
    print(f"  Python blocks: {total_python_blocks}")
    print(f"  Rust blocks: {total_rust_blocks}")
    print(f"  Total code blocks: {total_python_blocks + total_rust_blocks}")
    print(f"  Blocks with skip markers: {total_skipped}")
    print(f"  Test files generated: {files_generated}")
    print(f"\nRun tests with: pytest {args.output_dir} -v")
    print(f"Run Rust doc tests: cargo test --manifest-path {rust_crate_dir}/Cargo.toml")


if __name__ == "__main__":
    main()
