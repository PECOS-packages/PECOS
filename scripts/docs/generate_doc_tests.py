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
    <!--skip-if-no-cuda-->                 - Skip if CUDA not available
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
    """Convert a string to a valid Python identifier."""
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
                dedented_lines.append(line)
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
    # - Code with ellipsis (abbreviated)
    if "..." in code or "// ..." in code:
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


def _rust_wrap_snippet(code: str) -> str:
    """Wrap a Rust snippet in fn main() if needed.

    Keeps use/extern crate statements at module level and wraps the rest in main().
    If the code uses the ? operator, wraps in fn main() -> Result<...> with Ok(()) at end.
    """
    if "fn main()" in code or "fn main ()" in code:
        return code

    lines = code.strip().split("\n")
    module_level = []
    main_body = []
    in_multiline_use = False  # Track if we're inside a multi-line use statement

    for line in lines:
        stripped = line.strip()

        # Continue multi-line use statement until we see the closing };
        if in_multiline_use:
            module_level.append(line)
            # Count braces to handle nested structures
            in_multiline_use = not stripped.endswith("};")
            continue

        # Keep use statements, extern crate, and comments at module level
        if stripped.startswith(("use ", "extern crate ", "//")) or stripped == "":
            # But only if we haven't started the main body yet
            if not main_body or stripped.startswith("//") or stripped == "":
                module_level.append(line)
                # Check if this is the start of a multi-line use statement
                if stripped.startswith("use ") and "{" in stripped and "};" not in stripped:
                    in_multiline_use = True
            else:
                main_body.append(line)
        else:
            main_body.append(line)

    # Check if code uses ? operator (needs Result return type)
    main_body_str = "\n".join(main_body)
    uses_question_mark = "?" in main_body_str

    # Build the wrapped code
    result_lines = module_level.copy()
    if main_body:
        result_lines.append("")
        if uses_question_mark:
            result_lines.append("fn main() -> Result<(), Box<dyn std::error::Error>> {")
        else:
            result_lines.append("fn main() {")
        for line in main_body:
            result_lines.append("    " + line if line.strip() else line)
        if uses_question_mark:
            result_lines.append("    Ok(())")
        result_lines.append("}")
    else:
        # No body, just add empty main
        result_lines.append("")
        result_lines.append("fn main() {}")

    return "\n".join(result_lines)


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

    # Check for skip-if-no-cuda (must check before generic skip)
    if "skip-if-no-cuda" in comment_lower:
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
            '@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")',
        )

    lines.extend(f"@pytest.mark.{mark}" for mark in block.marks)

    # Function signature
    lines.append(f"def {func_name}():")

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

    return [
        '    code = """',
        *list(escaped_code.split("\n")),
        '"""',
        "    exec(code, {})",
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
        *list(escaped_code.split("\n")),
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
            *list(escaped_code.split("\n")),
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
            *list(escaped_code.split("\n")),
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
    """Generate test body for Rust code (compile and run)."""
    if _rust_needs_cargo(block.code):
        return _generate_rust_cargo_body(block)
    return _generate_rust_rustc_body(block)


def _generate_rust_rustc_body(block: CodeBlock) -> list[str]:
    """Generate test body for simple Rust code (compile with rustc)."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    # Check if code has a main function, if not wrap it
    has_main = "fn main()" in block.code

    lines = [
        "    import subprocess",
        "    import tempfile",
        "    import os",
        "    from pathlib import Path",
        "",
        '    code = """',
        *list(escaped_code.split("\n")),
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
            "        src_path = Path(tmpdir) / 'test.rs'",
            "        bin_path = Path(tmpdir) / 'test'",
            "        src_path.write_text(code)",
            "",
            "        # Compile with rustc",
            "        compile_result = subprocess.run(",
            "            ['rustc', str(src_path), '-o', str(bin_path)],",
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


def _generate_rust_cargo_body(block: CodeBlock) -> list[str]:
    """Generate test body for Rust code that requires cargo (uses pecos crate)."""
    # Wrap the code in fn main() if needed (keeps use statements at module level)
    wrapped_code = _rust_wrap_snippet(block.code)
    escaped_code = wrapped_code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    # Get the project root directory (where Cargo.toml is)
    lines = [
        "    import subprocess",
        "    import tempfile",
        "    import os",
        "    from pathlib import Path",
        "",
        '    code = """',
        *list(escaped_code.split("\n")),
        '"""',
        "",
        "    # Find the PECOS project root (where workspace Cargo.toml is)",
        "    project_root = Path(__file__).resolve()",
        "    while project_root.parent != project_root:",
        "        if (project_root / 'Cargo.toml').exists() and (project_root / 'crates').exists():",
        "            break",
        "        project_root = project_root.parent",
        "    else:",
        '        pytest.skip("Could not find PECOS project root")',
        "",
        "    # Create a temporary cargo project that depends on local pecos crate",
        "    with tempfile.TemporaryDirectory() as tmpdir:",
        "        tmpdir = Path(tmpdir)",
        "",
        "        # Create Cargo.toml with pecos as a path dependency",
        '        cargo_toml = f"""',
        "[package]",
        'name = "doctest"',
        'version = "0.1.0"',
        'edition = "2021"',
        "",
        "[dependencies]",
        "# Enable runtime and hugr features for full API access",
        'pecos = {{ path = "{project_root}/crates/pecos", features = ["runtime", "hugr", "wasm"] }}',
        "# Also include internal crates that docs may reference directly",
        'pecos-hugr = {{ path = "{project_root}/crates/pecos-hugr" }}',
        'pecos-engines = {{ path = "{project_root}/crates/pecos-engines" }}',
        'pecos-num = {{ path = "{project_root}/crates/pecos-num" }}',
        'pecos-decoders = {{ path = "{project_root}/crates/pecos-decoders", features = ["ldpc"] }}',
        'pecos-decoder-core = {{ path = "{project_root}/crates/pecos-decoder-core" }}',
        "# Common external crates used in documentation examples",
        'serde_json = "1.0"',
        '"""',
        "        (tmpdir / 'Cargo.toml').write_text(cargo_toml)",
        "",
        "        # Create src directory and main.rs",
        "        (tmpdir / 'src').mkdir()",
        "        (tmpdir / 'src' / 'main.rs').write_text(code)",
        "",
        "        # Copy test data files if needed",
        "        test_data_dir = project_root / 'docs' / 'assets' / 'test-data'",
        "        hugr_test_data_dir = project_root / 'crates' / 'pecos' / 'tests' / 'test_data' / 'hugr'",
        "        python_generated_dir = Path('/tmp/pecos-doc-tests')",
    ]

    # Add file copy commands for each test data file
    if block.test_data:
        for test_file in block.test_data:
            escaped_file = test_file.replace('"', '\\"')
            lines.extend(
                [
                    f"        # Look for {escaped_file} in multiple locations",
                    "        src_file = None",
                    "        for search_dir in [python_generated_dir, test_data_dir, hugr_test_data_dir]:",
                    f'            candidate = search_dir / "{escaped_file}"',
                    "            if candidate.exists():",
                    "                src_file = candidate",
                    "                break",
                    "        if src_file:",
                    "            import shutil",
                    f'            shutil.copy(src_file, tmpdir / "{escaped_file}")',
                    "        else:",
                    f'            pytest.skip(f"Test data file not found: {escaped_file}")',
                ],
            )

    lines.extend(
        [
            "",
            "        # Build and run with cargo",
            "        build_result = subprocess.run(",
            "            ['cargo', 'build', '--release'],",
            "            cwd=tmpdir,",
            "            capture_output=True,",
            "            text=True,",
            "            timeout=300,",
            "            check=False,",
            "        )",
            "        if build_result.returncode != 0:",
            '            pytest.fail(f"Cargo build failed:\\n{build_result.stderr}")',
            "",
            "        run_result = subprocess.run(",
            "            ['cargo', 'run', '--release'],",
            "            cwd=tmpdir,",
            "            capture_output=True,",
            "            text=True,",
            "            timeout=60,",
            "            check=False,",
            "        )",
            "        if run_result.returncode != 0:",
            '            pytest.fail(f"Cargo run failed:\\n{run_result.stderr}")',
        ],
    )

    return lines


def _generate_rust_expect_error_body(block: CodeBlock) -> list[str]:
    """Generate test body for Rust code that expects a compilation error."""
    escaped_code = block.code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    escaped_pattern = block.expect_error.replace('"', '\\"') if block.expect_error else ""

    return [
        "    import subprocess",
        "    import tempfile",
        "    import re",
        "    from pathlib import Path",
        "",
        '    code = """',
        *list(escaped_code.split("\n")),
        '"""',
        f'    expected_pattern = r"{escaped_pattern}"',
        "",
        "    # Create temp directory for Rust compilation",
        "    with tempfile.TemporaryDirectory() as tmpdir:",
        "        src_path = Path(tmpdir) / 'test.rs'",
        "        bin_path = Path(tmpdir) / 'test'",
        "        src_path.write_text(code)",
        "",
        "        # Compile with rustc (expect failure)",
        "        compile_result = subprocess.run(",
        "            ['rustc', str(src_path), '-o', str(bin_path)],",
        "            capture_output=True,",
        "            text=True,",
        "            timeout=60,",
        "            check=False,",
        "        )",
        "        assert compile_result.returncode != 0, 'Expected Rust compilation to fail but it succeeded'",
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
            *list(escaped_code.split("\n")),
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
            *list(escaped_code.split("\n")),
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

    lines = [
        '"""',
        f"Auto-generated tests from {file_path}",
        "",
        "DO NOT EDIT - Generated by scripts/docs/generate_doc_tests.py",
        '"""',
        "",
        "import pytest",
        "",
    ]

    # Only include cuda_available if needed
    if needs_cuda_check:
        lines.extend(
            [
                "",
                "# CUDA availability check (inlined to avoid import issues)",
                "_CUDA_AVAILABLE = None",
                "",
                "",
                "def cuda_available():",
                '    """Check if CUDA is available."""',
                "    global _CUDA_AVAILABLE",
                "    if _CUDA_AVAILABLE is None:",
                "        import subprocess",
                "        import sys",
                "        try:",
                "            # Check CUDA toolkit",
                "            result = subprocess.run(",
                '                ["cargo", "run", "-p", "pecos", "--features", "cli", "--", "cuda", "check", "-q"],',
                "                capture_output=True, timeout=30, check=False)",
                "            if result.returncode != 0:",
                "                _CUDA_AVAILABLE = False",
                "            else:",
                "                # Check cupy",
                "                result = subprocess.run(",
                '                    [sys.executable, "-c", "import cupy; print(cupy.cuda.is_available())"],',
                "                    capture_output=True, text=True, timeout=10, check=False)",
                '                _CUDA_AVAILABLE = result.returncode == 0 and "True" in result.stdout',
                "        except Exception:",
                "            _CUDA_AVAILABLE = False",
                "    return _CUDA_AVAILABLE",
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
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = _check_cuda_available()
    return _CUDA_AVAILABLE


@pytest.fixture(scope="session")
def cuda_check():
    """Fixture that returns CUDA availability."""
    return cuda_available()


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore the current working directory after each test.

    Some tests (e.g., WASM examples) change the working directory,
    which can interfere with other tests that rely on path resolution.
    """
    import os
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")


def pytest_collection_modifyitems(config, items):
    """Print CUDA status at collection time."""
    cuda = cuda_available()
    print(f"\\nCUDA available: {cuda}")
'''


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
        (args.output_dir.parent / "__init__.py").touch()
        (args.output_dir / "__init__.py").touch()

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
        # Count skipped blocks including Rust that requires cargo
        total_skipped += sum(
            1
            for b in all_blocks
            if b.skip or b.skip_if_no_cuda or (b.language == "rust" and _rust_is_incomplete(b.code))
        )

        # Generate test file
        test_content = generate_test_file(md_file, all_blocks)

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
            output_path.write_text(test_content)
            files_generated += 1
            print(
                f"Generated: {output_path} ({len(python_blocks)} Python, {len(rust_blocks)} Rust blocks)",
            )

    print("\nSummary:")
    print(f"  Python blocks: {total_python_blocks}")
    print(f"  Rust blocks: {total_rust_blocks}")
    print(f"  Total code blocks: {total_python_blocks + total_rust_blocks}")
    print(f"  Blocks with skip markers: {total_skipped}")
    print(f"  Test files generated: {files_generated}")
    print(f"\nRun tests with: pytest {args.output_dir} -v")


if __name__ == "__main__":
    main()
