#!/usr/bin/env python3
"""Check Zlup code snippets in documentation for syntax/semantic errors.

Extracts fenced code blocks from docs/**/*.md and validates them:
  - ``zlup``           → zlup check (parse + semantic)
  - ``zlup_fragment``  → zlup parse (syntax only), wrapped in fn
  - ``zlup_nocheck``   → skipped

Non-Zlup fences (bash, rust, zig, json, etc.) are ignored.

Usage:
    python3 scripts/check_docs.py [--verbose]
"""

import argparse
import glob
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZLUP_TAGS = {"zlup", "zlup_fragment", "zlup_nocheck"}

# Heuristic: lines starting with these indicate a complete (top-level) program
_TOPLEVEL_PREFIXES = (
    "fn ", "pub ", "inline fn ", "@attr", "gate ", "declare gate",
    "test ", "extern fn",
)

_TOPLEVEL_CONTAINS = (
    ":= struct", ":= enum", ":= error", ":= fault",
    ":= union", ":= @import",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_zlup_binary() -> str:
    """Locate the zlup binary (workspace target/debug or release)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zlup_dir = os.path.dirname(script_dir)  # exp/zlup

    # Walk up to find workspace root (contains target/)
    candidate = zlup_dir
    for _ in range(5):
        target = os.path.join(candidate, "target", "debug", "zlup")
        if os.path.isfile(target):
            return target
        target_rel = os.path.join(candidate, "target", "release", "zlup")
        if os.path.isfile(target_rel):
            return target_rel
        candidate = os.path.dirname(candidate)

    # Fallback: hope it's on PATH
    return "zlup"


def is_complete_program(source: str) -> bool:
    """Heuristic: does this snippet look like a complete top-level program?"""
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("///"):
            continue
        # First non-comment line
        for prefix in _TOPLEVEL_PREFIXES:
            if stripped.startswith(prefix):
                return True
        for pattern in _TOPLEVEL_CONTAINS:
            if pattern in stripped:
                return True
        return False
    return False


def wrap_fragment(source: str) -> str:
    """Wrap a code fragment in a function body for parsing."""
    # Replace `{ ... }` placeholder bodies with `{ }` so they parse
    wrapped = re.sub(r'\{\s*\.\.\.\s*\}', '{ }', source)
    # Also replace bare `// ...` comment-only placeholders
    wrapped = re.sub(r'//\s*\.\.\.', '// placeholder', wrapped)
    return f"fn __snippet__() -> unit {{\n{wrapped}\nreturn;\n}}"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r'^```(\w+)?\s*$')


def extract_blocks(filepath: str) -> list[dict]:
    """Extract fenced code blocks from a markdown file."""
    blocks = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    in_block = False
    tag = None
    start_line = 0
    block_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        if not in_block:
            m = FENCE_RE.match(line)
            if m:
                tag = m.group(1) or ""
                in_block = True
                start_line = i
                block_lines = []
        else:
            if line.rstrip() == "```":
                blocks.append({
                    "file": filepath,
                    "line": start_line,
                    "tag": tag,
                    "source": "".join(block_lines),
                })
                in_block = False
            else:
                block_lines.append(line)

    return blocks


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------

def run_zlup(binary: str, cmd: str, source: str) -> tuple[bool, str]:
    """Run zlup check/parse on source via stdin. Returns (ok, output)."""
    try:
        result = subprocess.run(
            [binary, cmd, "-"],
            input=source,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, f"zlup binary not found: {binary}"


def check_block(binary: str, block: dict) -> tuple[str, str]:
    """Check a single code block. Returns (status, detail).

    status is one of: "OK", "FAIL", "SKIP"
    """
    tag = block["tag"]
    source = block["source"]

    if tag not in ZLUP_TAGS:
        return "SKIP", "non-zlup fence"

    if tag == "zlup_nocheck":
        return "SKIP", "nocheck"

    if not source.strip():
        return "SKIP", "empty block"

    if tag == "zlup":
        # Full check (parse + semantic)
        ok, output = run_zlup(binary, "check", source)
        return ("OK" if ok else "FAIL"), output

    if tag == "zlup_fragment":
        # Wrap and parse-only
        wrapped = wrap_fragment(source)
        ok, output = run_zlup(binary, "parse", wrapped)
        return ("OK" if ok else "FAIL"), output

    return "SKIP", f"unknown tag: {tag}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show result for each snippet")
    args = parser.parse_args()

    # Find docs directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(os.path.dirname(script_dir), "docs")

    if not os.path.isdir(docs_dir):
        print(f"Error: docs directory not found at {docs_dir}", file=sys.stderr)
        sys.exit(1)

    binary = find_zlup_binary()

    # Quick check that the binary works
    ok, _ = run_zlup(binary, "parse", "fn test() -> unit { return; }")
    if not ok:
        print(f"Error: zlup binary not working ({binary})", file=sys.stderr)
        print("Run 'just build' first.", file=sys.stderr)
        sys.exit(1)

    # Collect all markdown files
    md_files = sorted(glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True))
    if not md_files:
        print("No markdown files found in docs/", file=sys.stderr)
        sys.exit(1)

    # Process
    counts = {"OK": 0, "FAIL": 0, "SKIP": 0}
    failures: list[dict] = []

    for md_file in md_files:
        blocks = extract_blocks(md_file)
        for block in blocks:
            status, detail = check_block(binary, block)
            counts[status] += 1

            rel = os.path.relpath(block["file"], os.path.dirname(docs_dir))

            if args.verbose:
                tag_info = f"[{block['tag']}]"
                if status == "FAIL":
                    print(f"  FAIL  {rel}:{block['line']}  {tag_info}")
                    # Show first few lines of error
                    for err_line in detail.splitlines()[:6]:
                        print(f"        {err_line}")
                elif status == "SKIP":
                    print(f"  SKIP  {rel}:{block['line']}  {tag_info} ({detail})")
                else:
                    print(f"  OK    {rel}:{block['line']}  {tag_info}")

            if status == "FAIL":
                failures.append({
                    "file": rel,
                    "line": block["line"],
                    "tag": block["tag"],
                    "detail": detail,
                })

    # Summary
    total = counts["OK"] + counts["FAIL"] + counts["SKIP"]
    print()
    print(f"check-docs: {total} snippets — "
          f"{counts['OK']} ok, {counts['FAIL']} failed, {counts['SKIP']} skipped")

    if failures:
        print()
        print("Failures:")
        for f in failures:
            print(f"  {f['file']}:{f['line']}  [{f['tag']}]")
            for err_line in f["detail"].splitlines()[:4]:
                print(f"    {err_line}")
        print()
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
