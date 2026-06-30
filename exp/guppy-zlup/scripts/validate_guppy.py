#!/usr/bin/env python3
"""Validate Guppy source files using guppylang.

This script validates that Guppy source code is semantically correct
by attempting to compile it with guppylang.

Usage:
    python validate_guppy.py <file.py>

Exit codes:
    0 - Valid Guppy code
    1 - Invalid Guppy code (errors printed to stderr)
    2 - File not found or other IO error
"""

import importlib.util
import json
import sys
from pathlib import Path


def validate_guppy_file(filepath: str) -> tuple[bool, list[dict]]:
    """Validate a Guppy source file using guppylang.

    Returns:
        (is_valid, errors) where errors is a list of error dicts
    """
    path = Path(filepath)
    if not path.exists():
        return False, [{"error": "FileNotFound", "message": f"File not found: {filepath}"}]

    try:
        spec = importlib.util.spec_from_file_location("guppy_module", str(path))
        if spec is None or spec.loader is None:
            return False, [{"error": "ImportError", "message": "Could not load module"}]

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find guppy functions and compile them
        compiled = []
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, "compile") and hasattr(obj, "check"):
                # This is a GuppyFunctionDefinition
                obj.compile()
                compiled.append(name)

        if not compiled:
            # No guppy functions found - just syntax checking passed
            return True, []

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        # Try to extract useful info from guppy errors
        error_info = {
            "error": error_type,
            "message": error_msg,
        }

        # Parse guppy error details if available
        if "var=" in error_msg:
            import re

            match = re.search(r"var='(\w+)'", error_msg)
            if match:
                error_info["variable"] = match.group(1)

        return False, [error_info]
    else:
        return True, []


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_guppy.py <file.py>", file=sys.stderr)
        sys.exit(2)

    filepath = sys.argv[1]
    output_json = "--json" in sys.argv

    is_valid, errors = validate_guppy_file(filepath)

    if output_json:
        result = {
            "valid": is_valid,
            "errors": errors,
            "file": filepath,
        }
        print(json.dumps(result))
    else:
        if is_valid:
            print(f"Valid: {filepath}")
        else:
            print(f"Invalid: {filepath}", file=sys.stderr)
            for err in errors:
                print(f"  {err['error']}: {err['message'][:200]}", file=sys.stderr)

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
