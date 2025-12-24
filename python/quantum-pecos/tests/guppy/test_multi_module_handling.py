"""Test how Selene and PECOS handle multiple modules in HUGR.

This test explores whether Selene supports multiple modules or processes just the first one,
similar to the PECOS compiler behavior we observed.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Check if we have the required dependencies
try:
    from guppylang import GuppyModule, guppy

    GUPPY_AVAILABLE = True
except ImportError:
    GUPPY_AVAILABLE = False
    GuppyModule = None
    guppy = None

# Import quantum operations separately to avoid import error when guppylang isn't available
if GUPPY_AVAILABLE:
    try:
        from guppylang.stdlib.quantum import cx, h, measure, qubit
    except ImportError:
        # Fallback for different guppylang versions
        from guppylang.std.quantum import cx, h, measure, qubit

try:
    from selene_hugr_qis_compiler import compile_to_llvm_ir as selene_compile

    SELENE_AVAILABLE = True
except ImportError:
    SELENE_AVAILABLE = False

from pecos_rslib import compile_hugr_to_llvm_rust as rust_compile


def count_modules_in_hugr(hugr_str: str) -> tuple[int, list[str]]:
    """Count modules and extract their function names from HUGR string.

    Args:
        hugr_str: HUGR in string format (may be JSON or binary-prefixed)

    Returns:
        (module_count, list_of_function_names)
    """
    try:
        # HUGR string format seems to have a binary prefix, try to extract JSON
        if hugr_str.startswith("HUGRi"):
            # Find the JSON part after the binary prefix
            json_start = hugr_str.find('{"modules"')
            if json_start == -1:
                return 0, []
            hugr_str = hugr_str[json_start:]

        data = json.loads(hugr_str)
        modules = data.get("modules", [])

        # Extract function names from all modules
        function_names = [
            node["name"]
            for module in modules
            for node in module.get("nodes", [])
            if node.get("op") == "FuncDefn"
            and "name" in node
            and node["name"] != "__main__"
        ]

        return len(modules), function_names
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Failed to parse HUGR: {e}")
        print(f"First 200 chars: {hugr_str[:200]}")
        return 0, []


def extract_function_calls_from_llvm(llvm_ir: str) -> set[str]:
    """Extract function calls from LLVM IR.

    This helps us identify which quantum functions are actually being called
    in the compiled LLVM IR, which indicates which modules were processed.
    """
    import re

    # Look for various patterns that indicate function calls
    patterns = [
        r"call.*@(\w+)\(",  # Direct function calls
        r"define.*@(\w+)\(",  # Function definitions
        r"___(\w+)(?:\.|%|\()",  # QIS function names
    ]

    function_calls = set()
    for raw_line in llvm_ir.split("\n"):
        line = raw_line.strip()
        if "call" in line or "define" in line or "___" in line:
            for pattern in patterns:
                matches = re.findall(pattern, line)
                function_calls.update(matches)

    return function_calls


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
def test_single_module_baseline() -> None:
    """Test baseline behavior with a single module for comparison."""

    @guppy
    def single_hadamard() -> bool:
        """Simple single-module function."""
        q = qubit()
        h(q)
        return measure(q)

    hugr = single_hadamard.compile()
    hugr_json = hugr.to_str() if hasattr(hugr, "to_str") else str(hugr)

    # Analyze the HUGR structure
    module_count, function_names = count_modules_in_hugr(hugr_json)

    print(f"Single module test - Modules: {module_count}, Functions: {function_names}")
    assert module_count >= 1, "Should have at least one module"
    assert "single_hadamard" in function_names, "Should contain the main function"


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
def test_multiple_functions_compilation() -> None:
    """Test compiling multiple functions using current guppylang API."""

    # Define multiple functions separately
    @guppy
    def create_bell_pair() -> tuple[bool, bool]:
        """Create a Bell pair and measure both qubits."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        m0 = measure(q0)
        m1 = measure(q1)
        return m0, m1

    @guppy
    def single_qubit_test() -> bool:
        """Single qubit Hadamard test."""
        q = qubit()
        h(q)
        return measure(q)

    # Compile each function separately
    bell_hugr = create_bell_pair.compile()
    single_hugr = single_qubit_test.compile()

    # Analyze each HUGR structure
    bell_hugr_str = (
        bell_hugr.to_str() if hasattr(bell_hugr, "to_str") else str(bell_hugr)
    )
    single_hugr_str = (
        single_hugr.to_str() if hasattr(single_hugr, "to_str") else str(single_hugr)
    )

    bell_modules, bell_functions = count_modules_in_hugr(bell_hugr_str)
    single_modules, single_functions = count_modules_in_hugr(single_hugr_str)

    print(f"Bell pair - Modules: {bell_modules}, Functions: {bell_functions}")
    print(f"Single qubit - Modules: {single_modules}, Functions: {single_functions}")

    # Each compiled function should have its own module with its function
    assert bell_modules >= 1, "Bell pair should have at least one module"
    assert single_modules >= 1, "Single qubit should have at least one module"

    assert (
        "create_bell_pair" in bell_functions
    ), "Bell HUGR should contain create_bell_pair"
    assert (
        "single_qubit_test" in single_functions
    ), "Single HUGR should contain single_qubit_test"


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
@pytest.mark.skipif(
    not SELENE_AVAILABLE,
    reason="selene_hugr_qis_compiler not available",
)
def test_compiler_comparison_simple() -> None:
    """Test how Selene vs PECOS handle HUGR compilation."""

    # Create a simple function to test both compilers
    @guppy
    def test_function() -> tuple[bool, bool]:
        """Test function that creates a Bell state."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        m0 = measure(q0)
        m1 = measure(q1)
        return m0, m1

    # Compile to HUGR
    hugr = test_function.compile()
    hugr_binary = hugr.to_bytes()  # Binary format for Selene
    hugr_str = hugr.to_str() if hasattr(hugr, "to_str") else str(hugr)

    # Analyze HUGR structure
    module_count, function_names = count_modules_in_hugr(hugr_str)
    print(f"HUGR Analysis - Modules: {module_count}, Functions: {function_names}")

    # Compile with both compilers
    try:
        selene_llvm = selene_compile(hugr_binary)
        print(f"Selene compilation succeeded, produced {len(selene_llvm)} chars")
    except Exception as e:
        pytest.fail(f"Selene compilation failed: {e}")

    try:
        # For Rust compiler, we need to extract the JSON part from HUGR string
        if hugr_str.startswith("HUGRi"):
            json_start = hugr_str.find('{"modules"')
            if json_start != -1:
                hugr_json = hugr_str[json_start:]
            else:
                pytest.fail("Could not extract JSON from HUGR string")
        else:
            hugr_json = hugr_str

        rust_llvm = rust_compile(hugr_json.encode("utf-8"), None)
        print(f"Rust compilation succeeded, produced {len(rust_llvm)} chars")
    except Exception as e:
        pytest.fail(f"Rust compilation failed: {e}")

    # Extract function calls from both LLVM outputs
    selene_functions = extract_function_calls_from_llvm(selene_llvm)
    rust_functions = extract_function_calls_from_llvm(rust_llvm)

    print(f"Selene LLVM functions: {sorted(selene_functions)}")
    print(f"Rust LLVM functions: {sorted(rust_functions)}")

    # Check if both compilers found the same functions
    # This will help us understand if they process modules differently
    common_functions = selene_functions & rust_functions
    selene_only = selene_functions - rust_functions
    rust_only = rust_functions - selene_functions

    print(f"Common functions: {sorted(common_functions)}")
    print(f"Selene-only functions: {sorted(selene_only)}")
    print(f"Rust-only functions: {sorted(rust_only)}")

    # Save debug output
    debug_dir = Path(tempfile.gettempdir()) / "compiler_comparison_debug"
    debug_dir.mkdir(exist_ok=True)

    (debug_dir / "hugr.txt").write_text(hugr_str)

    if hugr_str.startswith("HUGRi"):
        json_start = hugr_str.find('{"modules"')
        if json_start != -1:
            (debug_dir / "hugr.json").write_text(hugr_str[json_start:])

    (debug_dir / "selene.ll").write_text(selene_llvm)
    (debug_dir / "rust.ll").write_text(rust_llvm)

    print(f"Debug files saved to: {debug_dir}")

    # For now, just ensure both compilers produced valid output
    assert len(selene_llvm) > 0, "Selene should produce LLVM output"
    assert len(rust_llvm) > 0, "Rust should produce LLVM output"

    # Report the differences for analysis
    if selene_only or rust_only:
        print("WARNING: Compilers produced different function sets!")
        print("This suggests different compilation behavior.")


@pytest.mark.skipif(not GUPPY_AVAILABLE, reason="guppylang not available")
def test_hugr_structure_analysis() -> None:
    """Analyze the structure of HUGR to understand the format."""

    @guppy
    def test_func() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    hugr = test_func.compile()
    hugr_str = hugr.to_str() if hasattr(hugr, "to_str") else str(hugr)

    print("HUGR string format analysis:")
    print(f"- Length: {len(hugr_str)}")
    print(f"- Starts with: {hugr_str[:20]}")

    # Extract JSON from HUGR string
    hugr_json = hugr_str
    if hugr_str.startswith("HUGRi"):
        json_start = hugr_str.find('{"modules"')
        if json_start != -1:
            hugr_json = hugr_str[json_start:]
        else:
            print("No JSON found in HUGR string")
            return

    # Parse and analyze the JSON structure
    try:
        data = json.loads(hugr_json)
        print("HUGR JSON Structure Analysis:")
        print(f"- Top-level keys: {list(data.keys())}")

        if "modules" in data:
            modules = data["modules"]
            print(f"- Number of modules: {len(modules)}")

            for i, module_data in enumerate(modules):
                print(f"- Module {i} keys: {list(module_data.keys())}")

                if "nodes" in module_data:
                    nodes = module_data["nodes"]
                    func_nodes = [n for n in nodes if n.get("op") == "FuncDefn"]
                    print(f"  - Function definition nodes: {len(func_nodes)}")

                    for func_node in func_nodes:
                        func_name = func_node.get("name", "unnamed")
                        print(f"    - Function: {func_name}")

        # Save the full structure for manual inspection
        debug_file = Path(tempfile.gettempdir()) / "hugr_structure.json"
        debug_file.write_text(json.dumps(data, indent=2))
        print(f"Full HUGR structure saved to: {debug_file}")

    except json.JSONDecodeError as e:
        print(f"Failed to parse HUGR JSON: {e}")
        print(f"First 1000 chars: {hugr_json[:1000]}")


if __name__ == "__main__":
    # Manual testing
    if GUPPY_AVAILABLE:
        print("Running manual multi-module tests...")

        # Test 1: Single module baseline
        print("\n=== Test 1: Single Module ===")
        test_single_module_baseline()

        # Test 2: Multi-function compilation
        print("\n=== Test 2: Multi-Function Compilation ===")
        test_multiple_functions_compilation()

        # Test 3: Structure analysis
        print("\n=== Test 3: Structure Analysis ===")
        test_hugr_structure_analysis()

        # Test 4: Compiler comparison (if available)
        if SELENE_AVAILABLE:
            print("\n=== Test 4: Compiler Comparison ===")
            test_compiler_comparison_simple()
        else:
            print("\n=== Test 4: Skipped (missing compilers) ===")
            print(f"Selene available: {SELENE_AVAILABLE}")
            print("Rust compiler available: True")
    else:
        print("Guppylang not available, skipping manual tests")
