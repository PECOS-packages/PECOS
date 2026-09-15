"""Test how Selene and PECOS handle multiple modules in HUGR.

This test explores whether Selene supports multiple modules or processes just the first one,
similar to the PECOS compiler behavior we observed.
"""

import json
import re
import tempfile
from pathlib import Path

import pytest
from guppylang import GuppyModule, guppy
from guppylang.std.builtins import owned
from hugr.package import Package
from pecos.compilation_pipeline import compile_hugr_to_qis as pecos_compile
from pecos_rslib.hugr_lowering import PECOS_HELPER_ABIS
from selene_hugr_qis_compiler import compile_to_llvm_ir as selene_compile

# Import quantum operations - try stdlib first, fall back to std
try:
    from guppylang.stdlib.quantum import cx, h, measure, qubit
except ImportError:
    from guppylang.std.quantum import cx, h, measure, qubit


def count_modules_in_hugr(pkg: Package) -> tuple[int, list[str]]:
    """Count modules and extract their function names from a HUGR package.

    Args:
        pkg: Compiled HUGR as a hugr.package.Package

    Returns:
        (module_count, list_of_function_names)
    """
    function_names: list[str] = []
    for module in pkg.modules:
        for node in module.nodes():
            n = node[0] if isinstance(node, tuple) else node
            op = module[n].op
            if type(op).__name__ == "FuncDefn" and op.f_name != "__main__":
                function_names.append(op.f_name)

    return len(pkg.modules), function_names


def llvm_function_names(llvm_ir: str, kind: str) -> set[str]:
    """Read function names from Selene's canonical LLVM assembly output."""
    pattern = rf'^{kind}\b[^@\n]*@("(?:[^"\\]|\\.)*"|[a-zA-Z0-9_.$-]+)\s*\('
    names = set(re.findall(pattern, llvm_ir, re.MULTILINE))
    return {
        (
            re.sub(r"\\([0-9a-fA-F]{2})", lambda match: chr(int(match[1], 16)), name[1:-1])
            if name.startswith('"')
            else name
        )
        for name in names
    }


def test_single_module_baseline() -> None:
    """Test baseline behavior with a single module for comparison."""

    @guppy
    def single_hadamard() -> bool:
        """Simple single-module function."""
        q = qubit()
        h(q)
        return measure(q).read()

    pkg = single_hadamard.compile()

    # Analyze the HUGR structure
    module_count, function_names = count_modules_in_hugr(pkg)

    print(f"Single module test - Modules: {module_count}, Functions: {function_names}")
    assert module_count >= 1, "Should have at least one module"
    assert any(fn.endswith("single_hadamard") for fn in function_names), "Should contain the main function"


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
        m0 = measure(q0).read()
        m1 = measure(q1).read()
        return m0, m1

    @guppy
    def single_qubit_test() -> bool:
        """Single qubit Hadamard test."""
        q = qubit()
        h(q)
        return measure(q).read()

    # Compile each function separately
    bell_pkg = create_bell_pair.compile()
    single_pkg = single_qubit_test.compile()

    # Analyze each HUGR structure
    bell_modules, bell_functions = count_modules_in_hugr(bell_pkg)
    single_modules, single_functions = count_modules_in_hugr(single_pkg)

    print(f"Bell pair - Modules: {bell_modules}, Functions: {bell_functions}")
    print(f"Single qubit - Modules: {single_modules}, Functions: {single_functions}")

    # Each compiled function should have its own module with its function
    assert bell_modules >= 1, "Bell pair should have at least one module"
    assert single_modules >= 1, "Single qubit should have at least one module"

    assert any(fn.endswith("create_bell_pair") for fn in bell_functions), "Bell HUGR should contain create_bell_pair"
    assert any(
        fn.endswith("single_qubit_test") for fn in single_functions
    ), "Single HUGR should contain single_qubit_test"


def test_normalizer_keeps_selene_functions() -> None:
    """Preserve all definitions and declarations except renamed helper symbols."""

    @guppy.declare
    def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...

    @guppy
    def test_function() -> tuple[bool, bool]:
        q0 = qubit()
        q1 = qubit()
        q0 = pecos_qis_runtime_barrier_qubit_hugr(q0)
        h(q0)
        cx(q0, q1)
        m0 = measure(q0).read()
        m1 = measure(q1).read()
        return m0, m1

    data = test_function.compile().to_bytes()
    raw = selene_compile(data)
    normalized = pecos_compile(data)
    raw_definitions = llvm_function_names(raw, "define")
    assert raw_definitions
    assert llvm_function_names(normalized, "define") == raw_definitions

    raw_declarations = llvm_function_names(raw, "declare")
    declarations = llvm_function_names(normalized, "declare")
    expected = set()
    for name in raw_declarations:
        if name in declarations:
            expected.add(name)
            continue
        components = name.split(".")
        if components[-1].isascii() and components[-1].isdigit():
            components.pop()
        assert components[0] == "__hugr__"
        assert components[-1] in PECOS_HELPER_ABIS
        expected.add(components[-1])
    assert declarations == expected
    assert "pecos_qis_runtime_barrier_qubit_hugr" in declarations


def test_hugr_structure_analysis() -> None:
    """Analyze the structure of HUGR to understand the format."""

    @guppy
    def test_func() -> bool:
        q = qubit()
        h(q)
        return measure(q).read()

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
