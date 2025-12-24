#!/usr/bin/env python3
"""Example of using PECOS's execute_llvm module for HUGR->LLVM compilation.

PECOS provides an execute_llvm module that implements the same interface as
the external execute_llvm package, but uses PECOS's own HUGR compilation
infrastructure.
"""

import pecos as pc


def main() -> None:
    """Demonstrate execute_llvm functionality."""
    print("PECOS execute_llvm Module Demo")
    print("=" * 50)

    # Check if execute_llvm functionality is available
    if pc.execute_llvm.is_available():
        print("execute_llvm functionality is available")
    else:
        print("No HUGR->LLVM backend available")
        print("  Build PECOS with HUGR support or install external compiler")
        return

    # In a real scenario, you would get HUGR bytes from compiling a Guppy function
    # For this demo, we'll use dummy data
    dummy_hugr_bytes = b"HUGR data would go here"

    print("\nCompiling HUGR to LLVM IR...")
    try:
        # This would normally work with real HUGR data
        llvm_ir = pc.execute_llvm.compile_module_to_string(dummy_hugr_bytes)
        print(f"Generated {len(llvm_ir)} characters of LLVM IR")

    except RuntimeError as e:
        print(f"Compilation failed (expected with dummy data): {e}")

    print("\nThe execute_llvm module provides:")
    print("  - compile_module_to_string(hugr_bytes) -> str")
    print("  - compile_module_to_file(hugr_bytes, output_path)")
    print("  - compile_hugr_file_to_string(hugr_path) -> str")
    print("  - compile_hugr_file_to_file(hugr_path, output_path)")
    print("  - is_available() -> bool")

    print("\nThis integrates seamlessly with PECOS's Guppy frontend!")


if __name__ == "__main__":
    main()
