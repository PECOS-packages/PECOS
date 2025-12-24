#!/usr/bin/env python3
"""Simple run_guppy() API Demo (Backward Compatibility).

This example demonstrates the simple, qasm_sim-like API for running Guppy
quantum programs on PECOS. This API is provided for backward compatibility.

NOTE: For new code, consider using the unified API instead:
    from pecos_rslib import selene_engine
    from pecos_rslib.programs import HugrProgram

    # Convert Guppy to HUGR and run
    hugr_program = HugrProgram.from_bytes(guppy_to_hugr_bytes(my_func))
    results = selene_engine().program(hugr_program).to_sim().run(shots)

The backward compatibility API provides:
- run_guppy(function, shots) - Simple execution
- guppy_sim(function, shots) - Alias for consistency with PECOS APIs
- run_guppy_batch([functions], shots) - Batch execution
- get_guppy_backends() - Backend availability check.
"""

# Check availability first
try:
    import pecos

    print("[OK] PECOS available")
    print(f"Guppy integration: {pecos.GUPPY_INTEGRATION_AVAILABLE}")
except ImportError:
    print("[WARNING] PECOS not available")

# Try to import and run examples
try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, h, measure, qubit

    print("[OK] Guppy available")
    DEMO_ENABLED = True
except ImportError:
    print(
        "[WARNING] Guppy not available - install with: pip install quantum-pecos[guppy]",
    )
    DEMO_ENABLED = False


def demo_simple_api() -> None:
    """Demonstrate the simple run_guppy() API."""
    if not DEMO_ENABLED:
        print("Skipping demo - Guppy not available")
        return

    print("\n=== Simple run_guppy() API Demo ===")

    # Define quantum functions
    @guppy
    def random_bit() -> bool:
        """Generate a random bit using quantum superposition."""
        q = qubit()
        h(q)
        return measure(q)

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create Bell state and measure both qubits."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)

    @guppy
    def ghz_state() -> tuple[bool, bool, bool]:
        """Create GHZ state with three qubits."""
        q0, q1, q2 = qubit(), qubit(), qubit()
        h(q0)
        cx(q0, q1)
        cx(q1, q2)
        return measure(q0), measure(q1), measure(q2)

    try:
        # Import the simple API
        from pecos import get_guppy_backends, guppy_sim, run_guppy, run_guppy_batch

        # Check available backends
        print("Backend availability:")
        backends = get_guppy_backends()
        for name, status in backends.items():
            print(f"  {name}: {status}")

        # Demo 1: Simple single function execution
        print(f"\n1. Running {random_bit.__name__} with run_guppy()")
        result = run_guppy(random_bit, shots=100, verbose=True)

        print("Results summary:")
        print(f"  Function: {result['function_name']}")
        print(f"  Shots: {result['shots']}")
        print(f"  Backend: {result['backend_used']}")
        print(f"  Compilation time: {result['compilation_time']:.4f}s")
        print(f"  Sample results: {result['results'][:10]}")

        # Analyze results
        true_count = sum(result["results"])
        print(f"  True/False ratio: {true_count}/{result['shots'] - true_count}")
        print("  Expected ~50/50 for random bit")

        # Demo 2: Bell state with correlation analysis
        print(f"\n2. Running {bell_state.__name__} with guppy_sim() alias")
        result = guppy_sim(
            bell_state,
            shots=200,
            backend="rust",
        )  # Force Rust if available

        # Analyze Bell state correlations
        correlated = sum(1 for r in result["results"] if r[0] == r[1])
        correlation_rate = correlated / result["shots"]
        print(f"  Correlation rate: {correlation_rate:.2%}")
        print("  Expected: ~100% for perfect Bell state")
        print(f"  Sample results: {result['results'][:5]}")

        # Demo 3: Batch execution
        print("\n3. Batch execution with run_guppy_batch()")
        batch_results = run_guppy_batch(
            [random_bit, bell_state, ghz_state],
            shots=50,
            verbose=False,
        )

        print("Batch results:")
        for func_name, result in batch_results.items():
            if "error" in result:
                print(f"  {func_name}: [ERROR] {result['error']}")
            else:
                print(
                    f"  {func_name}: [OK] {result['shots']} shots, backend: {result['backend_used']}",
                )

        # Demo 4: Different backends
        print("\n4. Backend comparison")
        try:
            # Try Rust backend
            rust_result = run_guppy(random_bit, shots=50, backend="rust", verbose=False)
            print(f"  Rust backend: {rust_result['compilation_time']:.4f}s compilation")
        except RuntimeError as e:
            print(f"  Rust backend: Not available ({e})")
        except Exception as e:
            print(f"  Rust backend: Not available (unexpected error: {e})")

        try:
            # Try external backend
            ext_result = run_guppy(
                random_bit,
                shots=50,
                backend="external",
                verbose=False,
            )
            print(
                f"  External backend: {ext_result['compilation_time']:.4f}s compilation",
            )
        except RuntimeError as e:
            print(f"  External backend: Not available ({e})")
        except Exception as e:
            print(f"  External backend: Not available (unexpected error: {e})")

    except ImportError as e:
        print(f"[ERROR] Simple API not available: {e}")
        print("This is expected if dependencies are not installed")


def demo_comparison_with_qasm() -> None:
    """Show how run_guppy() compares to existing PECOS APIs."""
    print("\n=== API Comparison ===")

    print("PECOS QASM API (Old):")
    print("```python")
    print("from pecos_rslib.qasm_sim import qasm_sim")
    print("results = qasm_sim(qasm_code).run(shots=1000)")
    print("```")

    print("\nPECOS Unified API (New):")
    print("```python")
    print("from pecos_rslib import qasm_engine")
    print("from pecos_rslib.programs import QasmProgram")
    print(
        "results = qasm_engine().program(QasmProgram.from_string(qasm_code)).to_sim().run(1000)",
    )
    print("```")

    print("\nPECOS Guppy API (Old - Backward Compatibility):")
    print("```python")
    print("from pecos import run_guppy")
    print("from guppylang import guppy")
    print("")
    print("@guppy")
    print("def my_circuit() -> bool:")
    print("    q = qubit()")
    print("    h(q)")
    print("    return measure(q)")
    print("")
    print("results = run_guppy(my_circuit, shots=1000)")
    print("```")

    print("\nPECOS Guppy API (New - Unified):")
    print("```python")
    print("from pecos_rslib import selene_engine")
    print("from pecos_rslib.programs import HugrProgram")
    print("from pecos.frontends.guppy_frontend import GuppyFrontend")
    print("")
    print("# Convert Guppy function to HUGR")
    print("frontend = GuppyFrontend()")
    print("hugr_bytes = frontend.guppy_to_hugr(my_circuit)")
    print("hugr_program = HugrProgram.from_bytes(hugr_bytes)")
    print("")
    print("# Run simulation")
    print("results = selene_engine().program(hugr_program).to_sim().run(1000)")
    print("```")

    print("\nBoth return similar result dictionaries with:")
    print("- 'results': List of measurement outcomes")
    print("- 'shots': Number of executions")
    print("- Backend information and timing")


def demo_error_handling() -> None:
    """Demonstrate error handling in the simple API."""
    if not DEMO_ENABLED:
        return

    print("\n=== Error Handling Demo ===")

    # Test with non-guppy function
    def regular_function() -> bool:
        return True

    try:
        from pecos import run_guppy

        run_guppy(regular_function, shots=10)
    except ValueError as e:
        print(f"[OK] Correctly caught error for non-@guppy function: {e}")
    except ImportError as e:
        print(f"[WARNING] API not available: {e}")

    # Test with invalid backend
    @guppy
    def simple() -> bool:
        return measure(qubit())

    try:
        run_guppy(simple, shots=10, backend="invalid_backend")
    except ValueError as e:
        print(f"[OK] Backend validation works: {type(e).__name__}")
    except Exception as e:
        print(f"[OK] Backend validation caught unexpected error: {type(e).__name__}")
    except ImportError:
        print("[WARNING] API not available for backend test")


def main() -> None:
    """Run all demos."""
    print("PECOS Simple Guppy API Demo (Backward Compatibility)")
    print("=" * 50)
    print("NOTE: This demonstrates the backward compatibility API.")
    print("      For new code, use the unified selene_engine() API.")
    print("=" * 50)

    demo_simple_api()
    demo_comparison_with_qasm()
    demo_error_handling()

    print("\n" + "=" * 40)
    print("Demo complete!")

    print("\nQuick start guide:")
    print("1. Install: pip install quantum-pecos[guppy]")
    print("2. Import: from pecos import run_guppy")
    print("3. Use: results = run_guppy(my_guppy_function, shots=1000)")


if __name__ == "__main__":
    main()
