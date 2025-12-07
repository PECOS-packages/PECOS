#!/usr/bin/env python3
"""PECOS Rust HUGR Backend Example.

This example demonstrates the high-performance Rust backend for HUGR compilation
and QIR execution in PECOS.

Features demonstrated:
1. Automatic backend selection (Rust vs external tools)
2. Direct HUGR compilation using Rust
3. QIR engine creation and execution
4. Performance comparison between backends.
"""

import time

# Check availability
try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, h, measure, qubit

    GUPPY_AVAILABLE = True
    print("[OK] Guppy available")
except ImportError:
    GUPPY_AVAILABLE = False
    print("[WARNING] Guppy not available")

try:
    from pecos_rslib import (
        RUST_HUGR_AVAILABLE,
        RustHugrCompiler,
        RustHugrQirEngine,
        check_rust_hugr_availability,
    )

    print("[OK] Rust HUGR backend available")
except ImportError:
    RUST_HUGR_AVAILABLE = False
    print("[WARNING] Rust HUGR backend not available")

try:
    from pecos._compilation import GuppyFrontend

    print("[OK] PECOS Guppy frontend available")
except ImportError:
    print("[WARNING] PECOS Guppy frontend not available")


def example_rust_backend_usage() -> None:
    """Demonstrate direct usage of Rust backend components."""
    if not GUPPY_AVAILABLE or not RUST_HUGR_AVAILABLE:
        print("Skipping Rust backend example - dependencies not available")
        return

    print("\n=== Rust Backend Direct Usage ===")

    # Define a simple quantum function
    @guppy
    def quantum_random() -> bool:
        """Generate a random bit using quantum superposition."""
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    compiled = guppy.compile(quantum_random)
    hugr_bytes = compiled.package.to_bytes()
    print(f"[OK] Compiled to HUGR: {len(hugr_bytes)} bytes")

    # Check Rust backend availability
    available, message = check_rust_hugr_availability()
    print(f"Rust backend status: {available} - {message}")

    if not available:
        print("Cannot proceed with Rust backend demo")
        return

    try:
        # Use Rust compiler directly
        compiler = RustHugrCompiler(debug_info=False, llvm_convention="qir")
        print(
            f"[OK] Created Rust compiler with LLVM convention: {compiler.get_llvm_convention()}",
        )

        # Compile HUGR to QIR
        start_time = time.time()
        qir_code = compiler.compile_bytes_to_qir(hugr_bytes)
        rust_compile_time = time.time() - start_time

        print(f"[OK] Compiled to QIR in {rust_compile_time:.4f}s")
        print(f"QIR length: {len(qir_code)} characters")
        print("QIR preview:")
        print(qir_code[:200] + "..." if len(qir_code) > 200 else qir_code)

        # Create QIR engine
        engine = RustHugrQirEngine(hugr_bytes, shots=1000)
        print(f"[OK] Created QIR engine with {engine.get_shots()} shots")

        # Run (note: this is a placeholder implementation)
        results = engine.run()
        print(f"[OK] Execution completed: {len(results)} results")

    except RuntimeError as e:
        print(f"[ERROR] Rust backend runtime error: {e}")
    except ValueError as e:
        print(f"[ERROR] Rust backend value error: {e}")
    except Exception as e:
        print(f"[ERROR] Rust backend unexpected error: {e}")


def example_frontend_comparison() -> None:
    """Compare Rust backend vs external tools in GuppyFrontend."""
    if not GUPPY_AVAILABLE:
        print("Skipping frontend comparison - Guppy not available")
        return

    print("\n=== Frontend Backend Comparison ===")

    @guppy
    def bell_state() -> tuple[bool, bool]:
        """Create Bell state and measure both qubits."""
        q0 = qubit()
        q1 = qubit()
        h(q0)
        cx(q0, q1)
        return measure(q0), measure(q1)

    # Test with Rust backend (if available)
    if RUST_HUGR_AVAILABLE:
        try:
            frontend_rust = GuppyFrontend(use_rust_backend=True)
            info = frontend_rust.get_backend_info()
            print(f"Rust frontend info: {info}")

            start_time = time.time()
            qir_file_rust = frontend_rust.compile_function(bell_state)
            rust_time = time.time() - start_time

            print(f"[OK] Rust backend compilation: {rust_time:.4f}s")
            print(f"Output file: {qir_file_rust}")

        except RuntimeError as e:
            print(f"[ERROR] Rust backend compilation failed: {e}")
        except Exception as e:
            print(f"[ERROR] Rust backend compilation failed with unexpected error: {e}")

    # Test with external tools (fallback)
    try:
        frontend_external = GuppyFrontend(use_rust_backend=False)
        info = frontend_external.get_backend_info()
        print(f"External frontend info: {info}")

        # This will likely fail without external tools configured
        print("External tools compilation would require hugr-to-llvm binary")

    except ImportError as e:
        print(f"External backend import error: {e}")
    except Exception as e:
        print(f"External backend setup error: {e}")


def example_performance_benefits() -> None:
    """Demonstrate performance benefits of Rust backend."""
    if not GUPPY_AVAILABLE or not RUST_HUGR_AVAILABLE:
        print("Skipping performance demo - dependencies not available")
        return

    print("\n=== Performance Benefits ===")

    @guppy
    def larger_circuit() -> tuple[bool, bool, bool, bool]:
        """Larger quantum circuit for performance testing."""
        qubits = [qubit() for _ in range(4)]

        # Apply Hadamards
        for q in qubits:
            h(q)

        # Apply some entangling gates
        cx(qubits[0], qubits[1])
        cx(qubits[1], qubits[2])
        cx(qubits[2], qubits[3])

        # Measure all
        return tuple(measure(q) for q in qubits)  # type: ignore[arg-type]

    # Compile to HUGR once
    compiled = guppy.compile(larger_circuit)
    hugr_bytes = compiled.package.to_bytes()
    print(f"Circuit compiled to {len(hugr_bytes)} byte HUGR")

    # Test Rust backend performance
    compiler = RustHugrCompiler()

    # Warm up
    compiler.compile_bytes_to_qir(hugr_bytes)

    # Benchmark
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        qir = compiler.compile_bytes_to_qir(hugr_bytes)
    rust_total_time = time.time() - start_time

    print(f"[OK] Rust backend: {num_runs} compilations in {rust_total_time:.4f}s")
    print(f"  Average: {rust_total_time/num_runs:.4f}s per compilation")
    print(f"  QIR size: {len(qir)} characters")

    # Performance characteristics
    print("\nRust Backend Advantages:")
    print("- No subprocess overhead")
    print("- No temporary file I/O")
    print("- Direct memory operations")
    print("- Optimized HUGR parsing")
    print("- Integrated error handling")


def main() -> None:
    """Run all examples."""
    print("PECOS Rust HUGR Backend Examples")
    print("=" * 50)

    # Show availability status
    print(f"Guppy available: {GUPPY_AVAILABLE}")
    print(f"Rust HUGR backend available: {RUST_HUGR_AVAILABLE}")

    if RUST_HUGR_AVAILABLE:
        _available, message = check_rust_hugr_availability()
        print(f"Backend status: {message}")

    # Run examples
    example_rust_backend_usage()
    example_frontend_comparison()
    example_performance_benefits()

    print("\n" + "=" * 50)
    print("Examples complete!")

    if not RUST_HUGR_AVAILABLE:
        print("\nTo enable Rust backend:")
        print("1. Build PECOS with HUGR support:")
        print("   cd python/pecos-rslib && cargo build --features hugr")
        print("2. Install with HUGR support:")
        print("   pip install -e .")


if __name__ == "__main__":
    main()
