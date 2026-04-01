#!/usr/bin/env python3
"""Demonstration of PECOS namespace organization.

This example shows how the namespace modules make the API more discoverable
and organized.
"""

import pecos_rslib

# Import namespace modules for Example 3 demonstration
from pecos_rslib import engines, noise, quantum


def explore_namespaces() -> None:
    """Show what's available in each namespace."""
    print("PECOS Namespace Organization")
    print("=" * 50)

    # Engines namespace
    print("\n1. ENGINES namespace (pecos_rslib.engines):")
    print("   Available engine builders:")
    for item in dir(pecos_rslib.engines):
        if not item.startswith("_"):
            print(f"     - engines.{item}")

    # Noise namespace
    print("\n2. NOISE namespace (pecos_rslib.noise):")
    print("   Available noise model builders:")
    for item in dir(pecos_rslib.noise):
        if not item.startswith("_"):
            print(f"     - noise.{item}")

    # Quantum namespace
    print("\n3. QUANTUM namespace (pecos_rslib.quantum):")
    print("   Available quantum engine builders:")
    for item in dir(pecos_rslib.quantum):
        if not item.startswith("_"):
            print(f"     - quantum.{item}")

    # Programs namespace
    print("\n4. PROGRAMS namespace (pecos_rslib.programs):")
    print("   Available program types:")
    for item in dir(pecos_rslib.programs):
        if not item.startswith("_") and item[0].isupper():
            print(f"     - programs.{item}")


def namespace_usage_examples() -> None:
    """Show practical usage of namespaces."""
    print("\n\nPractical Namespace Usage")
    print("=" * 50)

    # Example 1: Using engines namespace
    print("\n1. Creating different engines:")
    print("   qasm_eng = pecos_rslib.engines.qasm()")
    print("   llvm_eng = pecos_rslib.engines.llvm()")
    print("   selene_eng = pecos_rslib.engines.selene()")

    # Example 2: Using noise namespace
    print("\n2. Creating noise models:")
    print("   simple_noise = pecos_rslib.noise.general()")
    print("   depol_noise = pecos_rslib.noise.depolarizing()")
    print("   biased_noise = pecos_rslib.noise.biased_depolarizing()")

    # Example 3: Using quantum namespace
    print("\n3. Creating quantum engines:")
    print("   state_vec = pecos_rslib.quantum.state_vector()")
    print("   sparse_stab = pecos_rslib.quantum.sparse_stab()")
    print("   # Alias: pecos_rslib.quantum.sparse_stab()")

    # Example 4: Complete workflow
    print("\n4. Complete workflow with namespaces:")
    print(
        """
    # Import what you need
    from pecos_rslib import engines, noise, quantum, programs

    # Create program
    prog = programs.Qasm.from_string(qasm_code)

    # Build simulation with clear namespace usage
    results = engines.qasm()\\
        .program(prog)\\
        .to_sim()\\
        .seed(42)\\
        .quantum_engine(quantum.sparse_stab())\\
        .noise(noise.depolarizing()
               .with_prep_probability(0.001)
               .with_p1_probability(0.01))\\
        .run(1000)
    """,
    )


def run_example_simulations() -> None:
    """Run actual simulations using namespaces."""
    print("\n\nRunning Example Simulations")
    print("=" * 50)

    # Simple Bell state program
    bell_state = pecos_rslib.programs.Qasm.from_string(
        """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """,
    )

    # Example 1: State vector simulation
    print("\n1. State vector simulation:")
    results = (
        pecos_rslib.engines.qasm()
        .program(bell_state)
        .to_sim()
        .quantum_engine(pecos_rslib.quantum.state_vector())
        .run(1000)
    )
    print(f"   Ran 1000 shots, got {len(results)} results")

    # Example 2: Sparse stabilizer with noise
    print("\n2. Sparse stabilizer with depolarizing noise:")
    results = (
        pecos_rslib.engines.qasm()
        .program(bell_state)
        .to_sim()
        .quantum_engine(pecos_rslib.quantum.sparse_stab())
        .noise(
            pecos_rslib.noise.depolarizing()
            .with_prep_probability(0.001)
            .with_meas_probability(0.001)
            .with_p1_probability(0.002)
            .with_p2_probability(0.01),
        )
        .run(1000)
    )
    print(f"   Ran 1000 shots with noise, got {len(results)} results")

    # Example 3: Using namespace imports for cleaner code
    print("\n3. Using namespace imports:")
    # (imports were moved to top of file)

    # Much cleaner!
    sim = engines.qasm().program(bell_state).to_sim()
    sim.seed(12345)
    sim.quantum_engine(quantum.sparse_stab())  # Using the alias
    sim.noise(noise.general().with_p1_probability(0.001))
    results = sim.run(500)
    print("   Ran 500 shots with imported namespaces")


def compare_with_direct_imports() -> None:
    """Compare namespace usage with direct imports."""
    print("\n\nNamespace vs Direct Import Comparison")
    print("=" * 50)

    print("\nOld style (direct imports):")
    print(
        "  from pecos_rslib import qasm_engine, sparse_stab, depolarizing_noise",
    )
    print("  # Less organized, harder to discover related functions")

    print("\nNew style (namespace imports):")
    print("  from pecos_rslib import engines, quantum, noise")
    print("  # Organized, discoverable, clear categories")

    print("\nBenefit: IDE autocomplete shows related functions:")
    print("  engines.<TAB>  # Shows: qasm, llvm, selene")
    print("  quantum.<TAB>  # Shows: state_vector, sparse_stab, sparse_stab")
    print("  noise.<TAB>    # Shows: general, depolarizing, biased_depolarizing")


if __name__ == "__main__":
    explore_namespaces()
    namespace_usage_examples()
    run_example_simulations()
    compare_with_direct_imports()

    print("\n\nConclusion: Namespaces make the API more discoverable and organized!")
