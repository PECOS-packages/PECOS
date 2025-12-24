"""Example demonstrating the new namespace-based API for PECOS.

This example shows how to use the namespace modules for better discoverability
and cleaner code organization.
"""

import pecos_rslib


def main() -> None:
    print("PECOS Namespace API Example")
    print("=" * 40)

    # 1. Using the engines namespace
    print("\n1. Engine builders via namespace:")
    print("   pecos_rslib.engines.qasm()")
    print("   pecos_rslib.engines.llvm()")
    print("   pecos_rslib.engines.selene()")

    # 2. Using the quantum namespace
    print("\n2. Quantum engine builders via namespace:")
    print("   pecos_rslib.quantum.state_vector()")
    print("   pecos_rslib.quantum.sparse_stabilizer()")
    print("   pecos_rslib.quantum.sparse_stab()  # alias")

    # 3. Using the noise namespace
    print("\n3. Noise model builders via namespace:")
    print("   pecos_rslib.noise.general()")
    print("   pecos_rslib.noise.depolarizing()")
    print("   pecos_rslib.noise.biased_depolarizing()")

    # 4. Complete example: Bell state with noise
    print("\n4. Running a complete example:")

    # Create a Bell state QASM program
    qasm_code = """
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[2];
    creg c[2];

    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """

    # Create program
    program = pecos_rslib.programs.Qasm.from_string(qasm_code)

    # Configure depolarizing noise
    noise_model = (
        pecos_rslib.noise.depolarizing()
        .with_prep_probability(0.001)  # State preparation errors
        .with_meas_probability(0.005)  # Measurement errors
        .with_p1_probability(0.002)  # Single-qubit gate errors
        .with_p2_probability(0.01)  # Two-qubit gate errors
    )

    # Run simulation using namespace API
    results = (
        pecos_rslib.engines.qasm()
        .program(program)
        .to_sim()
        .seed(42)  # For reproducibility
        .workers(4)  # Use 4 threads
        .quantum_engine(pecos_rslib.quantum.sparse_stabilizer())
        .noise(noise_model)
        .run(1000)
    )

    print(f"   Simulation complete! Got {len(results)} shots")
    print(f"   Result type: {type(results).__name__}")

    # 5. Alternative: Direct imports still work
    print("\n5. Direct imports are still available:")
    print("   from pecos_rslib import qasm_engine, sparse_stabilizer")

    # 6. Class-based instantiation
    print("\n6. Direct class instantiation:")
    print("   builder = pecos_rslib.engines.QasmEngineBuilder()")
    print("   quantum = pecos_rslib.quantum.StateVectorBuilder()")
    print("   noise = pecos_rslib.noise.GeneralNoiseModelBuilder()")


if __name__ == "__main__":
    main()
