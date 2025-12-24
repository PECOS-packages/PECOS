"""Demonstration of SLR to Guppy improvements.

This script shows how the three improvements work together to produce
better code generation with precise, element-level analysis.
"""

from pecos.qeclib import qubit
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.data_flow import DataFlowAnalyzer
from pecos.slr.gen_codes.guppy.ir_analyzer import IRAnalyzer


def demo_scenario(name: str, prog: Main, variables: dict) -> None:
    """Demonstrate analysis for a scenario."""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"{'='*70}\n")

    # Run data flow analysis
    data_flow_analyzer = DataFlowAnalyzer()
    data_flow = data_flow_analyzer.analyze(prog, variables)

    # Run IR analysis (which integrates data flow)
    ir_analyzer = IRAnalyzer()
    ir_analyzer.analyze_block(prog, variables)

    # Show results for each array
    for array_name in sorted(variables.keys()):
        if array_name in ir_analyzer.array_info:
            info = ir_analyzer.array_info[array_name]
            print(
                f"\nArray '{array_name}' (size={info.size}, classical={info.is_classical})",
            )
            print(f"  Element accesses: {sorted(info.element_accesses)}")
            print(f"  Elements consumed: {sorted(info.elements_consumed)}")

            # NEW: Show precise conditional tracking
            if hasattr(info, "conditionally_accessed_elements"):
                print(
                    f"  Conditionally accessed: {sorted(info.conditionally_accessed_elements)}",
                )

            # Show data flow insights
            requires_unpacking_flow = data_flow.array_requires_unpacking(array_name)
            requires_unpacking_decision = info.needs_unpacking

            print(f"  Data flow says unpack: {requires_unpacking_flow}")
            print(f"  Decision tree says unpack: {requires_unpacking_decision}")

            if requires_unpacking_decision:
                print("  WILL be unpacked")
            else:
                print("  Will NOT be unpacked")


def main() -> None:
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("SLR TO GUPPY IMPROVEMENTS DEMONSTRATION")
    print("=" * 70)
    print("\nShowing how the three improvements work together:")
    print("1. Rule-Based Decision Tree")
    print("2. Data Flow Analysis")
    print("3. Conditional Refinement")

    # Scenario 1: Syndrome Extraction (False positive eliminated!)
    print("\n" + "=" * 70)
    print("IMPROVEMENT: Syndrome extraction no longer causes false positives")
    print("=" * 70)

    prog1 = Main(
        data := QReg("data", 3),
        ancilla := QReg("ancilla", 2),
        syndrome := CReg("syndrome", 2),
        # Entangle
        qubit.CX(data[0], ancilla[0]),
        qubit.CX(data[1], ancilla[0]),
        # Measure ancillas
        qubit.Measure(ancilla[0]) > syndrome[0],
        qubit.Measure(ancilla[1]) > syndrome[1],
        # Continue using data qubits (different from ancillas!)
        qubit.H(data[0]),
        qubit.H(data[1]),
    )

    demo_scenario(
        "Syndrome Extraction",
        prog1,
        {"data": data, "ancilla": ancilla, "syndrome": syndrome},
    )

    print("\nBEFORE: Would unpack 'data' (false positive)")
    print("AFTER: 'data' NOT unpacked (correct!)")

    # Scenario 2: Partial Conditional (Element-level precision!)
    print("\n" + "=" * 70)
    print("IMPROVEMENT: Only conditionally accessed elements tracked")
    print("=" * 70)

    prog2 = Main(
        q := QReg("q", 4),
        c := CReg("c", 1),
        # Use q[0], q[1], q[2] unconditionally
        qubit.H(q[0]),
        qubit.H(q[1]),
        qubit.H(q[2]),
        qubit.Measure(q[0]) > c[0],
        # Only q[3] is conditional
        If(c[0]).Then(
            qubit.X(q[3]),
        ),
    )

    demo_scenario("Partial Conditional Access", prog2, {"q": q, "c": c})

    print("\nBEFORE: Would mark ALL elements as conditional")
    print("AFTER: Only q[3] marked as conditional (precise!)")

    # Scenario 3: Measure-Prep-Use (Replacement tracked!)
    print("\n" + "=" * 70)
    print("IMPROVEMENT: Prep replacement tracked in data flow")
    print("=" * 70)

    prog3 = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qubit.H(q[0]),
        qubit.Measure(q[0]) > c[0],
        qubit.Prep(q[0]),  # Replacement!
        qubit.H(q[0]),  # Use after replacement - OK!
    )

    demo_scenario("Measure-Prep-Use Pattern", prog3, {"q": q, "c": c})

    print("\nBEFORE: Would unpack 'q' (false positive)")
    print("AFTER: 'q' NOT unpacked because of Prep (correct!)")

    # Scenario 4: Different Element Usage (Element-level tracking!)
    print("\n" + "=" * 70)
    print("IMPROVEMENT: Different elements tracked separately")
    print("=" * 70)

    prog4 = Main(
        q := QReg("q", 3),
        c := CReg("c", 1),
        qubit.Measure(q[0]) > c[0],  # Measure q[0]
        qubit.X(q[1]),  # Use q[1] (different!)
        qubit.H(q[2]),  # Use q[2] (different!)
    )

    demo_scenario("Different Element Usage", prog4, {"q": q, "c": c})

    print("\nBEFORE: Would unpack 'q' (operation after measurement)")
    print("AFTER: 'q' NOT unpacked (different elements!)")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Improvements:")
    print("1. Element-level precision (not array-level)")
    print("2. Data flow tracking (replacement detection)")
    print("3. Conditional refinement (specific indices)")
    print("4. Rule-based decisions (explainable)")
    print()


if __name__ == "__main__":
    main()
