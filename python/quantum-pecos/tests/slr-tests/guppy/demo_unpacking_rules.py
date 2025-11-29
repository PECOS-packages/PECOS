"""Demonstration of the rule-based unpacking decision tree.

This script shows how the new system provides clear, explainable decisions
about array unpacking, replacing the complex heuristic logic.
"""

from pecos.slr.gen_codes.guppy.ir_analyzer import ArrayAccessInfo
from pecos.slr.gen_codes.guppy.unpacking_rules import should_unpack_array


def demo_scenario(name: str, info: ArrayAccessInfo) -> None:
    """Demonstrate unpacking decision for a scenario."""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"{'='*70}")
    print(f"Array: {info.array_name} (size={info.size}, classical={info.is_classical})")
    print(f"Element accesses: {sorted(info.element_accesses)}")
    print(f"Elements consumed: {sorted(info.elements_consumed)}")
    print(f"Full array accesses: {info.full_array_accesses}")
    print(f"Has operations between: {info.has_operations_between}")
    print(f"Has conditionals between: {info.has_conditionals_between}")
    print()

    # Get decision with verbose output
    should_unpack_array(info, verbose=True)


def main() -> None:
    """Run demonstrations of various scenarios."""
    print("\n" + "=" * 70)
    print("RULE-BASED UNPACKING DECISION TREE DEMONSTRATION")
    print("=" * 70)

    # Scenario 1: Full array measurement
    info1 = ArrayAccessInfo(array_name="q", size=5, is_classical=False)
    info1.full_array_accesses.append(10)
    info1.element_accesses.add(0)  # Also has individual access
    demo_scenario("Full Array Measurement (prevents unpacking)", info1)

    # Scenario 2: Individual quantum measurements
    info2 = ArrayAccessInfo(array_name="q", size=3, is_classical=False)
    info2.element_accesses.add(0)
    info2.element_accesses.add(1)
    info2.elements_consumed.add(0)
    info2.elements_consumed.add(1)
    demo_scenario("Individual Quantum Measurements (requires unpacking)", info2)

    # Scenario 3: Operations after measurement
    info3 = ArrayAccessInfo(array_name="q", size=3, is_classical=False)
    info3.element_accesses.add(0)
    info3.elements_consumed.add(0)
    info3.has_operations_between = True
    demo_scenario("Operations After Measurement (requires unpacking)", info3)

    # Scenario 4: Conditional access
    info4 = ArrayAccessInfo(array_name="c", size=4, is_classical=True)
    info4.element_accesses.add(0)
    info4.element_accesses.add(1)
    info4.has_conditionals_between = True
    demo_scenario("Conditional Element Access (requires unpacking)", info4)

    # Scenario 5: Single element only
    info5 = ArrayAccessInfo(array_name="q", size=5, is_classical=False)
    info5.element_accesses.add(2)
    demo_scenario("Single Element Access (use direct indexing)", info5)

    # Scenario 6: Multiple classical accesses
    info6 = ArrayAccessInfo(array_name="c", size=4, is_classical=True)
    info6.element_accesses.add(0)
    info6.element_accesses.add(1)
    info6.element_accesses.add(2)
    demo_scenario("Multiple Classical Accesses (cleaner when unpacked)", info6)

    # Scenario 7: Partial array usage (high ratio)
    info7 = ArrayAccessInfo(array_name="q", size=5, is_classical=False)
    info7.element_accesses.add(0)
    info7.element_accesses.add(2)
    info7.element_accesses.add(4)
    demo_scenario("Partial Array Usage - High Ratio (60%)", info7)

    # Scenario 8: Partial array usage (low ratio)
    info8 = ArrayAccessInfo(array_name="q", size=10, is_classical=False)
    info8.element_accesses.add(0)
    info8.element_accesses.add(5)
    demo_scenario("Partial Array Usage - Low Ratio (20%)", info8)

    # Scenario 9: No individual access
    info9 = ArrayAccessInfo(array_name="q", size=5, is_classical=False)
    demo_scenario("No Individual Element Access", info9)

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Improvements:")
    print("1. Clear, explainable decisions with reasoning")
    print("2. Explicit rules instead of complex heuristics")
    print("3. Easy to test and validate each rule")
    print("4. Maintainable and extensible")
    print()


if __name__ == "__main__":
    main()
