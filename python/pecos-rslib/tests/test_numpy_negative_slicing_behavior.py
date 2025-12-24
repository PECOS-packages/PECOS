"""Test script to understand exactly how NumPy handles negative step slicing.

This will help us implement matching behavior in PecosArray.
"""

import numpy as np


def test_numpy_slicing_behavior():
    """Explore NumPy's behavior with negative step slicing."""
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    print(f"Original array: {arr}\n")

    # Test 1: Basic reverse with [::-1]
    print("=" * 60)
    print("Test 1: arr[::-1] - basic reverse")
    result = arr[::-1]
    print(f"  Result: {result}")
    print("  Expected: [3.0, 2.0, 1.0, 0.0]")
    print(f"  Match: {np.array_equal(result, [3.0, 2.0, 1.0, 0.0])}\n")

    # Test 2: What does slice.indices() give us for [::-1]?
    print("=" * 60)
    print("Test 2: slice(None, None, -1).indices(4)")
    s = slice(None, None, -1)
    indices = s.indices(4)
    print(f"  Result: {indices}")
    print(f"  This gives: start={indices[0]}, stop={indices[1]}, step={indices[2]}\n")

    # Test 3: Try to manually use those indices
    print("=" * 60)
    print(f"Test 3: arr[{indices[0]}:{indices[1]}:{indices[2]}]")
    result = arr[indices[0] : indices[1] : indices[2]]
    print(f"  Result: {result}")
    print(f"  Match [::-1]: {np.array_equal(result, arr[::-1])}\n")

    # Test 4: What about very negative stop values?
    print("=" * 60)
    print("Test 4: Very negative stop values")
    for stop in [-1, -2, -3, -4, -5, -10, -100]:
        s = slice(3, stop, -1)
        result = arr[s]
        print(f"  arr[3:{stop}:-1] = {result}")
    print()

    # Test 5: What does slice.indices() give for various negative stops?
    print("=" * 60)
    print("Test 5: slice.indices() for various negative stops")
    for stop in [-1, -2, -3, -4, -5, -10, -100]:
        s = slice(3, stop, -1)
        indices = s.indices(4)
        result = arr[s]
        print(f"  slice(3, {stop}, -1).indices(4) = {indices}")
        print(f"    arr[3:{stop}:-1] = {result}")
    print()

    # Test 6: Start from end with -1
    print("=" * 60)
    print("Test 6: slice(-1, None, -1)")
    s = slice(-1, None, -1)
    result = arr[s]
    indices = s.indices(4)
    print(f"  arr[-1::-1] = {result}")
    print(f"  slice.indices(4) = {indices}\n")

    # Test 7: What about stop=None?
    print("=" * 60)
    print("Test 7: slice(3, None, -1)")
    s = slice(3, None, -1)
    result = arr[s]
    indices = s.indices(4)
    print(f"  arr[3::-1] = {result}")
    print(f"  slice.indices(4) = {indices}\n")

    # Test 8: Understand the pattern for "go to beginning"
    print("=" * 60)
    print("Test 8: Pattern for 'go to beginning' with negative step")
    print(f"  arr[::-1] = {arr[::-1]}")
    print(f"  slice(None, None, -1).indices(4) = {slice(None, None, -1).indices(4)}")
    print(f"  arr[3::-1] = {arr[3::-1]}")
    print(f"  slice(3, None, -1).indices(4) = {slice(3, None, -1).indices(4)}")

    # Key insight: when stop is None with negative step, indices() returns stop=-1
    # But arr[3:-1:-1] gives empty array!
    print(f"\n  BUT: arr[3:-1:-1] = {arr[3:-1:-1]}  <- This is EMPTY!")
    print(f"  slice(3, -1, -1).indices(4) = {slice(3, -1, -1).indices(4)}")
    print()

    # Test 9: The magic value for "go to beginning"
    print("=" * 60)
    print("Test 9: Finding the magic stop value")
    print("  When using negative step to go to beginning:")
    print(
        f"    slice(3, None, -1).indices(4) gives stop={slice(3, None, -1).indices(4)[1]}"
    )
    print("  But we can't use -1 directly in arr[3:-1:-1]")
    print("  We need a value that means 'before index 0'")
    print(f"    arr[3:-5:-1] = {arr[3:-5:-1]}")
    print(f"    slice(3, -5, -1).indices(4) = {slice(3, -5, -1).indices(4)}")
    print()

    # Test 10: Does NumPy ever raise errors?
    print("=" * 60)
    print("Test 10: Does NumPy raise errors for extreme values?")
    try:
        result = arr[3:-1000:-1]
        print(f"  arr[3:-1000:-1] = {result}")
        print(f"  slice(3, -1000, -1).indices(4) = {slice(3, -1000, -1).indices(4)}")
        print("  No error - NumPy handles extreme negative values gracefully")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

    # Test 11: Understanding the -1 special case
    print("=" * 60)
    print("Test 11: Understanding why arr[3:-1:-1] is empty")
    print("  Negative indices are relative to end:")
    print(f"    -1 means index {4-1} = 3")
    print(f"    -2 means index {4-2} = 2")
    print("  So arr[3:-1:-1] means arr[3:3:-1] which is empty (start==stop)")
    print(f"    arr[3:3:-1] = {arr[3:3:-1]}")
    print()

    # Test 12: The actual conversion rule
    print("=" * 60)
    print("Test 12: The conversion rule from slice.indices()")
    print("  slice.indices(length) normalizes slice parameters")
    print("  For negative step, it converts None to appropriate values:")
    s = slice(None, None, -1)
    print(f"    slice(None, None, -1).indices(4) = {s.indices(4)}")
    print("    Meaning: start at index 3, stop before index -1")
    print("  But stop=-1 in the result is NOT a Python index!")
    print("  It's a sentinel meaning 'go past index 0'")
    print()


if __name__ == "__main__":
    test_numpy_slicing_behavior()
