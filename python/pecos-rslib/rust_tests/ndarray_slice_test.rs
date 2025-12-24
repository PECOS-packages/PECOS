//! Test to understand ndarray's Slice behavior with negative steps

use ndarray::{Array1, Axis, Slice, s};

#[test]
fn test_ndarray_negative_step_slicing() {
    println!("\n{}", "=".repeat(60));
    println!("Testing ndarray Slice with negative steps");
    println!("{}\n", "=".repeat(60));

    // Create a simple 1D array
    let arr = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    println!("Original array: {arr:?}\n");

    // Test 1: Using the s![] macro with [::-1] equivalent
    println!("Test 1: s![..;-1] (reverse entire array)");
    let slice1 = arr.slice(s![..;-1]);
    println!("  Result: {slice1:?}");
    println!("  Expected: [3.0, 2.0, 1.0, 0.0]");
    println!("  Match: {}\n", slice1.to_vec() == vec![3.0, 2.0, 1.0, 0.0]);

    // Test 2: Using Slice::new with what Python gives us
    println!("Test 2: Slice::new(3, Some(-1), -1) - Python's slice.indices(4) for [::-1]");
    let slice2_info = Slice::new(3, Some(-1), -1);
    let slice2 = arr.slice_axis(Axis(0), slice2_info);
    println!("  Result: {slice2:?}");
    println!("  Expected: [3.0, 2.0, 1.0, 0.0]");
    println!("  Match: {}\n", slice2.to_vec() == vec![3.0, 2.0, 1.0, 0.0]);

    // Test 3: What about None for end?
    println!("Test 3: Slice::new(3, None, -1)");
    let slice3_info = Slice::new(3, None, -1);
    let slice3 = arr.slice_axis(Axis(0), slice3_info);
    println!("  Result: {slice3:?}");
    println!("  Expected: [3.0, 2.0, 1.0, 0.0]");
    println!("  Match: {}\n", slice3.to_vec() == vec![3.0, 2.0, 1.0, 0.0]);

    // Test 4: Try start=-1, end=None, step=-1
    println!("Test 4: Slice::new(-1, None, -1) - start from last element");
    let slice4_info = Slice::new(-1, None, -1);
    let slice4 = arr.slice_axis(Axis(0), slice4_info);
    println!("  Result: {slice4:?}\n");

    // Test 5: What does s![3..;-1] give us?
    println!("Test 5: s![3..;-1] - start at index 3, step backward");
    let slice5 = arr.slice(s![3..;-1]);
    println!("  Result: {slice5:?}\n");

    // Test 6: What about s![3..0;-1]?
    // This is intentionally testing reversed/empty ranges to understand ndarray behavior
    println!("Test 6: s![3..0;-1] - start at 3, end before 0, step backward");
    #[allow(clippy::reversed_empty_ranges)]
    let slice6 = arr.slice(s![3..0;-1]);
    println!("  Result: {slice6:?}\n");

    // Test 7: Check what 0 as end actually means
    println!("Test 7: Slice::new(3, Some(0), -1) - end at index 0 (exclusive)");
    let slice7_info = Slice::new(3, Some(0), -1);
    let slice7 = arr.slice_axis(Axis(0), slice7_info);
    println!("  Result: {slice7:?}");
    println!("  Expected: [3.0, 2.0, 1.0]\n");

    // Test 8: Try various negative end values
    // NOTE: ndarray panics with overflow for end values <= -5 with negative steps
    println!("Test 8: Try various negative end values");
    for end in [-1, -2, -3, -4] {
        println!("  Slice::new(3, Some({end}), -1)");
        let test_slice_info = Slice::new(3, Some(end), -1);
        let slice = arr.slice_axis(Axis(0), test_slice_info);
        println!("    Result: {slice:?}");
    }
    println!("  Slice::new(3, Some(-5), -1) and beyond: Skipped (ndarray overflow)");
    println!();

    // Test 9: What if we use a very negative number?
    // NOTE: ndarray panics with "attempt to subtract with overflow" for very negative
    // end values, so we skip this test. This is an ndarray limitation, not our bug.
    println!("Test 9: Slice::new(3, Some(-10), -1) - very negative end");
    println!("  Skipped: ndarray doesn't handle very negative end values with negative steps\n");

    println!("{}", "=".repeat(60));
}
