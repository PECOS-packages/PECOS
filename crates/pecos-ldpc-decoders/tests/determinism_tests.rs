//! Determinism tests for LDPC decoders
//!
//! These tests ensure that all LDPC decoders provide:
//! 1. Deterministic results with fixed seeds
//! 2. Thread safety in parallel execution
//! 3. Independence between decoder instances
//! 4. Reproducible behavior across different execution patterns

use ndarray::{Array1, arr1};
use pecos_ldpc_decoders::{
    BeliefFindDecoder, BpLsdDecoder, BpMethod, BpOsdDecoder, BpSchedule, FlipDecoder,
    InputVectorType, OsdMethod, SoftInfoBpDecoder, SparseMatrix, UfMethod, UnionFindDecoder,
};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Create a simple test parity check matrix (Hamming 7,4 code)
fn create_hamming_7_4_pcm() -> SparseMatrix {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    // Row 0: columns 0, 2, 4, 6
    row_indices.extend(&[0, 0, 0, 0]);
    col_indices.extend(&[0, 2, 4, 6]);

    // Row 1: columns 1, 2, 5, 6
    row_indices.extend(&[1, 1, 1, 1]);
    col_indices.extend(&[1, 2, 5, 6]);

    // Row 2: columns 3, 4, 5, 6
    row_indices.extend(&[2, 2, 2, 2]);
    col_indices.extend(&[3, 4, 5, 6]);

    SparseMatrix::from_coo(3, 7, row_indices, col_indices).unwrap()
}

/// Create a larger test matrix for stress testing
fn create_large_test_pcm() -> SparseMatrix {
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    // Create a 10x20 sparse matrix
    for row in 0..10 {
        for col in 0..20 {
            if (row + col) % 3 == 0 {
                row_indices.push(row);
                col_indices.push(col);
            }
        }
    }

    SparseMatrix::from_coo(10, 20, row_indices, col_indices).unwrap()
}

fn create_test_syndrome_small() -> Array1<u8> {
    arr1(&[1, 0, 1])
}

fn create_test_syndrome_large() -> Array1<u8> {
    arr1(&[1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
}

// ============================================================================
// BP-OSD Decoder Tests
// ============================================================================

#[test]
fn test_bp_osd_sequential_determinism() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    // Run multiple times with same seed
    for run in 0..15 {
        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(0.1),                 // error_rate
            None,                      // error_channel
            10,                        // max_iter
            BpMethod::ProductSum,      // bp_method
            BpSchedule::Serial,        // bp_schedule (deterministic)
            1.0,                       // ms_scaling_factor
            OsdMethod::Off,            // osd_method
            0,                         // osd_order
            InputVectorType::Syndrome, // input_vector_type
            None,                      // omp_thread_count
            None,                      // serial_schedule_order
            Some(42),                  // random_schedule_seed
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push((result.decoding.clone(), result.converged, result.iterations));

        if run < 3 {
            println!(
                "BP-OSD run {}: decoding={:?}, converged={}, iterations={}",
                run, result.decoding, result.converged, result.iterations
            );
        }
    }

    // All results should be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(first.0, result.0, "BP-OSD run {i} gave different decoding");
        assert_eq!(
            first.1, result.1,
            "BP-OSD run {i} gave different convergence"
        );
        assert_eq!(
            first.2, result.2,
            "BP-OSD run {i} gave different iterations"
        );
    }
}

#[test]
fn test_bp_osd_parallel_independence() {
    const NUM_THREADS: usize = 8;
    const NUM_ITERATIONS: usize = 10;

    let pcm = Arc::new(create_hamming_7_4_pcm());
    let syndrome = Arc::new(create_test_syndrome_small());
    let results = Arc::new(Mutex::new(Vec::new()));

    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for iteration in 0..NUM_ITERATIONS {
                let seed = 100 + i32::try_from(thread_id).expect("thread_id too large"); // Each thread uses unique seed

                let mut decoder = BpOsdDecoder::new(
                    &pcm_clone,
                    Some(0.1),
                    None,
                    10,
                    BpMethod::ProductSum,
                    BpSchedule::Serial,
                    1.0,
                    OsdMethod::Off,
                    0,
                    InputVectorType::Syndrome,
                    None,
                    None,
                    Some(seed),
                )
                .unwrap();

                let result = decoder.decode(&syndrome_clone.view()).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    seed,
                    result.decoding.clone(),
                    result.converged,
                    result.iterations,
                ));

                // Small delay to encourage interleaving
                thread::sleep(Duration::from_micros(10));
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check that each thread got consistent results for its seed
    for thread_id in 0..NUM_THREADS {
        let thread_results: Vec<_> = final_results
            .iter()
            .filter(|(tid, _, _, _, _, _)| *tid == thread_id)
            .collect();

        let first_result = &thread_results[0];
        for (i, result) in thread_results.iter().enumerate() {
            assert_eq!(
                first_result.3, result.3,
                "Thread {thread_id} iteration {i} gave different decoding"
            );
            assert_eq!(
                first_result.4, result.4,
                "Thread {thread_id} iteration {i} gave different convergence"
            );
            assert_eq!(
                first_result.5, result.5,
                "Thread {thread_id} iteration {i} gave different iterations"
            );
        }

        if thread_id < 2 {
            println!(
                "Thread {} (seed {}): consistent across {} iterations",
                thread_id, first_result.2, NUM_ITERATIONS
            );
        }
    }
}

// ============================================================================
// BP-LSD Decoder Tests
// ============================================================================

#[test]
fn test_bp_lsd_sequential_determinism() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    for run in 0..15 {
        let mut decoder = BpLsdDecoder::new(
            &pcm,
            Some(0.1),                 // error_rate
            None,                      // error_channel
            10,                        // max_iter
            BpMethod::ProductSum,      // bp_method
            BpSchedule::Serial,        // bp_schedule
            1.0,                       // ms_scaling_factor
            OsdMethod::OsdE,           // lsd_method
            3,                         // lsd_order
            1,                         // bits_per_step
            InputVectorType::Syndrome, // input_vector_type
            None,                      // omp_thread_count
            None,                      // serial_schedule_order
            Some(42),                  // random_schedule_seed
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push((result.decoding.clone(), result.converged, result.iterations));

        if run < 3 {
            println!(
                "BP-LSD run {}: decoding={:?}, converged={}, iterations={}",
                run, result.decoding, result.converged, result.iterations
            );
        }
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(first.0, result.0, "BP-LSD run {i} gave different decoding");
        assert_eq!(
            first.1, result.1,
            "BP-LSD run {i} gave different convergence"
        );
        assert_eq!(
            first.2, result.2,
            "BP-LSD run {i} gave different iterations"
        );
    }
}

// ============================================================================
// Flip Decoder Tests
// ============================================================================

#[test]
fn test_flip_decoder_determinism() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    for run in 0..15 {
        let mut decoder = FlipDecoder::new(
            &pcm, 10, // max_iter
            5,  // pfreq (perturbation frequency)
            42, // seed for random perturbations
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push((result.decoding.clone(), result.converged, result.iterations));

        if run < 3 {
            println!(
                "Flip run {}: decoding={:?}, converged={}, iterations={}",
                run, result.decoding, result.converged, result.iterations
            );
        }
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Flip decoder run {i} gave different decoding"
        );
        assert_eq!(
            first.1, result.1,
            "Flip decoder run {i} gave different convergence"
        );
        assert_eq!(
            first.2, result.2,
            "Flip decoder run {i} gave different iterations"
        );
    }
}

#[test]
fn test_flip_decoder_different_seeds() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    // Test with seed 42
    let mut decoder42 = FlipDecoder::new(&pcm, 10, 5, 42).unwrap();
    let result42 = decoder42.decode(&syndrome.view()).unwrap();

    // Test with seed 99
    let mut decoder99 = FlipDecoder::new(&pcm, 10, 5, 99).unwrap();
    let result99 = decoder99.decode(&syndrome.view()).unwrap();

    // Test with seed 42 again
    let mut decoder42_again = FlipDecoder::new(&pcm, 10, 5, 42).unwrap();
    let result42_again = decoder42_again.decode(&syndrome.view()).unwrap();

    println!("Seed 42 first: {:?}", result42.decoding);
    println!("Seed 99: {:?}", result99.decoding);
    println!("Seed 42 again: {:?}", result42_again.decoding);

    // Same seed should give same result
    assert_eq!(
        result42.decoding, result42_again.decoding,
        "Same seed gave different results"
    );
}

// ============================================================================
// Soft Info BP Decoder Tests
// ============================================================================

#[test]
fn test_soft_info_bp_determinism() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();
    let cutoff = 0.01;

    let mut results = Vec::new();

    for run in 0..15 {
        let mut decoder = SoftInfoBpDecoder::new(
            &pcm,
            Some(0.1),            // error_rate
            None,                 // error_channel
            10,                   // max_iter
            BpMethod::ProductSum, // bp_method
            1.0,                  // ms_scaling_factor
            None,                 // omp_thread_count
            None,                 // serial_schedule_order
            Some(42),             // random_schedule_seed
        )
        .unwrap();

        // Convert syndrome to soft syndrome (LLRs)
        let soft_syndrome: Vec<f64> = syndrome
            .iter()
            .map(|&s| if s == 1 { -2.0 } else { 2.0 })
            .collect();

        let result = decoder.decode(&soft_syndrome, cutoff, 1.0).unwrap();
        results.push((result.decoding.clone(), result.converged, result.iterations));

        if run < 3 {
            println!(
                "SoftInfo BP run {}: decoding={:?}, converged={}, iterations={}",
                run, result.decoding, result.converged, result.iterations
            );
        }
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "SoftInfo BP run {i} gave different decoding"
        );
        assert_eq!(
            first.1, result.1,
            "SoftInfo BP run {i} gave different convergence"
        );
        assert_eq!(
            first.2, result.2,
            "SoftInfo BP run {i} gave different iterations"
        );
    }
}

// ============================================================================
// BeliefFind Decoder Tests
// ============================================================================

#[test]
fn test_belief_find_determinism() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    for run in 0..15 {
        let mut decoder = BeliefFindDecoder::new(
            &pcm,
            Some(0.1),            // error_rate
            None,                 // error_channel
            10,                   // max_iter
            BpMethod::ProductSum, // bp_method
            1.0,                  // ms_scaling_factor
            BpSchedule::Serial,   // bp_schedule
            None,                 // omp_thread_count
            None,                 // serial_schedule_order
            Some(42),             // random_schedule_seed
            UfMethod::Peeling,    // uf_method
            10,                   // uf_max_iter
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push((result.decoding.clone(), result.converged, result.iterations));

        if run < 3 {
            println!(
                "BeliefFind run {}: decoding={:?}, converged={}, iterations={}",
                run, result.decoding, result.converged, result.iterations
            );
        }
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "BeliefFind run {i} gave different decoding"
        );
        assert_eq!(
            first.1, result.1,
            "BeliefFind run {i} gave different convergence"
        );
        assert_eq!(
            first.2, result.2,
            "BeliefFind run {i} gave different iterations"
        );
    }
}

// ============================================================================
// Union Find Decoder Tests
// ============================================================================

#[test]
fn test_union_find_determinism() {
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    for run in 0..15 {
        let mut decoder = UnionFindDecoder::new(&pcm, UfMethod::Inversion).unwrap();

        // Union Find doesn't use random seeds, should be inherently deterministic
        let result = decoder.decode(&syndrome.view(), &[], 0).unwrap();
        results.push((result.decoding.clone(), result.converged, result.iterations));

        if run < 3 {
            println!(
                "UnionFind run {}: decoding={:?}, converged={}, iterations={}",
                run, result.decoding, result.converged, result.iterations
            );
        }
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "UnionFind run {i} gave different decoding"
        );
        assert_eq!(
            first.1, result.1,
            "UnionFind run {i} gave different convergence"
        );
        assert_eq!(
            first.2, result.2,
            "UnionFind run {i} gave different iterations"
        );
    }
}

// ============================================================================
// Multi-Decoder Independence Tests
// ============================================================================

#[test]
#[allow(clippy::too_many_lines)]
fn test_multi_decoder_independence() {
    // Test that multiple different decoder types can run simultaneously
    // without interfering with each other's determinism

    const NUM_THREADS: usize = 6; // One for each decoder type
    const NUM_ITERATIONS: usize = 5;

    let pcm = Arc::new(create_hamming_7_4_pcm());
    let syndrome = Arc::new(create_test_syndrome_small());
    let results = Arc::new(Mutex::new(Vec::new()));

    let mut handles = vec![];

    // Thread 0: BP-OSD
    {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for i in 0..NUM_ITERATIONS {
                let mut decoder = BpOsdDecoder::new(
                    &pcm_clone,
                    Some(0.1),
                    None,
                    10,
                    BpMethod::ProductSum,
                    BpSchedule::Serial,
                    1.0,
                    OsdMethod::Off,
                    0,
                    InputVectorType::Syndrome,
                    None,
                    None,
                    Some(50),
                )
                .unwrap();

                let result = decoder.decode(&syndrome_clone.view()).unwrap();
                results_clone
                    .lock()
                    .unwrap()
                    .push((0, i, result.decoding.clone()));
                thread::sleep(Duration::from_millis(1));
            }
        });
        handles.push(handle);
    }

    // Thread 1: BP-LSD
    {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for i in 0..NUM_ITERATIONS {
                let mut decoder = BpLsdDecoder::new(
                    &pcm_clone,
                    Some(0.1),
                    None,
                    10,
                    BpMethod::ProductSum,
                    BpSchedule::Serial,
                    1.0,
                    OsdMethod::OsdE,
                    3,
                    1,
                    InputVectorType::Syndrome,
                    None,
                    None,
                    Some(51),
                )
                .unwrap();

                let result = decoder.decode(&syndrome_clone.view()).unwrap();
                results_clone
                    .lock()
                    .unwrap()
                    .push((1, i, result.decoding.clone()));
                thread::sleep(Duration::from_millis(1));
            }
        });
        handles.push(handle);
    }

    // Thread 2: Flip
    {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for i in 0..NUM_ITERATIONS {
                let mut decoder = FlipDecoder::new(&pcm_clone, 10, 5, 52).unwrap();
                let result = decoder.decode(&syndrome_clone.view()).unwrap();
                results_clone
                    .lock()
                    .unwrap()
                    .push((2, i, result.decoding.clone()));
                thread::sleep(Duration::from_millis(1));
            }
        });
        handles.push(handle);
    }

    // Thread 3: SoftInfo BP
    {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for i in 0..NUM_ITERATIONS {
                let mut decoder = SoftInfoBpDecoder::new(
                    &pcm_clone,
                    Some(0.1),
                    None,
                    10,
                    BpMethod::ProductSum,
                    1.0,
                    None,
                    None,
                    Some(53),
                )
                .unwrap();

                let soft_syndrome: Vec<f64> = syndrome_clone
                    .iter()
                    .map(|&s| if s == 1 { -2.0 } else { 2.0 })
                    .collect();

                let result = decoder.decode(&soft_syndrome, 0.01, 1.0).unwrap();
                results_clone
                    .lock()
                    .unwrap()
                    .push((3, i, result.decoding.clone()));
                thread::sleep(Duration::from_millis(1));
            }
        });
        handles.push(handle);
    }

    // Thread 4: BeliefFind
    {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for i in 0..NUM_ITERATIONS {
                let mut decoder = BeliefFindDecoder::new(
                    &pcm_clone,
                    Some(0.1),
                    None,
                    10,
                    BpMethod::ProductSum,
                    1.0,
                    BpSchedule::Serial,
                    None,
                    None,
                    Some(54),
                    UfMethod::Peeling,
                    10,
                )
                .unwrap();

                let result = decoder.decode(&syndrome_clone.view()).unwrap();
                results_clone
                    .lock()
                    .unwrap()
                    .push((4, i, result.decoding.clone()));
                thread::sleep(Duration::from_millis(1));
            }
        });
        handles.push(handle);
    }

    // Thread 5: UnionFind
    {
        let pcm_clone = Arc::clone(&pcm);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for i in 0..NUM_ITERATIONS {
                let mut decoder = UnionFindDecoder::new(&pcm_clone, UfMethod::Inversion).unwrap();
                let result = decoder.decode(&syndrome_clone.view(), &[], 0).unwrap();
                results_clone
                    .lock()
                    .unwrap()
                    .push((5, i, result.decoding.clone()));
                thread::sleep(Duration::from_millis(1));
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check that each decoder type got consistent results
    let decoder_names = [
        "BP-OSD",
        "BP-LSD",
        "Flip",
        "SoftInfo",
        "BeliefFind",
        "UnionFind",
    ];

    for (decoder_id, decoder_name) in decoder_names.iter().enumerate().take(NUM_THREADS) {
        let decoder_results: Vec<_> = final_results
            .iter()
            .filter(|(did, _, _)| *did == decoder_id)
            .collect();

        let first_result = &decoder_results[0].2;
        for (i, (_, _iter, result)) in decoder_results.iter().enumerate() {
            assert_eq!(
                first_result, result,
                "{decoder_name} iteration {i} gave different result"
            );
        }

        println!("{decoder_name}: consistent across {NUM_ITERATIONS} iterations");
    }
}

// ============================================================================
// Stress Tests with Larger Matrices
// ============================================================================

#[test]
fn test_large_matrix_determinism() {
    let pcm = create_large_test_pcm();
    let syndrome = create_test_syndrome_large();

    // Test BP-OSD with larger matrix
    let mut results = Vec::new();

    for _run in 0..10 {
        let mut decoder = BpOsdDecoder::new(
            &pcm,
            Some(0.1),
            None,
            20,
            BpMethod::ProductSum,
            BpSchedule::Serial,
            1.0,
            OsdMethod::Off,
            0,
            InputVectorType::Syndrome,
            None,
            None,
            Some(42),
        )
        .unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push(result.decoding.clone());
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first, result,
            "Large matrix BP-OSD run {i} gave different result"
        );
    }

    println!(
        "Large matrix determinism test passed for {}×{} matrix",
        pcm.rows, pcm.cols
    );
}

#[test]
fn test_seed_isolation_across_decoder_types() {
    // Verify that different decoder types don't interfere with each other's seeding
    let pcm = create_hamming_7_4_pcm();
    let syndrome = create_test_syndrome_small();

    // Run BP-OSD with seed 42
    let mut bp_osd = BpOsdDecoder::new(
        &pcm,
        Some(0.1),
        None,
        10,
        BpMethod::ProductSum,
        BpSchedule::Serial,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None,
        None,
        Some(42),
    )
    .unwrap();
    let bp_osd_result1 = bp_osd.decode(&syndrome.view()).unwrap();

    // Run Flip decoder with seed 99 (different type, different seed)
    let mut flip = FlipDecoder::new(&pcm, 10, 5, 99).unwrap();
    let _flip_result = flip.decode(&syndrome.view()).unwrap();

    // Run BP-OSD again with seed 42 - should get same result as first
    let mut bp_osd2 = BpOsdDecoder::new(
        &pcm,
        Some(0.1),
        None,
        10,
        BpMethod::ProductSum,
        BpSchedule::Serial,
        1.0,
        OsdMethod::Off,
        0,
        InputVectorType::Syndrome,
        None,
        None,
        Some(42),
    )
    .unwrap();
    let bp_osd_result2 = bp_osd2.decode(&syndrome.view()).unwrap();

    assert_eq!(
        bp_osd_result1.decoding, bp_osd_result2.decoding,
        "BP-OSD results changed after running different decoder type"
    );

    println!("Seed isolation test passed - different decoder types don't interfere");
}
