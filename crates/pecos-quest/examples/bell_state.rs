//! Example: Creating and measuring a Bell state using `QuEST` with PECOS-style API

use pecos_quest::{CliffordGateable, QuantumSimulator, QuestStateVec};

fn main() {
    println!("QuEST Bell State Example");
    println!("========================");

    // Create a 2-qubit quantum state vector
    let mut state = QuestStateVec::new(2);
    println!("Created {} qubit state vector", state.num_qubits());

    // Explicitly reset the state to make sure it's initialized
    state.reset();
    println!("Reset state explicitly");
    println!();

    // Display initial state probabilities
    println!("Initial state |00⟩:");
    display_state_probabilities(&state);

    // Check individual probabilities
    println!("  Probability |00⟩: {:.6}", state.probability(0b00));
    println!("  Probability |01⟩: {:.6}", state.probability(0b01));
    println!("  Probability |10⟩: {:.6}", state.probability(0b10));
    println!("  Probability |11⟩: {:.6}", state.probability(0b11));

    let amp00 = state.get_amplitude(0b00);
    let amp01 = state.get_amplitude(0b01);
    println!("  Amplitude |00⟩: {:.6} + {:.6}i", amp00.re, amp00.im);
    println!("  Amplitude |01⟩: {:.6} + {:.6}i", amp01.re, amp01.im);
    println!();

    // Create Bell state: (|00⟩ + |11⟩)/√2
    println!("Creating Bell state...");
    state.h(0); // Apply Hadamard to qubit 0
    println!("Applied Hadamard to qubit 0");

    state.cx(0, 1); // Apply CNOT with control=0, target=1
    println!("Applied CNOT(0, 1)");
    println!();

    // Display Bell state probabilities
    println!("Bell state probabilities:");
    display_state_probabilities(&state);
    println!();

    // Display the state amplitudes
    println!("Bell state amplitudes:");
    for i in 0..4 {
        let amp = state.get_amplitude(i);
        let prob = amp.norm_sqr();
        println!(
            "  |{:02b}⟩: {:.3} + {:.3}i (prob = {:.3})",
            i, amp.re, amp.im, prob
        );
    }
    println!();

    // Measure the qubits and demonstrate entanglement correlation
    println!("Measuring qubits to demonstrate entanglement:");

    // Create multiple copies to demonstrate correlation
    for measurement_round in 1..=5 {
        // Reset and recreate Bell state for each measurement
        let mut measurement_state: QuestStateVec = QuestStateVec::with_seed(2, measurement_round);
        measurement_state.h(0).cx(0, 1);

        let result0 = measurement_state.mz(0);
        let result1 = measurement_state.mz(1);

        println!(
            "  Round {}: Qubit 0: {} | Qubit 1: {} | Correlated: {}",
            measurement_round,
            if result0.outcome { "1" } else { "0" },
            if result1.outcome { "1" } else { "0" },
            if result0.outcome == result1.outcome {
                "✓"
            } else {
                "✗"
            }
        );
    }
    println!();

    // Demonstrate other PECOS-style operations
    println!("Demonstrating other quantum operations:");

    // Reset and apply different gates
    state.reset();
    println!("Reset to |00⟩");

    // Create |++⟩ state
    state.h(0).h(1);
    println!("Applied H⊗H to create |++⟩");
    println!("Probability of |00⟩: {:.3}", state.probability(0b00));
    println!("Probability of |01⟩: {:.3}", state.probability(0b01));
    println!("Probability of |10⟩: {:.3}", state.probability(0b10));
    println!("Probability of |11⟩: {:.3}", state.probability(0b11));
    println!();

    // Apply some Pauli gates
    state.reset();
    state.x(0); // |10⟩
    println!("Applied X(0) to create |10⟩");
    println!("Probability of |10⟩: {:.3}", state.probability(0b01));

    state.z(0); // Add phase to |10⟩
    println!("Applied Z(0) (adds phase, probability unchanged)");
    println!("Probability of |10⟩: {:.3}", state.probability(0b01));
    println!();

    // Demonstrate method chaining
    println!("Demonstrating method chaining:");
    state.reset().h(0).cx(0, 1).z(1);
    println!("Applied: reset().h(0).cx(0,1).z(1)");
    display_state_probabilities(&state);
}

fn display_state_probabilities(state: &QuestStateVec) {
    let num_states = 1 << state.num_qubits();
    for i in 0..num_states {
        let prob = state.probability(i);
        if prob > 1e-10 {
            // Only show non-zero probabilities
            println!(
                "  |{:0width$b}⟩: {:.6}",
                i,
                prob,
                width = state.num_qubits()
            );
        }
    }
}
