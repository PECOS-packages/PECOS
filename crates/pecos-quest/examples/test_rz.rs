use pecos_quest::{ArbitraryRotationGateable, CliffordGateable, QuestStateVec};
use std::f64::consts::PI;

fn main() {
    println!("Testing RZ gate behavior");

    // Test 1: Apply RZ(π) to |0⟩
    println!("\nTest 1: RZ(π) on |0⟩");
    let mut sim = QuestStateVec::new(1);
    sim.rz(PI, 0);
    println!("|0⟩ amplitude: {:?}", sim.get_amplitude(0));
    println!("|1⟩ amplitude: {:?}", sim.get_amplitude(1));

    // Test 2: Apply RZ(π) to |1⟩
    println!("\nTest 2: RZ(π) on |1⟩");
    let mut sim = QuestStateVec::new(1);
    sim.x(0); // Create |1⟩
    sim.rz(PI, 0);
    println!("|0⟩ amplitude: {:?}", sim.get_amplitude(0));
    println!("|1⟩ amplitude: {:?}", sim.get_amplitude(1));

    // Test 3: Apply RZ(π) to |+⟩
    println!("\nTest 3: RZ(π) on |+⟩ = (|0⟩ + |1⟩)/√2");
    let mut sim = QuestStateVec::new(1);
    sim.h(0); // Create |+⟩
    println!("Before RZ:");
    println!("|0⟩ amplitude: {:?}", sim.get_amplitude(0));
    println!("|1⟩ amplitude: {:?}", sim.get_amplitude(1));

    sim.rz(PI, 0);
    println!("After RZ(π):");
    println!("|0⟩ amplitude: {:?}", sim.get_amplitude(0));
    println!("|1⟩ amplitude: {:?}", sim.get_amplitude(1));

    // Expected: |+⟩ -> |-⟩ = (|0⟩ - |1⟩)/√2
    let expected_0 = 1.0 / 2.0_f64.sqrt();
    let expected_1 = -1.0 / 2.0_f64.sqrt();
    println!("\nExpected after RZ(π):");
    println!("|0⟩ amplitude: {expected_0}");
    println!("|1⟩ amplitude: {expected_1}");
}
