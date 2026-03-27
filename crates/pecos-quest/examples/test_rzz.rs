use pecos_core::{Angle64, QubitId, qid};
use pecos_quest::{ArbitraryRotationGateable, CliffordGateable, QuestStateVec};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

fn main() {
    println!("Testing RZZ gate behavior");

    // Test RZZ(π/2) on |11⟩
    println!("\nTest: RZZ(π/2) on |11⟩");
    let mut sim = QuestStateVec::new(2);

    // Prepare |11⟩ state
    sim.x(&qid(0)).x(&qid(1));
    println!("Initial |11⟩ amplitude: {:?}", sim.get_amplitude(0b11));

    // Apply RZZ(π/2)
    sim.rzz(
        Angle64::from_radians(FRAC_PI_2),
        &[(QubitId(0), QubitId(1))],
    );
    println!("After RZZ(π/2):");
    println!("|00⟩ amplitude: {:?}", sim.get_amplitude(0b00));
    println!("|01⟩ amplitude: {:?}", sim.get_amplitude(0b01));
    println!("|10⟩ amplitude: {:?}", sim.get_amplitude(0b10));
    println!("|11⟩ amplitude: {:?}", sim.get_amplitude(0b11));

    // Check the phase
    let amp11 = sim.get_amplitude(0b11);
    let phase = amp11.im.atan2(amp11.re);
    let magnitude = (amp11.re * amp11.re + amp11.im * amp11.im).sqrt();
    println!("\n|11⟩ magnitude: {magnitude}");
    println!("|11⟩ phase: {} (in units of π: {})", phase, phase / PI);
    println!("Expected phase -π/4 = {}", -FRAC_PI_4);
}
