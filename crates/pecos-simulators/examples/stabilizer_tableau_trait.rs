// Example demonstrating the StabilizerTableauSimulator trait
//
// This example shows how different stabilizer simulators can implement
// the same trait interface for accessing tableau information.

use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, SparseStab, StabilizerTableauSimulator};

/// Generic function that works with any stabilizer tableau simulator
fn print_bell_state_tableaux<T>(name: &str, mut sim: T)
where
    T: StabilizerTableauSimulator + CliffordGateable,
{
    println!("=== {name} ===");

    // Create Bell state |00> + |11>
    sim.h(&[QubitId(0)]);
    sim.cx(&[QubitId(0), QubitId(1)]);

    println!("Number of qubits: {}", sim.num_qubits());
    println!("\nStabilizers:");
    println!("{}", sim.stab_tableau());
    println!("Destabilizers:");
    println!("{}", sim.destab_tableau());

    // The full tableau method combines both
    println!("\nFull tableau:");
    println!("{}", sim.full_tableau());
    println!();
}

fn main() {
    // The trait allows us to work with different implementations uniformly
    let sim = SparseStab::new(2);
    print_bell_state_tableaux("Pure Rust Stabilizer Simulator", sim);

    // Future implementations can be added here:
    // let other_sim = OtherStabilizerImpl::new(2);
    // print_bell_state_tableaux("Other Implementation", other_sim);
}
