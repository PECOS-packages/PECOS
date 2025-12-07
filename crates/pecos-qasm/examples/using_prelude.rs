// Using the prelude - all common types are available with one import
use pecos_programs::Qasm;
use pecos_qasm::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // No need to import Shot, ShotVec, ShotMap, or ShotMapDisplayExt
    let mut shot_vec = ShotVec::new();

    for i in 0..4 {
        let mut shot = Shot::default();
        shot.add_register("q", i, 2);
        shot_vec.shots.push(shot);
    }

    // Convert to ShotMap for columnar access
    let shot_map: ShotMap = shot_vec.try_as_shot_map()?;

    // ShotMapDisplayExt trait is in scope
    println!("Default (decimal): {}", shot_map.display());
    println!("Binary: {}", shot_map.display().bitvec_binary());
    println!("Hex: {}", shot_map.display().bitvec_hex());

    // Can also run QASM simulations
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .run(10)?;
    let shot_map = shot_vec.try_as_shot_map()?;

    println!("\nQASM simulation results:");
    println!("{}", shot_map.display().bitvec_binary());

    Ok(())
}
