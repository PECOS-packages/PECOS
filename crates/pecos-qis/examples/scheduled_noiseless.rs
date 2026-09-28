//! Run with `cargo run -p pecos-qis --example scheduled_noiseless -- soft-rz`.
#[cfg(feature = "selene")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pecos_engines::runtime_frame::ShotContext;
    use pecos_qis::{
        pecos_qis_ffi_types::{Operation, QuantumOp},
        scheduled_execution::NoiselessScheduledExecutor,
    };
    let runtime = match std::env::args().nth(1).as_deref() {
        None | Some("simple") => pecos_qis::selene_simple_runtime()?,
        Some("soft-rz") => pecos_qis::selene_soft_rz_runtime()?,
        _ => return Err("expected simple or soft-rz".into()),
    };
    let mut executor = NoiselessScheduledExecutor::new(Box::new(runtime), 2)?;
    executor.start_shot(
        ShotContext {
            run: 1,
            worker: 0,
            shot: 0,
        },
        42,
        123,
    )?;
    let first = executor.submit(&[
        Operation::AllocateQubit { id: 7 },
        QuantumOp::RXY(std::f64::consts::PI, 0.0, 7).into(),
        QuantumOp::Measure(7, 91).into(),
    ])?;
    let (last, shot) = executor.finish_shot()?;
    println!(
        "Executed {} native batches; result 91 = {:?}",
        first.batches.len() + last.batches.len(),
        shot.measurements.get(&91)
    );
    if shot.measurements.get(&91) != Some(&true) {
        return Err("unexpected noiseless result".into());
    }
    Ok(())
}
#[cfg(not(feature = "selene"))]
fn main() {
    eprintln!("This example requires the selene feature");
}
