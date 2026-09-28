//! Run with `cargo run -p pecos-qis --example scheduled_runtime -- soft-rz idle-z`.
#[cfg(feature = "selene")]
#[path = "../tests/support/scheduled_execution.rs"]
mod scheduled_execution;

#[cfg(feature = "selene")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use pecos_engines::runtime_frame::ShotContext;
    use pecos_qis::pecos_qis_ffi_types::{Operation, QuantumOp};
    use scheduled_execution::ScheduledExecutor;
    let runtime = match std::env::args().nth(1).as_deref() {
        None | Some("simple") => pecos_qis::selene_simple_runtime()?,
        Some("soft-rz") => pecos_qis::selene_soft_rz_runtime()?,
        _ => return Err("expected simple or soft-rz".into()),
    };
    // Optional narrow timing consumer; this is not a device-noise profile.
    let mut executor = match std::env::args().nth(2).as_deref() {
        Some("idle-z") => ScheduledExecutor::with_idle_z(
            Box::new(runtime),
            2,
            scheduled_execution::IdleZNoise {
                coherent: 0.25,
                ..Default::default()
            },
        )?,
        None | Some("ideal") => ScheduledExecutor::new(Box::new(runtime), 2)?,
        _ => return Err("expected ideal or idle-z".into()),
    };
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
        "Host {:?}: executed {} native batches, {} feedback results; result 91 = {:?}",
        first.context,
        first.batches.len() + last.batches.len(),
        first.measurements.len() + last.measurements.len(),
        shot.measurements.get(&91)
    );
    if shot.measurements.get(&91) != Some(&true) {
        return Err("unexpected Z-idle smoke-test result".into());
    }
    Ok(())
}
#[cfg(not(feature = "selene"))]
fn main() {
    eprintln!("This example requires the selene feature");
}
