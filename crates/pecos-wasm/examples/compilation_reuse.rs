//! Compare compilation per shot with fresh instances of reusable compiled code.
//! Run with `cargo run --release -p pecos-wasm --example compilation_reuse`.
use pecos_wasm::{CompiledWasmModule, ForeignObject, WasmForeignObject};
use std::fmt::Write;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut source = String::from(
        "(module (global $n (mut i64) (i64.const 0))
        (func (export \"next\") (result i64)
            global.get $n i64.const 1 i64.add global.set $n global.get $n)",
    );
    for i in 0..1000 {
        write!(
            source,
            "(func (export \"padding{i}\") (param i64) (result i64) local.get 0 i64.const {i} i64.add)"
        )?;
    }
    source.push(')');
    for reuse in [false, true] {
        let mut prepared = None;
        let mut preparation = Duration::ZERO;
        let mut execution = Duration::ZERO;
        for _ in 0..20 {
            let start = Instant::now();
            let mut instance = if reuse {
                if prepared.is_none() {
                    prepared = Some(CompiledWasmModule::new(source.as_bytes())?);
                }
                prepared.as_ref().expect("compiled above").instantiate()?
            } else {
                WasmForeignObject::from_bytes(source.as_bytes())?
            };
            preparation += start.elapsed();
            let start = Instant::now();
            for expected in 1..=100 {
                // Every shot must start with fresh globals, even when code is shared.
                assert_eq!(instance.exec("next", &[])?, [expected]);
            }
            execution += start.elapsed();
        }
        println!("reuse={reuse}: 20 shots, preparation={preparation:?}, 2000 calls={execution:?}");
    }
    Ok(())
}
