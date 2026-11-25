use pecos_wasm::{CompiledWasmModule, ForeignObject};

#[test]
fn compiled_instances_have_independent_limits_and_clocks_survive_teardown() {
    let module = CompiledWasmModule::new(
        br#"(module
            (memory 1)
            (func (export "grow") (result i64) i32.const 1 memory.grow i64.extend_i32_s)
            (func (export "spin") (loop $again br $again)))"#,
    )
    .unwrap();
    let mut small = module.instantiate_with_limits(0.05, Some(65_536)).unwrap();
    let mut large = module.instantiate_with_limits(0.05, Some(131_072)).unwrap();
    assert_eq!(small.exec("grow", &[]).unwrap(), [-1]);
    assert_eq!(large.exec("grow", &[]).unwrap(), [1]);
    small.teardown();
    drop(small);
    drop(module);
    // Dropping the compilation/cache owner or another instance must not stop
    // this instance's clock. An infinite loop must still produce a timeout.
    assert!(large.exec("spin", &[]).is_err());
}

#[test]
fn cached_module_rearms_start_deadline_after_idle_time() {
    let module = CompiledWasmModule::new(
        br#"(module
            (global $n (mut i64) (i64.const 0))
            (func $start (local $i i32)
                (loop $again
                    local.get $i i32.const 1 i32.add local.tee $i
                    i32.const 1000 i32.lt_u br_if $again)
                i64.const 7 global.set $n)
            (start $start)
            (func (export "value") (result i64) global.get $n))"#,
    )
    .unwrap();
    let mut first = module.instantiate_with_limits(0.1, None).unwrap();
    assert_eq!(first.exec("value", &[]).unwrap(), [7]);
    first.teardown();
    drop(first);
    std::thread::sleep(std::time::Duration::from_millis(150));
    let mut second = module.instantiate_with_limits(0.1, None).unwrap();
    assert_eq!(second.exec("value", &[]).unwrap(), [7]);
}
