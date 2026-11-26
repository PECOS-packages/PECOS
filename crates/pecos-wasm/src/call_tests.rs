use crate::{CompiledWasmModule, ForeignObject, WasmForeignObject, WasmScalarType as Ty};
use pecos_core::errors::PecosError;

fn object(source: &str) -> WasmForeignObject {
    WasmForeignObject::from_bytes_with_limits(source.as_bytes(), 0.2, None).unwrap()
}

fn generic(result: Result<Vec<i64>, PecosError>, expected: &str) {
    let Err(PecosError::Generic(message)) = result else {
        panic!("expected Generic({expected:?}), got {result:?}");
    };
    assert_eq!(message, expected);
}

fn input<T: std::fmt::Debug>(result: Result<T, PecosError>, expected: &str) {
    let Err(PecosError::Input(message)) = result else {
        panic!("expected Input({expected:?}), got {result:?}");
    };
    assert_eq!(message, expected);
}

#[test]
fn cached_functions_follow_replaced_instance() {
    let mut wasm = object(
        r#"(module
        (global $value (mut i64) (i64.const 10))
        (func (export "next") (result i64)
            global.get $value i64.const 1 i64.add global.set $value global.get $value))"#,
    );
    assert_eq!(wasm.exec("next", &[]).unwrap(), [11]);
    assert_eq!(wasm.exec("next", &[]).unwrap(), [12]);
    wasm.new_instance().unwrap();
    assert_eq!(wasm.exec("next", &[]).unwrap(), [11]);
    wasm.new_instance().unwrap();
    assert_eq!(
        wasm.exec_scalar_bits("next", &[], &[Ty::I64], &[]).unwrap(),
        [11]
    );
}

#[test]
fn legacy_call_outcomes_are_preserved() {
    let mut wasm = object(
        r#"(module
        (memory (export "memory") 1)
        (func (export "int") (param i32) (result i32) local.get 0)
        (func (export "float") (param f64))
        (func (export "float32") (param f32))
        (func (export "negative") (result i32) i32.const -7)
        (func (export "result64") (result f64) f64.const 3.5)
        (func (export "result32") (result f32) f32.const 2.5)
        (func (export "void")))"#,
    );
    for _ in 0..2 {
        for name in ["missing", "memory"] {
            let error = wasm.exec(name, &[]).unwrap_err();
            let PecosError::Resource(message) = error else {
                panic!("{error:?}")
            };
            assert_eq!(message, format!("WebAssembly function '{name}' not found"));
        }
        generic(
            wasm.exec("int", &[1, 2]),
            "WebAssembly function 'int' failed: expected 1 arguments, got 2",
        );
        generic(
            wasm.exec("int", &[]),
            "WebAssembly function 'int' failed: expected 1 arguments, got 0",
        );
        // Wrong arity takes precedence even when the declared parameter is unsupported.
        generic(
            wasm.exec("float", &[]),
            "WebAssembly function 'float' failed: expected 1 arguments, got 0",
        );
        generic(
            wasm.exec("float", &[0]),
            "WebAssembly function 'float' failed: argument type mismatch",
        );
        generic(
            wasm.exec("float32", &[0]),
            "WebAssembly function 'float32' failed: argument type mismatch",
        );
        assert_eq!(
            wasm.exec("int", &[i64::MAX]).unwrap(),
            [i64::from(i32::MAX)]
        );
        assert_eq!(
            wasm.exec("int", &[i64::MIN]).unwrap(),
            [i64::from(i32::MIN)]
        );
        assert_eq!(wasm.exec("negative", &[]).unwrap(), [-7]);
        assert_eq!(wasm.exec("result64", &[]).unwrap(), [0]);
        assert_eq!(wasm.exec("result32", &[]).unwrap(), [0]);
        assert_eq!(wasm.exec("void", &[]).unwrap(), [0]);
    }
}

#[test]
fn unchecked_scalar_lanes_preserve_bits_and_validation() {
    let mut wasm = object(
        r#"(module
        (func (export "int") (param i32) (result i32) local.get 0)
        (func (export "float") (param f64) (result f64) local.get 0)
        (func (export "mixed") (param i32 i64) (result i32 i64) local.get 0 local.get 1)
        (func (export "void"))
        (func (export "float32") (param f32) (result f32) local.get 0))"#,
    );
    for _ in 0..2 {
        assert_eq!(
            wasm.exec_scalar_bits("int", &[Ty::I32], &[Ty::I32], &[-7])
                .unwrap(),
            [-7]
        );
        assert_eq!(
            wasm.exec_scalar_bits("int", &[Ty::I32], &[Ty::I32], &[0x1_ffff_fff9])
                .unwrap(),
            [-7]
        );
        for bits in [
            0x8000_0000_0000_0000_u64,
            0x7ff8_0123_4567_89ab,
            3.5_f64.to_bits(),
        ] {
            let bits = i64::from_ne_bytes(bits.to_ne_bytes());
            assert_eq!(
                wasm.exec_scalar_bits("float", &[Ty::F64], &[Ty::F64], &[bits])
                    .unwrap(),
                [bits]
            );
        }
        assert_eq!(
            wasm.exec_scalar_bits(
                "mixed",
                &[Ty::IntegerAuto; 2],
                &[Ty::IntegerAuto; 2],
                &[-7, i64::MIN]
            )
            .unwrap(),
            [-7, i64::MIN]
        );
        assert_eq!(
            wasm.exec_scalar_bits("void", &[], &[], &[]).unwrap(),
            Vec::<i64>::new()
        );
        for (name, inputs, outputs, args) in [
            ("int", vec![Ty::I64], vec![Ty::I32], vec![1]),
            ("int", vec![Ty::I32], vec![Ty::F64], vec![1]),
            ("int", vec![Ty::I32], vec![Ty::I32], vec![]),
            ("int", vec![], vec![Ty::I32], vec![]),
            ("int", vec![Ty::I32], vec![], vec![1]),
            ("float32", vec![Ty::F64], vec![Ty::F64], vec![1]),
        ] {
            input(
                wasm.exec_scalar_bits(name, &inputs, &outputs, &args),
                &format!("WASM scalar signature mismatch for {name}"),
            );
        }
    }
}

#[test]
fn legacy_non_numeric_results_use_checked_marshalling() {
    let mut wasm = object(
        r#"(module
        (func (export "function") (param i64) (result funcref) ref.null func)
        (func (export "external") (param i32) (result externref) ref.null extern)
        (func (export "vector") (param i64) (result v128) v128.const i32x4 1 2 3 4)
        (func (export "reference_param") (param funcref)))"#,
    );
    for _ in 0..2 {
        for name in ["function", "external", "vector"] {
            let before = wasm.checked_fallback_calls();
            assert_eq!(wasm.exec(name, &[7]).unwrap(), [0]);
            assert_eq!(wasm.checked_fallback_calls(), before + 1);
        }
        generic(
            wasm.exec("reference_param", &[0]),
            "WebAssembly function 'reference_param' failed: argument type mismatch",
        );
    }
}

#[test]
fn cached_export_queries_keep_errors_and_positions() {
    let mut wasm = object(
        r#"(module
        (memory (export "memory") 1)
        (func (export "int") (param i64) (result i64) local.get 0)
        (global (export "global") i64 (i64.const 0))
        (func (export "float") (param f64) (result f64) local.get 0))"#,
    );
    for _ in 0..3 {
        assert_eq!(wasm.get_funcs(), ["int", "float"]);
        assert!(wasm.has_func("int"));
        assert!(!wasm.has_func("memory"));
        assert!(!wasm.has_func("missing"));
        wasm.validate_i64_signature("int", 1, 1).unwrap();
        for (name, inputs, outputs) in [("int", 0, 1), ("int", 1, 0), ("float", 1, 1)] {
            input(
                wasm.validate_i64_signature(name, inputs, outputs),
                &format!("WASM i64 signature mismatch for {name}"),
            );
        }
        for name in ["missing", "memory"] {
            input(
                wasm.validate_i64_signature(name, 0, 0),
                &format!("Missing WASM function export: {name}"),
            );
        }
        assert_eq!(wasm.exported_function_name(1).unwrap(), "int");
        assert_eq!(wasm.exported_function_name(3).unwrap(), "float");
        input(
            wasm.exported_function_name(0),
            "WASM export index 0 is not a function (memory)",
        );
        input(
            wasm.exported_function_name(2),
            "WASM export index 2 is not a function (global)",
        );
        input(
            wasm.exported_function_name(4),
            "WASM export index 4 is out of range",
        );
        input(
            wasm.exported_function_name(u64::MAX),
            "WASM export index 18446744073709551615 is out of range",
        );
        assert_eq!(wasm.exec("int", &[42]).unwrap(), [42]);
        // Now queries also use a populated function cache.
    }
    wasm.new_instance().unwrap();
    wasm.validate_i64_signature("int", 1, 1).unwrap();
    assert_eq!(wasm.exported_function_name(1).unwrap(), "int");
}

#[test]
fn compiled_instances_share_export_metadata() {
    let compiled = CompiledWasmModule::new(
        br#"(module
        (memory (export "memory") 1)
        (func (export "init"))
        (func (export "identity") (param i64) (result i64) local.get 0)
        (func (export "float") (param f64) (result f64) local.get 0))"#,
    )
    .unwrap();
    let mut first = compiled.instantiate().unwrap();
    let second = compiled.instantiate().unwrap();
    let third = compiled.clone().instantiate().unwrap();
    assert!(first.shares_export_metadata_with(&second));
    assert!(first.shares_export_metadata_with(&third));
    drop(compiled);
    for wasm in [&first, &second, &third] {
        assert!(wasm.has_func("init"));
        assert!(!wasm.has_func("memory"));
        assert!(!wasm.has_func("missing"));
        assert_eq!(wasm.get_funcs(), ["init", "identity", "float"]);
        assert_eq!(wasm.exported_function_name(2).unwrap(), "identity");
        input(
            wasm.exported_function_name(0),
            "WASM export index 0 is not a function (memory)",
        );
        wasm.validate_i64_signature("identity", 1, 1).unwrap();
        input(
            wasm.validate_i64_signature("float", 1, 1),
            "WASM i64 signature mismatch for float",
        );
        input(
            wasm.validate_i64_signature("memory", 0, 0),
            "Missing WASM function export: memory",
        );
    }
    assert_eq!(first.exec("identity", &[42]).unwrap(), [42]);
    first.new_instance().unwrap();
    assert!(first.shares_export_metadata_with(&second));
    assert_eq!(first.exec("identity", &[-7]).unwrap(), [-7]);
}
