//! Export-level tests with the same thread registration used by the executor.

use crate::ffi::*;
use crate::*;

struct RegisteredContext(*mut ExecutionContext);

impl RegisteredContext {
    fn new() -> Self {
        let ptr = pecos_create_execution_context();
        unsafe { pecos_register_execution_context(ptr) };
        Self(ptr)
    }

    fn context(&self) -> &ExecutionContext {
        unsafe { &*self.0 }
    }
}

impl Drop for RegisteredContext {
    fn drop(&mut self) {
        unsafe {
            pecos_register_execution_context(std::ptr::null_mut());
            pecos_destroy_execution_context(self.0);
        }
    }
}

fn json_results() -> BTreeMap<String, NamedResult> {
    let ptr = pecos_get_named_results_json();
    assert!(!ptr.is_null());
    let json = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes();
    let parsed = serde_json::from_slice(json);
    unsafe { pecos_free_named_results_json(ptr) };
    parsed.expect("typed named results JSON")
}

macro_rules! test_exports {
    ($test:ident, $scalar:ident, $array:ident, $selene_scalar:ident, $selene_array:ident,
     $dense:ident, $variant:ident, $prefix:literal, $a:expr, $b:expr) => {
        #[test]
        fn $test() {
            let registered = RegisteredContext::new();
            let ctx = registered.context();
            let label = concat!("USER:", $prefix, ":value");
            let len = i64::try_from(label.len()).expect("label length");
            let mut tket_label = vec![u8::try_from(label.len()).expect("label length")];
            tket_label.extend_from_slice(label.as_bytes());
            let array_label = concat!("USER:", $prefix, "ARR:array");
            let array_len = i64::try_from(array_label.len()).expect("label length");
            let mut tket_array_label = vec![u8::try_from(array_label.len()).expect("label length")];
            tket_array_label.extend_from_slice(array_label.as_bytes());
            let values = [$a, $b];
            let dense = $dense {
                x: 2,
                y: 1,
                data: values.as_ptr(),
                mask: std::ptr::null(),
            };
            unsafe {
                ctx.record_result_read(1);
                $scalar(tket_label.as_ptr(), len, $a);
                ctx.record_result_read(2);
                $selene_scalar(label.as_ptr(), len, $b);
                ctx.record_result_read(3);
                ctx.record_result_read(4);
                $array(tket_array_label.as_ptr(), array_len, &dense);
                ctx.record_result_read(5);
                ctx.record_result_read(6);
                $selene_array(array_label.as_ptr(), array_len, values.as_ptr(), 2);
            }
            if let NamedResult::Bool(values) = NamedResult::$variant(vec![$a, $b]) {
                let expected = vec![
                    NamedResultTrace {
                        name: "value".to_string(),
                        values: vec![values[0]],
                        result_ids: vec![1],
                    },
                    NamedResultTrace {
                        name: "value".to_string(),
                        values: vec![values[1]],
                        result_ids: vec![2],
                    },
                    NamedResultTrace {
                        name: "array".to_string(),
                        values: values.clone(),
                        result_ids: vec![3, 4],
                    },
                    NamedResultTrace {
                        name: "array".to_string(),
                        values,
                        result_ids: vec![5, 6],
                    },
                ];
                assert_eq!(
                    serde_json::to_string(&ctx.get_named_result_traces()).expect("trace JSON"),
                    serde_json::to_string(&expected).expect("expected trace JSON")
                );
            } else {
                assert!(ctx.get_named_result_traces().is_empty());
                assert_eq!(
                    *ctx.pending_result_reads.lock().expect("reads"),
                    vec![1, 2, 3, 4, 5, 6]
                );
            }
            let expected = BTreeMap::from([
                ("value".to_string(), NamedResult::$variant(vec![$a, $b])),
                (
                    "array".to_string(),
                    NamedResult::$variant(vec![$a, $b, $a, $b]),
                ),
            ]);
            assert_eq!(ctx.get_named_results(), expected);
            assert_eq!(json_results(), expected);

            // An array with no elements still establishes its element type and tag.
            let empty = $dense {
                x: 0,
                y: 1,
                data: std::ptr::null(),
                mask: std::ptr::null(),
            };
            let traces_before = ctx.get_named_result_traces();
            let reads_before = ctx.pending_result_reads.lock().expect("reads").clone();
            unsafe {
                $array(b"\x05empty".as_ptr(), 5, &empty);
                $selene_array(b"empty".as_ptr(), 5, std::ptr::null(), 0);
            }
            assert_eq!(json_results()["empty"], NamedResult::$variant(vec![]));
            // A bool call always traces, so an empty bool array adds one
            // zero-length trace per call (as on `dev`); other types add none.
            let mut expected_traces = traces_before;
            if matches!(NamedResult::$variant(vec![]), NamedResult::Bool(_)) {
                for _ in 0..2 {
                    expected_traces.push(NamedResultTrace {
                        name: "empty".to_string(),
                        values: vec![],
                        result_ids: vec![],
                    });
                }
            }
            assert_eq!(ctx.get_named_result_traces(), expected_traces);
            assert_eq!(
                *ctx.pending_result_reads.lock().expect("reads"),
                reads_before
            );
        }
    };
}

test_exports!(
    bool_exports,
    print_bool,
    print_bool_arr,
    print_bool_selene,
    print_bool_arr_selene,
    Dense1DArrayBool,
    Bool,
    "BOOL",
    false,
    true
);
test_exports!(
    int_exports,
    print_int,
    print_int_arr,
    print_int_selene,
    print_int_arr_selene,
    Dense1DArrayInt,
    I64,
    "INT",
    i64::MIN,
    i64::MAX
);
test_exports!(
    uint_exports,
    print_uint,
    print_uint_arr,
    print_uint_selene,
    print_uint_arr_selene,
    Dense1DArrayUint,
    U64,
    "UINT",
    7,
    u64::MAX
);
test_exports!(
    float_exports,
    print_float,
    print_float_arr,
    print_float_selene,
    print_float_arr_selene,
    Dense1DArrayFloat,
    F64,
    "FLOAT",
    -1.5,
    2.5
);

#[test]
fn mixed_type_error_is_sticky() {
    let registered = RegisteredContext::new();
    unsafe {
        print_bool(b"\x03tag".as_ptr(), 3, true);
        print_float_selene(b"tag".as_ptr(), 3, 2.5);
        print_bool_selene(b"tag".as_ptr(), 3, false);
    }
    let ctx = registered.context();
    assert_eq!(
        ctx.get_named_results()["tag"],
        NamedResult::Bool(vec![true])
    );
    let error = ctx
        .program_error
        .lock()
        .expect("error lock")
        .clone()
        .expect("type error");
    assert!(error.to_string().contains("tag"));
    assert!(error.to_string().contains("bool"));
    assert!(error.to_string().contains("f64"));
    ctx.reset();
    assert!(ctx.program_error.lock().expect("error lock").is_none());
}

#[test]
fn numeric_output_preserves_measurement_provenance_for_detector() {
    let registered = RegisteredContext::new();
    let ctx = registered.context();
    // m = measure(q).read(); result("v", int(m) + 5); result("det", m)
    reset_interface();
    with_interface(|interface| interface.store_result(0, true));
    let m = unsafe { ___read_future_bool(0) };
    unsafe {
        print_int(b"\x01v".as_ptr(), 1, i64::from(m) + 5);
        print_uint_selene(b"u".as_ptr(), 1, 6);
        print_float_selene(b"f".as_ptr(), 1, 2.5);
    }
    assert!(ctx.get_named_result_traces().is_empty());
    assert_eq!(*ctx.pending_result_reads.lock().expect("reads"), vec![0]);
    unsafe { print_bool(b"\x03det".as_ptr(), 3, m) };
    let json = serde_json::to_string(&ctx.get_named_result_traces()).expect("trace JSON");
    assert_eq!(json, r#"[{"name":"det","values":[true],"result_ids":[0]}]"#);
    println!("{json}");
    assert!(ctx.pending_result_reads.lock().expect("reads").is_empty());
}

macro_rules! integer_detector_trace_test {
    ($test:ident, $ty:ty, $dense:ident, $variant:ident,
     $scalar:ident, $array:ident, $selene_scalar:ident, $selene_array:ident) => {
        #[test]
        fn $test() {
            for selene in [false, true] {
                let registered = RegisteredContext::new();
                let ctx = registered.context();
                let label = if selene { &b"i"[..] } else { &b"\x01i"[..] };
                let scalar = if selene { $selene_scalar } else { $scalar };
                ctx.record_result_read(10);
                unsafe { scalar(label.as_ptr(), 1, 0) };
                assert_eq!(
                    ctx.get_named_result_traces(),
                    vec![NamedResultTrace {
                        name: "i".to_string(),
                        values: vec![false],
                        result_ids: vec![10],
                    }]
                );
                let values: [$ty; 2] = [0, 1];
                let dense = $dense {
                    x: 2,
                    y: 1,
                    data: values.as_ptr(),
                    mask: std::ptr::null(),
                };
                ctx.record_result_read(11);
                ctx.record_result_read(12);
                unsafe {
                    if selene {
                        $selene_array(label.as_ptr(), 1, values.as_ptr(), 2);
                    } else {
                        $array(label.as_ptr(), 1, &dense);
                    }
                }
                assert_eq!(
                    ctx.get_named_result_traces()[1],
                    NamedResultTrace {
                        name: "i".to_string(),
                        values: vec![false, true],
                        result_ids: vec![11, 12],
                    }
                );
                // Bool output appends to the declared integer storage under the same tag.
                unsafe { print_bool_selene(b"i".as_ptr(), 1, true) };
                assert_eq!(json_results()["i"], NamedResult::$variant(vec![0, 0, 1, 1]));
                assert!(ctx.program_error.lock().expect("error").is_none());
                assert!(ctx.pending_result_reads.lock().expect("reads").is_empty());

                ctx.reset();
                ctx.record_result_read(13);
                unsafe { scalar(label.as_ptr(), 1, 7) };
                assert!(ctx.get_named_result_traces().is_empty());
                assert_eq!(*ctx.pending_result_reads.lock().expect("reads"), vec![13]);
                let mixed: [$ty; 2] = [0, 7];
                let dense = $dense {
                    x: 2,
                    y: 1,
                    data: mixed.as_ptr(),
                    mask: std::ptr::null(),
                };
                unsafe {
                    if selene {
                        $selene_array(label.as_ptr(), 1, mixed.as_ptr(), 2);
                    } else {
                        $array(label.as_ptr(), 1, &dense);
                    }
                }
                assert_eq!(json_results()["i"], NamedResult::$variant(vec![7, 0, 7]));
                assert!(ctx.get_named_result_traces().is_empty());
                assert_eq!(*ctx.pending_result_reads.lock().expect("reads"), vec![13]);
                // A later detector call keeps numeric storage and consumes the pending read.
                unsafe { scalar(label.as_ptr(), 1, 1) };
                assert!(ctx.program_error.lock().expect("error").is_none());
                assert_eq!(json_results()["i"], NamedResult::$variant(vec![7, 0, 7, 1]));
                assert_eq!(ctx.get_named_result_traces()[0].result_ids, vec![13]);
                assert!(ctx.pending_result_reads.lock().expect("reads").is_empty());

                ctx.reset();
                ctx.record_result_read(17);
                unsafe { print_bool_selene(b"i".as_ptr(), 1, true) };
                ctx.record_result_read(18);
                unsafe { scalar(label.as_ptr(), 1, 0) };
                assert_eq!(json_results()["i"], NamedResult::$variant(vec![1, 0]));
                assert_eq!(
                    ctx.get_named_result_traces(),
                    vec![
                        NamedResultTrace {
                            name: "i".to_string(),
                            values: vec![true],
                            result_ids: vec![17]
                        },
                        NamedResultTrace {
                            name: "i".to_string(),
                            values: vec![false],
                            result_ids: vec![18]
                        },
                    ]
                );

                ctx.reset();
                let empty = $dense {
                    x: 0,
                    y: 1,
                    data: std::ptr::null(),
                    mask: std::ptr::null(),
                };
                ctx.record_result_read(19);
                unsafe {
                    if selene {
                        $selene_array(label.as_ptr(), 1, std::ptr::null(), 0);
                    } else {
                        $array(label.as_ptr(), 1, &empty);
                    }
                }
                assert_eq!(json_results()["i"], NamedResult::$variant(vec![]));
                assert!(ctx.get_named_result_traces().is_empty());
                assert_eq!(*ctx.pending_result_reads.lock().expect("reads"), vec![19]);
                unsafe { print_bool_selene(b"i".as_ptr(), 1, true) };
                assert_eq!(json_results()["i"], NamedResult::$variant(vec![1]));
                assert_eq!(ctx.get_named_result_traces()[0].result_ids, vec![19]);
                assert!(ctx.pending_result_reads.lock().expect("reads").is_empty());
            }
        }
    };
}

integer_detector_trace_test!(
    signed_integer_detector_traces,
    i64,
    Dense1DArrayInt,
    I64,
    print_int,
    print_int_arr,
    print_int_selene,
    print_int_arr_selene
);
integer_detector_trace_test!(
    unsigned_integer_detector_traces,
    u64,
    Dense1DArrayUint,
    U64,
    print_uint,
    print_uint_arr,
    print_uint_selene,
    print_uint_arr_selene
);

#[test]
fn float_zero_and_one_never_produce_detector_traces() {
    let registered = RegisteredContext::new();
    let ctx = registered.context();
    let values = [0.0, 1.0];
    let dense = Dense1DArrayFloat {
        x: 2,
        y: 1,
        data: values.as_ptr(),
        mask: std::ptr::null(),
    };
    for selene in [false, true] {
        ctx.reset();
        ctx.record_result_read(1);
        unsafe {
            if selene {
                print_float_selene(b"f".as_ptr(), 1, 0.0);
            } else {
                print_float(b"\x01f".as_ptr(), 1, 0.0);
            }
        }
        assert!(ctx.get_named_result_traces().is_empty());
        assert_eq!(ctx.pending_result_reads.lock().expect("reads").len(), 1);
        ctx.record_result_read(2);
        ctx.record_result_read(3);
        unsafe {
            if selene {
                print_float_arr_selene(b"f".as_ptr(), 1, values.as_ptr(), 2);
            } else {
                print_float_arr(b"\x01f".as_ptr(), 1, &raw const dense);
            }
        }
        assert!(ctx.get_named_result_traces().is_empty());
        assert_eq!(
            *ctx.pending_result_reads.lock().expect("reads"),
            vec![1, 2, 3]
        );
    }
}

#[test]
fn unknown_json_element_type_is_rejected() {
    assert!(
        serde_json::from_str::<BTreeMap<String, NamedResult>>(
            r#"{"tag":{"type":"unknown","values":[1]}}"#
        )
        .is_err()
    );
}

#[test]
fn float_json_preserves_all_bits() {
    let _registered = RegisteredContext::new();
    let values = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_0042),
    ];
    unsafe { print_float_arr_selene(b"f".as_ptr(), 1, values.as_ptr(), 5) };
    let NamedResult::F64(round_trip) = &json_results()["f"] else {
        panic!("expected f64 result");
    };
    assert_eq!(
        round_trip.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn panic_decodes_length_prefixed_message_and_clears_at_shot_start() {
    let registered = RegisteredContext::new();
    for code in [0, 1, 999, 1000, 1001, 1002, -1] {
        let expected = if (0..=1000).contains(&code) {
            ProgramError::Exit {
                code,
                message: "hello".to_string(),
            }
        } else {
            ProgramError::Panic {
                code,
                message: "hello".to_string(),
            }
        };
        // No execution guard is active here, so the export only records.
        unsafe { panic(code, b"\x05helloNOT_PART_OF_MESSAGE".as_ptr().cast()) };
        assert_eq!(
            *registered
                .context()
                .program_error
                .lock()
                .expect("error lock"),
            Some(expected.clone())
        );
        let ptr = pecos_get_program_error_json();
        assert!(!ptr.is_null());
        let parsed: ProgramError =
            serde_json::from_slice(unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes())
                .expect("error JSON");
        unsafe { pecos_free_named_results_json(ptr) };
        assert_eq!(parsed, expected);
        pecos_clear_program_error();
        assert!(pecos_get_program_error_json().is_null());
    }
    unsafe { panic(0, std::ptr::null()) };
    assert!(
        registered
            .context()
            .program_error
            .lock()
            .expect("error lock")
            .as_ref()
            .expect("null message panic")
            .to_string()
            .contains("null panic message")
    );
}

#[test]
fn collection_reset_clears_reads_and_preserves_operation_reset_semantics() {
    let registered = RegisteredContext::new();
    let ctx = registered.context();
    for _ in 0..3 {
        unsafe { pecos_qis_reset_interface() };
        assert!(ctx.pending_result_reads.lock().expect("reads").is_empty());
        assert!(with_interface(|interface| interface.operations.is_empty()));
        with_interface(|interface| interface.store_result(0, true));
        assert!(unsafe { ___read_future_bool(0) });
        assert_eq!(*ctx.pending_result_reads.lock().expect("reads"), vec![0]);
        unsafe { __quantum__rt__qubit_allocate() };
        assert!(!with_interface(|interface| interface.operations.is_empty()));
    }
}

#[test]
fn destroying_registered_context_unregisters_it() {
    let ctx = pecos_create_execution_context();
    unsafe {
        pecos_register_execution_context(ctx);
        pecos_destroy_execution_context(ctx);
    }
    assert!(get_execution_context().is_none());
}

#[test]
fn declared_integer_storage_does_not_depend_on_values() {
    let registered = RegisteredContext::new();
    for selene in [false, true] {
        registered.context().reset();
        let label = if selene { &b"c"[..] } else { &b"\x01c"[..] };
        let print = if selene { print_int_selene } else { print_int };
        for value in 0..4 {
            unsafe { print(label.as_ptr(), 1, value) };
        }
        assert_eq!(json_results()["c"], NamedResult::I64(vec![0, 1, 2, 3]));
        for values in [&[0, 1][..], &[0, 1, 2][..]] {
            let dense = Dense1DArrayInt {
                x: i32::try_from(values.len()).expect("length"),
                y: 1,
                data: values.as_ptr(),
                mask: std::ptr::null(),
            };
            unsafe {
                if selene {
                    print_int_arr_selene(b"t".as_ptr(), 1, values.as_ptr(), values.len() as u64);
                } else {
                    print_int_arr(b"\x01t".as_ptr(), 1, &raw const dense);
                }
            }
        }
        assert_eq!(json_results()["t"], NamedResult::I64(vec![0, 1, 0, 1, 2]));
        assert!(
            registered
                .context()
                .program_error
                .lock()
                .expect("error")
                .is_none()
        );
    }
}

#[test]
fn declared_types_widen_only_between_bool_and_integers() {
    let registered = RegisteredContext::new();
    let ctx = registered.context();
    let types = [
        NamedResult::Bool(vec![true]),
        NamedResult::I64(vec![7]),
        NamedResult::U64(vec![7]),
        NamedResult::F64(vec![2.5]),
    ];
    for first in &types {
        for second in &types {
            ctx.reset();
            ctx.store_named_result("tag", first.clone());
            ctx.store_named_result("tag", second.clone());
            let expected = match (first, second) {
                (NamedResult::Bool(_), NamedResult::Bool(_)) => {
                    Some(NamedResult::Bool(vec![true, true]))
                }
                (NamedResult::Bool(_), NamedResult::I64(_)) => Some(NamedResult::I64(vec![1, 7])),
                (NamedResult::Bool(_), NamedResult::U64(_)) => Some(NamedResult::U64(vec![1, 7])),
                (NamedResult::I64(_), NamedResult::Bool(_)) => Some(NamedResult::I64(vec![7, 1])),
                (NamedResult::U64(_), NamedResult::Bool(_)) => Some(NamedResult::U64(vec![7, 1])),
                (NamedResult::I64(_), NamedResult::I64(_)) => Some(NamedResult::I64(vec![7, 7])),
                (NamedResult::U64(_), NamedResult::U64(_)) => Some(NamedResult::U64(vec![7, 7])),
                (NamedResult::F64(_), NamedResult::F64(_)) => {
                    Some(NamedResult::F64(vec![2.5, 2.5]))
                }
                _ => None,
            };
            if let Some(expected) = expected {
                assert_eq!(json_results()["tag"], expected);
                assert!(ctx.program_error.lock().expect("error").is_none());
            } else {
                assert_eq!(json_results()["tag"], *first);
                let error = ctx.program_error.lock().expect("error");
                let message = error.as_ref().expect("incompatible types").to_string();
                assert!(message.contains(first.element_type()));
                assert!(message.contains(second.element_type()));
            }
        }
    }
}
