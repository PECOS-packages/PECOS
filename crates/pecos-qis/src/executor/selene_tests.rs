//! ABI and recovery checks against the dynamically loaded runtime only.

use super::*;
use crate::test_env::ENV_MUTEX;

// Explicit public ABI surface: adding or removing an entry requires review.
const SELENE_EXPORTS: [&str; 42] = [
    "selene_qalloc",
    "selene_qfree",
    "selene_rxy",
    "selene_rz",
    "selene_rzz",
    "selene_qubit_reset",
    "selene_qubit_measure",
    "selene_qubit_lazy_measure",
    "selene_qubit_lazy_measure_leaked",
    "selene_future_read_bool",
    "selene_future_read_u64",
    "selene_refcount_increment",
    "selene_refcount_decrement",
    "selene_print_bool",
    "selene_print_i64",
    "selene_print_u64",
    "selene_print_f64",
    "selene_print_bool_array",
    "selene_print_i64_array",
    "selene_print_u64_array",
    "selene_print_f64_array",
    "selene_print_panic",
    "selene_dump_state",
    "selene_set_tc",
    "selene_get_tc",
    "selene_get_current_shot",
    "selene_local_barrier",
    "selene_global_barrier",
    "selene_shot_count",
    "selene_on_shot_start",
    "selene_on_shot_end",
    "selene_load_config",
    "selene_exit",
    "selene_print_exit",
    "selene_random_seed",
    "selene_random_advance",
    "selene_random_u32",
    "selene_random_u32_bounded",
    "selene_random_f64",
    "selene_custom_runtime_call",
    "pecos_call_qmain_with_setjmp",
    "pecos_call_void_main_with_setjmp",
];

#[cfg(unix)]
fn symbol_object(address: *const std::ffi::c_void) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;

    #[repr(C)]
    struct DlInfo {
        filename: *const std::ffi::c_char,
        base: *mut std::ffi::c_void,
        symbol: *const std::ffi::c_char,
        symbol_address: *mut std::ffi::c_void,
    }
    unsafe extern "C" {
        fn dladdr(address: *const std::ffi::c_void, info: *mut DlInfo) -> std::ffi::c_int;
    }
    let mut info = std::mem::MaybeUninit::<DlInfo>::uninit();
    unsafe {
        assert_ne!(dladdr(address, info.as_mut_ptr()), 0);
        let name = std::ffi::CStr::from_ptr(info.assume_init().filename);
        PathBuf::from(std::ffi::OsStr::from_bytes(name.to_bytes()))
    }
}

#[cfg(windows)]
fn symbol_object(address: *const std::ffi::c_void) -> PathBuf {
    use std::os::windows::ffi::OsStringExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(
            flags: u32,
            address: *const u16,
            module: *mut *mut std::ffi::c_void,
        ) -> i32;
        fn GetModuleFileNameW(module: *mut std::ffi::c_void, filename: *mut u16, size: u32) -> u32;
    }
    let mut module = std::ptr::null_mut();
    let mut filename = [0_u16; 32768];
    unsafe {
        // FROM_ADDRESS | UNCHANGED_REFCOUNT: inspect without changing ownership.
        assert_ne!(GetModuleHandleExW(6, address.cast(), &raw mut module), 0);
        let len = GetModuleFileNameW(module, filename.as_mut_ptr(), 32768);
        assert!(len > 0 && len < 32768);
        PathBuf::from(std::ffi::OsString::from_wide(&filename[..len as usize]))
    }
}

#[test]
fn all_selene_exports_are_defined_by_the_ffi() {
    let _env_lock = ENV_MUTEX.lock().expect("environment lock");
    let ffi = QisHeliosInterface::get_qis_ffi_lib_singleton().expect("FFI library");
    let expected = QisHeliosInterface::pinned_qis_ffi_lib_path()
        .expect("FFI path")
        .canonicalize()
        .expect("canonical FFI path");
    for name in SELENE_EXPORTS {
        let symbol: Symbol<*const std::ffi::c_void> =
            unsafe { ffi.get(name.as_bytes()).expect(name) };
        assert_eq!(
            symbol_object(*symbol)
                .canonicalize()
                .expect("symbol object"),
            expected,
            "{name}"
        );
    }
}

#[cfg(unix)]
fn compile_oracle(dir: &Path, executable: bool) -> PathBuf {
    let ffi = QisHeliosInterface::pinned_qis_ffi_lib_path().expect("FFI path");
    let output_path = dir.join(if executable {
        "byte_oracle"
    } else {
        "abi_oracle.so"
    });
    let mut cc = Command::new("cc");
    cc.args(["-std=c11", "-Wall", "-Wextra", "-Werror"]);
    if executable {
        cc.arg("-DBYTE_TEST_MAIN");
    } else {
        cc.args(["-shared", "-fPIC"]);
    }
    let output = cc
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/selene_abi.c"
        ))
        .arg(&ffi)
        .arg(format!(
            "-Wl,-rpath,{}",
            ffi.parent().expect("FFI directory").display()
        ))
        .arg("-o")
        .arg(&output_path)
        .output()
        .expect("run C compiler");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output_path
}

#[cfg(unix)]
unsafe extern "C" fn set_oracle_measurements() {
    let ffi = QisHeliosInterface::get_qis_ffi_lib_singleton().expect("FFI library");
    let set: Symbol<unsafe extern "C-unwind" fn(*const (usize, bool), usize)> = unsafe {
        ffi.get(b"pecos_qis_set_measurements\0")
            .expect("set results")
    };
    let results = [(1, true), (2, true)];
    unsafe { set(results.as_ptr(), results.len()) };
}

#[cfg(unix)]
#[test]
fn independent_c_abi_oracle() {
    use pecos_qis_ffi_types::QuantumOp;
    let env_lock = ENV_MUTEX.lock().expect("environment lock");
    let ffi = QisHeliosInterface::get_qis_ffi_lib_singleton().expect("FFI library");
    let dir = tempfile::tempdir().expect("fixture directory");
    let path = compile_oracle(dir.path(), false);
    let fixture = unsafe { Library::new(path).expect("C fixture") };
    let failure = unsafe {
        let reset: Symbol<ResetInterfaceFn> =
            ffi.get(b"pecos_qis_reset_interface\0").expect("reset");
        reset();
        let oracle: Symbol<unsafe extern "C" fn(unsafe extern "C" fn()) -> std::ffi::c_int> =
            fixture.get(b"abi_oracle\0").expect("oracle");
        oracle(set_oracle_measurements)
    };
    // A reported C failure must not poison the environment lock for other tests.
    drop(env_lock);
    assert_eq!(failure, 0, "C ABI oracle failed at selene_abi.c:{failure}");
    let ops = QisHeliosInterface::collect_operations_from_lib(ffi.inner()).expect("operations");
    assert_eq!(ops.operations[2], QuantumOp::RXY(0.25, 0.75, 1).into());
    assert_eq!(ops.operations[3], QuantumOp::RZ(1.25, 0).into());
    assert_eq!(ops.operations[4], QuantumOp::RZZ(1.75, 1, 0).into());
    assert_eq!(ops.operations[5], QuantumOp::Reset(1).into());
    assert_eq!(ops.operations[7], QuantumOp::Measure(0, 0).into());
    assert_eq!(ops.operations[9], QuantumOp::Measure(1, 1).into());
    assert_eq!(ops.operations[11], QuantumOp::MeasureLeaked(0, 2).into());
}

#[cfg(unix)]
#[test]
fn selene_exit_bytes_and_panic_with_and_without_guard() {
    let _env_lock = ENV_MUTEX.lock().expect("environment lock");
    let dir = tempfile::tempdir().expect("fixture directory");
    let output = Command::new(compile_oracle(dir.path(), true))
        .output()
        .expect("run byte oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stderr,
        b"EXIT [4294967295]: abc\nEXIT [17]: d\nEXIT [0]: \xff\x80z\nEXIT [8]: long\nEXIT [9]: max\n"
    );
}

#[test]
fn generated_program_selene_binds_to_selected_ffi() {
    let _env_lock = ENV_MUTEX.lock().expect("environment lock");
    let ffi = QisHeliosInterface::get_qis_ffi_lib_singleton().expect("FFI library");
    let mut interface = QisHeliosInterface::new();
    let import = if cfg!(windows) { "dllimport" } else { "" };
    interface.program = format!(
        r"
        declare {import} {{ i32, i64 }} @selene_qalloc(ptr)
        define i64 @qmain(i64 %arg) {{
            %address = ptrtoint ptr @selene_qalloc to i64
            ret i64 %address
        }}
    "
    )
    .into_bytes();
    interface.format = ProgramFormat::LlvmIrText;
    let path = interface.create_shared_library().expect("link program");
    let (_handle, program) =
        QisHeliosInterface::load_library(&path, "load probe", false).expect("load program");
    unsafe {
        let address: Symbol<unsafe extern "C" fn(u64) -> u64> =
            program.get(b"qmain\0").expect("address probe");
        let expected: Symbol<*const std::ffi::c_void> =
            ffi.get(b"selene_qalloc\0").expect("Selene allocation");
        assert_eq!(address(0), *expected as usize as u64);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SeleneString {
    data: *const u8,
    length: u64,
    owned: bool,
}

#[repr(C)]
struct VoidResult {
    error_code: u32,
}

#[repr(C)]
struct U32Result {
    error_code: u32,
    value: u32,
}

type Terminate =
    unsafe extern "C-unwind" fn(*mut std::ffi::c_void, SeleneString, u32) -> VoidResult;
type Handler = unsafe extern "C-unwind" fn();
type QGuard = unsafe extern "C" fn(unsafe extern "C-unwind" fn(u64) -> u64) -> u64;
type VGuard = unsafe extern "C" fn(unsafe extern "C-unwind" fn()) -> u64;

// Copy function pointers out of TLS before any adapter can transfer. There is
// no TLS borrow, library lookup, or owned value inside a skipped Rust frame.
#[derive(Clone, Copy)]
struct Adapters {
    exit: Terminate,
    panic: Terminate,
    invalid: unsafe extern "C-unwind" fn(*mut std::ffi::c_void, u64, f64, f64) -> VoidResult,
    random: unsafe extern "C-unwind" fn(*mut std::ffi::c_void) -> U32Result,
    seed: unsafe extern "C-unwind" fn(*mut std::ffi::c_void, u64) -> VoidResult,
    allocate: unsafe extern "C-unwind" fn(i64) -> *mut std::ffi::c_void,
}

thread_local! {
    static ADAPTERS: std::cell::Cell<Option<Adapters>> = const { std::cell::Cell::new(None) };
    static TRANSFER_MODE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static REACHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

unsafe extern "C-unwind" fn previous_handler() {}

unsafe extern "C-unwind" fn adapter_entry() {
    let adapters = ADAPTERS.get().expect("installed test adapters");
    let instance = std::ptr::null_mut();
    let message = SeleneString {
        data: b"done".as_ptr(),
        length: 4,
        owned: false,
    };
    unsafe {
        (adapters.allocate)(23);
        match TRANSFER_MODE.get() {
            0 => {
                (adapters.exit)(instance, message, 42);
            }
            1 => {
                (adapters.panic)(instance, message, 3);
            }
            2 => {
                (adapters.panic)(instance, message, 1001);
            }
            3 => {
                (adapters.invalid)(instance, u64::MAX, 0.2, 0.3);
            }
            4 => {
                (adapters.random)(instance);
            }
            _ => {
                (adapters.seed)(instance, 42);
                (adapters.random)(instance);
            }
        }
    }
    REACHED.set(true);
}

unsafe extern "C-unwind" fn adapter_qentry(_arg: u64) -> u64 {
    unsafe { adapter_entry() };
    0
}

fn run_adapter_transfers() {
    let ffi = QisHeliosInterface::get_qis_ffi_lib_singleton().expect("FFI library");
    let context = QisHeliosInterface::create_execution_context(ffi.inner()).expect("context");
    unsafe {
        let register: Symbol<RegisterExecutionContextFn> = ffi
            .get(b"pecos_register_execution_context\0")
            .expect("register");
        register(context.0);
        ADAPTERS.set(Some(Adapters {
            exit: *ffi.get(b"selene_print_exit\0").expect("print exit"),
            panic: *ffi.get(b"selene_print_panic\0").expect("print panic"),
            invalid: *ffi.get(b"selene_rxy\0").expect("rxy"),
            random: *ffi.get(b"selene_random_u32\0").expect("random"),
            seed: *ffi.get(b"selene_random_seed\0").expect("seed"),
            allocate: *ffi.get(b"heap_alloc\0").expect("allocate"),
        }));
        let qguard: Symbol<QGuard> = ffi
            .get(b"pecos_call_qmain_with_setjmp\0")
            .expect("qmain guard");
        let vguard: Symbol<VGuard> = ffi
            .get(b"pecos_call_void_main_with_setjmp\0")
            .expect("main guard");
        let freed: Symbol<unsafe extern "C" fn() -> usize> = ffi
            .get(b"pecos_get_context_allocation_free_count\0")
            .expect("free count");
        let get_handler: Symbol<unsafe extern "C" fn() -> Option<Handler>> = ffi
            .get(b"pecos_get_program_panic_handler\0")
            .expect("get handler");
        let set_handler: Symbol<unsafe extern "C" fn(Option<Handler>)> = ffi
            .get(b"pecos_set_program_panic_handler\0")
            .expect("set handler");
        let get_error: Symbol<GetProgramErrorJsonFn> =
            ffi.get(b"pecos_get_program_error_json\0").expect("error");
        let free: Symbol<FreeNamedResultsJsonFn> =
            ffi.get(b"pecos_free_named_results_json\0").expect("free");
        let invoke = |is_void| {
            if is_void {
                vguard(adapter_entry)
            } else {
                qguard(adapter_qentry)
            }
        };
        for is_void in [false, true] {
            for (mode, error_text) in [
                (0, None),
                (1, Some("Exit")),
                (2, Some("Panic")),
                (3, Some("-1")),
                (4, Some("random_seed")),
            ] {
                TRANSFER_MODE.set(mode);
                REACHED.set(false);
                let before = freed();
                set_handler(Some(previous_handler));
                assert_eq!(invoke(is_void), u64::from(mode >= 2));
                assert_eq!(REACHED.get(), mode == 0);
                assert_eq!(freed(), before + 1);
                assert!(std::ptr::fn_addr_eq(
                    get_handler().expect("restored handler"),
                    previous_handler as Handler
                ));
                let error = get_error();
                if let Some(expected) = error_text {
                    assert!(!error.is_null());
                    assert!(
                        std::ffi::CStr::from_ptr(error)
                            .to_string_lossy()
                            .contains(expected)
                    );
                    free(error);
                } else {
                    assert!(error.is_null());
                }
                TRANSFER_MODE.set(5);
                REACHED.set(false);
                assert_eq!(invoke(is_void), 0);
                assert!(REACHED.get());
                assert_eq!(freed(), before + 2);
                assert!(get_error().is_null());
                assert!(std::ptr::fn_addr_eq(
                    get_handler().expect("restored handler"),
                    previous_handler as Handler
                ));
            }
        }
        set_handler(None);
        ADAPTERS.set(None);
        register(std::ptr::null_mut());
    }
}

#[test]
fn rust_adapter_transfers_through_both_guards() {
    let _env_lock = ENV_MUTEX.lock().expect("environment lock");
    run_adapter_transfers();
}

#[test]
fn rust_adapter_transfers_are_thread_local() {
    let _env_lock = ENV_MUTEX.lock().expect("environment lock");
    // Initialize the singleton while environment mutations are excluded.
    QisHeliosInterface::get_qis_ffi_lib_singleton().expect("FFI library");
    let barrier = std::sync::Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                barrier.wait();
                for _ in 0..8 {
                    run_adapter_transfers();
                }
            });
        }
    });
}
