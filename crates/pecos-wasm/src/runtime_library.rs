//! Keep process-wide trap handlers, fork handlers and the epoch ticker mapped.

// Pin only where it was verified: dynamically linked glibc Linux and macOS.
#[cfg(any(
    all(
        target_os = "linux",
        target_env = "gnu",
        not(target_feature = "crt-static")
    ),
    target_os = "macos"
))]
pub(crate) fn pin() -> Result<(), String> {
    use std::ffi::{CStr, c_void};
    use std::mem::MaybeUninit;

    let address = (pin as *const ()).cast::<c_void>();
    let mut own = MaybeUninit::<libc::Dl_info>::uninit();
    // SAFETY: address points into this loaded object; own is writable Dl_info.
    if unsafe { libc::dladdr(address, own.as_mut_ptr()) } == 0 {
        // No dynamic object was found: no pin is needed or possible.
        return Ok(());
    }
    // SAFETY: Successful dladdr initialized own, including its loader-owned name.
    let own = unsafe { own.assume_init() };
    #[cfg(target_os = "linux")]
    {
        let mut main = MaybeUninit::<libc::Dl_info>::uninit();
        // SAFETY: AT_PHDR is the main executable's program-header address, and
        // main is writable Dl_info. No pointer is dereferenced by Rust here.
        let found = unsafe {
            libc::dladdr(
                libc::getauxval(libc::AT_PHDR) as *const c_void,
                main.as_mut_ptr(),
            )
        };
        if found == 0 {
            // Static executables may not have loader metadata to pin.
            return Ok(());
        }
        // SAFETY: Successful dladdr initialized main.
        if unsafe { main.assume_init() }.dli_fbase == own.dli_fbase {
            return Ok(());
        }
    }
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn _dyld_get_image_header(index: u32) -> *const c_void;
        }
        // SAFETY: Image zero is the process's main executable and stays loaded.
        if unsafe { _dyld_get_image_header(0) } == own.dli_fbase.cast_const() {
            return Ok(());
        }
    }
    // SAFETY: dladdr supplied a live, NUL-terminated loader path. NOLOAD requires
    // this existing object; NODELETE and the deliberately retained handle pin it.
    // Called only by Engine's initializer, never again in an inheriting child.
    let handle = unsafe {
        libc::dlopen(
            own.dli_fname,
            libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE,
        )
    };
    if handle.is_null() {
        // SAFETY: dlerror returns a thread-local NUL-terminated error, if present.
        let error = unsafe { libc::dlerror() };
        let detail = if error.is_null() {
            "dlopen returned a null handle".into()
        } else {
            // SAFETY: A non-null dlerror pointer remains valid until the next loader call.
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        };
        return Err(format!(
            "Failed to pin the WebAssembly runtime library: {detail}"
        ));
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn pin() -> Result<(), String> {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, address: *const u16, handle: *mut *mut c_void) -> i32;
    }
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
    const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x1;
    let mut handle = std::ptr::null_mut();
    // SAFETY: FROM_ADDRESS interprets the pointer as a code address, not a UTF-16
    // string. It points into this module, and handle is writable. PIN is permanent.
    let result = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            (pin as *const ()).cast::<u16>(),
            &raw mut handle,
        )
    };
    if result == 0 {
        return Err(format!(
            "Failed to pin the WebAssembly runtime library: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

// Static binaries need no pin (their loader APIs may be stubs). musl never
// unloads on dlclose, and dlopen's path re-check can fail after a file replacement.
// Other targets (including musl-derived environments) have no pinning implementation.
#[cfg(not(any(
    windows,
    all(
        target_os = "linux",
        target_env = "gnu",
        not(target_feature = "crt-static")
    ),
    target_os = "macos"
)))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "shares the fallible signature of the platforms that pin"
)]
pub(crate) fn pin() -> Result<(), String> {
    Ok(())
}
