//! Preserve ordinary TLS cleanup, but skip inherited fork state and Windows shutdown.

use std::mem::ManuallyDrop;
use std::ops::Deref;

pub(crate) struct Value<T>(ManuallyDrop<T>);

impl<T> Value<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(ManuallyDrop::new(value))
    }
}

impl<T> Deref for Value<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Drop for Value<T> {
    fn drop(&mut self) {
        #[cfg(unix)]
        if crate::inherited() {
            // A child that never re-enters the ABI must leak everything, including
            // compiled code: its inherited workers cannot be joined or waited on.
            return;
        }
        #[cfg(windows)]
        {
            #[link(name = "ntdll")]
            unsafe extern "system" {
                fn RtlDllShutdownInProgress() -> u8;
            }
            // SAFETY: This documented, argument-free Windows query has no
            // preconditions. BOOLEAN is an unsigned byte in the Windows ABI.
            if unsafe { RtlDllShutdownInProgress() } != 0 {
                // ExitProcess can forcibly terminate sibling threads before DLL
                // teardown. They may still own channel locks and Rust return
                // packets, so even dropping a reply receiver can be unsafe here.
                // Leave all owned state for the terminating process to reclaim.
                // Normal reset, failure, and owner-thread exit still clean up.
                return;
            }
        }
        // SAFETY: Value exclusively owns this initialized value. It is dropped
        // exactly once here and cannot be accessed after its owner is destroyed.
        unsafe { ManuallyDrop::drop(&mut self.0) };
    }
}
