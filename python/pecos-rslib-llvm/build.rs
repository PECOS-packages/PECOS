//! Build script for pecos-rslib-llvm.
//!
//! Injects the Python distribution version so the extension module's `__version__` matches
//! the wheel users install.

fn main() {
    pecos_build::python::emit_python_version();
}
