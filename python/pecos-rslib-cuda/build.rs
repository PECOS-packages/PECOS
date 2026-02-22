//! Build script for pecos-rslib-cuda
//!
//! Sets RPATH on the final cdylib so that cuQuantum, cuTensor, and CUDA
//! shared libraries can be found at runtime without requiring
//! `LD_LIBRARY_PATH`.
//!
//! Uses `--disable-new-dtags` to emit `DT_RPATH` instead of `DT_RUNPATH`.
//! RPATH is inherited by transitive shared library dependencies, which is
//! necessary because `libcutensornet.so` loads `libcutensor.so` at runtime
//! from a different directory.

fn main() {
    env_logger::init();
    // RPATH configuration is Linux-only (ELF). macOS uses different mechanisms
    // (@rpath / install_name_tool) and doesn't support --disable-new-dtags.
    if cfg!(target_os = "linux") {
        // Use RPATH (not RUNPATH) so paths propagate to transitive dependencies.
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");

        // cuQuantum
        if let Some(cuquantum_path) = pecos_build::cuquantum::find_cuquantum()
            && let Some(lib_dir) = pecos_build::cuquantum::get_lib_dir(&cuquantum_path)
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }

        // cuTensor (transitive dependency of cuTensorNet)
        if let Ok(cutensor_path) = pecos_build::cutensor::ensure_cutensor()
            && let Some(lib_dir) = pecos_build::cutensor::get_lib_dir(&cutensor_path)
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }

        // CUDA runtime
        if let Some(cuda_path) = pecos_build::cuda::find_cuda() {
            let lib64 = cuda_path.join("lib64");
            if lib64.exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib64.display());
            } else {
                let lib = cuda_path.join("lib");
                if lib.exists() {
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
                }
            }
        }
    }

    println!("cargo:rerun-if-env-changed=CUQUANTUM_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
