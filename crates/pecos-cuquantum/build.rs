//! Build script for pecos-cuquantum
//!
//! Sets rpath so that test and binary targets can find cuQuantum,
//! cuTensor, and CUDA shared libraries at runtime.

fn main() {
    env_logger::init();

    // RPATH configuration is Linux-only (ELF). macOS uses different mechanisms
    // (@rpath / install_name_tool) and doesn't support -Wl,-rpath.
    if cfg!(target_os = "linux") {
        // cuQuantum
        if let Some(cuquantum_path) = pecos_build::cuquantum::find_cuquantum()
            && let Some(lib_dir) = pecos_build::cuquantum::get_lib_dir(&cuquantum_path)
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        } else if pecos_build::cuda::find_cuda().is_some() {
            // CUDA available but cuQuantum not found -- try auto-install
            match pecos_build::cuquantum::ensure_cuquantum() {
                Ok(cuquantum_path) => {
                    if let Some(lib_dir) = pecos_build::cuquantum::get_lib_dir(&cuquantum_path) {
                        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
                    }
                }
                // Never swallow this. The installers verify archive checksums, and their
                // own output goes to cargo's captured stdout, which is invisible without
                // -vv; discarding the error too would mean a failed integrity check
                // produced no signal at all on the one path an ordinary `cargo build`
                // uses. cargo:warning= is the only channel cargo always shows.
                Err(error) => {
                    println!("cargo:warning=cuQuantum auto-install failed: {error}");
                }
            }
        }

        // cuTensor (transitive dependency of cuTensorNet)
        match pecos_build::cutensor::ensure_cutensor() {
            Ok(cutensor_path) => {
                if let Some(lib_dir) = pecos_build::cutensor::get_lib_dir(&cutensor_path) {
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
                }
            }
            Err(error) => {
                println!("cargo:warning=cuTensor auto-install failed: {error}");
            }
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
