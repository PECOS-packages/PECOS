//! Build script for pecos-cuquantum
//!
//! Sets rpath so that test and binary targets can find cuQuantum,
//! cuTensor, and CUDA shared libraries at runtime.

fn main() {
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

    println!("cargo:rerun-if-env-changed=CUQUANTUM_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
