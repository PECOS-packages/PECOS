fn main() {
    // Build C++ source files
    cc::Build::new()
        .cpp(true)
        .file("src/sparsesim.cpp")
        .file("src/cxx_shim.cpp")
        .include("src")
        .std("c++14")
        .compile("sparsesim");

    // Generate cxx bridge code
    cxx_build::bridge("src/lib.rs")
        .file("src/cxx_shim.cpp")
        .std("c++14")
        .compile("cppsparsesim-bridge");

    // Tell cargo to rerun if source files change
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/sparsesim.cpp");
    println!("cargo:rerun-if-changed=src/sparsesim.h");
    println!("cargo:rerun-if-changed=src/cxx_shim.cpp");
    println!("cargo:rerun-if-changed=src/cxx_shim.h");
}
