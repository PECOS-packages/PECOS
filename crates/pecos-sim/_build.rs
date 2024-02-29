
/*
fn main() {
    cc::Build::new()
        .cpp(true) // Indicates we're compiling C++ code
        .file("../../cpp/sparsesim/sparsesim.cpp") // Adjust the path to your C++ source file
        .include("../../cpp/sparsesim") // If your C++ code depends on additional headers
        .flag("-std=c++17") // Adjust according to the C++ version your code requires
        .compile("sparsesim");
    println!("cargo:rerun-if-changed=../../cpp/sparsesim/sparsesim.cpp");
    println!("cargo:rerun-if-changed=../../cpp/sparsesim/sparsesim.h");
}
*/


/*
extern crate cxx_build;

fn main() {
    // Tell Cargo to rerun this script if the source changes.
    println!("cargo:rerun-if-changed=../../cpp/sparsesim/sparsesim.cpp");
    println!("cargo:rerun-if-changed=../../cpp/sparsesim/sparsesim.h");

    cxx_build::bridge("src/lib.rs")  // Path to your bridge module
        .file("../../cpp/sparsesim/sparsesim.cpp") // Path to your C++ source file
        .include("../../cpp/sparsesim/") // Path to your C++ header files
        .flag_if_supported("-std=c++17") // Optional: specify C++ standard or other compiler flags
        .compile("sparsesim"); // Name of the generated library

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/lib.rs");
}
*/
