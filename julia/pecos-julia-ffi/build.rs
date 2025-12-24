fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // For macOS, link against the system C++ library from dyld shared cache
    if std::env::var("TARGET")
        .unwrap_or_default()
        .contains("darwin")
    {
        // Prioritize /usr/lib to prevent opportunistic linking to Homebrew's libunwind
        println!("cargo:rustc-link-search=native=/usr/lib");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-arg=-Wl,-search_paths_first");
    }
}
