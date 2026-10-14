fn main() {
    println!("cargo:rerun-if-changed=src/execution_guard.c");
    cc::Build::new()
        .file("src/execution_guard.c")
        .std("c11")
        .compile("pecos_execution_guard");
}
