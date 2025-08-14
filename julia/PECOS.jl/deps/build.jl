# Build script for PECOS.jl
# This is run automatically when the package is installed

println("PECOS.jl: Checking for required library...")

# Check if library exists
lib_path = joinpath(@__DIR__, "..", "..", "..", "target", "release")
lib_name =
    Sys.iswindows() ? "pecos_julia.dll" :
    Sys.isapple() ? "libpecos_julia.dylib" : "libpecos_julia.so"
lib_file = joinpath(lib_path, lib_name)

if !isfile(lib_file)
    # Try to build automatically if we're in a Git clone
    if isdir(joinpath(@__DIR__, "..", "..", "..", ".git"))
        println("Detected Git repository. Attempting to build Rust library...")

        # Check for cargo
        try
            run(`cargo --version`)

            # Build the library
            cd(joinpath(@__DIR__, "..", "..", "pecos-julia-ffi")) do
                run(`cargo build --release`)
            end
            println("Successfully built PECOS Rust library!")
        catch
            println("Could not build automatically. Please install Rust or build manually:")
            println("  cd julia/pecos-julia-ffi && cargo build --release")
        end
    else
        println("Library not found. This appears to be a packaged installation.")
        println("  Binary distribution support coming soon!")
    end
else
    println("PECOS library found!")
end
