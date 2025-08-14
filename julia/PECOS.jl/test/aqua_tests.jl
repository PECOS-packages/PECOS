# Run Aqua.jl quality checks
using Pkg

# Activate test environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Add parent directory to LOAD_PATH to find PECOS
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using PECOS
using Aqua

println("Running Aqua.jl quality checks...")

Aqua.test_all(
    PECOS;
    ambiguities = false,
    unbound_args = true,
    undefined_exports = true,
    project_extras = true,
    stale_deps = false,
    deps_compat = true,
    piracies = true,
    persistent_tasks = false,
)

println("Aqua.jl checks completed successfully!")
