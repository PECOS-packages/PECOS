#!/usr/bin/env julia
# Demonstration of PECOS.jl functionality

using PECOS

println("PECOS.jl Demonstration")
println("======================\n")

# 1. Version information
println("1. Version Information:")
version = pecos_version()
println("   $version")
println()

# 2. Creating qubits
println("2. Creating Qubits:")
qubits = QubitId[]
for i = 0:4
    q = QubitId(i)
    push!(qubits, q)
    println("   Created: $q")
end
println()

# 3. Error handling
println("3. Error Handling:")
try
    q_invalid = QubitId(-1)
catch e
    println("   Caught expected error: $e")
end
println()

# 4. Working with collections
println("4. Working with Qubit Collections:")
println("   Number of qubits: $(length(qubits))")
println("   First qubit: $(qubits[1])")
println("   Last qubit: $(qubits[end])")
println()

# 5. Direct FFI demonstration
println("5. Direct FFI Calls (Advanced):")
result = ccall((:add_two_numbers, PECOS.libpecos_julia), Int64, (Int64, Int64), 10, 32)
println("   10 + 32 = $result (called via FFI)")
println()

println("PECOS.jl is ready for quantum error correction simulations!")
