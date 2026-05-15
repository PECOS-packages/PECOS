# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
    PECOS.jl

Julia interface for PECOS quantum error correction library.
"""
module PECOS

# Artifacts will be used in future releases
# using Pkg.Artifacts

export pecos_version, QubitId, libpecos_julia

include("Decoder.jl")
include("Simulator.jl")

# Determine library path based on environment
const libpecos_julia = begin
    lib_name = if Sys.iswindows()
        "pecos_julia.dll"
    elseif Sys.isapple()
        "libpecos_julia.dylib"
    else
        "libpecos_julia.so"
    end

    explicit_lib_dir = get(ENV, "PECOS_JULIA_LIB_DIR", "")
    candidate_dirs = if isempty(explicit_lib_dir)
        [
            joinpath(@__DIR__, "..", "..", "..", "target", "release"),
            joinpath(@__DIR__, "..", "..", "..", "target", "native"),
            joinpath(@__DIR__, "..", "..", "..", "target", "debug"),
        ]
    else
        [explicit_lib_dir]
    end

    candidates = [joinpath(candidate_dir, lib_name) for candidate_dir in candidate_dirs]
    found_index = findfirst(isfile, candidates)

    if found_index === nothing
        searched = join(candidate_dirs, "\n                    ")
        error("""
            PECOS Julia library not found!

            Searched:
                $searched

            Build it with:
                just julia-build release

            Or select a specific build directory with PECOS_JULIA_LIB_DIR.
        """)
    end

    candidates[found_index]
end

struct QubitId
    index::Int64

    # Inner constructor with validation
    function QubitId(index::Integer)
        index < 0 && throw(ArgumentError("QubitId index must be non-negative"))
        new(Int64(index))
    end
end

Base.show(io::IO, q::QubitId) = print(io, "QubitId($(q.index))")

function pecos_version()
    ptr = ccall((:pecos_version, libpecos_julia), Ptr{UInt8}, ())
    version = unsafe_string(ptr)
    ccall((:free_rust_string, libpecos_julia), Cvoid, (Ptr{UInt8},), ptr)
    return version
end

function __init__()
    # Verify library can be loaded
    if !isfile(libpecos_julia)
        error("PECOS Julia library not found at: $libpecos_julia")
    end
end

end # module PECOS
