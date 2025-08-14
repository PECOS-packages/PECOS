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

module PECOS_julia_jll

using Libdl

# This is a development JLL that looks for the library in the workspace build location
const workspace_root = joinpath(@__DIR__, "..", "..", "..", "..", "..", "..", "PECOS")
const libpecos_julia_path = joinpath(workspace_root, "target", "debug")
const libpecos_julia = joinpath(
    libpecos_julia_path,
    Sys.iswindows() ? "pecos_julia.dll" :
    Sys.isapple() ? "libpecos_julia.dylib" : "libpecos_julia.so",
)

function __init__()
    if !isfile(libpecos_julia)
        @warn """
        PECOS Julia library not found at: $libpecos_julia

        Please build it first:
        cd julia/pecos-julia-ffi
        cargo build
        """
    else
        # Verify we can load it
        try
            dlopen(libpecos_julia)
            @info "Loaded PECOS library from: $libpecos_julia"
        catch e
            @error "Failed to load PECOS library" exception=e
        end
    end
end

# Re-export for compatibility
export libpecos_julia

end # module
