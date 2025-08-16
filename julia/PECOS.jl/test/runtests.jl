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

using Test
using PECOS

# Check if we should run Aqua tests only
if "aqua" in ARGS
    using Aqua
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
else
    @testset "PECOS.jl Tests" begin
        @testset "Version Information" begin
            version = pecos_version()
            @test version isa String
            @test occursin("PECOS", version)
            @test occursin("v", lowercase(version))  # Should contain version indicator
        end

        @testset "QubitId Type" begin
            # Test valid construction
            q0 = QubitId(0)
            @test q0.index == 0

            q5 = QubitId(5)
            @test q5.index == 5

            # Test invalid construction
            @test_throws ArgumentError QubitId(-1)
            @test_throws ArgumentError QubitId(-10)

            # Test display
            @test string(QubitId(42)) == "QubitId(42)"
        end

        @testset "Type Stability" begin
            # Ensure our functions are type-stable
            @test @inferred(pecos_version()) isa String
            @test @inferred(QubitId(5)) isa QubitId
        end
    end
end
