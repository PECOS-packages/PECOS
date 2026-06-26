# Copyright 2026 The PECOS Developers
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
Foreign simulator plugin interface for Julia.

Implement the `AbstractCliffordSimulator` interface and PECOS will use your
simulator through its native trait system.

# Clifford simulator (5 methods)

    sz!(sim, qubits::Vector{Int})
    h!(sim, qubits::Vector{Int})
    cx!(sim, pairs::Vector{Tuple{Int,Int}})
    mz(sim, qubits::Vector{Int}) -> Vector{MeasurementResult}
    reset!(sim)

All 52 other Clifford gates are automatically decomposed into these.

# Universal simulator (add 3 more)

Also subtype `AbstractRotationSimulator` and implement:

    rx!(sim, theta::Float64, qubits::Vector{Int})
    rz!(sim, theta::Float64, qubits::Vector{Int})
    rzz!(sim, theta::Float64, pairs::Vector{Tuple{Int,Int}})

# Example

```julia
mutable struct MyStabSim <: PECOS.AbstractCliffordSimulator
    state::Vector{Bool}
end

PECOS.sz!(s::MyStabSim, qubits) = nothing  # phase gate
PECOS.h!(s::MyStabSim, qubits) = for q in qubits; s.state[q+1] = !s.state[q+1]; end
PECOS.cx!(s::MyStabSim, pairs) = for (c,t) in pairs; s.state[c+1] && (s.state[t+1] = !s.state[t+1]); end
PECOS.mz(s::MyStabSim, qubits) = [PECOS.MeasurementResult(s.state[q+1], true) for q in qubits]
PECOS.reset!(s::MyStabSim) = fill!(s.state, false)
```
"""

export AbstractCliffordSimulator, AbstractRotationSimulator, MeasurementResult
export sz!, h!, cx!, mz, reset!, rx!, rz!, rzz!

"""
Result of a Z-basis measurement.
"""
struct MeasurementResult
    """
    Measurement outcome: false = |0>, true = |1>.
    """
    outcome::Bool
    """
    Whether the outcome was deterministic.
    """
    is_deterministic::Bool
end

"""
Abstract type for Clifford-only simulators.

Subtypes must implement: `sz!`, `h!`, `cx!`, `mz`, `reset!`
"""
abstract type AbstractCliffordSimulator end

"""
Abstract type for universal (Clifford + rotation) simulators.

Subtypes must also implement: `rx!`, `rz!`, `rzz!`
"""
abstract type AbstractRotationSimulator <: AbstractCliffordSimulator end

# Interface methods (to be implemented by subtypes)
function sz! end
function h! end
function cx! end
function mz end
function reset! end
function rx! end
function rz! end
function rzz! end

# ============================================================================
# Registry (same pattern as Decoder.jl)
# ============================================================================

const _sim_registry = Dict{UInt,AbstractCliffordSimulator}()
const _sim_next_handle = Ref{UInt}(0)
const _sim_lock = ReentrantLock()

function _register_simulator(s::AbstractCliffordSimulator)::Ptr{Cvoid}
    lock(_sim_lock) do
        _sim_next_handle[] += 1
        h = _sim_next_handle[]
        _sim_registry[h] = s
        return Ptr{Cvoid}(h)
    end
end

function _unregister_simulator(handle::Ptr{Cvoid})
    lock(_sim_lock) do
        delete!(_sim_registry, UInt(handle))
    end
end

function _lookup_simulator(handle::Ptr{Cvoid})::Union{AbstractCliffordSimulator,Nothing}
    lock(_sim_lock) do
        get(_sim_registry, UInt(handle), nothing)
    end
end

# C-ABI measurement result (matches PecosMeasurementResult)
struct CMeasurementResult
    outcome::UInt8
    is_deterministic::UInt8
end
