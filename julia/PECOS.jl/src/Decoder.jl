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
Foreign decoder plugin interface for Julia.

Implement the `AbstractDecoder` interface and PECOS will use your decoder
through its native trait system. Only 3 methods needed:

    decode(d::MyDecoder, syndrome::Vector{UInt8}) -> DecodingResult
    check_count(d::MyDecoder) -> Int
    bit_count(d::MyDecoder) -> Int

# Example

```julia
struct MyDecoder <: PECOS.AbstractDecoder
    checks::Int
    bits::Int
end

function PECOS.decode(d::MyDecoder, syndrome::Vector{UInt8})
    observable = zeros(UInt8, d.bits)
    # ... your decoding logic ...
    return PECOS.DecodingResult(observable, 1.0, true)
end

PECOS.check_count(d::MyDecoder) = d.checks
PECOS.bit_count(d::MyDecoder) = d.bits
```
"""

export AbstractDecoder, DecodingResult, decode, check_count, bit_count

"""
Result of a decoding operation.
"""
struct DecodingResult
    """
    Decoded observable/correction vector.
    """
    observable::Vector{UInt8}
    """
    Weight/cost of the solution.
    """
    weight::Float64
    """
    Whether the decoder converged (nothing = unknown).
    """
    converged::Union{Bool,Nothing}
end

DecodingResult(observable, weight) = DecodingResult(observable, weight, nothing)

"""
Abstract type for all PECOS-compatible decoders.

Subtypes must implement:

  - `decode(d, syndrome::Vector{UInt8}) -> DecodingResult`
  - `check_count(d) -> Int`
  - `bit_count(d) -> Int`
"""
abstract type AbstractDecoder end

# Interface methods (to be implemented by subtypes)
function decode end
function check_count end
function bit_count end

# ============================================================================
# C ABI callback wrappers
#
# These functions are called from Rust via function pointers. They look up
# the Julia decoder and dispatch to the user's methods.
# ============================================================================

# Global registry: maps Ptr{Cvoid} handles to Julia decoder objects.
# Julia objects can't cross FFI directly, so we use integer handles.
const _decoder_registry = Dict{UInt,AbstractDecoder}()
const _decoder_next_handle = Ref{UInt}(0)
const _decoder_lock = ReentrantLock()

function _register_decoder(d::AbstractDecoder)::Ptr{Cvoid}
    lock(_decoder_lock) do
        _decoder_next_handle[] += 1
        h = _decoder_next_handle[]
        _decoder_registry[h] = d
        return Ptr{Cvoid}(h)
    end
end

function _unregister_decoder(handle::Ptr{Cvoid})
    lock(_decoder_lock) do
        delete!(_decoder_registry, UInt(handle))
    end
end

function _lookup_decoder(handle::Ptr{Cvoid})::Union{AbstractDecoder,Nothing}
    lock(_decoder_lock) do
        get(_decoder_registry, UInt(handle), nothing)
    end
end

# C-ABI result struct (matches PecosDecodingResultRaw)
struct CDecodingResultRaw
    observable_ptr::Ptr{UInt8}
    observable_len::Csize_t
    weight::Cdouble
    converged::Int8
    error_ptr::Ptr{UInt8}
    error_len::Csize_t
end
