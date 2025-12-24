# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Type annotations for SLR blocks to specify return types for code generation.

This module provides type annotations that allow Block subclasses to declare their
return types, which is essential for proper code generation in languages with strict
type systems (like Guppy).

Example:
    from pecos.slr import Block
    from pecos.slr.types import Array, Qubit

    class PrepEncodingFTZero(Block):
        # Declares that this block returns two quantum arrays: size 2 and size 7
        returns = (Array[Qubit, 2], Array[Qubit, 7])

        def __init__(self, data, ancilla, init_bit):
            # ... implementation ...
"""

from __future__ import annotations


class TypeAnnotation:
    """Base class for SLR type annotations."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ElementType(TypeAnnotation):
    """Represents a quantum or classical element type."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name


class ArrayType(TypeAnnotation):
    """Represents an array type with element type and size.

    Usage:
        Array[Qubit, 5]  # Array of 5 qubits
        Array[Bit, 3]    # Array of 3 classical bits
    """

    def __init__(self, elem_type: ElementType, size: int):
        self.elem_type = elem_type
        self.size = size

    def __repr__(self) -> str:
        return f"Array[{self.elem_type}, {self.size}]"

    def __class_getitem__(cls, params):
        """Support Array[Qubit, N] syntax."""
        if not isinstance(params, tuple) or len(params) != 2:
            msg = "Array requires exactly 2 parameters: Array[ElementType, size]"
            raise TypeError(msg)
        elem_type, size = params
        if not isinstance(elem_type, ElementType):
            msg = f"First parameter must be an ElementType, got {type(elem_type)}"
            raise TypeError(msg)
        if not isinstance(size, int):
            msg = f"Second parameter must be an int, got {type(size)}"
            raise TypeError(msg)
        return cls(elem_type, size)

    def to_guppy_type(self) -> str:
        """Convert to Guppy type string."""
        return f"array[quantum.qubit, {self.size}]"


# Predefined element types
Qubit = ElementType("Qubit")
Bit = ElementType("Bit")

# Aliases for clarity when used in type annotations alongside slr.Qubit/slr.Bit variables
QubitType = Qubit
BitType = Bit

# Export the Array class for use in annotations
Array = ArrayType


class _ReturnNotSetType:
    """Sentinel type indicating that block_returns has not been explicitly set."""

    def __repr__(self) -> str:
        return "ReturnNotSet"

    def __bool__(self) -> bool:
        return False


# Sentinel value for blocks that haven't declared their return type
ReturnNotSet = _ReturnNotSetType()
