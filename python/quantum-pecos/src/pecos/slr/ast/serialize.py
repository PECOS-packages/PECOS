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

"""AST serialization and deserialization.

This module provides functions to convert AST nodes to and from JSON format,
enabling persistence and interoperability.

Example:
    >>> from pecos.slr import Main, QReg
    >>> from pecos.slr.qeclib import qubit as qb
    >>> from pecos.slr.ast import slr_to_ast
    >>> from pecos.slr.ast.serialize import ast_to_json, json_to_ast
    >>>
    >>> prog = Main(q := QReg("q", 2), qb.H(q[0]), qb.CX(q[0], q[1]))
    >>> ast = slr_to_ast(prog)
    >>> json_str = ast_to_json(ast)
    >>> restored = json_to_ast(json_str)
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from typing import Any

from pecos.slr.ast.nodes import (
    AllocatorDecl,
    AllocatorTypeExpr,
    ArrayTypeExpr,
    AssignOp,
    AstNode,
    BarrierOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    BitTypeExpr,
    CommentOp,
    ForStmt,
    GateKind,
    GateOp,
    IfStmt,
    LiteralExpr,
    MeasureOp,
    ParallelBlock,
    PermuteOp,
    PrepareOp,
    Program,
    QubitTypeExpr,
    RegisterDecl,
    RepeatStmt,
    ReturnOp,
    SlotRef,
    SourceLocation,
    UnaryExpr,
    UnaryOp,
    VarExpr,
    WhileStmt,
)

# Mapping from class names to classes for deserialization
_NODE_CLASSES: dict[str, type[AstNode]] = {
    "AllocatorDecl": AllocatorDecl,
    "AllocatorTypeExpr": AllocatorTypeExpr,
    "ArrayTypeExpr": ArrayTypeExpr,
    "AssignOp": AssignOp,
    "BarrierOp": BarrierOp,
    "BinaryExpr": BinaryExpr,
    "BitExpr": BitExpr,
    "BitRef": BitRef,
    "BitTypeExpr": BitTypeExpr,
    "CommentOp": CommentOp,
    "ForStmt": ForStmt,
    "GateOp": GateOp,
    "IfStmt": IfStmt,
    "LiteralExpr": LiteralExpr,
    "MeasureOp": MeasureOp,
    "ParallelBlock": ParallelBlock,
    "PermuteOp": PermuteOp,
    "PrepareOp": PrepareOp,
    "Program": Program,
    "QubitTypeExpr": QubitTypeExpr,
    "RegisterDecl": RegisterDecl,
    "RepeatStmt": RepeatStmt,
    "ReturnOp": ReturnOp,
    "SlotRef": SlotRef,
    "SourceLocation": SourceLocation,
    "UnaryExpr": UnaryExpr,
    "VarExpr": VarExpr,
    "WhileStmt": WhileStmt,
}

# Mapping from enum names to enum classes
_ENUM_CLASSES: dict[str, type] = {
    "GateKind": GateKind,
    "BinaryOp": BinaryOp,
    "UnaryOp": UnaryOp,
}


def ast_to_dict(node: AstNode | SourceLocation) -> dict[str, Any]:
    """Convert an AST node to a JSON-serializable dictionary.

    Args:
        node: The AST node to convert.

    Returns:
        A dictionary representation of the node that can be serialized to JSON.
    """
    if not is_dataclass(node):
        msg = f"Expected dataclass, got {type(node)}"
        raise TypeError(msg)

    result: dict[str, Any] = {"_type": type(node).__name__}

    for f in fields(node):
        value = getattr(node, f.name)
        result[f.name] = _serialize_value(value)

    return result


def _serialize_value(value: Any) -> Any:
    """Serialize a single value."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (GateKind, BinaryOp, UnaryOp)):
        return {"_enum": type(value).__name__, "value": value.name}
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return ast_to_dict(value)
    msg = f"Cannot serialize value of type {type(value)}: {value}"
    raise TypeError(msg)


def dict_to_ast(data: dict[str, Any]) -> AstNode:
    """Reconstruct an AST node from a dictionary.

    Args:
        data: Dictionary representation of an AST node.

    Returns:
        The reconstructed AST node.

    Raises:
        ValueError: If the dictionary has an unknown node type.
    """
    node_type = data.get("_type")
    if node_type is None:
        msg = "Dictionary missing '_type' field"
        raise ValueError(msg)

    if node_type == "SourceLocation":
        return _deserialize_source_location(data)

    node_class = _NODE_CLASSES.get(node_type)
    if node_class is None:
        msg = f"Unknown node type: {node_type}"
        raise ValueError(msg)

    # Get field info from the dataclass
    field_info = {f.name: f for f in fields(node_class)}

    # Build kwargs for constructor
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if name == "_type":
            continue
        if name not in field_info:
            continue  # Skip unknown fields for forward compatibility

        kwargs[name] = _deserialize_value(value, name, field_info)

    return node_class(**kwargs)


def _deserialize_source_location(data: dict[str, Any]) -> SourceLocation:
    """Deserialize a SourceLocation."""
    return SourceLocation(
        line=data["line"],
        column=data["column"],
        file=data.get("file"),
    )


def _deserialize_value(value: Any, field_name: str, field_info: dict) -> Any:
    """Deserialize a single value."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool, str)):
        # Handle polymorphic string fields (like target: BitRef | str)
        return value
    if isinstance(value, dict):
        if "_enum" in value:
            # Enum value
            enum_class = _ENUM_CLASSES.get(value["_enum"])
            if enum_class is None:
                msg = f"Unknown enum type: {value['_enum']}"
                raise ValueError(msg)
            return enum_class[value["value"]]
        if "_type" in value:
            # Nested AST node
            return dict_to_ast(value)
        msg = f"Unknown dict format: {value}"
        raise ValueError(msg)
    if isinstance(value, list):
        # Determine if this should be a tuple based on field info
        items = [_deserialize_value(v, field_name, field_info) for v in value]
        # Most AST node tuple fields store tuples
        return tuple(items)
    msg = f"Cannot deserialize value of type {type(value)}: {value}"
    raise TypeError(msg)


def ast_to_json(program: Program, *, indent: int | None = 2) -> str:
    """Serialize an AST Program to a JSON string.

    Args:
        program: The AST Program to serialize.
        indent: JSON indentation level (None for compact output).

    Returns:
        JSON string representation of the program.
    """
    return json.dumps(ast_to_dict(program), indent=indent)


def json_to_ast(json_str: str) -> Program:
    """Deserialize a JSON string to an AST Program.

    Args:
        json_str: JSON string representation of a program.

    Returns:
        The reconstructed Program node.

    Raises:
        ValueError: If the JSON doesn't represent a valid Program.
    """
    data = json.loads(json_str)
    result = dict_to_ast(data)
    if not isinstance(result, Program):
        msg = f"Expected Program, got {type(result).__name__}"
        raise ValueError(msg)
    return result
