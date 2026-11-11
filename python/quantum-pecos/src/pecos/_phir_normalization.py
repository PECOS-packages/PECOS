# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Canonical PHIR operation names at JSON program input boundaries."""

import copy


def canonicalize_measurement_qops(program: dict) -> dict:
    """Accept MeasureZ while preserving upstream Measure output and caller data.

    Only paths to changed operations are copied; unchanged objects are reused.
    Metadata and classical expressions are untouched, including any strings or
    arbitrary Python objects they contain.
    Structural errors remain the responsibility of the existing validators.
    """

    def visit(operations: object) -> object:
        if not isinstance(operations, list):
            return operations
        result = operations
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                continue
            node = operation
            if node.get("qop") == "MeasureZ":
                node = copy.copy(operation)
                node["qop"] = "Measure"
            if "block" in node:
                for field in ("ops", "true_branch", "false_branch"):
                    if field in node:
                        children = visit(node[field])
                        if children is not node[field]:
                            if node is operation:
                                node = copy.copy(operation)
                            node[field] = children
            if node is not operation:
                if result is operations:
                    result = list(operations)
                result[index] = node
        return result

    if "ops" not in program:
        return program
    ops = visit(program["ops"])
    if ops is program["ops"]:
        return program
    result = copy.copy(program)
    result["ops"] = ops
    return result
