# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Static cvar_export validation and output-to-source selection."""


def all_operations(operations: list[dict]) -> list[dict]:
    """Collect operations in every block and both branches, in source order."""
    result = []
    if not isinstance(operations, list):
        # Structural errors belong to the schema, as in declaration validation.
        return result
    for node in operations:
        if not isinstance(node, dict):
            continue
        result.append(node)
        if "block" in node:
            for field in ("ops", "true_branch", "false_branch"):
                result.extend(all_operations(node.get(field) or []))
    return result


def decode_export(node: dict) -> tuple[list[str], list[str] | None]:
    """Decode a directive's shape before constructing its Python representation."""
    variables = node.get("variables")
    to = node.get("to", variables)
    for field, names in (("variables", variables), ("to", to)):
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            msg = f"cvar_export: invalid {field} {names!r}; supply an array of strings"
            raise ValueError(msg)
    if len(to) != len(variables):
        msg = f"cvar_export: invalid to {to!r}; supply one output name for each of {len(variables)} variables"
        raise ValueError(msg)
    return variables, node.get("to")


def collect_exports(operations: list[dict]) -> dict[str, str] | None:
    """Validate every export against program-wide declarations and Result names."""
    nodes = all_operations(operations)
    if not any(node.get("data") == "cvar_export" for node in nodes):
        return None
    declarations = {
        node["variable"]: node["data"]
        for node in nodes
        if node.get("data") in {"cvar_define", "qvar_define"} and isinstance(node.get("variable"), str)
    }
    destinations = {
        ref if isinstance(ref, str) else ref[0]
        for node in nodes
        if node.get("cop") == "Result"
        for ref in node.get("returns", [])
    }
    outputs = {}
    for node in nodes:
        if node.get("data") != "cvar_export":
            continue
        variables, to = decode_export(node)
        for source, output in zip(variables, variables if to is None else to, strict=True):
            if declarations.get(source) != "cvar_define":
                problem = f"unknown classical variable {source!r}; declare it with cvar_define"
            elif output.startswith("_width_"):
                problem = f"reserved output {output!r}; choose a name without the _width_ prefix"
            elif output != source and output in declarations:
                problem = f"output {output!r} conflicts with a declared variable; choose a different output name"
            elif output in destinations:
                problem = f"output {output!r} conflicts with a Result destination; choose a different output name"
            elif output in outputs and outputs[output] != source:
                problem = f"output {output!r} has different sources; choose a distinct output name for {source!r}"
            else:
                outputs[output] = source
                continue
            msg = f"cvar_export: {problem}"
            raise ValueError(msg)
    return outputs
