# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Static cvar_export validation and output-to-source selection."""

import json

from pecos_rslib import validate_phir_result_counts, validate_phir_result_reference


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
        if isinstance(names, list) and any(name is None for name in names):
            msg = f"cvar_export: invalid {field} entry <missing or null>; supply an array of strings"
            raise ValueError(msg)
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            bad = "<missing or null>" if names is None else repr(names)
            msg = f"cvar_export: invalid {field} {bad}; supply an array of strings"
            raise ValueError(msg)
    if len(to) != len(variables):
        msg = f"cvar_export: invalid to {to!r}; supply one output name for each of {len(variables)} variables"
        raise ValueError(msg)
    return variables, node.get("to")


def collect_exports(operations: list[dict]) -> dict[str, str] | None:
    """Validate every export against program-wide declarations and Result names."""
    nodes = all_operations(operations)
    if not any(node.get("data") == "cvar_export" for node in nodes):
        validate_declaration_placement(operations)
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
            elif output != source and output in destinations:
                problem = f"output {output!r} conflicts with a Result destination; choose a different output name"
            elif output in outputs and outputs[output] != source:
                problem = f"output {output!r} has different sources; choose a distinct output name for {source!r}"
            else:
                outputs[output] = source
                continue
            msg = f"cvar_export: {problem}"
            raise ValueError(msg)
    return outputs


def validate_declaration_placement(operations: list[dict]) -> None:
    """Reject block-local declarations unless static exports enable their initialization."""
    if any(node.get("data") == "cvar_export" for node in all_operations(operations)):
        return
    if not isinstance(operations, list):
        return
    for node in operations:
        if not isinstance(node, dict) or "block" not in node:
            continue
        for field in ("ops", "true_branch", "false_branch"):
            for child in all_operations(node.get(field) or []):
                if child.get("data") in ("cvar_define", "qvar_define"):
                    name = json.dumps(child.get("variable"))
                    msg = (
                        f"Nested declaration {child['data']} {name}: move the declaration to the top level; "
                        "declarations inside blocks are only supported in programs that use cvar_export"
                    )
                    raise ValueError(msg)


def collect_result_types(operations: list[dict]) -> dict[str, tuple[str, int]]:
    """Resolve Result storage statically, including copies in unexecuted branches."""
    nodes = all_operations(operations)
    types = {node["variable"]: (node["data_type"], node["size"]) for node in nodes if node.get("data") == "cvar_define"}
    declared = set(types)
    quantum = {node["variable"] for node in nodes if node.get("data") == "qvar_define"}
    pairs = []
    for node in nodes:
        if node.get("cop") == "Result":
            args, returns = node.get("args", []), node.get("returns", [])
            validate_phir_result_counts(len(args), len(returns))
            pairs.extend(
                (
                    validate_phir_result_reference(json.dumps(source), "args"),
                    validate_phir_result_reference(json.dumps(dest), "returns"),
                )
                for source, dest in zip(args, returns, strict=True)
            )
    pending = [(source, dest) for (source, _), (dest, _) in pairs]
    source_types = {}
    while pending:
        unresolved = []
        for source, dest in pending:
            if dest in quantum:
                msg = (
                    "PHIR interpreter failed: Result: destination "
                    f"{json.dumps(dest, ensure_ascii=False)} is quantum; "
                    "use a classical variable or a new export name"
                )
                raise ValueError(msg)
            if source not in types:
                unresolved.append((source, dest))
                continue
            dtype, size = types[source]
            signature = (size, dtype.startswith("i"))
            previous = source_types.get(dest)
            if previous is not None and (previous[0] != size or (dest not in declared and previous != signature)):
                msg = (
                    "PHIR interpreter failed: Result: destination "
                    f"{json.dumps(dest, ensure_ascii=False)} has incompatible source widths or signedness; "
                    "use sources with the same width and signedness"
                )
                raise ValueError(msg)
            source_types[dest] = signature
            if dest not in types:
                types[dest] = (dtype, size)
        if len(unresolved) == len(pending):
            source, dest = unresolved[0]
            msg = (
                "PHIR interpreter failed: Result: unknown classical variable "
                f"{json.dumps(source, ensure_ascii=False)} "
                f"for destination {json.dumps(dest, ensure_ascii=False)}; "
                "declare it with cvar_define"
            )
            raise ValueError(msg)
        pending = unresolved
    for pair in pairs:
        for name, index in pair:
            size = types[name][1]
            if index is not None and index >= size:
                msg = (
                    f"PHIR interpreter failed: Result: bit index {index} is invalid for register "
                    f"{json.dumps(name, ensure_ascii=False)} "
                    f"of declared size {size}; use an index below {size}"
                )
                raise ValueError(msg)
    return {dest: types[dest] for _, (dest, _) in pairs}


def validate_measure_returns(name: str, args: list, returns: list | None) -> None:
    """Require an explicit destination for each measured qubit before execution."""
    if name == "Measure" and (not returns or len(returns) != len(args)):
        msg = f"Measure: invalid returns {returns!r} for {len(args)} args; give each measured qubit a destination"
        raise ValueError(msg)
