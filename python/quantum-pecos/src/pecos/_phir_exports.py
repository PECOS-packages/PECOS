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
    widths = {name: size for name, (_, size) in types.items()}
    data_types = {name: dtype for name, (dtype, _) in types.items()}
    # Signedness is a bit set: 1 for unsigned sources, 2 for signed sources.
    signs = {name: 2 if dtype.startswith("i") else 1 for name, (dtype, _) in types.items()}
    indexed = {dest for _, (dest, index) in pairs if index is not None}
    whole = {dest for (_, index), (dest, _) in pairs if index is None}
    for (_source, source_index), (dest, index) in pairs:
        if dest in quantum:
            msg = (
                "PHIR interpreter failed: Result: destination "
                f"{json.dumps(dest, ensure_ascii=False)} is quantum; "
                "use a classical variable or a new export name"
            )
            raise ValueError(msg)
        if dest in declared:
            continue
        if index is not None:
            if index >= 64:
                _validate_result_index(dest, index, 64)
            widths[dest] = max(widths.get(dest, 0), 32, index + 1)
        if source_index is not None:
            widths[dest] = max(widths.get(dest, 0), 1)
        if dest in indexed or dest not in whole:
            signs[dest] = 1
            data_types[dest] = "u64"
    _propagate_result_types(pairs, declared, indexed, widths, signs, data_types)
    # Default unanchored cycles only after resolving anchored types, then propagate
    # again so those defaults can widen downstream mixed-type copies.
    for _, (dest, _) in pairs:
        if not signs.get(dest):
            signs[dest] = 1
        data_types.setdefault(dest, "i64" if signs[dest] == 2 else "u64")
    _propagate_result_types(pairs, declared, indexed, widths, signs, data_types)
    for (source, _), _ in pairs:
        if source not in widths:
            msg = (
                "PHIR interpreter failed: Result or measurement: unknown classical variable "
                f"{json.dumps(source, ensure_ascii=False)}; declare it with cvar_define before use"
            )
            raise ValueError(msg)
    for _, (dest, _) in pairs:
        if dest in declared:
            continue
        sources = {other: signs[other] for (other, index), (target, _) in pairs if target == dest and index is None}
        signed = next((name for name in sorted(sources) if sources[name] & 2), None)
        unsigned = next((name for name in sorted(sources) if sources[name] & 1 and name != signed), None)
        if signed is not None and unsigned is not None:
            msg = (
                "PHIR interpreter failed: Result: destination "
                f"{json.dumps(dest, ensure_ascii=False)} has conflicting signedness from sources "
                f"{json.dumps(signed, ensure_ascii=False)} and {json.dumps(unsigned, ensure_ascii=False)}; "
                "declare the destination or use whole-register sources with the same signedness"
            )
            raise ValueError(msg)
        width = widths.get(dest)
        if width is None:
            msg = (
                f"Result width invariant broken for destination {dest!r}: every source has a width "
                "after unknown-source validation, so every destination must have a propagated width"
            )
            raise RuntimeError(msg)
        types[dest] = (data_types[dest], width)
    for pair in pairs:
        for name, index in pair:
            _validate_result_index(name, index, types[name][1])
    return {dest: types[dest] for _, (dest, _) in pairs}


def _propagate_result_types(
    pairs: list[tuple[tuple[str, int | None], tuple[str, int | None]]],
    declared: set[str],
    indexed: set[str],
    widths: dict[str, int],
    signs: dict[str, int],
    data_types: dict[str, str],
) -> None:
    while True:
        changed = False
        for (source, source_index), (dest, _) in pairs:
            if dest in declared:
                continue
            if source in widths:
                size = 1 if source_index is not None else widths[source]
                changed |= widths.get(dest, -1) < size
                widths[dest] = max(widths.get(dest, 0), size)
            if source_index is None and dest not in indexed:
                sign = signs.get(source, 0)
                previous = signs.get(dest, 0)
                changed |= previous | sign != previous
                signs[dest] = previous | sign
                if source in data_types:
                    dtype = data_types[source]
                    if dest in data_types and data_types[dest] != dtype:
                        dtype = "i64" if signs[dest] == 2 else "u64"
                    changed |= data_types.get(dest) != dtype
                    data_types[dest] = dtype
        if not changed:
            return


def _validate_result_index(name: str, index: int | None, size: int) -> None:
    if index is not None and index >= size:
        msg = (
            f"PHIR interpreter failed: Result: bit index {index} is invalid for register "
            f"{json.dumps(name, ensure_ascii=False)} of size {size}; use an index below {size}"
        )
        raise ValueError(msg)


def flatten_measure_args(args: list) -> list:
    """Flatten PHIR qubit groups and resolved QOp integer groups in source order."""
    return [
        qubit
        for arg in args
        for qubit in (arg if isinstance(arg, list | tuple) and (not arg or not isinstance(arg[0], str)) else [arg])
    ]


def validate_measure_returns(name: str, args: list, returns: list | None) -> None:
    """Require array fields and one explicit return per flattened measured qubit."""
    if name != "Measure":
        return
    if (
        not isinstance(args, list)
        or not isinstance(returns, list)
        or not returns
        or len(returns) != len(flatten_measure_args(args))
    ):
        encoded_args = json.dumps(args, separators=(",", ":"), ensure_ascii=False)
        encoded_returns = json.dumps(returns, separators=(",", ":"), ensure_ascii=False)
        msg = (
            f"Measure args {encoded_args}: invalid args or returns {encoded_returns}; "
            "supply arrays and give each measured qubit a destination"
        )
        raise ValueError(msg)
