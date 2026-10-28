# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Shared quantum argument expansion, grouping and slot identity for SLR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.qalloc import QAlloc, QubitRef
from pecos.slr.vars import QReg, Qubit, SymbolicElem

if TYPE_CHECKING:
    from collections.abc import Sequence


def qubit_key(qubit: Qubit | QubitRef) -> tuple[str, int]:
    """Identify the same named slot across registers and allocator references."""
    return qubit.reg.sym, qubit.index


def _reject_symbolic_indices(qargs: Sequence, gate_name: str) -> None:
    for arg in qargs:
        match arg:
            case SymbolicElem():
                msg = f"Gate {gate_name}: loop-variable qubit indices are not supported by this conversion: {arg}"
                raise ValueError(msg)
            case tuple() | list():
                _reject_symbolic_indices(arg, gate_name)


def expand_qubit_args(qargs: Sequence, gate_name: str) -> list[Qubit | QubitRef]:
    """Expand register/allocator contents and nested collections into qubits.

    This collects operands, without assigning them to multi-qubit applications.
    QAlloc iteration includes only available, non-reserved slots.
    """
    _reject_symbolic_indices(qargs, gate_name)
    expanded = []
    for arg in qargs:
        match arg:
            case tuple() | list():
                expanded.extend(expand_qubit_args(arg, gate_name))
            case QReg() | QAlloc():
                expanded.extend(arg)
            case Qubit() | QubitRef():
                expanded.append(arg)
            case _:
                msg = f"Gate {gate_name}: expected a qubit argument, got {arg!s} ({arg!r})"
                raise ValueError(msg)
    return expanded


def gate_target_groups(qargs: Sequence, arity: int, gate_name: str) -> list[tuple[Qubit | QubitRef, ...]]:
    """Resolve gate applications, preserving explicit groups and register indices."""
    _reject_symbolic_indices(qargs, gate_name)
    if arity == 1:
        qubits = expand_qubit_args(qargs, gate_name)
        if not qubits:
            msg = f"Gate {gate_name} has no qubit targets for arguments {qargs!r}"
            raise ValueError(msg)
        return [(qubit,) for qubit in qubits]

    invalid_args = (
        f"Gate {gate_name} requires {arity} individual qubits or equally sized QRegs "
        f"per application (optionally grouped in tuples); got arguments {qargs!r}"
    )
    if any(isinstance(arg, tuple) for arg in qargs):
        if not all(isinstance(arg, tuple) for arg in qargs):
            raise ValueError(invalid_args)
        applications = qargs
    else:
        applications = (qargs,)

    groups = []
    for args in applications:
        if len(args) != arity:
            raise ValueError(invalid_args)
        if all(isinstance(arg, QReg) for arg in args):
            if any(arg.size != args[0].size for arg in args):
                sizes = ", ".join(f"{arg.sym} (size {arg.size})" for arg in args)
                msg = f"Gate {gate_name} requires equally sized registers; got {sizes}"
                raise ValueError(msg)
            target_groups = zip(*(arg.elems for arg in args), strict=True)
        elif all(isinstance(arg, (Qubit, QubitRef)) for arg in args):
            target_groups = (tuple(args),)
        else:
            raise ValueError(invalid_args)
        for group in target_groups:
            seen = set()
            for qubit in group:
                key = qubit_key(qubit)
                if key in seen:
                    msg = f"Gate {gate_name} repeats qubit {key[0]}[{key[1]}]"
                    raise ValueError(msg)
                seen.add(key)
            groups.append(group)
    return groups
