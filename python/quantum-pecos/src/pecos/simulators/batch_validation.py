# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Spelling-based batch checks shared by the Python simulator wrappers.

Bindings still decide which gates a wrapper supports. Overriding a known
spelling retains its arity; a different-arity override must update GATE_ARITIES.
"""

from collections.abc import Iterable, Iterator, Mapping, Sized
from operator import index

from pecos_rslib import _gate_bindings_symbols

from pecos.simulators.gate_syms import alt_symbols

# Native tuple layout: (spelling, is_measurement, parameter_mode, qubit_count).
GATE_ARITIES = {symbol: arity for symbol, _measurement, _parameters, arity in _gate_bindings_symbols()}
# Explicit Python spellings/policies, including those also present natively.
GATE_ARITIES.update(
    {
        "Measure +X": 1,
        "Measure -X": 1,
        "Measure +Y": 1,
        "Measure -Y": 1,
        "Measure +Z": 1,
        "Measure -Z": 1,
        "force output": 0,
        "U1q": 1,
        "MS": 2,
        "MSXX": 2,
        "MSYY": 2,
        "MSZZ": 2,
        "ZZ": 2,
        "ZZMax": 2,
        "ZZPhase": 2,
        "I": 1,
    },
)
for _alias in alt_symbols:
    _target = _alias
    _seen = set()
    while _target not in GATE_ARITIES and _target in alt_symbols and _target not in _seen:
        _seen.add(_target)
        _target = alt_symbols[_target]
    if _target in GATE_ARITIES:
        GATE_ARITIES[_alias] = GATE_ARITIES[_target]


def replayable_locations(locations: Iterable) -> Iterable:
    """Buffer one-shot iterators at dispatch boundaries, retaining order and objects.

    Ordinary caller containers are returned unchanged. A generator needs a
    replay buffer so validation can finish before any binding is invoked.
    """
    return tuple(locations) if isinstance(locations, Iterator) else locations


def is_empty_batch(locations: object) -> bool:
    """Check container length without invoking array truth-value conversion.

    Preserve the historical falsey bare-zero/None shortcut. Iterators have no
    length and are truthy, so their replay buffer determines whether they run.
    """
    try:
        return len(locations) == 0
    except TypeError:
        # Zero-dimensional arrays advertise __len__ but have no usable length.
        return not locations


def validate_batch(
    symbol: str,
    locations: Iterable,
    capacity: int | None,
    *,
    allow_singletons: bool = True,
    allow_sequences: bool = False,
) -> None:
    """Check all locations without changing their container or dispatching gates.

    States select whether singleton locations and general sized sequences are
    accepted by their bindings. Unknown spellings and location-free operations skip
    checks. None capacity means growable, but indices must still be nonnegative.
    Only qubits within a single location must be distinct.
    """
    arity = GATE_ARITIES.get(symbol)
    if arity is None or arity == 0:
        return
    if _plain_batch_is_valid(arity, locations, capacity):
        return
    _validate_each(
        symbol,
        arity,
        locations,
        capacity,
        allow_singletons=allow_singletons,
        allow_sequences=allow_sequences,
    )


def _plain_batch_is_valid(arity: int, locations: Iterable, capacity: int | None) -> bool:
    """Accept the common shapes (plain ints, or 2-tuples of plain ints) at builtin speed.

    Returns False for anything else, including invalid input, so the caller can
    run the general check and raise its precise error. One-shot iterators also
    take the general, single-pass check, because this path reads the batch twice.
    """
    if isinstance(locations, Iterator):
        return False
    if arity == 1:
        if not all(type(q) is int for q in locations):
            return False
        low = min(locations, default=0)
        high = max(locations, default=-1)
        return low >= 0 and (capacity is None or high < capacity)
    if arity == 2:
        for location in locations:
            if type(location) is not tuple or len(location) != 2:
                return False
            a, b = location
            if type(a) is not int or type(b) is not int or a == b or a < 0 or b < 0:
                return False
            if capacity is not None and (a >= capacity or b >= capacity):
                return False
        return True
    return False


def _validate_each(
    symbol: str,
    arity: int,
    locations: Iterable,
    capacity: int | None,
    *,
    allow_singletons: bool,
    allow_sequences: bool,
) -> None:
    for location in locations:
        try:
            index(location)
        except TypeError:
            is_sequence = isinstance(location, (tuple, list)) or (
                allow_sequences
                and isinstance(location, Sized)
                and hasattr(location, "__getitem__")
                and not isinstance(location, (str, bytes, bytearray, Mapping))
            )
        else:
            # Integer-like objects take precedence over their sequence methods.
            is_sequence = False
        qubits = location if is_sequence else (location,)
        try:
            size = len(qubits)
        except TypeError as exc:
            msg = f"{symbol}: expected a qubit index or a sized sequence, got {location!r}"
            raise ValueError(msg) from exc
        if arity == 1 and is_sequence and not allow_singletons and size == 1:
            msg = f"{symbol}: expected a scalar qubit index, got {location!r}"
            raise ValueError(msg)
        if size != arity:
            hint = "; use II for the two-qubit identity" if symbol == "I" and size == 2 else ""
            msg = f"{symbol}: expected {arity} qubit(s) per location, got {location!r}{hint}"
            raise ValueError(msg)
        seen = set()
        for qubit in qubits:
            try:
                q = index(qubit)
            except TypeError as exc:
                msg = f"{symbol}: expected a non-negative qubit index, got {qubit!r}"
                raise ValueError(msg) from exc
            # Results are keyed by location, and Python simulators store qubits
            # in sets, so reject unhashable indices before dispatch.
            try:
                hash(qubit)
            except TypeError as exc:
                msg = f"{symbol}: expected a hashable qubit index, got {qubit!r}"
                raise ValueError(msg) from exc
            if q < 0 or (capacity is not None and q >= capacity):
                msg = f"{symbol}: qubit {q} is out of bounds (capacity={capacity})"
                raise ValueError(msg)
            if q in seen:
                msg = f"{symbol} requires distinct qubits; repeated qubit {q}"
                raise ValueError(msg)
            seen.add(q)
