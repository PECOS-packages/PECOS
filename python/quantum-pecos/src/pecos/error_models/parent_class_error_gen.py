# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Simple error generator meant to demonstrate a basic error generator that produces errors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pecos as pc
from pecos.error_models.class_errors_circuit import ErrorCircuits

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pecos.circuits import QuantumCircuit
    from pecos.typing import GateParams, LocationSet


class ParentErrorModel:
    """A simple error generator for the depolarizing model.

    This error generator does not allow much modification of the error model.
    """

    def __init__(self) -> None:
        """Initialize a ParentErrorModel instance.

        Sets up the base error model with placeholders for error circuits,
        error parameters, quantum circuit, and the generator class.
        """
        self.error_circuits = None

        self.error_params = None
        self.circuit = None
        self.generator_class = Generator

    def start(
        self,
        circuit: QuantumCircuit,
        error_params: dict[str, Any],
    ) -> ErrorCircuits:
        """Start up at the beginning of a circuit simulation.

        Args:
            circuit: Quantum circuit to simulate.
            error_params: Dictionary of error parameters.

        """
        self.error_circuits = ErrorCircuits()
        self.circuit = circuit
        self.error_params = error_params

        return self.error_circuits

    def generate_tick_errors(
        self,
        _tick_circuit: QuantumCircuit,
        _time: int,
        **_params: GateParams,
    ) -> dict:
        """Returns before errors, after errors, and replaced locations for the given key (args).

        This method should be overridden in subclasses.

        Args:
            tick_circuit: The tick circuit containing gate operations
            time: The time index or tuple indicating when errors occur
            **params: Additional parameters for error generation

        Raises:
            NotImplementedError: This base implementation should be overridden
        """
        msg = "Subclasses must implement generate_tick_errors"
        raise NotImplementedError(msg)


class Generator:
    """Class that tracks and generates errors for gates and gate groups.

    Keeps track of how errors are generated for each gate and groups of gates.
    It also has a method for generating errors.
    """

    def __init__(self) -> None:
        """Initialize a Generator instance.

        Sets up empty gate groups, error function dictionary, and default
        error tuple for tracking and generating errors.
        """
        self.gate_groups = {}
        self.error_func_dict = {}
        self.default_error_tuple = (False, "p")

    def set_gate_group(self, group_symbol: str, gate_set: Iterable[str]) -> None:
        """Set a group of gates associated with a symbol.

        Args:
            group_symbol: Symbol representing the group.
            gate_set: Iterable of gate symbols to include in the group.

        Returns: None

        """
        self.gate_groups[group_symbol] = set(gate_set)

    def in_group(self, group_symbol: str, gate_symbol: str) -> bool:
        """Returns whether the `gate_symbol` is in the group represented by `group_symbol`.

        Args:
            group_symbol: Symbol representing the group to check.
            gate_symbol: Symbol of the gate to check membership for.

        """
        return gate_symbol in self.gate_groups[group_symbol]

    def set_gate_error(
        self,
        gate_symbol: str,
        error_func: Callable[..., Any] | str,
        error_param: str = "p",
        *,
        after: bool = True,
    ) -> None:
        """Sets the errors for a gate.

        Args:
            gate_symbol: The gate symbol that is being evaluated for errors.
            error_func: A callable to generate errors or an iterable of gate symbols from which errors are uniformly
                drawn from. It can also be a str that represents an gate error that is always returned if an error
                occurs.
            error_param: What error parameter determines if an error occurs or not. Error functions will be given the
                error_params are an argument so more detailed error distributions can be created.
            after: If True, apply errors after the gate; if False, apply before.

        Returns: None

        """
        if error_func is True:
            self.error_func_dict[gate_symbol] = (True, error_param)

        elif error_func is False:
            self.error_func_dict[gate_symbol] = False

        elif isinstance(error_func, str):
            error_func = self.ErrorStaticSymbol(error_func, after=after).error_func
            self.error_func_dict[gate_symbol] = (error_func, error_param)

        elif hasattr(error_func, "__iter__"):
            error_func = list(error_func)

            first = error_func[0]
            if (
                isinstance(first, str)
                and first not in {"CNOT", "II", "CZ", "SWAP", "G2"}
            ) or not hasattr(
                first,
                "__iter__",
            ):
                error_func = self.ErrorSet(error_func, after=after).error_func
            else:
                error_func = self.ErrorSetMultiQuditGate(
                    error_func,
                    after=after,
                ).error_func

            self.error_func_dict[gate_symbol] = (error_func, error_param)

        else:
            self.error_func_dict[gate_symbol] = (error_func, error_param)

    def set_group_error(
        self,
        group_symbol: str,
        error_func: Callable[..., Any] | str,
        error_param: str = "p",
        *,
        after: bool = True,
    ) -> None:
        """Sets the errors for a group of gates.

        Args:
            group_symbol: Symbol identifying the gate group.
            error_func: Error function to apply to the gates in the group.
            error_param (str): Parameter name for the error function.
            after (bool): If True, apply errors after the gate; if False, apply before.

        Returns: None

        """
        for symbol in self.gate_groups[group_symbol]:
            if symbol in self.error_func_dict:
                logger.warning("Overriding gate error for gate: %s.", symbol)

            self.set_gate_error(symbol, error_func, error_param, after)

    def set_default_error(
        self,
        error_func: Callable[..., Any] | str,
        error_param: str = "p",
    ) -> None:
        """Sets the default error if a gate is not found.

        Args:
            error_func: Default error function to use when a gate-specific error is not defined.
            error_param: Parameter name for the default error function.

        Returns: None

        """
        self.default_error_tuple = (error_func, error_param)

    def create_errors(
        self,
        err_gen: ParentErrorModel,
        gate_symbol: str,
        locations: LocationSet,
        after: dict[str, set[int]],
        before: dict[str, set[int]],
        replace: set[int],
        **kwargs: object,
    ) -> set | list | None:
        """Used to determine if an error occurs, and if so, calls the error function to determine errors.

        It also updates the `error_circuit` with the errors.

        Args:
            err_gen: Error generator instance.
            gate_symbol: Symbol of the gate to apply errors to.
            locations: Qubit locations where the gate is applied.
            after: Whether to apply errors after the gate.
            before: Whether to apply errors before the gate.
            replace: Whether to remove the gate.
            **kwargs: Additional keyword arguments passed to the error function.

        Returns: None

        """
        error_func, error_param = self.error_func_dict.get(
            gate_symbol,
            self.default_error_tuple,
        )

        if error_func is True:  # Default error
            # Use the default error function.
            error_func = self.default_error_tuple[0]
            # If no default error has been defined, then no error will be applied.

        if error_func is False:  # No errors
            return None

        p = err_gen.error_params[error_param]

        if p is True:  # Error always occurs
            for loc in locations:
                error_func(after, before, replace, loc, err_gen.error_params, **kwargs)

            return locations

        # Create len(locations) number of random float between 0 and 1.
        rand_nums = pc.random.random(len(locations))
        rand_nums = rand_nums <= p  # Boolean evaluation of random number <= p

        # TODO: Consider vectorizing this operation for better performance
        error_locations = set()

        for i, loc in enumerate(locations):
            if rand_nums[i]:
                error_locations.add(loc)
                error_func(
                    after,
                    before,
                    replace,
                    loc,
                    err_gen.error_params,
                    **kwargs,
                )

        return error_locations

    class ErrorStaticSymbol:
        """Class used to create a callable that just returns a symbol."""

        def __init__(self, symbol: str, *, after: bool = True) -> None:
            """Initialize an ErrorStaticSymbol.

            Args:
                symbol: The error symbol to apply.
                after: If True, apply error after the gate; if False, before.
            """
            self.data = symbol

            if after:
                self.error_func = self.error_func_after
            else:
                self.error_func = self.error_func_before

        def error_func_after(
            self,
            after: dict[str, set[int]],
            _before: dict[str, set[int]],
            _replace: set[int],
            location: int,
            _error_params: dict[str, Any],
        ) -> None:
            """Apply deterministic error after gate execution."""
            # Convert Pauli objects to strings for compatibility with gate symbols
            symbol = str(self.data) if hasattr(self.data, "__str__") else self.data
            after.update(symbol, {location}, emptyappend=True)

        def error_func_before(
            self,
            _after: dict[str, set[int]],
            before: dict[str, set[int]],
            _replace: set[int],
            location: int,
            _error_params: dict[str, Any],
        ) -> None:
            """Apply deterministic error before gate execution."""
            # Convert Pauli objects to strings for compatibility with gate symbols
            symbol = str(self.data) if hasattr(self.data, "__str__") else self.data
            before.update(symbol, {location}, emptyappend=True)

    class ErrorSet:
        """Class used to create a callable that returns an element from the error_set with uniform distribution."""

        def __init__(self, error_set: Iterable[str], *, after: bool = True) -> None:
            """Initialize an ErrorSet.

            Args:
                error_set: Collection of error symbols to choose from uniformly.
                after: If True, apply error after the gate; if False, before.
            """
            self.data = pc.array(list(error_set))

            if after:
                self.error_func = self.error_func_after
            else:
                self.error_func = self.error_func_before

        def error_func_after(
            self,
            after: dict[str, set[int]],
            _before: dict[str, set[int]],
            _replace: set[int],
            location: int,
            _error_params: dict[str, Any],
        ) -> None:
            """Apply random error after gate execution."""
            error_symbol = pc.random.choice(self.data, 1)[0]
            # Convert Pauli objects to strings for compatibility with gate symbols
            symbol = (
                str(error_symbol) if hasattr(error_symbol, "__str__") else error_symbol
            )
            after.update(symbol, {location}, emptyappend=True)

        def error_func_before(
            self,
            _after: dict[str, set[int]],
            before: dict[str, set[int]],
            _replace: set[int],
            location: int,
            _error_params: dict[str, Any],
        ) -> None:
            """Apply random error before gate execution."""
            error_symbol = pc.random.choice(self.data, 1)[0]
            # Convert Pauli objects to strings for compatibility with gate symbols
            symbol = (
                str(error_symbol) if hasattr(error_symbol, "__str__") else error_symbol
            )
            before.update(symbol, {location}, emptyappend=True)

    class ErrorSetMultiQuditGate:
        """Class used to create a callable that returns an element from the error_set with uniform distribution."""

        def __init__(
            self,
            error_set: Iterable[str | tuple[str, ...]],
            *,
            after: bool = True,
        ) -> None:
            """Initialize an ErrorSetMultiQuditGate.

            Args:
                error_set: Collection of error symbols or tuples of symbols for multi-qubit gates.
                after: If True, apply error after the gate; if False, before.
            """
            try:
                self.data = pc.array(list(error_set))
            except ValueError:
                error_set[0] = (error_set[0],)
                self.data = pc.array(list(error_set))

            if after:
                self.error_func = self.error_func_after
            else:
                self.error_func = self.error_func_before

        def error_func_after(
            self,
            after: dict[str, set[int]],
            _before: dict[str, set[int]],
            _replace: set[int],
            location: int | set[int],
            _error_params: dict[str, Any],
        ) -> None:
            """Apply sampled multi-qubit error after gate execution."""
            # Choose an error symbol or tuple of symbols:
            index = int(pc.random.choice(len(self.data), 1)[0])
            error_symbols = self.data[index]

            if isinstance(error_symbols, tuple | pc.Array) and len(error_symbols) > 1:
                for sym, loc in zip(error_symbols, location, strict=False):
                    if sym != "I":
                        after.update(sym, {loc}, emptyappend=True)

            elif isinstance(error_symbols, str):
                if error_symbols != "I":
                    after.update(error_symbols, {location}, emptyappend=True)

            elif isinstance(error_symbols, tuple) and len(error_symbols) == 1:
                error_symbols = error_symbols[0]
                if error_symbols != "I":
                    after.update(error_symbols, {location}, emptyappend=True)
            else:
                msg = "Only tuples and strings are currently accepted"
                raise Exception(msg)

        def error_func_before(
            self,
            _after: dict[str, set[int]],
            before: dict[str, set[int]],
            _replace: set[int],
            location: int | set[int],
            _error_params: dict[str, Any],
        ) -> None:
            """Apply sampled multi-qubit error before gate execution."""
            index = int(pc.random.choice(len(self.data), 1)[0])
            error_symbols = self.data[index]

            if isinstance(error_symbols, pc.Array) and len(error_symbols) > 1:
                for sym, loc in zip(error_symbols, location, strict=False):
                    if sym != "I":
                        before.update(sym, {loc}, emptyappend=True)
            elif isinstance(error_symbols, str):
                if error_symbols != "I":
                    before.update(error_symbols, {location}, emptyappend=True)

            elif isinstance(error_symbols, tuple) and len(error_symbols) == 1:
                error_symbols = error_symbols[0]
                if error_symbols != "I":
                    before.update(error_symbols, {location}, emptyappend=True)
            else:
                msg = "Only tuples and strings are currently accepted"
                raise Exception(msg)

    class ErrorSetTwoQuditTensorProduct(ErrorSetMultiQuditGate):
        """Created just to preserve the functionality of other error models.

        Creates a uniform distribution... not a tensor product.
        """

        def __init__(
            self,
            error_set: Iterable[str | tuple[str, ...]],
            *,
            after: bool = True,
        ) -> None:
            """Initialize an ErrorSetTwoQuditTensorProduct.

            Args:
                error_set: Collection of error symbols or tuples for two-qubit tensor products.
                after: If True, apply error after the gate; if False, before.
            """
            super().__init__(error_set, after=after)
