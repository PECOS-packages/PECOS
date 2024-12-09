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

import numpy as np

from pecos.error_models.class_errors_circuit import ErrorCircuits


class ParentErrorModel:
    """A simple error generator for the depolarizing model.

    This error generator does not allow much modification of the error model.
    """

    def __init__(self) -> None:
        """ """
        self.error_circuits = None

        self.error_params = None
        self.circuit = None
        self.generator_class = Generator

    def start(self, circuit, error_params):
        """Start up at the beginning of a circuit simulation.

        Args:
        ----
            circuit:
            error_params:

        Returns:
        -------

        """
        self.error_circuits = ErrorCircuits()
        self.circuit = circuit
        self.error_params = error_params

        return self.error_circuits

    def generate_tick_errors(self, tick_circuit, time, **params):
        """Returns before errors, after errors, and replaced locations for the given key (args).

        Returns:
        -------

        """
        return {}


class Generator:
    """Class that keeps track of how errors are generated for each gate and groups of gates. It also has a method for
    generating errors.
    """

    def __init__(self) -> None:
        self.gate_groups = {}
        self.error_func_dict = {}
        self.default_error_tuple = (False, "p")

    def set_gate_group(self, group_symbol, gate_set):
        """Args:
        ----
            group_symbol: Symbol representing the group.
            gate_set: Iterable of gate symbols.

        Returns: None

        """
        self.gate_groups[group_symbol] = set(gate_set)

    def in_group(self, group_symbol, gate_symbol):
        """Returns whether the `gate_symbol` is in the group represented by `group_symbol`.

        Args:
        ----
            group_symbol:
            gate_symbol:

        Returns:
        -------

        """
        return gate_symbol in self.gate_groups[group_symbol]

    def set_gate_error(self, gate_symbol, error_func, error_param="p", after=True):
        """Sets the errors for a gate.

        Args:
        ----
            gate_symbol: The gate symbol that is being evaluated for errors.
            error_func (callable, iterable, str, bool, None): A callable to generate errors or an iterable of gate
            symbols from which errors are uniformly drawn from. It can also be a str that represents an gate error that
            is always returned if an error occurs.
            error_param: What error parameter determines if an error occurs or not. Error functions will be given the
            error_params are an argument so more detailed error distributions can be created.
            after (bool):

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
                and first not in ["CNOT", "II", "CZ", "SWAP", "G2"]
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

    def set_group_error(self, group_symbol, error_func, error_param="p", after=True):
        """Sets the errors for a group of gates.

        Args:
        ----
            group_symbol:
            error_func:
            error_param (str):
            after (bool)

        Returns: None

        """
        for symbol in self.gate_groups[group_symbol]:
            if symbol in self.error_func_dict:
                print("Overriding gate error for gate: %s." % symbol)

            self.set_gate_error(symbol, error_func, error_param, after)

    def set_default_error(self, error_func, error_param="p"):
        """Sets the default error if gate is not found.

        Args:
        ----
            error_func:
            error_param:

        Returns: None

        """
        self.default_error_tuple = (error_func, error_param)

    def create_errors(
        self,
        err_gen,
        gate_symbol,
        locations,
        after,
        before,
        replace,
        **kwargs,
    ):
        """Used to determine if an error occurs and if so, calls the error function to determine errors.

        It also updates the `error_circuit` with the errors.

        Args:
        ----
            err_gen:
            gate_symbol:
            locations:
            after:
            before:
            replace:

        Returns: None

        """
        error_func, error_param = self.error_func_dict.get(
            gate_symbol,
            self.default_error_tuple,
        )

        if error_func is True:  # Default error
            # Use the default error function.
            error_func = self.default_error_tuple[0]
            # If no default error has been defined then no error will be applied.

        if error_func is False:  # No errors
            return None

        p = err_gen.error_params[error_param]

        if p is True:  # Error always occurs
            for loc in locations:
                error_func(after, before, replace, loc, err_gen.error_params, **kwargs)

            return locations

        else:
            # Create len(locations) number of random float between 0 and 1.
            rand_nums = np.random.random(len(locations))
            rand_nums = rand_nums <= p  # Boolean evaluation of random number <= p

            # TODO: Think about using the numpy function vectorize...
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

        def __init__(self, symbol, after=True) -> None:
            self.data = symbol

            if after:
                self.error_func = self.error_func_after
            else:
                self.error_func = self.error_func_before

        def error_func_after(self, after, before, replace, location, error_params):
            after.update(self.data, {location}, emptyappend=True)

        def error_func_before(self, after, before, replace, location, error_params):
            before.update(self.data, {location}, emptyappend=True)

    class ErrorSet:
        """Class used to create a callable that returns an element from the error_set with uniform distribution."""

        def __init__(self, error_set, after=True) -> None:
            self.data = np.array(list(error_set))

            if after:
                self.error_func = self.error_func_after
            else:
                self.error_func = self.error_func_before

        def error_func_after(self, after, before, replace, location, error_params):
            after.update(np.random.choice(self.data), {location}, emptyappend=True)

        def error_func_before(self, after, before, replace, location, error_params):
            before.update(np.random.choice(self.data), {location}, emptyappend=True)

    class ErrorSetMultiQuditGate:
        """Class used to create a callable that returns an element from the error_set with uniform distribution."""

        def __init__(self, error_set, after=True) -> None:
            try:
                self.data = np.array(list(error_set))
            except ValueError:
                error_set[0] = (error_set[0],)
                self.data = np.array(list(error_set))

            if after:
                self.error_func = self.error_func_after
            else:
                self.error_func = self.error_func_before

        def error_func_after(self, after, before, replace, location, error_params):
            # Choose an error symbol or tuple of symbols:
            indx = np.random.choice(len(self.data))
            error_symbols = self.data[indx]

            if (
                isinstance(error_symbols, (tuple, np.ndarray))
                and len(error_symbols) > 1
            ):
                for sym, loc in zip(error_symbols, location):
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

        def error_func_before(self, after, before, replace, location, error_params):
            indx = np.random.choice(len(self.data))
            error_symbols = self.data[indx]

            if isinstance(error_symbols, np.ndarray) and len(error_symbols) > 1:
                for sym, loc in zip(error_symbols, location):
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

        def __init__(self, error_set, after=True) -> None:
            super().__init__(error_set, after)
