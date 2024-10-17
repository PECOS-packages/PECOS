# Copyright 2019 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class SymbolLibrary:
    """A library of objects and constructors of objects, where the objects are specified by symbols (strings) and
    parameters that are used to construct the object.

    Attributes:
    ----------
        constructors (Dict[str, Callable]): A dictionary of constructors.
        default_constructor:
        library (DefaultDict[str, Set[object]]: A dictionary of objects.
    """

    def __init__(self) -> None:
        self.constructors = {}
        self.default_constructor = None
        self.library = defaultdict(set)

    def add(self, symbol: str, obj: Any, params: dict[str, Any]) -> None:
        """Adds an object to `library`.

        Args:
        ----
            symbol (str): A string representing an object constructed.
            obj (object): Object to be stored.
            params (Dict[str, Any]): Parameters used to construct the object corresponding to `symbol`.

        Returns:
        -------

        """
        self.library[symbol].add((obj, params))

    def get(
        self,
        symbol: str,
        params: dict[str, Any],
        default: Any | None = None,
    ) -> Any:
        """Get an instance associated with `symbol` that has the parameters `params`.

        Args:
        ----
            symbol (str): A string representing an object constructed.
            params (Dict[str, Any]): Parameters used to construct the object corresponding to `symbol`.
            default (Optional[Any]): Default value to return if a object could not be found or a constructed.

        Returns:
        -------

        """
        obj_set = self.library.get(symbol)

        if obj_set:
            for instance, inst_params in obj_set:
                if params == inst_params:
                    return instance
            return None

        else:
            constructor = self.constructors.get(symbol)

            if constructor:
                instance = constructor(**params)

            elif default is None and self.default_constructor is not None:
                instance = self.default_constructor(**params)

            else:
                return default

            self.library[symbol].add((instance, params))
            return instance

    def add_constructor(self, symbol: str, constructor: Callable[..., Any]) -> None:
        """Add a constructor of a circuit.

        Args:
        ----
            symbol (str): A string representing an object constructed.
            constructor (Callable): A callable that returns an object.

        Returns:
        -------
            None

        """
        if not isinstance(constructor, Callable):
            msg = f"Constructor must be a callable. Got type: {type(constructor)}"
            raise TypeError(msg)

        self.constructors[symbol] = constructor
