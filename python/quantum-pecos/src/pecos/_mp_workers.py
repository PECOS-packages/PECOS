# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Worker functions for multiprocessing with spawn context.

When using ``multiprocessing`` with the ``spawn`` start method (default on
macOS/Windows), worker functions must be importable by the child process.
Functions defined in test files are NOT importable because test directories
are not installed packages.  This module provides generic workers that
live inside the installed ``pecos`` package so child processes can always
find them.
"""

from __future__ import annotations

import pickle


def deserialize_and_call(args: tuple[bytes, str, tuple, str, tuple]) -> object:
    """Unpickle an object, call a method on it, and return a result.

    Args:
        args: A tuple of ``(data, method, method_args, result_method, result_args)``
            where *data* is the pickled object, *method* is the method to invoke
            on the deserialized object, *method_args* are its positional arguments,
            *result_method* is the method/property to call for the return value,
            and *result_args* are its positional arguments (empty tuple for properties).

    Returns:
        The value of ``getattr(obj, result_method)(*result_args)``.
    """
    data, method, method_args, result_method, result_args = args
    obj = pickle.loads(data)  # noqa: S301
    getattr(obj, method)(*method_args)
    result = getattr(obj, result_method)
    return result(*result_args) if callable(result) else result


def run_callable_worker(args: tuple[object, dict]) -> dict:
    """Worker that calls a callable with kwargs and returns the result dict.

    Mirrors the production pattern in
    ``pecos.engines.hybrid_engine_multiprocessing.worker_wrapper``
    where a callable and its keyword arguments are passed to each pool worker.

    Args:
        args: A tuple of ``(callable, kwargs)`` where *callable* is the function
            to invoke and *kwargs* is a dict of keyword arguments.

    Returns:
        The dict returned by ``callable(**kwargs)``.
    """
    fn, kwargs = args
    return fn(**kwargs)


def sim_run_from_bytes(**kwargs: object) -> dict:
    """Deserialize a simulator from bytes, run a gate, and return results.

    This is a test helper that mimics the pattern of ``HybridEngine.run``:
    a callable that receives kwargs, operates on a simulator, and returns
    a results dict.

    Expected kwargs:
        sim_bytes: Pickled simulator bytes.
        method: Method name to call on the deserialized simulator.
        method_args: Positional arguments for the method.
        result_attr: Attribute name to read from the simulator as the result.

    Returns:
        A dict with a ``"measurements"`` key containing a list with the result.
    """
    sim = pickle.loads(kwargs["sim_bytes"])  # noqa: S301
    getattr(sim, kwargs["method"])(*kwargs["method_args"])
    result = getattr(sim, kwargs["result_attr"])
    value = result() if callable(result) else result
    return {"measurements": [value]}
