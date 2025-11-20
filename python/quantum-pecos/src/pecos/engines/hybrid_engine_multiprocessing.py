# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Multiprocessing hybrid engine for parallel quantum-classical computation.

This module provides a multiprocessing-enabled hybrid engine for parallel
execution of quantum-classical algorithms across multiple CPU cores.
"""

from __future__ import annotations

import multiprocessing
import sys
from os import getpid
from typing import TYPE_CHECKING
from warnings import warn

import pecos as pc

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.managers import SyncManager
    from typing import Any

    from pecos.engines.hybrid_engine import HybridEngine, PHIRProgram

# TODO: Add runtime data to multisim_proc_info


def run_multisim(
    eng: HybridEngine,
    program: PHIRProgram,
    foreign_object: object = None,
    *,
    shots: int = 1,
    seed: int | None = None,
    pool_size: int = 1,
    reset_engine: bool = True,
) -> dict:
    """Parallelize the running of the sim."""
    if reset_engine:
        eng.reset_all()

    # Don't use more resource than is needed
    pool_size = min(pool_size, shots)

    # divide up the shots among the pool
    q, mod = divmod(shots, pool_size)
    multi_shots = [q] * pool_size
    for i in range(mod):
        multi_shots[i] += 1

    # TODO: Find a more elegant solution. At least rest eng
    if foreign_object and hasattr(foreign_object, "to_dict"):
        eng.cinterp.foreign_obj = None
        foreign_object = foreign_object.to_dict()

    # Create the list of kwargs to launch the run_sim instances
    kwargs = {
        "program": program,
        "foreign_object": foreign_object,
        "shots": shots,
        "seed": seed,
        "initialize": True,
    }

    if seed is not None:
        pc.random.seed(seed)
    # Use i32::MAX from Rust as max seed value
    max_value = pc.dtypes.i32.max

    manager = multiprocessing.get_context("spawn").Manager()
    queue = manager.Queue()
    args = []
    for i, sh in enumerate(multi_shots):
        # make a unique seed for each process
        sd = int(pc.random.randint(0, max_value, 1)[0])

        kwargs_temp = dict(kwargs)
        kwargs_temp.update(
            shots=sh,
            seed=sd,
        )
        args.append((queue, eng.run, kwargs_temp, i))

    # Launch multiple serial sims
    with multiprocessing.get_context("spawn").Pool(processes=pool_size) as pool:
        presults = pool.map(worker_wrapper, args)

    msg_dict = {}
    while not queue.empty():
        pid, msg_type, msg_data = queue.get()
        msg_dict.setdefault(pid, {})
        if isinstance(msg_data, str):
            msg_dict[pid].setdefault(msg_type, "")
            msg_dict[pid][msg_type] += msg_data
        else:
            msg = f"msg_data type {type(msg_data)} not currently handled!"
            raise TypeError(msg)

    # Combine results
    results = {}
    errors = []
    process_info = []
    for r, pinfo in presults:
        pid = pinfo["pid"]
        i = pinfo["i"]
        msgs = msg_dict.get(pid, {})
        pinfo["messages"] = msgs
        if "error" in msgs:
            errors.append((i, pid, msgs["error"]))

        process_info.append(pinfo)
        for key, value in r.items():
            if isinstance(value, list):
                results.setdefault(key, []).extend(value)
            else:
                msg = f"Unexpected results! Got a result dict with a value of type: {type(value)}"
                raise TypeError(msg)

    eng.multisim_process_info = process_info

    if errors:
        for i, pid, error_msg in errors:
            warn(
                f"process {i} with pid = {pid} had this error message: {error_msg}",
                stacklevel=2,
            )
        msg = "Processes experienced errors!"
        raise MultisimError(msg)

    return results


class MultisimError(Exception):
    """Exception raised when errors occur during multi-simulation execution.

    This exception is used to signal errors that occur during parallel
    simulation runs in the multiprocessing engine.
    """


def worker_wrapper(
    args: tuple[SyncManager.Queue, Callable[..., dict], dict[str, Any], int],
) -> tuple[dict, dict]:
    """A wrapper to pass kwargs onto run for multiprocess.pool.map."""
    queue, run, pkwargs, i = args
    pid = getpid()

    sys.stdout = WriteStream(queue, pid, "stdout")
    sys.stderr = WriteStream(queue, pid, "stderr")

    run_info = {"i": i, "pid": pid, "seed": pkwargs["seed"], "shots": pkwargs["shots"]}

    # TODO: Find a more elegant solution.
    foreign_object = pkwargs["foreign_object"]
    if isinstance(foreign_object, dict) and hasattr(
        foreign_object["fobj_class"],
        "from_dict",
    ):
        pkwargs["foreign_object"] = foreign_object["fobj_class"].from_dict(
            foreign_object,
        )

    results = {}
    try:
        results = run(**pkwargs)
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as e:
        queue.put((pid, "error", f"{type(e).__name__}: {e}"))
    # Must catch all exceptions in worker process to prevent silent failures
    except Exception as e:
        queue.put((pid, "error", f"Unexpected error: {type(e).__name__}: {e}"))

    return results, run_info


class WriteStream:
    """Custom stream for capturing and redirecting process output.

    This class captures stdout/stderr from worker processes and forwards
    the output through a multiprocessing queue for centralized handling.
    """

    def __init__(self, q: SyncManager.Queue, pid: int, stream_type: str) -> None:
        """Initialize a write stream for capturing process output.

        Args:
            q: The multiprocessing queue for sending messages.
            pid: The process ID of the worker process.
            stream_type: The type of stream ('stdout' or 'stderr').
        """
        self.queue = q
        self.stream_type = stream_type
        self.pid = pid

    def write(self, msg: str) -> None:
        """Write message to the output queue.

        Args:
            msg: Message string to write.
        """
        self.queue.put((self.pid, self.stream_type, msg))

    def flush(self) -> None:
        """Flush the output stream (no-op for queue-based output)."""
