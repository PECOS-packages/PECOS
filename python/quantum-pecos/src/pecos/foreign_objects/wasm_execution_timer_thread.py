# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from threading import Event, Thread

# These values multiplied should equal the intended maximum execution time
WASM_EXECUTION_TICK_LENGTH_S: float = 0.25
WASM_EXECUTION_MAX_TICKS: int = 4


class WasmExecutionTimerThread(Thread):
    def __init__(self, stop_event: Event, func) -> None:
        Thread.__init__(self, daemon=True)
        self._stop_event = stop_event
        self._func = func

    def run(self):
        while not self._stop_event.wait(WASM_EXECUTION_TICK_LENGTH_S):
            self._func()
