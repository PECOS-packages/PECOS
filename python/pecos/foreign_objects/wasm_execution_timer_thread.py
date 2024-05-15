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
