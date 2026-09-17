# Copyright 2026 The PECOS Developers

"""A single native provider call lets another Python thread keep running."""

import threading
import time

import pytest
from pecos_rslib_exp import bp_trellis, frontier

# A call this long spans many interpreter switch intervals, so a thread that
# stalls for half of it can only mean the call held the GIL.
MIN_CALL_SECONDS = 0.2


@pytest.fixture(params=[frontier(), bp_trellis()], ids=["frontier", "bp_trellis"])
def spec(request):
    return request.param


def synthetic_dem(length, stride):
    return "".join(f"error(0.3) D{i} D{i + stride} L0\nerror(0.2) D{i}\n" for i in range(length))


def largest_python_stall(native_call):
    """Return the call's duration and the longest pause of a concurrent Python thread."""
    ready = threading.Event()
    finished = threading.Event()
    largest_gap = [0.0]

    def observe():
        previous = time.perf_counter()
        ready.set()
        while True:
            now = time.perf_counter()
            largest_gap[0] = max(largest_gap[0], now - previous)
            previous = now
            # Record the interval crossing the call's return before stopping.
            if finished.is_set():
                return

    observer = threading.Thread(target=observe)
    observer.start()
    ready.wait()
    try:
        started = time.perf_counter()
        native_call()
        duration = time.perf_counter() - started
    finally:
        finished.set()
        observer.join(timeout=10)
    assert not observer.is_alive()
    return duration, largest_gap[0]


def assert_releases_gil(native_call_for_length):
    # Build profile and machine speed change the cost of a call by orders of
    # magnitude, so grow the model until one call is long enough to judge.
    for length in (500 * 2**doubling for doubling in range(8)):
        duration, stall = largest_python_stall(native_call_for_length(length))
        if duration > MIN_CALL_SECONDS:
            assert stall < duration / 2, f"Python thread stalled for {stall:.3f}s of a {duration:.3f}s native call"
            return
    pytest.fail(f"no native call exceeded {MIN_CALL_SECONDS}s; the largest took {duration:.3f}s")


def test_build_releases_gil(spec):
    def build_call(length):
        dem = synthetic_dem(length, 1)
        return lambda: spec._pecos_build_decoder(dem)

    assert_releases_gil(build_call)


def test_decode_releases_gil(spec):
    # A wide detector stride keeps many boundary states alive, making one decode substantial.
    stride = 64

    def decode_call(length):
        worker = spec._pecos_build_decoder(synthetic_dem(length, stride))
        syndrome = bytes(length + stride)
        return lambda: worker._pecos_decode_obs(syndrome)

    assert_releases_gil(decode_call)
