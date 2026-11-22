"""Copy outside the bundle and implement locally; never publish private imports."""

from contextlib import contextmanager


@contextmanager
def prepare(program, modules, settings, build_dir):  # noqa: ARG001 - Adapter contract for the local implementation.
    """Compile source and construct a runner before yielding its run callable.

    program: the Guppy definition; use the local stack's normal source frontend.
    modules: exact identifiers mapped to explicit WAT bytes; do not substitute files.
    settings: shots, repeats, workers, seed, qubits, platform, simulator_family,
    wasm_int_width (case-specific native integer width, 32 or 64).
    build_dir: temporary scratch space, removed after this case.

    Each run must execute settings['shots'] fresh shots of the same compiled
    program and consume every streamed record. Return {'shots': [record, ...]}
    with one dictionary of numeric result labels per shot. Reject duplicate
    labels before constructing dictionaries; never silently overwrite a record. Only map a recognized
    WASM unreachable trap to {'error': 'wasm_trap'}; raise all other errors.
    Respect workers and simulator settings or fail preparation explicitly.
    Release resources in a finally block around yield. Do not read the expected
    results from workloads.json to synthesize observations.
    """
    message = "Implement this adapter in the internal environment"
    raise NotImplementedError(message)
    yield  # pragma: no cover - Documents the context-manager interface.
