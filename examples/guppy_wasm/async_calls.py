"""Launch independent WASM calls before reading their results.

Use --heavy for a CPU-intensive overlap benchmark, and --async-workers 0 to
compare inline execution. Ordinary Guppy calls normally read results immediately;
this example adds explicit ordering to the public HUGR dataflow graph.
"""

from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module
from hugr.hugr import Hugr
from hugr.package import Package

WASM_PATH = Path(__file__).with_suffix(".wat")


@wasm_module(str(WASM_PATH))
class Work:
    @wasm
    def work(self, iterations: int) -> int: ...


@guppy
def independent_calls() -> None:
    first = Work(nat(0))
    second = Work(nat(1))
    a = first.work(1000000)
    b = second.work(2000000)
    first.discard()
    second.discard()
    result("first", a)
    result("second", b)


@guppy
def heavy_calls() -> None:
    first = Work(nat(0))
    second = Work(nat(1))
    a = first.work(1000000000)
    b = second.work(2000000000)
    first.discard()
    second.discard()
    result("first", a)
    result("second", b)


@guppy
def quantum_feedback() -> None:
    first = Work(nat(0))
    second = Work(nat(1))
    a = first.work(1000000)
    q = qubit()
    x(q)
    measured = measure(q).read()
    b = second.work(2000000 + int(measured))
    first.discard()
    second.discard()
    result("first", a)
    result("second", b)
    result("measured", measured)


def deferred_program(function=independent_calls) -> Package:
    """Order two independent public WASM calls before either result read."""
    package = Package.from_bytes(function.with_minimal_opt().compile().to_bytes())
    serial = package.modules[0]._to_serial()  # noqa: SLF001 - Versioned HUGR serialization API.
    document = serial.model_dump(mode="json")
    calls = [
        i
        for i, node in enumerate(document["nodes"])
        if node.get("extension") == "tket.wasm" and node.get("name") == "call"
    ]
    if len(calls) != 2:
        msg = "Expected exactly two independent WASM calls"
        raise ValueError(msg)
    reads = []
    for call in calls:
        targets = [target[0] for source, target in document["edges"] if source == [call, 0]]
        if len(targets) != 1 or document["nodes"][targets[0]].get("name") != "read_result":
            msg = "Expected a single result read for each WASM call"
            raise ValueError(msg)
        reads.append(targets[0])
    document["edges"].extend(
        [
            [[calls[0], None], [calls[1], None]],
            [[calls[1], None], [reads[0], None]],
            [[reads[0], None], [reads[1], None]],
        ],
    )
    allocations = [
        i
        for i, node in enumerate(document["nodes"])
        if node.get("extension") == "tket.quantum" and node.get("name") == "QAlloc"
    ]
    if allocations:
        if len(allocations) != 1 or len({document["nodes"][i]["parent"] for i in calls + reads + allocations}) != 1:
            msg = "Expected one quantum allocation in the same dataflow region as the WASM calls"
            raise ValueError(msg)
        # Start the first call before the measurement that feeds the second.
        document["edges"].append([[calls[0], None], [allocations[0], None]])
    graph = Hugr._from_serial(type(serial).model_validate(document))  # noqa: SLF001
    return Package(modules=[graph], extensions=package.extensions)


def expected_value(iterations: int) -> int:
    """Evaluate the recurrence by affine composition in logarithmic time."""
    multiplier, offset, value = 6364136223846793005, 1, 0
    mask = (1 << 64) - 1
    while iterations:
        if iterations & 1:
            value = (multiplier * value + offset) & mask
        multiplier, offset = (multiplier * multiplier) & mask, (offset * (multiplier + 1)) & mask
        iterations >>= 1
    return value if value < 1 << 63 else value - (1 << 64)


def main() -> None:
    import argparse
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("selene", "pecos"), default="selene")
    parser.add_argument("--async-workers", type=int, default=2)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--heavy", action="store_true")
    workload.add_argument("--feedback", action="store_true", help="Measure a qubit before the second WASM call")
    args = parser.parse_args()
    function = heavy_calls if args.heavy else quantum_feedback if args.feedback else independent_calls
    program = deferred_program(function)
    iterations = 1000000000 if args.heavy else 1000000
    expected = {"first": expected_value(iterations), "second": expected_value(2 * iterations)}
    if args.feedback:
        expected.update(second=expected_value(2 * iterations + 1), measured=True)
    modules = {str(WASM_PATH): WASM_PATH.read_bytes()}
    if args.backend == "selene":
        from pecos_selene_wasm import build
        from selene_sim import Quest

        with TemporaryDirectory() as directory:
            runner = build(
                program,
                modules,
                build_dir=Path(directory),
                wasm_async_workers=args.async_workers,
                wasm_timeout_seconds=10.0,
            )
            [observed] = [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=1)]
    else:
        import pecos as pc

        program = pc.Hugr.from_bytes(
            program.to_bytes(),
            wasm_modules=modules,
            wasm_async_workers=args.async_workers,
            wasm_timeout_seconds=10.0,
        )
        output = pc.sim(program).qubits(1).quantum(pc.state_vector()).workers(1).run(1).to_dict()
        observed = {key: values[0] for key, values in output.items()}
    if observed != expected:
        msg = "WASM outputs disagree with the independent arithmetic oracle"
        raise RuntimeError(msg)
    print(observed)


if __name__ == "__main__":
    main()
