"""Isolate external-result loop counter folding without importing PECOS.

This inspects a specific HUGR shape, not general compiler correctness. Exit 2
means a pass replaced the loop input with constant zero; other unexpected
shapes raise an error. Run under a process timeout. See runtime-decoding-guppy.md.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from hugr.package import Package
from selene_hugr_qis_compiler import compile_to_llvm_ir
from tket.passes import Normalize, _QSystemLLVMPass


@guppy.declare
def external_value(counter: int) -> int: ...


@guppy
def branch_loop() -> None:
    counter = 0
    while counter < 3:
        value = external_value(counter)
        if value != 0:
            result("nonzero", value)
        result("counter", counter)
        counter += 1


@guppy
def plain_loop() -> None:
    counter = 0
    while counter < 3:
        value = external_value(counter)
        result("value", value)
        result("counter", counter)
        counter += 1


def call_argument(graph):
    """Inspect the sole external call in this fixture; fail on shape changes."""
    [module] = graph["modules"]
    nodes, edges = module["nodes"], module["edges"]
    [declaration] = [
        index
        for index, node in enumerate(nodes)
        if node["op"] == "FuncDecl" and node["name"].endswith(".external_value")
    ]
    [call] = [target[0] for source, target in edges if source[0] == declaration and nodes[target[0]]["op"] == "Call"]
    [source] = [source for source, target in edges if target == [call, 0]]
    producer = nodes[source[0]]
    evidence = {"call_node": call, "source_node": source[0], "source_op": producer["op"]}
    if producer["op"] == "Input":
        evidence["kind"] = "input_retained"
    elif producer["op"] == "LoadConstant":
        [constant] = [edge_source[0] for edge_source, target in edges if target == [source[0], 0]]
        value = nodes[constant]["v"]["value"]
        if value != {"c": "ConstInt", "v": {"log_width": 6, "value": 0}}:
            msg = f"Unexpected external argument constant: {value!r}"
            raise ValueError(msg)
        evidence.update(kind="constant_zero", constant_node=constant)
    else:
        msg = f"Unexpected external argument producer: {producer!r}"
        raise ValueError(msg)
    return evidence


def inspect(case, artifacts):
    """Compare fresh copies of minimal HUGR with one pass option changed."""
    program = {"branch": branch_loop, "plain": plain_loop}[case]
    raw = program.with_minimal_opt().compile().to_bytes()
    stages = {}
    if artifacts is not None:
        artifacts.mkdir(parents=True, exist_ok=True)

    def record(name, package):
        data = package.to_bytes()
        # Private serialization is intentional here: a version-specific
        # diagnostic must fail visibly if the compiler's graph schema changes.
        graph = package._to_serial().model_dump(mode="json")  # noqa: SLF001
        stages[name] = {"sha256": hashlib.sha256(data).hexdigest(), **call_argument(graph)}
        if artifacts is not None:
            (artifacts / f"{name}.hugr").write_bytes(data)
            (artifacts / f"{name}.json").write_text(json.dumps(graph, indent=2) + "\n")

    record("minimal", Package.from_bytes(raw))
    record("default", program.compile())
    for name, compiler_pass in (
        ("normalize_no_fold", Normalize(constant_folding=False)),
        ("normalize_fold", Normalize(constant_folding=True)),
        ("qsystem_no_fold", _QSystemLLVMPass(constant_fold=False)),
        ("qsystem_fold", _QSystemLLVMPass(constant_fold=True)),
    ):
        package = Package.from_bytes(raw)
        compiler_pass.run(package.modules[0])
        record(name, package)

    llvm = {}
    for level in (0, 2):
        # This compiler itself runs HUGR passes, even at LLVM optimization 0.
        ir = compile_to_llvm_ir(raw, opt_level=level)
        llvm[str(level)] = hashlib.sha256(ir.encode()).hexdigest()
        if artifacts is not None:
            (artifacts / f"minimal-O{level}.ll").write_text(ir)

    return {
        "case": case,
        "outcome": "constant_zero" if any(s["kind"] == "constant_zero" for s in stages.values()) else "input_retained",
        "stages": stages,
        "llvm_sha256": llvm,
        "versions": {
            name: version(name)
            for name in ("guppylang", "guppylang-internals", "hugr", "tket", "selene-hugr-qis-compiler")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("branch", "plain"), default="branch")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    report = inspect(args.case, args.artifacts)
    print(json.dumps(report, sort_keys=True))
    return 2 if report["outcome"] == "constant_zero" else 0


if __name__ == "__main__":
    raise SystemExit(main())
