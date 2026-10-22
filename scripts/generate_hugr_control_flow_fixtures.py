#!/usr/bin/env -S uv run --frozen python
"""Regenerate the shared native/Python HUGR control-flow corpus."""

import json
from pathlib import Path

from hugr import ops, tys, val
from hugr.build.cfg import Cfg
from hugr.build.dfg import Dfg
from hugr.build.function import Module
from hugr.envelope import EnvelopeConfig, EnvelopeFormat
from tket_exts import measurement, quantum


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "crates/pecos-qec/tests/fixtures/hugr_control_flow"
    directory.mkdir(parents=True, exist_ok=True)
    result_tags = root / "python/quantum-pecos/tests/qec/fixtures/result_tags"
    dem_programs = root / "crates/pecos-qec/tests/fixtures/dem_programs"
    for name, source in {
        "straight_line": result_tags / "scrambled.hugr",
        "guppy_branch": dem_programs / "changing_schedule.hugr",
        "guppy_loop": result_tags / "looped.hugr",
        "func_decl": result_tags / "funcdecl.hugr",
    }.items():
        (directory / f"{name}.hugr").write_bytes(source.read_bytes())

    conditional = Dfg()
    qubit = conditional.add(quantum.qAlloc())
    measured = conditional.add(quantum.measure_free(qubit))
    bit = conditional.add(measurement.read(measured))
    branch = conditional.add_conditional(bit)
    for index in range(2):
        branch.add_case(index).set_outputs()
    conditional.set_outputs()

    loop = Dfg()
    body = loop.add_tail_loop([], [])
    body.set_loop_outputs(body.load(val.TRUE))
    loop.set_outputs()

    uncalled = Module()
    uncalled.define_main([]).set_outputs()
    helper = uncalled.define_function("uncalled", [])
    body = helper.add_tail_loop([], [])
    body.set_loop_outputs(body.load(val.TRUE))
    helper.set_outputs()

    indirect = Dfg(tys.FunctionType([tys.Bool], [tys.Bool]), tys.Bool)
    function, bit = indirect.inputs()
    indirect.set_outputs(indirect.add(ops.CallIndirect()(function, bit)))

    single = Cfg()
    entry = single.add_entry()
    entry.set_single_succ_outputs()
    single.branch_exit(entry[0])

    multiple = Cfg(tys.Bool)
    entry = multiple.add_entry()
    entry.set_outputs(*entry.inputs())
    for branch in range(2):
        successor = multiple.add_successor(entry[branch])
        successor.set_single_succ_outputs()
        multiple.branch_exit(successor[0])

    for name, builder in {
        "uncalled_loop": uncalled,
        "conditional": conditional,
        "tail_loop": loop,
        "call_indirect": indirect,
        "cfg_one_row": single,
        "cfg_multi_row": multiple,
    }.items():
        data = builder.hugr.to_bytes(EnvelopeConfig(format=EnvelopeFormat.S_EXPRESSION))
        (directory / f"{name}.hugr").write_bytes(data)

    verdicts = {
        "straight_line.hugr": False,
        "guppy_branch.hugr": True,
        "guppy_loop.hugr": True,
        "conditional.hugr": True,
        "tail_loop.hugr": True,
        "uncalled_loop.hugr": True,
        "func_decl.hugr": True,
        "call_indirect.hugr": True,
        "cfg_multi_row.hugr": True,
        "cfg_one_row.hugr": False,
    }
    (directory / "verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")


if __name__ == "__main__":
    main()
