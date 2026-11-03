"""Probe Guppy decoder admission; this is not a supported frontend API.

The only hand-written LLVM connects two synthetic external declarations to the
existing checked site helper. Physical operations and control come from Guppy.
Run in a subprocess with a timeout: an incorrect compiler can emit a loop that
never reaches the decoder's instance guard. See runtime-decoding-guppy.md.
"""

import argparse
import hashlib
import json
import re
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import Measurement, measure, qubit, x
from pecos import Qis, selene_engine, sim
from pecos.compilation_pipeline import compile_guppy_to_hugr, compile_hugr_to_qis
from pecos.qec import (
    RuntimeDecodeBatchRequest,
    RuntimeDecodeBinding,
    RuntimeDecodeQisSite,
    RuntimeDecoder,
    RuntimeDecodeSchema,
    RuntimeDecodeWindow,
    RuntimeDetectorBinding,
    RuntimeRecordRef,
)
from pecos_rslib import DagCircuit
from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder


@guppy.declare
def probe_observed(m: Measurement) -> bool: ...


@guppy.declare
def probe_decode(instance: int, observed: bool) -> int: ...


@guppy
def feedback_round(instance: int) -> None:
    q = qubit()
    x(q)
    observed = probe_observed(measure(q))
    frame = probe_decode(instance, observed)
    out = qubit()
    x(out)
    if frame != 0:
        x(out)
    result("after", measure(out).read())


@guppy
def single() -> None:
    feedback_round(0)


@guppy
def shared_measurement() -> None:
    q = qubit()
    x(q)
    m = measure(q)
    observed = probe_observed(m)
    again = m.read()
    frame = probe_decode(0, observed)
    out = qubit()
    if again:
        x(out)
    if frame != 0:
        x(out)
    result("after", measure(out).read())


@guppy
def helper_loop() -> None:
    for instance in range(3):
        feedback_round(instance)


@guppy
def branch_loop() -> None:
    instance = 0
    while instance < 3:
        q = qubit()
        x(q)
        observed = probe_observed(measure(q))
        frame = probe_decode(instance, observed)
        out = qubit()
        x(out)
        if frame != 0:
            x(out)
        result("after", measure(out).read())
        instance += 1


READ = RuntimeRecordRef(name="readout", occurrence=0)
SITE = RuntimeDecodeQisSite(
    name="probe",
    window_id=7,
    binding=RuntimeDecodeBinding(
        program_revision="synthetic-guppy-feedback-probe-v1",
        compiler_revision="diagnostic-shims-v1",
        decode_site="qis:probe/decode",
        schema=RuntimeDecodeSchema(
            detectors=(RuntimeDetectorBinding(detector_id="parity", records=(READ,)),),
            frame_ids=("frame_x",),
        ),
    ),
)


def supplied_window() -> RuntimeDecodeWindow:
    """One closed readout model with the independently known relation L0 = D0."""
    dag = DagCircuit()
    dag.pz([0])
    dag.mz([0])
    model = (
        DemBuilder(DagFaultAnalyzer(dag).build_influence_map())
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-1]}]')
        .with_observables_json('[{"id":0,"records":[-1]}]')
        .build()
    )
    return RuntimeDecodeWindow(
        window_id=7,
        model=model,
        decoder="pecos_uf",
        detector_ids=("parity",),
        output_ids=("frame_x",),
        detector_order=(0,),
        output_map=((0,),),
        fault_ids=("readout",),
        commit=("parity",),
        independent=True,
    )


def link_probe(ir: str) -> str:
    """Define exactly the two probe imports, preserving compiler-emitted code.

    This deliberately narrow fixture linker is not a general LLVM transform.
    Each Measurement argument is consumed once by the read shim, including its
    native decref. Copy/borrow semantics of a production primitive are deferred.
    """
    bodies = {
        "probe_observed": (
            "i1",
            "i64",
            "i64 %m",
            SITE.read_ir(READ, future="%m")
            + "call void @___dec_future_refcount(i64 %m)\n"
            + f"ret i1 {SITE.read_value(READ)}",
        ),
        "probe_decode": (
            "i64",
            "i64, i1",
            "i64 %instance, i1 %observed",
            SITE.entry_ir()
            + SITE.decode_ir(instance="%instance").replace(SITE.read_value(READ), "%observed")
            + f"ret i64 {SITE.frame_value}",
        ),
    }
    for name, (ret, abi, arguments, body) in bodies.items():
        declaration = re.compile(
            rf"^declare {ret} (@__hugr__\.[a-zA-Z0-9_.$]+\.{name}\.[0-9]+)\({abi}\)(?: local_unnamed_addr)?$",
            re.MULTILINE,
        )
        matches = list(declaration.finditer(ir))
        if len(matches) != 1:
            msg = f"probe requires exactly one recognized {name} declaration, got {len(matches)}"
            raise ValueError(msg)
        match = matches[0]
        definition = f"define {ret} {match[1]}({arguments}) {{\nentry:\n{body}\n}}"
        ir = ir[: match.start()] + definition + ir[match.end() :]
    return ir + "\n" + SITE.declarations_ir() + SITE.globals_ir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("single", "shared-measurement", "helper-loop", "branch-loop"),
        default="single",
    )
    parser.add_argument(
        "--runtime",
        choices=("selene_simple_runtime", "selene_soft_rz_runtime"),
        default="selene_simple_runtime",
    )
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--inject-failure", action="store_true")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    definition = {
        "single": single,
        "shared-measurement": shared_measurement,
        "helper-loop": helper_loop,
        "branch-loop": branch_loop,
    }[args.case]
    hugr = compile_guppy_to_hugr(definition)
    compiled = compile_hugr_to_qis(hugr)
    linked = link_probe(compiled)
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
        (args.artifacts / "program.hugr").write_bytes(hugr)
        (args.artifacts / "compiled.ll").write_text(compiled)
        (args.artifacts / "linked.ll").write_text(linked)
    report = {
        "case": args.case,
        "versions": {p: version(p) for p in ("guppylang", "hugr", "tket", "selene-hugr-qis-compiler")},
        "hugr_sha256": hashlib.sha256(hugr).hexdigest(),
        "compiled_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
        "linked_sha256": hashlib.sha256(linked.encode()).hexdigest(),
        "sessions": [],
        "results": [],
    }
    window = supplied_window()
    if args.no_correction:
        window = replace(window, output_map=((),))
    fail = args.inject_failure

    def factory():
        seen = []
        report["sessions"].append(seen)

        def provide(request):
            nonlocal fail
            if type(request) is not RuntimeDecodeBatchRequest:
                msg = "probe must not claim compiler control provenance"
                raise TypeError(msg)
            seen.append(request.instance_id)
            if fail:
                fail = False
                msg = "synthetic probe provider failure"
                raise ValueError(msg)
            if request.instance_id >= 3:
                msg = "probe exceeded its three-round contract"
                raise ValueError(msg)
            # An explicit expectation for this fixed probe only: two source
            # measurements per round. Never derive this from captured reads.
            return SITE.bind_window(window, source_ids={READ: 2 * request.instance_id})

        return provide

    try:
        runner = (
            sim(Qis(linked))
            .classical(
                selene_engine(args.runtime).runtime_decoder(
                    RuntimeDecoder(provider_factory=factory),
                    capture_batches=True,
                    capture_control_flow=args.control,
                ),
            )
            .qubits(1)
            .workers(2)
            .build()
        )
        if args.inject_failure:
            try:
                runner.run(1)
            except Exception as error:
                if "synthetic probe provider failure" not in str(error):
                    raise
                report["injected_failure"] = str(error)
            else:
                msg = "injected provider failure was not propagated"
                raise RuntimeError(msg)
        for _ in range(2):
            report["results"].append(runner.run(2).to_dict()["after"])
    except Exception as error:
        report.update(outcome="rejected", error=str(error))
        print(json.dumps(report, sort_keys=True))
        return 2
    rounds = 3 if args.case.endswith("loop") else 1
    bit = int(args.no_correction)
    expected = [bit, bit] if rounds == 1 else [[bit] * rounds] * 2
    expected_sessions = ([[0]] if args.inject_failure else []) + [list(range(rounds))] * 4
    if report["results"] != [expected] * 2 or report["sessions"] != expected_sessions:
        report.update(outcome="rejected", error="probe completed with incorrect outputs or decoder instances")
        print(json.dumps(report, sort_keys=True))
        return 2
    report["outcome"] = "completed"
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
