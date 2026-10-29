"""Run an observed-data decoder call before a later physical measurement.

The explicit X is a deterministic perturbation with the same detector/output
action as the supplied one-bit model. This illustrates feedback, not a code's
fault tolerance or a noise-rate experiment.
"""

from pecos import Qis, selene_engine, sim
from pecos.qec import (
    RuntimeDecodeBinding,
    RuntimeDecodeContext,
    RuntimeDecoder,
    RuntimeDecodeSchema,
    RuntimeDecodeWindow,
    RuntimeDetectorBinding,
    RuntimeRecordRef,
)
from pecos_rslib import DagCircuit
from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

PROGRAM = r"""
@label = private constant [5 x i8] c"after"
declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i1 @___read_future_bool(i64)
declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {
    %data = alloca i8
    call void @__quantum__qis__x__body(i64 0)
    %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %observed = call i1 @___read_future_bool(i64 0)
    %bit = zext i1 %observed to i8
    store i8 %bit, ptr %data
    %frame = call i64 @pecos_runtime_decode(i64 7, i64 0, ptr %data, i64 1)
    %correct = icmp ne i64 %frame, 0
    br i1 %correct, label %apply, label %tail
apply:
    call void @__quantum__qis__x__body(i64 0)
    br label %tail
tail:
    %after = call i32 @__quantum__qis__m__body(i64 0, i64 1)
    %value = call i1 @___read_future_bool(i64 1)
    call void @print_bool_selene(ptr @label, i64 5, i1 %value)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
"""


def main(*, versioned: bool = False) -> None:
    """Build a supplied model and use its prediction in live execution."""
    circuit = DagCircuit()
    circuit.pz([0])
    circuit.mz([0])
    model = (
        DemBuilder(DagFaultAnalyzer(circuit).build_influence_map())
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-1]}]')
        .with_observables_json('[{"id":0,"records":[-1]}]')
        .build()
    )
    # A synthetic frontend declares the read that produced the observed byte.
    # This does not prove that the declaration matches arbitrary generated code.
    binding = (
        RuntimeDecodeBinding(
            program_revision="synthetic-feedback-v1",
            compiler_revision="synthetic-llvm-v1",
            decode_site="qis:qmain/decode",
            schema=RuntimeDecodeSchema(
                detectors=(
                    RuntimeDetectorBinding(
                        detector_id="parity",
                        records=(RuntimeRecordRef(name="observed", occurrence=0),),
                    ),
                ),
                frame_ids=("frame_x",),
            ),
        )
        if versioned
        else None
    )
    source = PROGRAM
    if binding is not None:
        context = RuntimeDecodeContext(
            version=1,
            program_revision=binding.program_revision,
            compiler_revision=binding.compiler_revision,
            decode_site=binding.decode_site,
            schema_id=binding.schema.schema_id,
            branches=(),
            loops=(),
            inputs=(),
        ).to_json()
        escaped = "".join(f"\\{byte:02X}" for byte in context)
        source = f'@context = private constant [{len(context)} x i8] c"{escaped}"\n' + source.replace(
            "declare i64 @pecos_runtime_decode(i64, i64, ptr, i64)",
            "declare i64 @pecos_runtime_decode_v1(i64, i64, ptr, i64, ptr, i64)",
        ).replace(
            "@pecos_runtime_decode(i64 7, i64 0, ptr %data, i64 1)",
            f"@pecos_runtime_decode_v1(i64 7, i64 0, ptr %data, i64 1, ptr @context, i64 {len(context)})",
        )
    window = RuntimeDecodeWindow(
        window_id=7,
        model=model,
        decoder="pecos_uf",
        detector_ids=("parity",),
        output_ids=("frame_x",),
        detector_order=(0,),
        output_map=((0,),),
        fault_ids=("readout",),
        independent=True,
        commit=("parity",),
        binding=binding,
    )
    results = (
        sim(Qis(source))
        .classical(
            selene_engine().runtime_decoder(RuntimeDecoder({7: window})),
        )
        .qubits(1)
        .seed(7)
        .run(4)
    )
    if results.to_dict()["after"] != [0, 0, 0, 0]:
        msg = "decoder feedback did not correct subsequent observations"
        raise RuntimeError(msg)
    print(results.to_dict())


if __name__ == "__main__":
    main()
    main(versioned=True)
