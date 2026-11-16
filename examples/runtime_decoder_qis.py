"""Lower decoder boilerplate from a schema and run live native QIS feedback.

The physical program uses deterministic X probes. Its composed readout model
has the known relation L0 = D0 XOR D1; this is not a fault-tolerance experiment.
"""

from pecos import Qis, selene_engine, sim
from pecos.qec import (
    RuntimeDecodeBinding,
    RuntimeDecodeQisSite,
    RuntimeDecoder,
    RuntimeDecodeSchema,
    RuntimeDecodeWindow,
    RuntimeDetectorBinding,
    RuntimeRecordRef,
)
from pecos_rslib import DagCircuit
from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder, DemSliceRoundSchedule

FIRST = RuntimeRecordRef(name="readout", occurrence=0)
SECOND = RuntimeRecordRef(name="readout", occurrence=1)


def decode_site() -> RuntimeDecodeQisSite:
    """Define detector semantics once for generated code and supplied windows."""
    return RuntimeDecodeQisSite(
        name="parity",
        window_id=7,
        binding=RuntimeDecodeBinding(
            program_revision="synthetic-parity-loop-v1",
            compiler_revision="example-frontend-v1",
            decode_site="qis:qmain/decode",
            schema=RuntimeDecodeSchema(
                detectors=(
                    RuntimeDetectorBinding(detector_id="first", records=(FIRST,)),
                    RuntimeDetectorBinding(detector_id="second", records=(SECOND,)),
                ),
                frame_ids=("frame_x",),
            ),
        ),
    )


def program(site: RuntimeDecodeQisSite) -> str:
    """The frontend controls physical operations, read placement and feedback."""
    return site.declarations_ir() + site.globals_ir() + f"""
@label = private constant [5 x i8] c"after"
declare i64 @___qalloc()
declare void @___qfree(i64)
declare void @___reset(i64)
declare void @___rxy(i64, double, double)
declare i64 @___lazy_measure(i64)
declare i1 @___read_future_bool(i64)
declare void @___dec_future_refcount(i64)
declare void @print_bool_selene(ptr, i64, i1)
define i64 @qmain(i64 %shot) #0 {{
entry:
    %q = call i64 @___qalloc()
    {site.entry_ir()}
    br label %loop
loop:
    %instance = phi i64 [0, %entry], [%next, %tail]
    call void @___reset(i64 %q)
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    %m0 = call i64 @___lazy_measure(i64 %q)
    {site.read_ir(FIRST, future="%m0")}
    call void @___dec_future_refcount(i64 %m0)
    call void @___reset(i64 %q)
    %second_flip = icmp eq i64 %instance, 1
    br i1 %second_flip, label %perturb, label %read_second
perturb:
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    br label %read_second
read_second:
    %m1 = call i64 @___lazy_measure(i64 %q)
    {site.read_ir(SECOND, future="%m1")}
    call void @___dec_future_refcount(i64 %m1)
    call void @___reset(i64 %q)
    %parity = xor i1 {site.read_value(FIRST)}, {site.read_value(SECOND)}
    br i1 %parity, label %prepare_output, label %decode
prepare_output:
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    br label %decode
decode:
    {site.decode_ir(instance="%instance")}
    %correct = icmp ne i64 {site.frame_value}, 0
    br i1 %correct, label %apply, label %tail
apply:
    call void @___rxy(i64 %q, double 3.141592653589793, double 0.0)
    br label %tail
tail:
    %after = call i64 @___lazy_measure(i64 %q)
    %value = call i1 @___read_future_bool(i64 %after)
    call void @print_bool_selene(ptr @label, i64 5, i1 %value)
    call void @___dec_future_refcount(i64 %after)
    %next = add i64 %instance, 1
    %more = icmp ult i64 %next, 3
    br i1 %more, label %loop, label %done
done:
    call void @___qfree(i64 %q)
    ret i64 0
}}
attributes #0 = {{ "EntryPoint" }}
"""


def supplied_window() -> RuntimeDecodeWindow:
    """Reuse the existing slice composer; explicitly declare physical ownership."""
    dag = DagCircuit()
    for _ in range(2):
        dag.pz([0])
        dag.mz([0])
    for i, node in enumerate(dag.nodes()):
        dag.set_gate_attr(node, "dem_slice_round", i // 2)
    influence = DagFaultAnalyzer(dag).build_influence_map()
    model = (
        DemBuilder(influence)
        .with_noise(0.0, 0.0, 0.1, 0.0)
        .with_detectors_json('[{"id":0,"records":[-2],"coords":[0,0,0]},{"id":1,"records":[-1],"coords":[0,0,1]}]')
        .with_observables_json('[{"id":0,"records":[-2,-1]}]')
        .build()
    )
    schedule = model.round_schedule(influence, dag)
    composed = DemSliceRoundSchedule.from_cached_slices(
        model,
        [(schedule.cached_slice(i), i) for i in range(2)],
        [0],
        [],
    ).compose(start_round=0, commit_rounds=2, buffer_rounds=0, forward_boundary="hard")
    return RuntimeDecodeWindow(
        window_id=7,
        model=composed,
        decoder="pecos_uf",
        detector_ids=("first", "second"),
        output_ids=("parity",),
        detector_order=(0, 1),
        output_map=((0,),),
        fault_ids=("first_readout", "second_readout"),
        commit=("first", "second"),
        independent=True,
    )


def main() -> None:
    """Run a repeated live decode with compiler-owned control and batch capture."""
    site = decode_site()
    window = supplied_window()

    def factory():
        def provide(request):
            # Explicit expectation for this fixed synthetic schedule only:
            # two detector measurements and one tail measurement per iteration.
            return site.bind_window(
                window,
                source_ids={FIRST: 3 * request.instance_id, SECOND: 3 * request.instance_id + 1},
            )

        return provide

    runner = (
        sim(Qis(program(site)))
        .classical(
            selene_engine().runtime_decoder(
                RuntimeDecoder(provider_factory=factory),
                capture_batches=True,
                capture_control_flow=True,
            ),
        )
        .qubits(1)
        .workers(2)
        .build()
    )
    if runner.run(4).to_dict()["after"] != [[0, 0, 0]] * 4:
        msg = "unexpected live correction result"
        raise RuntimeError(msg)
    print("All three live parity updates corrected the subsequent measurements.")


if __name__ == "__main__":
    main()
