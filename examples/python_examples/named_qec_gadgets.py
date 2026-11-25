# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Author and run a reusable ZZ measurement without writing Guppy source.

The data start in |+0>; the first ZZ outcome is random. Repeating the same
measurement must agree, and final Z readout must reproduce that parity.
This small example illustrates gadget composition, not a fault-tolerant code.
"""

from pecos import sim, stabilizer
from pecos.qec.generic import Gadget, GadgetFlow, GadgetProgram, Gate, NamedParity, SchedulePolicy


def repeated_zz_memory() -> GadgetProgram:
    """Compose preparation, two uses of one check gadget, and final readout."""
    measure_zz = Gadget(
        name="measure ZZ",
        roles=("left", "right", "ancilla"),
        inputs=("left", "right"),
        outputs=("left", "right"),
        moments=(
            (Gate("R", ("ancilla",)),),
            (Gate("CX", ("left", "ancilla")),),
            (Gate("CX", ("right", "ancilla")),),
            (Gate("M", ("ancilla",), outcome="syndrome"),),
        ),
        flows=(GadgetFlow("ZZ", NamedParity(("syndrome",))),),
    )
    program = GadgetProgram(("data[0]", "data[1]", "probe"))
    program.append(
        Gadget(
            name="prepare +0",
            roles=("data[0]", "data[1]"),
            inputs=(),
            outputs=("data[0]", "data[1]"),
            moments=(
                (Gate("R", ("data[0]",)), Gate("R", ("data[1]",))),
                (Gate("H", ("data[0]",)),),
            ),
        ),
        label="prepare",
    )
    bindings = {"left": "data[0]", "right": "data[1]", "ancilla": "probe"}
    first = program.append(measure_zz, label="first", bindings=bindings)
    second = program.append(measure_zz, label="second", bindings=bindings)
    readout = program.append(
        Gadget(
            name="readout Z",
            roles=("data[0]", "data[1]"),
            inputs=("data[0]", "data[1]"),
            outputs=(),
            moments=((Gate("M", ("data[0]",), "left"), Gate("M", ("data[1]",), "right")),),
        ),
        label="readout",
    )
    program.detector("repeat ZZ", first["syndrome"] ^ second.flow("ZZ"))
    program.detector("final ZZ", second["syndrome"] ^ readout["left"] ^ readout["right"])
    return program


def main() -> None:
    """Inspect the authored program and execute both available schedules."""
    program = repeated_zz_memory()
    print(program.describe())
    for policy in SchedulePolicy:
        plan = program.compile(policy)
        print(plan.describe())
        print(plan.resources)
        print(plan.explain_operation("second/moment[1]/op[0]"))
        results = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(42).run(32).to_dict()
        for shot in range(32):
            detectors, _ = plan.evaluate_named({tag: values[shot] for tag, values in results.items()})
            if any(detectors):
                msg = "Repeated ZZ measurement disagrees in an ideal shot"
                raise RuntimeError(msg)
        print(f"{policy.value}: 32 ideal shots passed")


if __name__ == "__main__":
    main()
