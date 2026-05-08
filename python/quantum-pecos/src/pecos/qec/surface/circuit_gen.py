# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Direct circuit generation from SurfacePatch geometry.

This module generates quantum circuits (Stim and DagCircuit) directly from
the SurfacePatch geometry, bypassing Guppy/HUGR. This provides:

1. Full control over the circuit structure (properly unrolled loops)
2. Correct detector annotations (using stabilizer structure)
3. A reference implementation to validate HUGR-based approaches

The generated circuits use the same 4-round parallel CNOT schedule as
the Guppy code generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.surface.patch import SurfacePatch
    from pecos.quantum import DagCircuit

from pecos.qec.surface.schedule import compute_cnot_schedule


def generate_stim_circuit(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    p_meas: float = 0.0,
    p_prep: float = 0.0,
) -> str:
    """Generate a Stim circuit from SurfacePatch geometry.

    This creates a memory experiment circuit that exactly mirrors the
    structure of the Guppy-generated code:
    - prep_z_basis/prep_x_basis: Creates data qubits
    - syndrome_extraction (repeated): Allocates ancillas, CNOT schedule, measures
    - measure_z_basis/measure_x_basis: Final data qubit measurement

    The circuit includes proper DETECTOR annotations based on the
    stabilizer structure and OBSERVABLE_INCLUDE for the logical operator.

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' for |0_L> state or 'X' for |+_L> state
        p1: Single-qubit gate depolarizing error rate
        p2: Two-qubit gate depolarizing error rate
        p_meas: Measurement error rate (X_ERROR before M)
        p_prep: Initialization error rate (X_ERROR after R)

    Returns:
        Stim circuit string with noise and detector annotations
    """
    geom = patch.geometry
    d = patch.distance
    num_data = geom.num_data
    num_x_anc = len(geom.x_stabilizers)
    num_z_anc = len(geom.z_stabilizers)

    # Qubit layout matches Guppy:
    # - Data qubits [0, num_data) allocated in prep_*_basis
    # - X ancillas [num_data, num_data + num_x_anc) allocated per syndrome round
    # - Z ancillas [num_data + num_x_anc, total) allocated per syndrome round
    def data_q(i: int) -> int:
        return i

    def x_anc_q(stab_idx: int) -> int:
        return num_data + stab_idx

    def z_anc_q(stab_idx: int) -> int:
        return num_data + num_x_anc + stab_idx

    total_qubits = num_data + num_x_anc + num_z_anc

    lines = []
    lines.append(f"# Surface code d={d} {basis}-basis memory experiment")
    basis_lower = basis.lower()
    lines.append(
        f"# Mirrors Guppy: prep_{basis_lower}_basis -> syndrome_extraction x{num_rounds} "
        f"-> measure_{basis_lower}_basis",
    )
    lines.append(
        f"# Qubits: {num_data} data + {num_x_anc} X ancilla + {num_z_anc} Z ancilla = {total_qubits}",
    )
    lines.append("")

    # Get CNOT schedule (same as Guppy generator uses)
    cnot_rounds = compute_cnot_schedule(patch)

    # Track measurement count for detector definitions
    meas_count = 0
    # Map (stab_type, stab_idx, round) -> measurement record index
    stab_meas_record: dict[tuple[str, int, int], int] = {}

    # =========================================================================
    # prep_z_basis / prep_x_basis
    # Guppy: data = array(qubit() for _ in range(num_data))
    #        for X basis: for i in range(num_data): h(data[i])
    # =========================================================================
    lines.append(f"# === prep_{basis.lower()}_basis ===")

    # Allocate data qubits (R = reset = qubit())
    for i in range(num_data):
        lines.append(f"R {data_q(i)}")
        if p_prep > 0:
            lines.append(f"X_ERROR({p_prep}) {data_q(i)}")

    # For X-basis: H on each data qubit
    if basis.upper() == "X":
        for i in range(num_data):
            lines.append(f"H {data_q(i)}")
            if p1 > 0:
                lines.append(f"DEPOLARIZE1({p1}) {data_q(i)}")

    lines.append("TICK")
    lines.append("")

    # =========================================================================
    # syndrome_extraction (called num_rounds times)
    # Guppy structure:
    #   - ax{i} = qubit() for each X stabilizer
    #   - az{i} = qubit() for each Z stabilizer
    #   - h(ax{i}) for each X stabilizer
    #   - 4 rounds of CX gates following schedule
    #   - h(ax{i}) for each X stabilizer
    #   - sx{i} = measure(ax{i}) for each X stabilizer
    #   - sz{i} = measure(az{i}) for each Z stabilizer
    # =========================================================================
    for rnd in range(num_rounds):
        lines.append(f"# === syndrome_extraction (round {rnd + 1}) ===")

        # Allocate ancilla qubits (fresh each round, matching Guppy)
        # Guppy: ax{i} = qubit() for each X stabilizer
        for s in geom.x_stabilizers:
            lines.append(f"R {x_anc_q(s.index)}")
            if p_prep > 0:
                lines.append(f"X_ERROR({p_prep}) {x_anc_q(s.index)}")

        # Guppy: az{i} = qubit() for each Z stabilizer
        for s in geom.z_stabilizers:
            lines.append(f"R {z_anc_q(s.index)}")
            if p_prep > 0:
                lines.append(f"X_ERROR({p_prep}) {z_anc_q(s.index)}")

        lines.append("")

        # Guppy: h(ax{i}) for each X stabilizer
        lines.append("# Hadamard on X ancillas")
        for s in geom.x_stabilizers:
            lines.append(f"H {x_anc_q(s.index)}")
            if p1 > 0:
                lines.append(f"DEPOLARIZE1({p1}) {x_anc_q(s.index)}")

        lines.append("TICK")

        # 4 CNOT rounds (exactly matching Guppy schedule)
        for rnd_idx, cx_round in enumerate(cnot_rounds):
            lines.append(f"# Round {rnd_idx + 1}")
            for stab_type, stab_idx, data_idx in cx_round:
                if stab_type == "X":
                    # Guppy: cx(ax{stab_idx}, surf.data[{data_idx}])
                    control = x_anc_q(stab_idx)
                    target = data_q(data_idx)
                else:
                    # Guppy: cx(surf.data[{data_idx}], az{stab_idx})
                    control = data_q(data_idx)
                    target = z_anc_q(stab_idx)
                lines.append(f"CX {control} {target}")
                if p2 > 0:
                    lines.append(f"DEPOLARIZE2({p2}) {control} {target}")
            lines.append("TICK")

        # Guppy: h(ax{i}) for each X stabilizer (second time)
        lines.append("# Hadamard on X ancillas")
        for s in geom.x_stabilizers:
            lines.append(f"H {x_anc_q(s.index)}")
            if p1 > 0:
                lines.append(f"DEPOLARIZE1({p1}) {x_anc_q(s.index)}")

        lines.append("")

        # Guppy: sx{i} = measure(ax{i}) for each X stabilizer
        lines.append("# Measure ancillas")
        for s in geom.x_stabilizers:
            if p_meas > 0:
                lines.append(f"X_ERROR({p_meas}) {x_anc_q(s.index)}")
            lines.append(f"M {x_anc_q(s.index)}")
            stab_meas_record[("X", s.index, rnd)] = meas_count
            meas_count += 1

        # Guppy: sz{i} = measure(az{i}) for each Z stabilizer
        for s in geom.z_stabilizers:
            if p_meas > 0:
                lines.append(f"X_ERROR({p_meas}) {z_anc_q(s.index)}")
            lines.append(f"M {z_anc_q(s.index)}")
            stab_meas_record[("Z", s.index, rnd)] = meas_count
            meas_count += 1

        lines.append("")

        # Add detector annotations
        # Detector = XOR of current measurement with previous round (or 0 for first round)
        for s in geom.x_stabilizers:
            curr_idx = stab_meas_record[("X", s.index, rnd)]
            curr_offset = meas_count - curr_idx
            if rnd == 0:
                # First round: compare to initial 0
                lines.append(f"DETECTOR({s.index}, 0, {rnd}) rec[{-curr_offset}]")
            else:
                # Compare to previous round
                prev_idx = stab_meas_record[("X", s.index, rnd - 1)]
                prev_offset = meas_count - prev_idx
                lines.append(
                    f"DETECTOR({s.index}, 0, {rnd}) rec[{-curr_offset}] rec[{-prev_offset}]",
                )

        for s in geom.z_stabilizers:
            curr_idx = stab_meas_record[("Z", s.index, rnd)]
            curr_offset = meas_count - curr_idx
            det_x = num_x_anc + s.index  # Offset X stabilizer count for unique x-coord
            if rnd == 0:
                lines.append(f"DETECTOR({det_x}, 1, {rnd}) rec[{-curr_offset}]")
            else:
                prev_idx = stab_meas_record[("Z", s.index, rnd - 1)]
                prev_offset = meas_count - prev_idx
                lines.append(
                    f"DETECTOR({det_x}, 1, {rnd}) rec[{-curr_offset}] rec[{-prev_offset}]",
                )

        lines.append("")

    # =========================================================================
    # measure_z_basis / measure_x_basis
    # Guppy:
    #   measure_z_basis: return measure_array(surf.data)
    #   measure_x_basis: for i in range(num_data): h(surf.data[i])
    #                    return measure_array(surf.data)
    # =========================================================================
    lines.append(f"# === measure_{basis.lower()}_basis ===")

    # For X-basis: H on each data qubit first
    if basis.upper() == "X":
        for i in range(num_data):
            lines.append(f"H {data_q(i)}")
            if p1 > 0:
                lines.append(f"DEPOLARIZE1({p1}) {data_q(i)}")

    # Measure all data qubits
    final_meas_start = meas_count
    for i in range(num_data):
        if p_meas > 0:
            lines.append(f"X_ERROR({p_meas}) {data_q(i)}")
        lines.append(f"M {data_q(i)}")
        meas_count += 1

    lines.append("")

    # Add final detectors comparing final measurement to last syndrome round
    # For Z-basis: Z stabilizers compare to final Z measurements
    # For X-basis: X stabilizers compare to final X measurements
    if basis.upper() == "Z":
        stabilizers = geom.z_stabilizers
        stab_type = "Z"
        logical_qubits = list(geom.logical_z.data_qubits) if geom.logical_z else []
    else:
        stabilizers = geom.x_stabilizers
        stab_type = "X"
        logical_qubits = list(geom.logical_x.data_qubits) if geom.logical_x else []

    lines.append(
        "# Final detectors: compare final data measurement parity to last syndrome",
    )
    for s in stabilizers:
        # Get final measurement values for data qubits in this stabilizer
        data_rec_offsets = []
        for dq in s.data_qubits:
            rec_idx = final_meas_start + dq
            data_rec_offsets.append(meas_count - rec_idx)

        # Get last syndrome measurement for this stabilizer
        last_syn_idx = stab_meas_record[(stab_type, s.index, num_rounds - 1)]
        syn_offset = meas_count - last_syn_idx

        # Detector compares syndrome to parity of data measurements
        rec_str = " ".join(f"rec[{-off}]" for off in data_rec_offsets)
        det_x = s.index if stab_type == "X" else num_x_anc + s.index
        det_y = 0 if stab_type == "X" else 1
        lines.append(
            f"DETECTOR({det_x}, {det_y}, {num_rounds}) {rec_str} rec[{-syn_offset}]",
        )

    # Observable: parity of logical operator measurements
    lines.append("")
    lines.append("# Logical observable")
    logical_rec_offsets = [meas_count - (final_meas_start + q) for q in logical_qubits]
    logical_rec_str = " ".join(f"rec[{-off}]" for off in logical_rec_offsets)
    lines.append(f"OBSERVABLE_INCLUDE(0) {logical_rec_str}")

    return "\n".join(lines)


def generate_circuit_level_dem(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    p: float = 0.01,
) -> str:
    """Generate a circuit-level DEM by building a Stim circuit and extracting the DEM.

    This uses Stim's built-in DEM generation which properly accounts for
    error propagation through the circuit.

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' or 'X' basis
        p: Uniform physical error rate

    Returns:
        DEM string in Stim format
    """
    try:
        import stim
    except ImportError as e:
        msg = "Stim is required for circuit-level DEM generation. Install with: pip install stim"
        raise ImportError(msg) from e

    # Generate circuit with noise
    circuit_str = generate_stim_circuit(
        patch,
        num_rounds,
        basis,
        p1=p,
        p2=p,
        p_meas=p,
        p_prep=p,
    )

    # Parse and generate DEM
    circuit = stim.Circuit(circuit_str)
    dem = circuit.detector_error_model()

    return str(dem)


def generate_dag_circuit(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
) -> DagCircuit:
    """Generate a PECOS DagCircuit from SurfacePatch geometry.

    This creates a memory experiment circuit matching the Stim circuit
    structure but in PECOS DagCircuit format.

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' for |0_L> state or 'X' for |+_L> state

    Returns:
        PECOS DagCircuit instance
    """
    from pecos_rslib import DagCircuit, Gate, GateType

    geom = patch.geometry
    num_data = geom.num_data
    num_x_anc = len(geom.x_stabilizers)
    num_z_anc = len(geom.z_stabilizers)
    total_qubits = num_data + num_x_anc + num_z_anc

    # Create circuit
    circuit = DagCircuit(total_qubits)

    # Qubit mappings
    def data_q(i: int) -> int:
        return i

    def x_anc_q(i: int) -> int:
        return num_data + i

    def z_anc_q(i: int) -> int:
        return num_data + num_x_anc + i

    # Get CNOT schedule
    cnot_rounds = compute_cnot_schedule(patch)

    # === State Preparation ===
    # Reset all qubits
    for q in range(total_qubits):
        circuit.append(Gate(GateType.RESET, [q]))

    # For X-basis, apply H to data qubits
    if basis.upper() == "X":
        for i in range(num_data):
            circuit.append(Gate(GateType.H, [data_q(i)]))

    # === Syndrome Extraction Rounds ===
    for _rnd in range(num_rounds):
        # Reset ancilla qubits
        for i in range(num_x_anc):
            circuit.append(Gate(GateType.RESET, [x_anc_q(i)]))
        for i in range(num_z_anc):
            circuit.append(Gate(GateType.RESET, [z_anc_q(i)]))

        # H on X ancillas
        for s in geom.x_stabilizers:
            circuit.append(Gate(GateType.H, [x_anc_q(s.index)]))

        # 4 CNOT rounds
        for cx_round in cnot_rounds:
            for stab_type, stab_idx, data_idx in cx_round:
                if stab_type == "X":
                    control = x_anc_q(stab_idx)
                    target = data_q(data_idx)
                else:
                    control = data_q(data_idx)
                    target = z_anc_q(stab_idx)
                circuit.append(Gate(GateType.CX, [control, target]))

        # H on X ancillas
        for s in geom.x_stabilizers:
            circuit.append(Gate(GateType.H, [x_anc_q(s.index)]))

        # Measure all ancillas
        for s in geom.x_stabilizers:
            circuit.append(Gate(GateType.MEASURE, [x_anc_q(s.index)]))
        for s in geom.z_stabilizers:
            circuit.append(Gate(GateType.MEASURE, [z_anc_q(s.index)]))

    # === Final Measurement ===
    if basis.upper() == "X":
        for i in range(num_data):
            circuit.append(Gate(GateType.H, [data_q(i)]))

    for i in range(num_data):
        circuit.append(Gate(GateType.MEASURE, [data_q(i)]))

    return circuit


def compare_dems(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    p: float = 0.01,
) -> dict:
    """Compare DEMs generated from different sources.

    Compares:
    1. Direct Stim circuit generation (reference)
    2. Phenomenological DEM (simpler model)

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        p: Physical error rate

    Returns:
        Dictionary with comparison results
    """
    from pecos.qec.surface.decode import NoiseModel, generate_surface_code_dem

    # Generate circuit-level DEM via Stim
    stim_dem = generate_circuit_level_dem(patch, num_rounds, basis, p=p)

    # Generate phenomenological DEM
    noise = NoiseModel(p1=p, p2=p, p_meas=p, p_prep=p)
    stab_type = "X" if basis.upper() == "X" else "Z"
    phenom_dem = generate_surface_code_dem(patch, num_rounds, noise, stab_type)

    # Parse and compare
    def parse_dem(dem_str: str) -> dict:
        """Parse DEM to extract statistics."""
        lines = [line.strip() for line in dem_str.split("\n") if line.strip() and not line.startswith("#")]
        errors = [line for line in lines if line.startswith("error")]
        detectors = [line for line in lines if line.startswith("detector")]
        observables = [line for line in lines if "logical" in line.lower() or "observable" in line.lower()]
        return {
            "error_count": len(errors),
            "detector_count": len(detectors),
            "observable_count": len(observables),
            "raw_errors": errors[:5],  # First 5 for inspection
        }

    stim_stats = parse_dem(stim_dem)
    phenom_stats = parse_dem(phenom_dem)

    return {
        "stim_circuit_level": stim_stats,
        "phenomenological": phenom_stats,
        "stim_dem_full": stim_dem,
        "phenom_dem_full": phenom_dem,
    }
