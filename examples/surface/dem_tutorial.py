"""DEM generation tutorial: build circuit, inspect DEM, sample, validate.

Demonstrates the full PECOS DEM pipeline using a d=3 surface code.

Usage:
    uv run python examples/surface/dem_tutorial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "quantum-pecos" / "src"))

from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos_rslib.qec import DemSampler, DetectorErrorModel
from pecos_rslib_exp import depolarizing, sim_neo, stabilizer


def main():
    # ================================================================
    # 1. Build a surface code circuit
    # ================================================================
    distance = 3
    patch = SurfacePatch.create(distance=distance)
    b = LogicalCircuitBuilder()
    b.add_patch(patch, "Q0")
    b.add_memory("Q0", rounds=distance, basis="Z")
    tc = b.to_tick_circuit()

    print(f"Surface code d={distance}, {tc.num_ticks()} ticks")
    print(
        f"  {int(tc.get_meta('num_measurements'))} measurements, "
        f"{len(json.loads(tc.get_meta('detectors')))} detectors",
    )

    # ================================================================
    # 2. Inspect measurement IDs on gates
    # ================================================================
    # Each MZ gate carries a MeasId — a stable identity for that
    # measurement result. These persist through all transformations.
    dag = tc.to_dag_circuit()

    print("\nMeasurement gates (first 5):")
    shown = 0
    for node_id in dag.nodes():
        gate = dag.gate(node_id)
        if gate and gate.gate_type.name == "MZ" and shown < 5:
            print(f"  node={node_id}: MZ qubit={list(gate.qubits)} meas_ids={gate.meas_ids}")
            shown += 1

    # ================================================================
    # 3. Inspect detector definitions
    # ================================================================
    # Detectors reference measurements by both:
    #   - "records": negative offsets (Stim compatibility)
    #   - "meas_ids": stable MeasId values (preferred)
    dets = json.loads(tc.get_meta("detectors"))
    print(f"\nDetector definitions (first 3 of {len(dets)}):")
    for det in dets[:3]:
        print(f"  D{det['id']}: meas_ids={det['meas_ids']}  records={det['records']}")

    # ================================================================
    # 4. Build and inspect the DEM (one line)
    # ================================================================
    p = 0.005
    dem = DetectorErrorModel.from_circuit(tc, p1=p, p2=p, p_meas=p, p_prep=p)
    print(f"\nDetectorErrorModel: {dem.num_detectors} detectors, {dem.num_observables} observables")

    dem_str = dem.to_string()
    error_lines = [line for line in dem_str.split("\n") if line.startswith("error(")]
    print(f"  {len(error_lines)} DEM events")
    print("  First 3 events:")
    for line in error_lines[:3]:
        print(f"    {line}")

    # ================================================================
    # 5. Sample from the DEM (one line)
    # ================================================================
    shots = 100_000
    sampler = DemSampler.from_circuit(tc, p1=p, p2=p, p_meas=p, p_prep=p)
    batch = sampler.generate_samples(num_shots=shots, seed=42)

    # Compute per-detector firing rates from DEM sampling
    num_dets = len(dets)
    dem_rates = [0.0] * num_dets
    for i in range(shots):
        syn = batch.get_syndrome(i)
        for d in range(min(num_dets, len(syn))):
            if syn[d]:
                dem_rates[d] += 1.0 / shots

    # ================================================================
    # 6. Validate against stabilizer simulation (ground truth)
    # ================================================================
    noise = depolarizing().p1(p).p2(p).p_meas(p).p_prep(p)
    results = sim_neo(tc).quantum(stabilizer()).noise(noise).shots(shots).seed(42).run()

    num_meas = int(tc.get_meta("num_measurements"))
    sim_rates = [0.0] * num_dets
    for r in results:
        meas = list(r)
        for i, det in enumerate(dets):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                sim_rates[i] += 1.0 / shots

    # ================================================================
    # 7. Compare
    # ================================================================
    print(f"\nPer-detector rates (p={p}, {shots} shots):")
    print(f"  {'Det':>4} {'DEM':>10} {'Stabilizer':>10} {'Ratio':>7}")
    max_rel = 0
    for d in range(num_dets):
        if sim_rates[d] > 0.003:
            ratio = dem_rates[d] / sim_rates[d]
            max_rel = max(max_rel, abs(1 - ratio))
            print(f"  D{d:>2} {dem_rates[d]:>10.5f} {sim_rates[d]:>10.5f} {ratio:>7.3f}")

    status = "PASS" if max_rel < 0.15 else f"FAIL (max_rel={max_rel*100:.0f}%)"
    print(f"\nValidation: {status}")
    print(f"  Max relative error: {max_rel*100:.1f}% (threshold: 15% for {shots} shots)")


if __name__ == "__main__":
    main()
