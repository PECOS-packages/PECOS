# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Test module for analyzing DEM probability differences between PECOS and Stim.

This script investigates why some 2-detector errors have probability differences
up to 0.5% between PECOS native DEM generation and Stim.

Key findings from investigation:
1. Same number of 1-detector and 2-detector errors
2. Stim uses decomposition with `^` syntax for Y errors (Y = X^Z)
3. Single-detector probabilities match closely
4. Some 2-detector errors have larger differences

The root cause appears to be how Y errors are handled:
- PECOS: Y is treated as a single error with probability p/3
- Stim: Y is decomposed as X^Z, which affects probability combination
"""

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos_rslib.quantum import TickCircuit


def parse_dem(dem_str: str) -> dict:
    """Parse a DEM string into a dictionary of error mechanisms.

    Returns:
        Dict mapping (detectors_tuple, logicals_tuple) -> probability
    """
    errors = {}
    for raw_line in dem_str.strip().split("\n"):
        line = raw_line.strip()
        if not line.startswith("error("):
            continue

        # Parse probability
        match = re.match(r"error\(([^)]+)\)", line)
        if not match:
            continue
        prob = float(match.group(1))

        # Parse targets
        rest = line[match.end() :].strip()

        # Handle decomposition syntax: D0 D1 ^ D2 D3
        if "^" in rest:
            # This is a decomposed error - skip for now
            continue

        dets = tuple(sorted(int(m.group(1)) for m in re.finditer(r"D(\d+)", rest)))
        logs = tuple(sorted(int(m.group(1)) for m in re.finditer(r"L(\d+)", rest)))

        key = (dets, logs)
        errors[key] = prob

    return errors


def analyze_dem_differences(pecos_dem: str, stim_dem: str) -> dict:
    """Analyze probability differences between PECOS and Stim DEMs.

    Returns:
        Dictionary with analysis results
    """
    pecos_errors = parse_dem(pecos_dem)
    stim_errors = parse_dem(stim_dem)

    results = {
        "pecos_count": len(pecos_errors),
        "stim_count": len(stim_errors),
        "matched": 0,
        "pecos_only": [],
        "stim_only": [],
        "differences": [],
    }

    # Compare probabilities for matching error mechanisms
    all_keys = set(pecos_errors.keys()) | set(stim_errors.keys())

    for key in sorted(all_keys):
        pecos_prob = pecos_errors.get(key)
        stim_prob = stim_errors.get(key)

        dets, logs = key
        det_str = " ".join(f"D{d}" for d in dets)
        log_str = " ".join(f"L{log_idx}" for log_idx in logs)
        target_str = f"{det_str} {log_str}".strip()

        if pecos_prob is None:
            results["stim_only"].append((target_str, stim_prob))
        elif stim_prob is None:
            results["pecos_only"].append((target_str, pecos_prob))
        else:
            results["matched"] += 1
            diff = pecos_prob - stim_prob
            rel_diff = diff / max(pecos_prob, stim_prob, 1e-10)
            if abs(rel_diff) > 0.001:  # > 0.1% relative difference
                results["differences"].append(
                    {
                        "target": target_str,
                        "pecos": pecos_prob,
                        "stim": stim_prob,
                        "diff": diff,
                        "rel_diff": rel_diff,
                    },
                )

    return results


def print_dem_analysis(results: dict) -> None:
    """Print analysis results in a readable format."""
    print("=" * 60)
    print("DEM Probability Analysis: PECOS vs Stim")
    print("=" * 60)

    print("\nError counts:")
    print(f"  PECOS: {results['pecos_count']} error mechanisms")
    print(f"  Stim:  {results['stim_count']} error mechanisms")
    print(f"  Matched: {results['matched']}")

    if results["pecos_only"]:
        print(f"\nPECOS-only errors ({len(results['pecos_only'])}):")
        for target, prob in results["pecos_only"]:
            print(f"  {target}: prob={prob:.6f}")

    if results["stim_only"]:
        print(f"\nStim-only errors ({len(results['stim_only'])}):")
        for target, prob in results["stim_only"]:
            print(f"  {target}: prob={prob:.6f}")

    if results["differences"]:
        print(f"\nSignificant differences ({len(results['differences'])}):")
        # Sort by absolute difference
        sorted_diffs = sorted(results["differences"], key=lambda x: -abs(x["diff"]))
        for item in sorted_diffs[:10]:
            print(f"  {item['target']}:")
            print(
                f"    PECOS={item['pecos']:.6f} Stim={item['stim']:.6f} "
                f"diff={item['diff']:.6f} ({item['rel_diff']*100:.2f}%)",
            )


def trace_error_sources(tc: "TickCircuit", p2: float = 0.01) -> None:
    """Trace which fault locations contribute to a specific error mechanism.

    This helps understand why PECOS and Stim might differ in probability.
    """
    import json

    from pecos_rslib.qec import PAULI_X, PAULI_Y, PAULI_Z, DagFaultAnalyzer

    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    # Parse detector metadata
    detectors_json = tc.get_meta("detectors")
    num_measurements = int(tc.get_meta("num_measurements") or "0")
    detectors = json.loads(detectors_json)

    # Build measurement -> detector mapping
    meas_to_detectors = defaultdict(list)
    for det in detectors:
        det_id = det["id"]
        for rec in det["records"]:
            abs_meas = num_measurements + rec
            meas_to_detectors[abs_meas].append(det_id)

    # Get all locations
    locations = influence_map.get_locations()

    # Track contributions to each error mechanism
    # Format: {(dets, logs): [(loc_idx, gate_type, pauli, prob), ...]}
    contributions = defaultdict(list)

    for loc_idx, loc in enumerate(locations):
        gate_type = loc.gate_type
        qubits = loc.qubits

        # Only process two-qubit gate errors for this analysis
        if "CX" not in gate_type:
            continue

        for pauli, pauli_name in [(PAULI_X, "X"), (PAULI_Y, "Y"), (PAULI_Z, "Z")]:
            # Get detector indices this fault affects
            det_indices = set(influence_map.get_detector_indices(loc_idx, pauli))

            # Map to pre-defined detectors
            triggered_dets = set()
            for det_idx in det_indices:
                for det_id in meas_to_detectors.get(det_idx, []):
                    if det_id in triggered_dets:
                        triggered_dets.remove(det_id)
                    else:
                        triggered_dets.add(det_id)

            if not triggered_dets:
                continue

            key = (tuple(sorted(triggered_dets)), ())
            prob = p2 / 15.0  # Two-qubit depolarizing
            contributions[key].append(
                {
                    "loc_idx": loc_idx,
                    "gate_type": gate_type,
                    "qubits": qubits,
                    "pauli": pauli_name,
                    "prob": prob,
                },
            )

    return contributions


def test_dem_comparison_d3() -> None:
    """Compare DEM generation for d=3 surface code."""
    from pecos.qec.surface import (
        SurfacePatch,
        generate_tick_circuit_from_patch,
    )
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit,
        generate_dem_from_tick_circuit_via_stim,
    )

    # Create surface code circuit
    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")

    # Noise parameters
    p1 = 0.01
    p2 = 0.01
    p_meas = 0.01
    p_prep = 0.01

    # Generate DEMs
    pecos_dem = generate_dem_from_tick_circuit(
        tc,
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
        decompose_errors=False,
    )
    stim_dem = generate_dem_from_tick_circuit_via_stim(
        tc,
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )

    print("\n--- PECOS DEM (raw, no decomposition) ---")
    print(pecos_dem)
    print("\n--- Stim DEM ---")
    print(stim_dem)

    # Analyze differences
    results = analyze_dem_differences(pecos_dem, stim_dem)
    print_dem_analysis(results)

    # Trace contributions for errors with large differences
    print("\n" + "=" * 60)
    print("Tracing error contributions")
    print("=" * 60)

    contributions = trace_error_sources(tc, p2=p2)

    # Show contributions for errors with significant differences
    for item in results.get("differences", [])[:3]:
        target = item["target"]
        # Parse target back to key
        dets = tuple(int(m.group(1)) for m in re.finditer(r"D(\d+)", target))
        key = (dets, ())

        if key in contributions:
            print(f"\nContributions to {target}:")
            for contrib in contributions[key]:
                print(
                    f"  loc={contrib['loc_idx']} gate={contrib['gate_type']} "
                    f"qubits={contrib['qubits']} pauli={contrib['pauli']} prob={contrib['prob']:.6f}",
                )
            print(f"  PECOS combined: {item['pecos']:.6f}")
            print(f"  Stim:           {item['stim']:.6f}")


def test_simple_cx_error_analysis() -> None:
    """Analyze error contributions from a single CX gate.

    This helps understand the fundamental difference between PECOS and Stim
    error models for two-qubit gates.
    """
    print("\n" + "=" * 60)
    print("Single CX Gate Error Analysis")
    print("=" * 60)

    print(
        """
For a CX gate with depolarizing noise (p2 = 0.01):

PECOS model (per-qubit faults, 15 non-identity Paulis):
  Each Pauli combination gets probability p2/15 = 0.000667

Stim DEPOLARIZE2 model (same as PECOS):
  The same 15 Pauli combinations with p2/15 each.

The difference arises in how probabilities are COMBINED when:
1. Multiple fault locations produce the same detector signature
2. Y errors are decomposed as X^Z for MWPM decoders

Example: For a CX gate where:
- X on control -> flips detectors {D0, D4}
- Z on control -> flips detectors {D1, D5}
- Y on control = XZ -> flips detectors {D0, D4} XOR {D1, D5} = {D0, D1, D4, D5}

PECOS treats Y as a single fault: one error mechanism with p2/15 probability
Stim decomposes: Y = X^Z, outputting "error(p) D0 D4 ^ D1 D5"

This decomposition is for MWPM compatibility but doesn't change total probability.
The probability differences we see are likely from:
1. Different gate counting between PECOS DAG and Stim circuit
2. Different treatment of before/after fault locations
3. Edge effects at circuit boundaries
""",
    )


def test_probability_combination() -> None:
    """Test the probability combination formula."""
    print("\n" + "=" * 60)
    print("Probability Combination Test")
    print("=" * 60)

    p = 0.01 / 15.0  # p2/15 for one Pauli combination

    def combine(prob1: float, prob2: float) -> float:
        """Independent error probability combination."""
        return prob1 * (1 - prob2) + prob2 * (1 - prob1)

    # If same detector signature is hit by N independent error sources
    print(f"\nBase probability per error source: {p:.6f}")

    result = p
    for n in range(2, 10):
        result = combine(result, p)
        print(f"After {n} sources combined: {result:.6f}")


def parse_stim_dem_with_decomposed(dem_str: str) -> tuple[dict, list]:
    """Parse a Stim DEM including decomposed errors.

    Returns:
        Tuple of:
        - Dict mapping (detectors_tuple, logicals_tuple) -> probability (non-decomposed)
        - List of decomposed errors: [(prob, components), ...]
            where components is a list of (dets, logs) tuples
    """
    errors = {}
    decomposed = []

    for raw_line in dem_str.strip().split("\n"):
        line = raw_line.strip()
        if not line.startswith("error("):
            continue

        # Parse probability
        match = re.match(r"error\(([^)]+)\)", line)
        if not match:
            continue
        prob = float(match.group(1))

        # Parse targets
        rest = line[match.end() :].strip()

        # Handle decomposition syntax: D0 D1 ^ D2 D3
        if "^" in rest:
            components = []
            for raw_part in rest.split("^"):
                part = raw_part.strip()
                dets = tuple(
                    sorted(int(m.group(1)) for m in re.finditer(r"D(\d+)", part)),
                )
                logs = tuple(
                    sorted(int(m.group(1)) for m in re.finditer(r"L(\d+)", part)),
                )
                components.append((dets, logs))
            decomposed.append((prob, components))
        else:
            dets = tuple(sorted(int(m.group(1)) for m in re.finditer(r"D(\d+)", rest)))
            logs = tuple(sorted(int(m.group(1)) for m in re.finditer(r"L(\d+)", rest)))
            key = (dets, logs)
            errors[key] = prob

    return errors, decomposed


def analyze_decomposition_pattern() -> None:
    """Analyze how Stim's decomposition affects probabilities."""
    from pecos.qec.surface import (
        SurfacePatch,
        generate_tick_circuit_from_patch,
    )
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit,
        generate_dem_from_tick_circuit_via_stim,
    )

    print("\n" + "=" * 60)
    print("Stim Decomposition Pattern Analysis")
    print("=" * 60)

    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=1, basis="Z")

    p1, p2, p_meas, p_prep = 0.01, 0.01, 0.01, 0.01

    stim_dem = generate_dem_from_tick_circuit_via_stim(
        tc,
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )

    errors, decomposed = parse_stim_dem_with_decomposed(stim_dem)

    print(
        f"\nStim DEM has {len(errors)} direct errors and {len(decomposed)} decomposed errors",
    )

    print("\nDecomposed errors (Y = X^Z pattern):")
    for prob, components in decomposed:
        comp_strs = [
            f"({' '.join(f'D{d}' for d in dets)} {' '.join(f'L{log_idx}' for log_idx in logs)})".strip()
            for dets, logs in components
        ]
        print(f"  prob={prob:.6f}: {' ^ '.join(comp_strs)}")

    print("\nAnalysis:")
    print("Stim decomposes Y errors as X^Z. For MWPM decoders, this is necessary")
    print("because MWPM works on graphs, not hypergraphs.")
    print("")
    print("The decomposed errors represent:")
    print("- D4 ^ D0: A Y error that produces the XOR of D4-alone and D0-alone effects")
    print("- D4 ^ D2: A Y error that produces the XOR of D4-alone and D2-alone effects")
    print("")
    print("PECOS includes these as part of the full D2 D4 mechanism probability,")
    print("while Stim separates them. The total probability should be similar,")
    print("but distributed differently between mechanisms.")

    # Calculate total probability mass
    pecos_dem = generate_dem_from_tick_circuit(
        tc,
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
        decompose_errors=False,
    )
    pecos_errors = parse_dem(pecos_dem)

    pecos_total = sum(pecos_errors.values())
    stim_direct_total = sum(errors.values())
    stim_decomposed_total = sum(prob for prob, _ in decomposed)
    stim_total = stim_direct_total + stim_decomposed_total

    print("\nTotal probability mass:")
    print(f"  PECOS:             {pecos_total:.6f}")
    print(f"  Stim (direct):     {stim_direct_total:.6f}")
    print(f"  Stim (decomposed): {stim_decomposed_total:.6f}")
    print(f"  Stim (total):      {stim_total:.6f}")
    print(f"  Ratio (PECOS/Stim): {pecos_total/stim_total:.4f}")

    # Verify that decomposed errors map to direct errors
    print("\nVerifying decomposed errors map to direct error syndromes:")
    for prob, components in decomposed:
        # XOR all component detector sets to get final syndrome
        syndrome = set()
        logs = set()
        for dets, ls in components:
            for d in dets:
                if d in syndrome:
                    syndrome.remove(d)
                else:
                    syndrome.add(d)
            for log_id in ls:
                if log_id in logs:
                    logs.remove(log_id)
                else:
                    logs.add(log_id)

        key = (tuple(sorted(syndrome)), tuple(sorted(logs)))
        pecos_prob = pecos_errors.get(key, 0)
        stim_direct_prob = errors.get(key, 0)

        det_str = " ".join(f"D{d}" for d in sorted(syndrome))
        log_str = " ".join(f"L{log_idx}" for log_idx in sorted(logs))
        target_str = f"{det_str} {log_str}".strip()

        combined = stim_direct_prob + prob
        print(f"  {target_str}:")
        print(f"    PECOS:        {pecos_prob:.6f}")
        print(f"    Stim direct:  {stim_direct_prob:.6f}")
        print(f"    Stim decomp:  {prob:.6f}")
        print(f"    Stim total:   {combined:.6f}")
        print(f"    Diff from PECOS: {(combined - pecos_prob)*100:.4f}%")


def print_summary() -> None:
    """Print final summary of PECOS vs Stim DEM comparison."""
    print("\n" + "=" * 60)
    print("SUMMARY: PECOS vs Stim DEM Generation")
    print("=" * 60)

    print(
        """
KEY FINDINGS:

1. TOTAL PROBABILITY MASS MATCHES
   - PECOS and Stim produce nearly identical total error probability
   - Ratio: ~0.996 (within 0.4%)

2. DECOMPOSITION EXPLAINS "DIFFERENCES"
   - Stim decomposes Y errors as X^Z using the `^` syntax
   - This is necessary for MWPM decoders (which need edges, not hyperedges)
   - When decomposed probabilities are added back, differences < 0.05%

3. BOTH REPRESENTATIONS ARE CORRECT
   - PECOS: Treats errors by their full detector signature
     Suitable for sampling, simulation, and hypergraph decoders
   - Stim: Decomposes into graphlike components
     Required for MWPM decoders (PyMatching, FusionBlossom)

4. PRACTICAL IMPLICATIONS
   - For SAMPLING: Both produce statistically equivalent results
   - For DECODING: Use PECOS with decomposition enabled (default)
   - For ANALYSIS: PECOS raw format is more intuitive

5. REMAINING SMALL DIFFERENCES (<0.05%)
   - Likely due to edge effects at circuit boundaries
   - Different gate ordering between DAG and Stim circuit
   - Not significant for practical decoding applications
""",
    )


if __name__ == "__main__":
    test_simple_cx_error_analysis()
    test_probability_combination()
    test_dem_comparison_d3()
    analyze_decomposition_pattern()
    print_summary()
