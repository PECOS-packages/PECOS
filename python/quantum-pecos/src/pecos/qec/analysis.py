# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""QEC result analysis and post-processing utilities.

This module provides utilities for analyzing quantum error correction
results, including logical operator extraction, fidelity calculation,
and syndrome processing.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def logical_x_from_data(d: int, data: Sequence[int]) -> int:
    """Extract logical X value from measurement data.

    For a surface code, the logical X operator is supported on the
    left column of data qubits (indices 0, d, 2d, ...).

    Args:
        d: Code distance.
        data: Measurement outcomes for d^2 data qubits.

    Returns:
        Logical X value (0 or 1) computed as XOR of left column.
    """
    if len(data) != d * d:
        msg = f"Expected {d*d} data qubits, got {len(data)}"
        raise ValueError(msg)

    result = 0
    for i in range(d):
        result ^= data[i * d]
    return result


def logical_z_from_data(d: int, data: Sequence[int]) -> int:
    """Extract logical Z value from measurement data.

    For a surface code, the logical Z operator is supported on the
    top row of data qubits (indices 0, 1, 2, ..., d-1).

    Args:
        d: Code distance.
        data: Measurement outcomes for d^2 data qubits.

    Returns:
        Logical Z value (0 or 1) computed as XOR of top row.
    """
    if len(data) != d * d:
        msg = f"Expected {d*d} data qubits, got {len(data)}"
        raise ValueError(msg)

    result = 0
    for i in range(d):
        result ^= data[i]
    return result


def logical_from_data(d: int, data: Sequence[int]) -> tuple[int, int]:
    """Extract both logical X and Z values from measurement data.

    Args:
        d: Code distance.
        data: Measurement outcomes for d^2 data qubits.

    Returns:
        Tuple of (logical_x, logical_z) values.
    """
    return logical_x_from_data(d, data), logical_z_from_data(d, data)


def logical_fidelity(
    outcomes: Sequence[Sequence[int]],
    d: int,
    basis: int,
    expected: int = 0,
) -> tuple[float, float]:
    """Compute logical fidelity from multiple measurement outcomes.

    Calculates the fraction of shots where the logical measurement
    matches the expected value, with binomial error bars.

    Args:
        outcomes: List of measurement outcomes, each with d^2 data qubits.
        d: Code distance.
        basis: Measurement basis (0=X, 1=Z).
        expected: Expected logical value (default 0).

    Returns:
        Tuple of (fidelity, error) where error is the standard deviation
        assuming binomial statistics: sqrt(f * (1-f) / num_shots).
    """
    if not outcomes:
        msg = "No outcomes provided"
        raise ValueError(msg)

    num_shots = len(outcomes)
    successes = 0

    for data in outcomes:
        logical = logical_x_from_data(d, data) if basis == 0 else logical_z_from_data(d, data)

        if logical == expected:
            successes += 1

    fidelity = successes / num_shots

    # Binomial error bar
    error = 0.0 if fidelity == 0.0 or fidelity == 1.0 else math.sqrt(fidelity * (1 - fidelity) / num_shots)

    return fidelity, error


def syndrome_difference(
    syndromes: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Compute syndrome differences between consecutive rounds.

    For MWPM decoding, we need the syndrome changes (differences)
    between rounds rather than the raw syndromes. The first round
    is compared against all-zeros (initialization).

    Args:
        syndromes: List of syndrome measurements, one per round.
            Each syndrome is a sequence of stabilizer measurement results.

    Returns:
        List of syndrome differences. The i-th difference is
        syndromes[i] XOR syndromes[i-1] (with syndromes[-1] = 0).
    """
    if not syndromes:
        return []

    num_stab = len(syndromes[0])
    result = []

    # First round: difference from all-zeros
    result.append(list(syndromes[0]))

    # Subsequent rounds: difference from previous
    for i in range(1, len(syndromes)):
        diff = [syndromes[i][j] ^ syndromes[i - 1][j] for j in range(num_stab)]
        result.append(diff)

    return result


def syndrome_to_detection_events(
    syndromes: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    """Convert syndrome history to detection event coordinates.

    Detection events are (stabilizer_index, round) pairs where
    the syndrome changed (difference is 1).

    Args:
        syndromes: List of syndrome measurements, one per round.

    Returns:
        List of (stabilizer_index, round) tuples for each detection event.
    """
    differences = syndrome_difference(syndromes)
    events = []

    for t, diff in enumerate(differences):
        for s, val in enumerate(diff):
            if val == 1:
                events.append((s, t))

    return events


def logical_error_rate(
    outcomes: Sequence[Sequence[int]],
    d: int,
    basis: int,
    expected: int = 0,
) -> tuple[float, float]:
    """Compute logical error rate from measurement outcomes.

    This is simply 1 - fidelity, with propagated error bars.

    Args:
        outcomes: List of measurement outcomes, each with d^2 data qubits.
        d: Code distance.
        basis: Measurement basis (0=X, 1=Z).
        expected: Expected logical value (default 0).

    Returns:
        Tuple of (error_rate, error_bar).
    """
    fidelity, error = logical_fidelity(outcomes, d, basis, expected)
    return 1 - fidelity, error


def lower_bound_fidelity(f1: float, f2: float) -> float:
    """Compute lower bound on true fidelity from two measurements.

    Uses the formula: bound = (4/5) * (f1 + f2) - (3/5)

    This provides a statistical lower bound accounting for
    measurement correlations when measuring in two different bases.

    Args:
        f1: Fidelity from first measurement basis.
        f2: Fidelity from second measurement basis.

    Returns:
        Lower bound on true fidelity.
    """
    return (4 / 5) * (f1 + f2) - (3 / 5)


# ---------------------------------------------------------------------------
# Detector flip frequency matrices
# ---------------------------------------------------------------------------


def detector_flip_matrix(
    detector_events: Sequence[Sequence[int]],
    num_detectors: int,
) -> list[list[float]]:
    """Compute the detector flip frequency matrix from sampled detector events.

    The matrix M has:
      - M[i][i] = P(detector i fires)                    (marginal rate)
      - M[i][j] = 0.5 * P(detector i AND j both fire)    (half joint rate)

    The factor 0.5 on off-diagonal entries makes the matrix interpretable
    as a covariance-like object: each correlated error mechanism contributes
    equally to M[i][j] and M[j][i].

    Args:
        detector_events: Per-shot detector outcomes. Each entry is a sequence
            of detector indices that fired in that shot. Alternatively, each
            entry can be a full-length binary vector (0/1) of length
            ``num_detectors``.
        num_detectors: Total number of detectors.

    Returns:
        ``num_detectors x num_detectors`` matrix as nested lists.
    """
    n = num_detectors
    shots = len(detector_events)
    if shots == 0:
        return [[0.0] * n for _ in range(n)]

    inv_shots = 1.0 / shots
    half_inv = 0.5 * inv_shots

    # Accumulate as flat list for speed
    m = [0.0] * (n * n)

    for events in detector_events:
        # Determine which detectors fired
        fired = [i for i in range(n) if events[i]] if len(events) == n else list(events)

        for a in fired:
            m[a * n + a] += inv_shots  # diagonal
            for b in fired:
                if b > a:
                    m[a * n + b] += half_inv
                    m[b * n + a] += half_inv

    return [m[i * n : (i + 1) * n] for i in range(n)]


def detector_flip_matrices_by_round(
    detector_events: Sequence[Sequence[int]],
    num_detectors: int,
    detectors_per_round: int,
) -> list[list[list[float]]]:
    """Compute per-round detector flip frequency matrices.

    Groups detectors into rounds of ``detectors_per_round`` each and
    computes a separate flip frequency matrix for each round.

    Args:
        detector_events: Per-shot detector outcomes (same format as
            :func:`detector_flip_matrix`).
        num_detectors: Total number of detectors.
        detectors_per_round: Number of detectors in each syndrome
            extraction round.

    Returns:
        List of matrices, one per round.
    """
    num_rounds = (num_detectors + detectors_per_round - 1) // detectors_per_round
    shots = len(detector_events)
    if shots == 0:
        return [[[0.0] * detectors_per_round for _ in range(detectors_per_round)] for _ in range(num_rounds)]

    inv_shots = 1.0 / shots
    half_inv = 0.5 * inv_shots
    k = detectors_per_round

    # One flat matrix per round
    matrices = [[0.0] * (k * k) for _ in range(num_rounds)]

    for events in detector_events:
        fired = [i for i in range(num_detectors) if events[i]] if len(events) == num_detectors else list(events)

        # Bin by round
        round_fired: dict[int, list[int]] = {}
        for d in fired:
            r = d // k
            local = d % k
            round_fired.setdefault(r, []).append(local)

        for r, local_ids in round_fired.items():
            if r >= num_rounds:
                continue
            mat = matrices[r]
            for a in local_ids:
                mat[a * k + a] += inv_shots
                for b in local_ids:
                    if b > a:
                        mat[a * k + b] += half_inv
                        mat[b * k + a] += half_inv

    return [[matrices[r][i * k : (i + 1) * k] for i in range(k)] for r in range(num_rounds)]


def compare_flip_matrices(
    sim_matrix: Sequence[Sequence[float]],
    dem_matrix: Sequence[Sequence[float]],
    *,
    min_rate: float = 0.0005,
) -> tuple[float, float, tuple[int, int]]:
    """Compare two detector flip frequency matrices.

    Args:
        sim_matrix: Ground truth matrix (from simulation).
        dem_matrix: Test matrix (from DEM sampling).
        min_rate: Minimum entry value to consider (avoids division by tiny
            numbers from statistical noise).

    Returns:
        Tuple of ``(max_relative_error, frobenius_relative_error,
        worst_element)`` where ``worst_element`` is the ``(i, j)`` index
        of the largest relative error.
    """
    n = len(sim_matrix)
    max_err = 0.0
    worst = (0, 0)
    sum_sq_diff = 0.0
    sum_sq_sim = 0.0

    for i in range(n):
        for j in range(n):
            s = sim_matrix[i][j]
            d = dem_matrix[i][j]
            diff = d - s
            sum_sq_diff += diff * diff
            sum_sq_sim += s * s
            if s > min_rate:
                rel = abs(diff / s)
                if rel > max_err:
                    max_err = rel
                    worst = (i, j)

    frob_rel = (sum_sq_diff**0.5) / max(sum_sq_sim**0.5, 1e-30)
    return max_err, frob_rel, worst


# ---------------------------------------------------------------------------
# Higher-order detector correlation analysis
# ---------------------------------------------------------------------------


def detector_k_body_rates(
    detector_events: Sequence[Sequence[int]],
    num_detectors: int,
    max_order: int = 3,
) -> dict[tuple[int, ...], float]:
    """Compute k-body detector firing rates up to a given order.

    For each subset of detectors of size 1..max_order that fires together
    in at least one shot, records the joint firing probability.

    - 1-body: ``(i,)`` -> P(Di fires)
    - 2-body: ``(i, j)`` -> P(Di AND Dj fire)
    - 3-body: ``(i, j, k)`` -> P(Di AND Dj AND Dk fire)

    Keys are sorted tuples of detector indices.

    Args:
        detector_events: Per-shot detector outcomes. Each entry is either
            a list of fired detector indices or a full binary vector.
        num_detectors: Total number of detectors.
        max_order: Maximum correlation order (default 3).

    Returns:
        Dict mapping detector index tuples to joint firing rates.
    """
    shots = len(detector_events)
    if shots == 0:
        return {}

    inv_shots = 1.0 / shots
    rates: dict[tuple[int, ...], float] = {}

    for events in detector_events:
        fired = [i for i in range(num_detectors) if events[i]] if len(events) == num_detectors else sorted(events)

        for k in range(1, min(max_order, len(fired)) + 1):
            for combo in combinations(fired, k):
                if combo in rates:
                    rates[combo] += inv_shots
                else:
                    rates[combo] = inv_shots

    return rates


def detector_k_body_rates_by_round(
    detector_events: Sequence[Sequence[int]],
    num_detectors: int,
    detectors_per_round: int,
    max_order: int = 3,
) -> list[dict[tuple[int, ...], float]]:
    """Compute per-round k-body detector firing rates.

    Groups detectors into rounds, then computes k-body rates within
    each round. Detector indices in the returned dicts are round-local
    (0..detectors_per_round-1).

    Args:
        detector_events: Per-shot detector outcomes.
        num_detectors: Total number of detectors.
        detectors_per_round: Number of detectors per round.
        max_order: Maximum correlation order (default 3).

    Returns:
        List of dicts, one per round, mapping local detector index
        tuples to joint firing rates.
    """
    k = detectors_per_round
    num_rounds = (num_detectors + k - 1) // k
    shots = len(detector_events)
    if shots == 0:
        return [{} for _ in range(num_rounds)]

    inv_shots = 1.0 / shots
    round_rates: list[dict[tuple[int, ...], float]] = [{} for _ in range(num_rounds)]

    for events in detector_events:
        fired = [i for i in range(num_detectors) if events[i]] if len(events) == num_detectors else sorted(events)

        # Bin fired detectors by round
        round_fired: dict[int, list[int]] = {}
        for d in fired:
            r = d // k
            local = d % k
            round_fired.setdefault(r, []).append(local)

        for r, local_ids in round_fired.items():
            if r >= num_rounds:
                continue
            rr = round_rates[r]
            for order in range(1, min(max_order, len(local_ids)) + 1):
                for combo in combinations(local_ids, order):
                    if combo in rr:
                        rr[combo] += inv_shots
                    else:
                        rr[combo] = inv_shots

    return round_rates


def compare_k_body_rates(
    sim_rates: dict[tuple[int, ...], float],
    dem_rates: dict[tuple[int, ...], float],
    *,
    min_rate: float = 0.0005,
    max_order: int | None = None,
) -> dict[int, tuple[float, float, tuple[int, ...]]]:
    """Compare k-body rates between simulation and DEM, grouped by order.

    Args:
        sim_rates: Ground truth rates from simulation.
        dem_rates: Rates from DEM sampling.
        min_rate: Minimum rate to consider for relative error.
        max_order: If set, only compare up to this order.

    Returns:
        Dict mapping order k to ``(max_relative_error,
        rms_relative_error, worst_event)`` for that order.
    """
    # Collect all keys
    all_keys = set(sim_rates.keys()) | set(dem_rates.keys())

    # Group by order
    by_order: dict[int, list[tuple[tuple[int, ...], float, float]]] = {}
    for key in all_keys:
        k = len(key)
        if max_order is not None and k > max_order:
            continue
        s = sim_rates.get(key, 0.0)
        d = dem_rates.get(key, 0.0)
        by_order.setdefault(k, []).append((key, s, d))

    result: dict[int, tuple[float, float, tuple[int, ...]]] = {}
    for k in sorted(by_order.keys()):
        entries = by_order[k]
        max_err = 0.0
        worst: tuple[int, ...] = ()
        sum_sq_rel = 0.0
        count = 0

        for key, s, d in entries:
            if s > min_rate:
                rel = abs(d / s - 1)
                if rel > max_err:
                    max_err = rel
                    worst = key
                sum_sq_rel += rel * rel
                count += 1

        rms = (sum_sq_rel / max(count, 1)) ** 0.5
        result[k] = (max_err, rms, worst)

    return result


# ---------------------------------------------------------------------------
# Simulation-based correlation table
# ---------------------------------------------------------------------------


def empirical_correlation_table(
    tick_circuit: object,
    noise_builder: object,
    shots: int,
    max_order: int = 2,
    *,
    backend: str = "stabilizer",
    seed: int = 42,
) -> list[tuple[tuple[int, ...], float]]:
    """Build an empirical correlation table from simulation.

    Runs ``sim_neo`` with the given noise model, extracts detector events
    per shot, and computes k-body joint detection rates. Same output format
    as :func:`exact_correlation_table` from the Heisenberg walk.

    Args:
        tick_circuit: A ``TickCircuit`` with detector metadata.
        noise_builder: A noise builder (e.g., ``depolarizing().p1(0.001).p2(0.01)``).
        shots: Number of simulation shots.
        max_order: Maximum correlation order (1 = marginals, 2 = pairwise, etc.).
        backend: Simulator backend — ``"stabilizer"``, ``"statevec"``,
            or ``"meas_sampling"``. The meas_sampling backend uses the fast
            whole-circuit DEM-based sampler (geometric/O(fired)) instead of
            gate-by-gate simulation.
        seed: RNG seed.

    Returns:
        List of ``(detector_indices_tuple, probability)`` pairs, same format
        as ``exact_correlation_table``.

    Example::

        from pecos_rslib_exp import depolarizing
        from pecos.qec.analysis import empirical_correlation_table

        table = empirical_correlation_table(
            tc, depolarizing().p1(0.001).p2(0.01), shots=100000, max_order=3,
        )
        for indices, prob in table:
            print(f"P({indices}) = {prob:.6f}")
    """
    from pecos_rslib_exp import (
        meas_sampling,
        sim_neo,
        stabilizer,
        statevec,
    )

    if backend == "meas_sampling":
        results = sim_neo(tick_circuit).quantum(meas_sampling()).noise(noise_builder).shots(shots).seed(seed).run()
    elif backend in ("stabilizer", "statevec"):
        backend_obj = stabilizer() if backend == "stabilizer" else statevec()
        results = sim_neo(tick_circuit).quantum(backend_obj).noise(noise_builder).shots(shots).seed(seed).run()
    else:
        supported = "'stabilizer', 'statevec', 'meas_sampling'"
        msg = f"Unknown backend {backend!r}. Supported: {supported}."
        raise ValueError(
            msg,
        )

    det_json = json.loads(tick_circuit.get_meta("detectors"))
    obs_json_str = tick_circuit.get_meta("observables")
    obs_json = json.loads(obs_json_str) if obs_json_str else []
    num_meas = int(tick_circuit.get_meta("num_measurements"))
    len(det_json)

    # Extract fired detectors and observables per shot
    fired_per_shot: list[list[int]] = []
    obs_per_shot: list[list[int]] = []
    for r in results:
        meas = list(r)
        fired: list[int] = []
        for i, det in enumerate(det_json):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                fired.append(i)
        fired_per_shot.append(fired)

        obs_fired: list[int] = []
        for i, obs in enumerate(obs_json):
            val = 0
            for rec in obs["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                obs_fired.append(i)
        obs_per_shot.append(obs_fired)

    # Compute detector k-body rates with string labels
    inv_shots = 1.0 / shots
    rates: dict[tuple[str, ...], float] = {}

    for fired in fired_per_shot:
        labels = [f"D{d}" for d in fired]
        for k in range(1, min(max_order, len(labels)) + 1):
            for combo in combinations(labels, k):
                rates[combo] = rates.get(combo, 0.0) + inv_shots

    # Observable marginals: P(Lj)
    for obs_fired in obs_per_shot:
        for o in obs_fired:
            key = (f"L{o}",)
            rates[key] = rates.get(key, 0.0) + inv_shots

    # Detector-observable pairwise: P(Di AND Lj)
    for fired, obs_fired in zip(fired_per_shot, obs_per_shot, strict=False):
        for d in fired:
            for o in obs_fired:
                key = (f"D{d}", f"L{o}")
                rates[key] = rates.get(key, 0.0) + inv_shots

    return sorted(rates.items())


def fit_dem_from_simulation(
    tick_circuit: object,
    noise_builder: object,
    shots: int,
    *,
    backend: str = "stabilizer",
    seed: int = 42,
    max_correlation_order: int = 2,
) -> str:
    """Build a DEM fitted to simulation data.

    Runs simulation to get empirical detection rates, uses the circuit's
    mechanism structure, and fits mechanism probabilities to match the
    empirical marginals and pairwise rates. Returns a Stim DEM string.

    This is the "hardware calibration" workflow: real noise statistics
    (or accurate simulation) combined with circuit-derived structure.

    Args:
        tick_circuit: A ``TickCircuit`` with detector metadata.
        noise_builder: A noise builder (e.g., ``depolarizing().p1(0.001)``).
        shots: Number of simulation shots.
        backend: Simulator backend — ``"stabilizer"``, ``"statevec"``,
            or ``"meas_sampling"``. The meas_sampling backend uses the fast
            whole-circuit DEM-based sampler instead of gate-by-gate simulation.
        seed: RNG seed.
        max_correlation_order: Max order for empirical rates (1 or 2).

    Returns:
        Stim-format DEM string with simulation-fitted probabilities.
    """
    if max_correlation_order < 1:
        msg = "max_correlation_order must be at least 1"
        raise ValueError(msg)

    from pecos_rslib.qec import (
        DagFaultAnalyzer,
        DemBuilder,
        fit_dem_to_marginals,
        mechanisms_to_dem_string,
    )
    from pecos_rslib_exp import (
        meas_sampling,
        sim_neo,
        stabilizer,
        statevec,
    )

    # Step 1: Get mechanism structure from circuit
    dag = tick_circuit.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence = analyzer.build_influence_map()
    # Use dummy noise to get mechanism structure (probabilities will be replaced)
    builder = DemBuilder(influence)
    builder = builder.with_noise(p1=0.01, p2=0.01, p_meas=0.01, p_prep=0.01)
    builder = builder.with_detectors_json(tick_circuit.get_meta("detectors"))
    builder = builder.with_observables_json(tick_circuit.get_meta("observables"))
    builder = builder.with_num_measurements(
        int(tick_circuit.get_meta("num_measurements")),
    )
    dem = builder.build()
    dem_str = dem.to_string()

    mechs: list[tuple[float, list[int], list[int]]] = []
    for raw_line in dem_str.strip().split("\n"):
        line = raw_line.strip()
        if line.startswith("error("):
            pe = line.index(")")
            prob = float(line[6:pe])
            toks = line[pe + 1 :].split()
            ds = sorted(int(t[1:]) for t in toks if t.startswith("D"))
            os = sorted(int(t[1:]) for t in toks if t.startswith("L"))
            mechs.append((prob, ds, os))

    # Step 2: Run simulation and extract empirical rates
    det_json = json.loads(tick_circuit.get_meta("detectors"))
    num_meas = int(tick_circuit.get_meta("num_measurements"))
    num_dets = len(det_json)

    if backend == "meas_sampling":
        results = sim_neo(tick_circuit).quantum(meas_sampling()).noise(noise_builder).shots(shots).seed(seed).run()
    elif backend in ("stabilizer", "statevec"):
        backend_obj = stabilizer() if backend == "stabilizer" else statevec()
        results = sim_neo(tick_circuit).quantum(backend_obj).noise(noise_builder).shots(shots).seed(seed).run()
    else:
        supported = "'stabilizer', 'statevec', 'meas_sampling'"
        msg = f"Unknown backend {backend!r}. Supported: {supported}."
        raise ValueError(msg)

    inv_shots = 1.0 / shots
    emp_marginals = [0.0] * num_dets
    for r in results:
        meas = list(r)
        for i, det in enumerate(det_json):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(meas):
                    val ^= meas[idx]
            if val:
                emp_marginals[i] += inv_shots

    # Step 3: Fit mechanism probabilities to empirical marginals
    fitted, _residuals = fit_dem_to_marginals(mechs, emp_marginals)

    return mechanisms_to_dem_string(fitted)


def build_adaptive_dem(
    tick_circuit: object,
    noise_params: dict[str, float],
    *,
    max_order: int = 2,
    prune: float = 1e-12,
) -> tuple[str, str]:
    """Build the best DEM using adaptive mechanism structure.

    Uses influence map mechanisms for stochastic noise sources and EEG
    backward extraction mechanisms for coherent (idle_rz) noise sources.
    Fits all mechanism probabilities to Heisenberg exact marginals + pairwise.

    Args:
        tick_circuit: A ``TickCircuit`` with detector metadata.
        noise_params: Dict with keys p1, p2, p_meas, p_prep, idle_rz.
        max_order: Correlation order for Heisenberg targets (default 2).
        prune: Pruning threshold for Heisenberg walks.

    Returns:
        Tuple of (json_str, dem_str) — noise characterization JSON and
        Stim DEM string.
    """
    idle_rz = noise_params.get("idle_rz", 0.0)

    if idle_rz == 0.0 or idle_rz < 1e-10:
        # Pure stochastic: from_circuit is best
        from pecos_rslib.qec import DemSampler

        DemSampler.from_circuit(
            tick_circuit,
            p1=noise_params.get("p1", 0.0),
            p2=noise_params.get("p2", 0.0),
            p_meas=noise_params.get("p_meas", 0.0),
            p_prep=noise_params.get("p_prep", 0.0),
        )
        # For consistency, also compute correlation table
        from pecos_rslib_exp import exact_correlation_table

        table = exact_correlation_table(tick_circuit, **noise_params, max_order=max_order)
        # Return a minimal JSON with correlations
        json_out = json.dumps(
            {
                "correlations": [{"nodes": list(labels), "probability": prob} for labels, prob in table],
            },
            indent=2,
        )
        # Get DEM string from the sampler's internal DEM
        dem_str = ""  # from_circuit doesn't expose DEM string directly
        # Rebuild via DemBuilder
        from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

        dag = tick_circuit.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dag)
        influence = analyzer.build_influence_map()
        builder = DemBuilder(influence)
        builder = builder.with_noise(
            p1=noise_params.get("p1", 0.0),
            p2=noise_params.get("p2", 0.0),
            p_meas=noise_params.get("p_meas", 0.0),
            p_prep=noise_params.get("p_prep", 0.0),
        )
        builder = builder.with_detectors_json(tick_circuit.get_meta("detectors"))
        builder = builder.with_observables_json(tick_circuit.get_meta("observables"))
        builder = builder.with_num_measurements(
            int(tick_circuit.get_meta("num_measurements")),
        )
        dem = builder.build()
        dem_str = dem.to_string()

        return json_out, dem_str

    # Has coherent noise: use noise_characterization (EEG structure + L-BFGS fit)
    from pecos_rslib_exp import noise_characterization

    return noise_characterization(
        tick_circuit,
        **noise_params,
        max_order=max_order,
        prune=prune,
    )
