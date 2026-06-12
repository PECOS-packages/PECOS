"""Canonical `(round, qubit) -> site_idx` mapping shared by the abstract
circuit (`circuit_builder.py`) and the Guppy runtime renderer
(`pecos.guppy.surface`).

Both tracks must agree byte-for-byte on the ordering of twirl-site
metadata: the abstract circuit's `tracked_pauli` annotations populate
rows of the `PauliFrameLookup` matrix `M`, and the Guppy program's
runtime Pauli applications plus per-shot `result()` recordings populate
the matching mask columns. If the two paths disagree about which
`(round, qubit)` maps to which row / column, the per-shot XOR
application silently runs the mask against the wrong tracked-Pauli
annotations and the decoder sees an incoherent syndrome.

The mapping is currently a simple `site_idx == round_idx` for the
`between_rounds` schedule (one site between each pair of consecutive
syndrome rounds, for a total of `num_rounds - 1` sites). The helpers
below abstract that so a future schedule only needs to update this module.

Encoding contract for the runtime mask (``"bool_array_v1"``):

- One result tag PER twirl site, named via :func:`pauli_mask_round_tag`
  (`f"{PAULI_MASK_TAG_PREFIX}:round:{round_idx}"`). Each tag is emitted
  exactly once per shot, so ``ShotVec.to_dict()`` cannot collapse it.
  (An earlier "shared-tag scalar bool" variant was attempted but
  unimplementable -- ``ShotVec.to_dict()`` collapses repeated same-tag
  ``result()`` calls in one shot to the last value, dropping every
  earlier bit on the floor.)
- Each per-round tag carries one bool array of length
  ``2 * num_data`` laid out as
  ``(qubit_0_lo, qubit_0_hi, qubit_1_lo, qubit_1_hi, ...)``, where
  ``lo = (m == 1) | (m == 3)`` and ``hi = (m == 2) | (m == 3)`` for a
  Pauli value ``m in {0=I, 1=X, 2=Y, 3=Z}`` drawn at runtime by the
  generated Guppy program's inline functional PCG helper and applied as
  the matching physical Pauli gate at that twirl site. This is the binary encoding
  of the enumeration code, not the Pauli's symplectic X/Z components.
- The decoder
  (:func:`pecos.qec.surface.decode._extract_pauli_masks_from_results`)
  reads each ``pauli_mask:round:{r}`` tag, packs each ``(lo, hi)`` pair
  back into the integer Pauli code via ``m = lo + 2 * hi``, and lays
  the result at column :func:`mask_col_for` in row-major
  ``(site, qubit)`` order so the output columns match the abstract
  ``PauliFrameLookup`` byte-for-byte.
- Side-band result tags emitted by twirl support (`pauli_mask:*`,
  `frame_mode:*`, `raw:*`) are not detector-bearing measurement tags.
  They must never contain ``":meas:"`` and must never be exactly
  ``"final"``; handoff result-provenance code treats only the surface
  measurement tag grammar and exact ``"final"`` as measurement records.
"""

from __future__ import annotations

# Base prefix for per-round mask result tags. The Guppy renderer emits one
# `result(f"{PAULI_MASK_TAG_PREFIX}:round:{r}", array(lo_q0, hi_q0, ...))`
# call per twirl site so each tag fires exactly once per shot (avoids the
# same-tag-multi-call overwrite trap in `ShotVec.to_dict()`).
PAULI_MASK_TAG_PREFIX = "pauli_mask"

# Backwards-compatible alias for callers that constructed the bare tag.
PAULI_MASK_TAG = PAULI_MASK_TAG_PREFIX


def pauli_mask_round_tag(round_idx: int) -> str:
    """Canonical `result()` tag for the twirl mask emitted between syndrome
    rounds ``round_idx`` and ``round_idx + 1``.
    """
    return f"{PAULI_MASK_TAG_PREFIX}:round:{round_idx}"


def num_twirl_sites(num_rounds: int) -> int:
    """Number of twirl sites for the `between_rounds` schedule.

    One site between each pair of consecutive syndrome rounds:
    `max(0, num_rounds - 1)`.
    """
    return max(0, num_rounds - 1)


def site_idx_for_round(round_idx: int) -> int:
    """Map the round-loop index of the round PRECEDING a twirl site to its
    canonical `site_idx`.

    For the `between_rounds` schedule, the twirl site that sits between
    syndrome round `r` and round `r + 1` is canonical `site_idx == r`.
    Identity by construction; abstracted so a future schedule can override.
    """
    return round_idx


def num_mask_bits_per_round(num_data: int) -> int:
    """Number of bool bits the Guppy program emits per twirl site.

    The bool-bits encoding uses two bits per data qubit (`lo, hi`) so the
    per-site contribution is `2 * num_data` bools.
    """
    return 2 * num_data


def num_mask_bits_per_shot(num_rounds: int, num_data: int) -> int:
    """Total bool-bit count one shot's `pauli_mask` result accumulates."""
    return num_twirl_sites(num_rounds) * num_mask_bits_per_round(num_data)


def num_pauli_sites(num_rounds: int, num_data: int) -> int:
    """Number of mask columns (one per (site, data qubit) cell).

    Matches the column count of the abstract `PauliFrameLookup` matrix `M`
    when twirling is enabled.
    """
    return num_twirl_sites(num_rounds) * num_data


def mask_col_for(site_idx: int, qubit_idx: int, num_data: int) -> int:
    """Canonical flat (site, qubit) -> mask column index.

    Row-major over (site, qubit). Decoder-side `sample_pauli_masks_from_guppy`
    uses this to assemble the `(num_shots, num_pauli_sites)` u8 array from a
    flat bool stream.
    """
    return site_idx * num_data + qubit_idx
