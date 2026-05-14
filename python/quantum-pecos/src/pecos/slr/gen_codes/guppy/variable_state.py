"""Unified variable-state tracking for the Guppy IR generator.

The Guppy generator translates SLR programs (high-level quantum DSL) to
Guppy source. Guppy uses linear types: every qubit must be used exactly
once, and arrays-of-qubits get "moved" into and out of operations rather
than mutated in place.

Translating SLR to Guppy means tracking, for each SLR variable, *what
Guppy variable currently holds it*. The form changes over the lifetime
of the SLR variable -- it might be a whole array, get unpacked into
element variables for individual access, get refreshed by a function
return, get partially consumed, etc.

Historically the IRGuppyGenerator did this with ~6+ separate dicts
(`unpacked_vars`, `refreshed_arrays`, `array_remapping`, `index_mapping`,
`variable_remapping`, `function_var_remapping`, `replaced_qubits`,
`fresh_variables_to_track`, ...). Different code generation sites
consult different subsets of these dicts; sites that miss a state
transition emit Guppy that violates linearity ("AlreadyUsedError",
"WrongNumberOfUnpacksError", etc.).

This module replaces that with one model: each SLR variable has a
*current binding* describing its Guppy form right now. Operations on the
variable consult the binding; transitions update it. Code-generation
sites that need the variable in a particular form call helpers like
`ensure_whole()` which emit reconstruction statements transparently.

The migration is incremental. While the legacy dicts still exist, this
module shadows them: writes go to both, reads prefer this module. Once
all read sites are migrated, the legacy dicts can be removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WholeArray:
    """SLR variable is currently bound to a single Guppy array variable.

    `guppy_name` is the live identifier; subsequent ops can reference
    `guppy_name` directly or index into it via `guppy_name[i]`.
    """

    guppy_name: str


@dataclass(frozen=True)
class UnpackedArray:
    """SLR variable was unpacked into per-element Guppy variables.

    `element_names[i]` is the Guppy variable for original SLR index
    `i` -- unless `index_mapping` is set, in which case mapping
    `original_index -> position_in_element_names` is used (this happens
    when a function call returned a partially-consumed array).
    """

    element_names: tuple[str, ...]
    index_mapping: tuple[tuple[int, int], ...] = ()  # (orig_idx, position)

    def position_for(self, original_index: int) -> int | None:
        """Return the position in `element_names` for an SLR index.

        With no `index_mapping`, returns `original_index` directly when in
        bounds. With a mapping, looks up the position; returns None for
        SLR indices that aren't present in the partial array.
        """
        if not self.index_mapping:
            return original_index if original_index < len(self.element_names) else None
        for orig, pos in self.index_mapping:
            if orig == original_index:
                return pos
        return None


@dataclass(frozen=True)
class Consumed:
    """SLR variable is fully consumed; subsequent references are bugs.

    `reason` is a short human-readable note for diagnostics ("measured",
    "passed to function as @owned", etc.).
    """

    reason: str = ""


Binding = WholeArray | UnpackedArray | Consumed


@dataclass
class VariableState:
    """Current Guppy bindings for SLR variables in one generation context.

    A "context" is typically one Guppy function being generated -- the
    main function or one of the extracted sub-block functions. Bindings
    are local to a context; the same SLR variable name in different
    contexts can have different bindings.
    """

    bindings: dict[str, Binding] = field(default_factory=dict)

    def bind_whole(self, slr_name: str, guppy_name: str) -> None:
        """Record that `slr_name` is currently held by Guppy var `guppy_name`."""
        self.bindings[slr_name] = WholeArray(guppy_name)

    def bind_unpacked(
        self,
        slr_name: str,
        element_names: list[str],
        index_mapping: dict[int, int] | None = None,
    ) -> None:
        """Record that `slr_name` was unpacked into per-element Guppy vars."""
        mapping_tuple = tuple(sorted(index_mapping.items())) if index_mapping else ()
        self.bindings[slr_name] = UnpackedArray(tuple(element_names), mapping_tuple)

    def bind_consumed(self, slr_name: str, reason: str = "") -> None:
        """Record that `slr_name` is no longer accessible."""
        self.bindings[slr_name] = Consumed(reason)

    def get(self, slr_name: str) -> Binding | None:
        """Return current binding, or None if `slr_name` is unknown here."""
        return self.bindings.get(slr_name)

    def is_unpacked(self, slr_name: str) -> bool:
        """True iff `slr_name` is currently in unpacked form."""
        return isinstance(self.bindings.get(slr_name), UnpackedArray)

    def is_consumed(self, slr_name: str) -> bool:
        """True iff `slr_name` has been consumed."""
        return isinstance(self.bindings.get(slr_name), Consumed)

    def ensure_whole(self, slr_name: str) -> tuple[list[str], str | None]:
        """Ensure `slr_name` is bound as a whole array; emit prep code if not.

        Returns (preparation_lines, guppy_name). The caller emits the
        preparation_lines (Guppy source as `array(elem_0, elem_1, ...)`
        repacking) before whatever it does with `guppy_name`. Returns
        ([], guppy_name) when already whole. Returns ([], None) when
        `slr_name` is consumed or unknown -- caller should treat as a
        programming error.

        After repack, the binding is updated to WholeArray so subsequent
        callers don't repack again.
        """
        binding = self.bindings.get(slr_name)
        if isinstance(binding, WholeArray):
            return [], binding.guppy_name
        if isinstance(binding, UnpackedArray):
            elements = ", ".join(binding.element_names)
            line = f"{slr_name} = array({elements})"
            self.bindings[slr_name] = WholeArray(slr_name)
            return [line], slr_name
        return [], None
