"""Analyzer for qubit usage patterns to optimize allocation strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr import Block as SLRBlock


class QubitRole(Enum):
    """Role classification for quantum registers."""

    DATA = "data"  # Long-lived data qubits
    ANCILLA = "ancilla"  # Short-lived ancilla qubits
    UNKNOWN = "unknown"  # Not yet classified


@dataclass
class QubitUsageStats:
    """Statistics about how a quantum register is used."""

    name: str
    size: int

    # Usage patterns
    measurement_count: int = 0
    reset_count: int = 0
    gate_count: int = 0

    # Lifetime tracking
    first_use_position: int | None = None
    last_use_position: int | None = None
    measurement_positions: list[int] = field(default_factory=list)
    reset_positions: list[int] = field(default_factory=list)

    # Access patterns
    individual_accesses: set[int] = field(default_factory=set)
    full_array_accesses: int = 0

    # Structural hints
    is_struct_member: bool = False
    struct_name: str | None = None

    @property
    def lifetime(self) -> int:
        """Calculate the lifetime of this register."""
        if self.first_use_position is None or self.last_use_position is None:
            return 0
        return self.last_use_position - self.first_use_position

    @property
    def measure_reset_ratio(self) -> float:
        """Ratio of measurements+resets to total operations."""
        total_ops = self.measurement_count + self.reset_count + self.gate_count
        if total_ops == 0:
            return 0.0
        return (self.measurement_count + self.reset_count) / total_ops

    @property
    def individual_access_ratio(self) -> float:
        """Ratio of individual element accesses to size."""
        if self.size == 0:
            return 0.0
        return len(self.individual_accesses) / self.size

    def classify_role(self) -> QubitRole:
        """Classify the role of this register based on usage patterns."""
        # Explicit ancilla naming patterns
        if any(pattern in self.name.lower() for pattern in ["ancilla", "anc", "syndrome", "flag"]):
            return QubitRole.ANCILLA

        # Explicit data naming patterns
        if any(pattern in self.name.lower() for pattern in ["data", "logical", "code"]):
            return QubitRole.DATA

        # Pattern-based classification
        # High measure/reset ratio suggests ancilla
        if self.measure_reset_ratio > 0.7:
            return QubitRole.ANCILLA

        # Short lifetime with measurements suggests ancilla
        if self.lifetime < 10 and self.measurement_count > 0:
            return QubitRole.ANCILLA

        # Part of a struct (like QEC code) suggests data
        if self.is_struct_member:
            return QubitRole.DATA

        # Default to data for long-lived qubits
        if self.lifetime > 20:
            return QubitRole.DATA

        return QubitRole.UNKNOWN


class QubitUsageAnalyzer:
    """Analyzes qubit usage patterns to inform allocation strategies."""

    def __init__(self):
        self.register_stats: dict[str, QubitUsageStats] = {}
        self.position_counter = 0

    def analyze_block(
        self,
        block: SLRBlock,
        struct_info: dict[str, dict] | None = None,
    ) -> dict[str, QubitUsageStats]:
        """Analyze a block and return usage statistics for each quantum register."""
        # Reset state
        self.register_stats.clear()
        self.position_counter = 0

        # First, collect all quantum registers
        if hasattr(block, "vars"):
            for var in block.vars:
                if type(var).__name__ == "QReg" and hasattr(var, "sym"):
                    stats = QubitUsageStats(
                        name=var.sym,
                        size=getattr(var, "size", 1),
                    )

                    # Check if part of a struct
                    if struct_info:
                        for struct_name, info in struct_info.items():
                            if var.sym in info.get("var_names", {}).values():
                                stats.is_struct_member = True
                                stats.struct_name = struct_name
                                break

                    self.register_stats[var.sym] = stats

        # Analyze operations
        if hasattr(block, "ops"):
            for op in block.ops:
                self._analyze_operation(op)
                self.position_counter += 1

        return self.register_stats

    def _analyze_operation(self, op) -> None:
        """Analyze a single operation for qubit usage patterns."""
        op_type = type(op).__name__

        if op_type == "Measure":
            self._analyze_measurement(op)
        elif op_type in ["Prep", "Reset"]:
            self._analyze_reset(op)
        elif hasattr(op, "qargs"):
            self._analyze_gate(op)
        elif hasattr(op, "ops"):
            # Nested block
            for nested_op in op.ops:
                self._analyze_operation(nested_op)

    def _analyze_measurement(self, meas) -> None:
        """Analyze measurement operations."""
        if hasattr(meas, "qargs") and meas.qargs:
            for qarg in meas.qargs:
                reg_name = self._get_register_name(qarg)
                if reg_name and reg_name in self.register_stats:
                    stats = self.register_stats[reg_name]
                    stats.measurement_count += 1
                    stats.measurement_positions.append(self.position_counter)
                    self._update_lifetime(stats)

                    # Track access pattern
                    if hasattr(qarg, "index"):
                        stats.individual_accesses.add(qarg.index)
                    elif hasattr(qarg, "size"):
                        stats.full_array_accesses += 1

    def _analyze_reset(self, reset_op) -> None:
        """Analyze reset/prep operations."""
        if hasattr(reset_op, "qargs") and reset_op.qargs:
            for qarg in reset_op.qargs:
                reg_name = self._get_register_name(qarg)
                if reg_name and reg_name in self.register_stats:
                    stats = self.register_stats[reg_name]
                    stats.reset_count += 1
                    stats.reset_positions.append(self.position_counter)
                    self._update_lifetime(stats)

    def _analyze_gate(self, gate) -> None:
        """Analyze gate operations."""
        if hasattr(gate, "qargs") and gate.qargs:
            for qarg in gate.qargs:
                # Handle tuple arguments (e.g., CX gates)
                if isinstance(qarg, tuple):
                    for sub_qarg in qarg:
                        self._record_gate_usage(sub_qarg)
                else:
                    self._record_gate_usage(qarg)

    def _record_gate_usage(self, qarg) -> None:
        """Record usage from a gate operation."""
        reg_name = self._get_register_name(qarg)
        if reg_name and reg_name in self.register_stats:
            stats = self.register_stats[reg_name]
            stats.gate_count += 1
            self._update_lifetime(stats)

            # Track access pattern
            if hasattr(qarg, "index"):
                stats.individual_accesses.add(qarg.index)

    def _get_register_name(self, qarg) -> str | None:
        """Extract register name from a qubit argument."""
        if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
            return qarg.reg.sym
        if hasattr(qarg, "sym"):
            return qarg.sym
        return None

    def _update_lifetime(self, stats: QubitUsageStats) -> None:
        """Update lifetime tracking for a register."""
        if stats.first_use_position is None:
            stats.first_use_position = self.position_counter
        stats.last_use_position = self.position_counter

    def get_allocation_recommendations(self) -> dict[str, dict]:
        """Get allocation recommendations based on usage analysis."""
        recommendations = {}

        for reg_name, stats in self.register_stats.items():
            role = stats.classify_role()

            if role == QubitRole.ANCILLA:
                # Ancillas benefit from dynamic allocation
                recommendations[reg_name] = {
                    "allocation": "dynamic",
                    "reason": f"High measure/reset ratio ({stats.measure_reset_ratio:.2f})",
                    "keep_packed": False,
                    "pre_allocate": False,
                }
            elif role == QubitRole.DATA:
                # Data qubits should stay bundled
                recommendations[reg_name] = {
                    "allocation": "static",
                    "reason": (
                        "Long-lived data qubits"
                        if stats.is_struct_member
                        else f"Low measure/reset ratio ({stats.measure_reset_ratio:.2f})"
                    ),
                    "keep_packed": True,
                    "pre_allocate": True,
                }
            else:
                # Default conservative approach
                recommendations[reg_name] = {
                    "allocation": "static",
                    "reason": "Unknown usage pattern",
                    "keep_packed": True,
                    "pre_allocate": True,
                }

        return recommendations
