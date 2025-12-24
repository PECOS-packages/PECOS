"""Data flow analysis for SLR to Guppy code generation.

This module provides data flow analysis to track how quantum and classical values
flow through a program, particularly tracking measurement results and their usage.

The key insight is that we need to distinguish between:
1. Operations BEFORE measurement (don't require unpacking)
2. Operations AFTER measurement that use the SAME qubit (require unpacking for replacement)
3. Operations AFTER measurement that use DIFFERENT qubits (don't require unpacking)

Current heuristics over-approximate by treating ANY operation after ANY measurement
as requiring unpacking, leading to unnecessary array unpacking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.slr import Block as SLRBlock


@dataclass
class ValueUse:
    """Represents a use of a value (qubit or classical bit)."""

    array_name: str
    index: int
    position: int  # Position in operation sequence
    operation_type: str  # e.g., "gate", "measurement", "condition"
    is_consuming: bool = (
        False  # True if this use consumes the value (e.g., measurement)
    )


@dataclass
class DataFlowInfo:
    """Information about data flow for a single array element."""

    array_name: str
    index: int
    is_classical: bool

    # Track all uses of this element
    uses: list[ValueUse] = field(default_factory=list)

    # Track consumption points (measurements)
    consumed_at: list[int] = field(default_factory=list)

    # Track replacements (e.g., Prep after measurement)
    replaced_at: list[int] = field(default_factory=list)

    def add_use(
        self,
        position: int,
        operation_type: str,
        *,
        is_consuming: bool = False,
    ) -> None:
        """Add a use of this value."""
        use = ValueUse(
            array_name=self.array_name,
            index=self.index,
            position=position,
            operation_type=operation_type,
            is_consuming=is_consuming,
        )
        self.uses.append(use)

        if is_consuming:
            self.consumed_at.append(position)

    def add_replacement(self, position: int) -> None:
        """Mark that this value is replaced at a position (e.g., Prep)."""
        self.replaced_at.append(position)

    def has_use_after_consumption(self) -> bool:
        """Check if this element is used after being consumed.

        This is the key analysis for determining if unpacking is needed.
        If a qubit is measured and then used again (not just replaced),
        we need unpacking to handle the replacement properly.
        """
        if not self.consumed_at:
            return False

        # Find the first consumption point
        first_consumption = min(self.consumed_at)

        # Check if there are any non-replacement uses after consumption
        for use in self.uses:
            if (
                use.position > first_consumption
                and use.position not in self.replaced_at
            ):
                # This is a real use after consumption, not just replacement
                # However, we need to check if it's AFTER replacement
                # Find if there's a replacement between consumption and this use
                replacements_between = [
                    r for r in self.replaced_at if first_consumption < r < use.position
                ]

                if not replacements_between:
                    # Use after consumption with no replacement in between
                    # This requires unpacking
                    return True

        return False

    def requires_unpacking_for_flow(self) -> bool:
        """Determine if this element requires unpacking based on data flow.

        This is more precise than the heuristic approach:
        - Classical values can be used multiple times without issue
        - Quantum values can be used multiple times if not measured
        - Quantum values that are measured and then used require unpacking
        """
        if self.is_classical:
            # Classical values can be read multiple times
            return False

        # Quantum values: check if used after consumption
        return self.has_use_after_consumption()


@dataclass
class DataFlowAnalysis:
    """Complete data flow analysis for a block."""

    # Map from (array_name, index) to DataFlowInfo
    element_flows: dict[tuple[str, int], DataFlowInfo] = field(default_factory=dict)

    # Track conditionally accessed elements
    conditional_accesses: set[tuple[str, int]] = field(default_factory=set)

    def get_or_create_flow(
        self,
        array_name: str,
        index: int,
        is_classical: bool,
    ) -> DataFlowInfo:
        """Get or create data flow info for an array element."""
        key = (array_name, index)
        if key not in self.element_flows:
            self.element_flows[key] = DataFlowInfo(
                array_name=array_name,
                index=index,
                is_classical=is_classical,
            )
        return self.element_flows[key]

    def add_gate_use(self, array_name: str, index: int, position: int) -> None:
        """Record a gate operation on an array element."""
        flow = self.get_or_create_flow(array_name, index, is_classical=False)
        flow.add_use(position, "gate", is_consuming=False)

    def add_measurement(
        self,
        quantum_array: str,
        quantum_index: int,
        position: int,
        classical_array: str | None = None,
        classical_index: int | None = None,
    ) -> None:
        """Record a measurement operation."""
        # Quantum side: consumption
        q_flow = self.get_or_create_flow(
            quantum_array,
            quantum_index,
            is_classical=False,
        )
        q_flow.add_use(position, "measurement", is_consuming=True)

        # Classical side: creation (if specified)
        if classical_array is not None and classical_index is not None:
            c_flow = self.get_or_create_flow(
                classical_array,
                classical_index,
                is_classical=True,
            )
            c_flow.add_use(position, "measurement_result", is_consuming=False)

    def add_preparation(self, array_name: str, index: int, position: int) -> None:
        """Record a preparation/reset operation (replaces a qubit)."""
        flow = self.get_or_create_flow(array_name, index, is_classical=False)
        flow.add_use(position, "preparation", is_consuming=False)
        flow.add_replacement(position)

    def add_conditional_use(
        self,
        array_name: str,
        index: int,
        position: int,
        is_classical: bool,
    ) -> None:
        """Record a conditional use of an array element."""
        flow = self.get_or_create_flow(array_name, index, is_classical)
        flow.add_use(position, "condition", is_consuming=False)
        self.conditional_accesses.add((array_name, index))

    def elements_requiring_unpacking(self) -> set[tuple[str, int]]:
        """Get the set of array elements that require unpacking based on data flow."""
        requiring_unpacking = set()

        for key, flow in self.element_flows.items():
            if flow.requires_unpacking_for_flow():
                requiring_unpacking.add(key)

        return requiring_unpacking

    def array_requires_unpacking(self, array_name: str) -> bool:
        """Check if an entire array requires unpacking based on data flow."""
        for key, flow in self.element_flows.items():
            if key[0] == array_name and flow.requires_unpacking_for_flow():
                return True
        return False


class DataFlowAnalyzer:
    """Analyzes data flow in SLR blocks."""

    def __init__(self):
        self.position_counter = 0
        self.in_conditional = False

    def analyze(
        self,
        block: SLRBlock,
        variable_context: dict[str, Any],
    ) -> DataFlowAnalysis:
        """Analyze data flow in a block.

        Args:
            block: The SLR block to analyze
            variable_context: Context of variables (QReg, CReg, etc.)

        Returns:
            DataFlowAnalysis containing all data flow information
        """
        analysis = DataFlowAnalysis()
        self.position_counter = 0
        self.in_conditional = False

        # Analyze all operations
        if hasattr(block, "ops"):
            for op in block.ops:
                self._analyze_operation(op, analysis, variable_context)
                self.position_counter += 1

        return analysis

    def _analyze_operation(
        self,
        op: Any,
        analysis: DataFlowAnalysis,
        variable_context: dict[str, Any],
    ) -> None:
        """Analyze a single operation."""
        op_type = type(op).__name__

        if op_type == "Measure":
            self._analyze_measurement(op, analysis)
        elif op_type == "If":
            self._analyze_if_block(op, analysis, variable_context)
        elif hasattr(op, "qargs"):
            # Check if this is a preparation operation
            if self._is_preparation(op):
                self._analyze_preparation(op, analysis)
            else:
                self._analyze_quantum_operation(op, analysis)
        elif hasattr(op, "ops"):
            # Nested block - recurse
            for nested_op in op.ops:
                self._analyze_operation(nested_op, analysis, variable_context)

    def _is_preparation(self, op: Any) -> bool:
        """Check if an operation is a preparation/reset."""
        op_name = type(op).__name__
        return op_name in ["Prep", "Init", "Reset"]

    def _analyze_measurement(self, meas: Any, analysis: DataFlowAnalysis) -> None:
        """Analyze a measurement operation."""
        # Get classical targets
        classical_targets = []
        if hasattr(meas, "cout") and meas.cout:
            classical_targets.extend(
                (cout.reg.sym, cout.index)
                for cout in meas.cout
                if hasattr(cout, "reg")
                and hasattr(cout.reg, "sym")
                and hasattr(cout, "index")
            )

        # Analyze quantum sources
        if hasattr(meas, "qargs") and meas.qargs:
            for i, qarg in enumerate(meas.qargs):
                # Individual element measurement
                if (
                    hasattr(qarg, "reg")
                    and hasattr(qarg.reg, "sym")
                    and hasattr(qarg, "index")
                ):
                    array_name = qarg.reg.sym
                    index = qarg.index

                    # Get corresponding classical target if exists
                    classical_array = None
                    classical_index = None
                    if i < len(classical_targets):
                        classical_array, classical_index = classical_targets[i]

                    analysis.add_measurement(
                        quantum_array=array_name,
                        quantum_index=index,
                        position=self.position_counter,
                        classical_array=classical_array,
                        classical_index=classical_index,
                    )

    def _analyze_preparation(self, op: Any, analysis: DataFlowAnalysis) -> None:
        """Analyze a preparation/reset operation."""
        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                if (
                    hasattr(qarg, "reg")
                    and hasattr(qarg.reg, "sym")
                    and hasattr(qarg, "index")
                ):
                    array_name = qarg.reg.sym
                    index = qarg.index
                    analysis.add_preparation(array_name, index, self.position_counter)

    def _analyze_quantum_operation(self, op: Any, analysis: DataFlowAnalysis) -> None:
        """Analyze a quantum gate operation."""
        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                if (
                    hasattr(qarg, "reg")
                    and hasattr(qarg.reg, "sym")
                    and hasattr(qarg, "index")
                ):
                    array_name = qarg.reg.sym
                    index = qarg.index

                    if self.in_conditional:
                        analysis.add_conditional_use(
                            array_name,
                            index,
                            self.position_counter,
                            is_classical=False,
                        )
                    else:
                        analysis.add_gate_use(array_name, index, self.position_counter)

    def _analyze_if_block(
        self,
        if_block: Any,
        analysis: DataFlowAnalysis,
        variable_context: dict[str, Any],
    ) -> None:
        """Analyze an if block."""
        prev_conditional = self.in_conditional
        self.in_conditional = True

        # Analyze condition
        if hasattr(if_block, "cond"):
            self._analyze_condition(if_block.cond, analysis)

        # Analyze then block
        if hasattr(if_block, "ops"):
            for op in if_block.ops:
                self._analyze_operation(op, analysis, variable_context)

        # Analyze else block
        if (
            hasattr(if_block, "else_block")
            and if_block.else_block
            and hasattr(if_block.else_block, "ops")
        ):
            for op in if_block.else_block.ops:
                self._analyze_operation(op, analysis, variable_context)

        self.in_conditional = prev_conditional

    def _analyze_condition(self, cond: Any, analysis: DataFlowAnalysis) -> None:
        """Analyze a condition expression."""
        cond_type = type(cond).__name__

        if (
            cond_type == "Bit"
            and hasattr(cond, "reg")
            and hasattr(cond.reg, "sym")
            and hasattr(cond, "index")
        ):
            array_name = cond.reg.sym
            index = cond.index
            analysis.add_conditional_use(
                array_name,
                index,
                self.position_counter,
                is_classical=True,
            )

        # Handle compound conditions
        if hasattr(cond, "left"):
            self._analyze_condition(cond.left, analysis)
        if hasattr(cond, "right"):
            self._analyze_condition(cond.right, analysis)
