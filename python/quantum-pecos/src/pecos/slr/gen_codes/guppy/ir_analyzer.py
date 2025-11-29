"""Analyzer for determining array unpacking and other transformations needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.slr import Block as SLRBlock


@dataclass
class ArrayAccessInfo:
    """Information about how an array is accessed."""

    array_name: str
    size: int
    is_classical: bool = False  # Track if this is a CReg

    # Track individual element accesses
    element_accesses: set[int] = field(default_factory=set)
    element_access_positions: dict[int, list[int]] = field(default_factory=dict)

    # Track full array accesses
    full_array_accesses: list[int] = field(default_factory=list)
    
    # Track if passed to blocks
    passed_to_blocks: bool = False

    # Track operations between accesses
    has_operations_between: bool = False
    has_conditionals_between: bool = False

    # Consumption info
    elements_consumed: set[int] = field(default_factory=set)
    fully_consumed: bool = False
    consumed_at_position: int | None = None

    @property
    def has_individual_access(self) -> bool:
        """Check if array has individual element access."""
        return len(self.element_accesses) > 0

    @property
    def all_elements_accessed(self) -> bool:
        """Check if all elements are accessed."""
        return len(self.element_accesses) == self.size

    @property
    def needs_unpacking(self) -> bool:
        """Determine if this array needs unpacking."""
        # Classical arrays (CReg) can be unpacked if they have individual element access
        # and are not used in result() as a full array
        if self.is_classical:
            # Check if used in result() as full array
            # For now, we'll allow unpacking for classical arrays with element access
            if not self.element_accesses:
                return False
            # If we have multiple element accesses, unpack for cleaner code
            if len(self.element_accesses) > 1:
                return True

        # If there's a full array measurement, don't unpack
        # Even if there are individual element accesses for gates
        if self.full_array_accesses:
            return False

        # Need unpacking if we have individual element access
        # and can't use measure_array
        if not self.has_individual_access:
            return False

        # If we have operations between measurements, need unpacking
        # (e.g., measure then use - requires qubit replacement)
        if self.has_operations_between:
            return True

        # If we have conditional access, need unpacking
        if self.has_conditionals_between:
            return True

        # Don't unpack if only one element is accessed - use direct indexing instead
        # This avoids the PlaceNotUsedError when we unpack all but only use one
        # But only if we don't have operations/conditionals between (checked above)
        if len(self.element_accesses) == 1:
            return False
        
        # For quantum arrays, if we have individual element measurements (consumed),
        # we need unpacking to avoid MoveOutOfSubscriptError
        if not self.is_classical and self.elements_consumed:
            # Individual measurements on quantum arrays require unpacking
            return True

        # If not all elements are accessed together, need unpacking
        return bool(not self.all_elements_accessed)


@dataclass
class UnpackingPlan:
    """Plan for unpacking arrays in a scope."""

    arrays_to_unpack: dict[str, ArrayAccessInfo] = field(default_factory=dict)
    unpack_at_start: set[str] = field(default_factory=set)
    renamed_variables: dict[str, str] = field(default_factory=dict)
    # Store all analyzed arrays, including those that don't need unpacking
    all_analyzed_arrays: dict[str, ArrayAccessInfo] = field(default_factory=dict)


class IRAnalyzer:
    """Analyzes SLR blocks to determine IR transformations needed."""

    def __init__(self):
        self.array_info: dict[str, ArrayAccessInfo] = {}
        self.position_counter = 0
        self.in_conditional = False
        self.reserved_names = {"result", "array", "quantum", "guppy", "owned"}
        self.has_nested_blocks = False

    def analyze_block(
        self,
        block: SLRBlock,
        variable_context: dict[str, Any],
    ) -> UnpackingPlan:
        """Analyze a block and return unpacking plan."""
        plan = UnpackingPlan()

        # Reset state
        self.array_info.clear()
        self.position_counter = 0
        
        # First, collect array information from variables
        self._collect_array_info(block, variable_context)

        # Analyze operations to determine access patterns
        if hasattr(block, "ops"):
            for op in block.ops:
                self._analyze_operation(op)
                self.position_counter += 1
        
        # Determine which arrays need unpacking
        # Special case: if we have nested blocks but @owned parameters, we must unpack
        # because @owned parameters require unpacking to access elements
        must_unpack_for_owned = (
            hasattr(self, 'has_nested_blocks_with_owned') and 
            self.has_nested_blocks_with_owned
        )
        
        # Store all analyzed arrays in the plan
        plan.all_analyzed_arrays = self.array_info.copy()
        
        if not self.has_nested_blocks or must_unpack_for_owned:
            for array_name, info in self.array_info.items():
                should_unpack = info.needs_unpacking
                
                # Force unpacking for @owned parameters even with nested blocks
                if (must_unpack_for_owned and 
                    hasattr(self, 'expected_owned_params') and 
                    array_name in self.expected_owned_params):
                    should_unpack = True
                    
                if should_unpack:
                    plan.arrays_to_unpack[array_name] = info
                    plan.unpack_at_start.add(array_name)

        # Check for variable name conflicts
        self._check_name_conflicts(block, plan)

        return plan

    def _collect_array_info(
        self,
        block: SLRBlock,
        variable_context: dict[str, Any],
    ) -> None:
        """Collect information about arrays in the block."""
        # From block variables
        if hasattr(block, "vars"):
            for var in block.vars:
                var_type = type(var).__name__
                if (
                    var_type in ["QReg", "CReg"]
                    and hasattr(var, "sym")
                    and hasattr(var, "size")
                ):
                    self.array_info[var.sym] = ArrayAccessInfo(
                        array_name=var.sym,
                        size=var.size,
                        is_classical=(var_type == "CReg"),
                    )

        # From variable context
        if variable_context:
            for var_name, var in variable_context.items():
                var_type = type(var).__name__
                if (
                    var_type in ["QReg", "CReg"]
                    and hasattr(var, "size")
                    and var_name not in self.array_info
                ):
                    self.array_info[var_name] = ArrayAccessInfo(
                        array_name=var_name,
                        size=var.size,
                        is_classical=(var_type == "CReg"),
                    )

    def _analyze_operation(self, op: Any) -> None:
        """Analyze a single operation."""
        op_type = type(op).__name__

        if op_type == "Measure":
            self._analyze_measurement(op)
        elif op_type == "If":
            self._analyze_if_block(op)
        elif hasattr(op, "qargs"):
            self._analyze_quantum_operation(op)
        elif hasattr(op, "ops"):
            # Check if this is a nested Block
            if hasattr(op, "__class__"):
                from pecos.slr import Block as SlrBlock
                try:
                    if issubclass(op.__class__, SlrBlock):
                        # Mark that we have nested blocks
                        self.has_nested_blocks = True
                except:
                    pass
            
            # Nested block - recurse into its operations
            for nested_op in op.ops:
                self._analyze_operation(nested_op)

    def _analyze_measurement(self, meas: Any) -> None:
        """Analyze a measurement operation."""
        # Analyze classical targets if present
        if hasattr(meas, "cout") and meas.cout:
            for cout in meas.cout:
                if hasattr(cout, "reg") and hasattr(cout.reg, "sym"):
                    array_name = cout.reg.sym
                    if array_name in self.array_info and hasattr(cout, "index"):
                        info = self.array_info[array_name]
                        # Track individual classical element access
                        info.element_accesses.add(cout.index)

        # Analyze quantum sources
        if hasattr(meas, "qargs") and meas.qargs:
            for qarg in meas.qargs:
                # Handle full array measurement (QReg directly)
                if hasattr(qarg, "sym") and hasattr(qarg, "size"):
                    array_name = qarg.sym
                    if array_name in self.array_info:
                        info = self.array_info[array_name]
                        # Full array measurement
                        info.full_array_accesses.append(self.position_counter)
                        info.fully_consumed = True
                        info.consumed_at_position = self.position_counter

                        # Mark all elements as consumed
                        for i in range(info.size):
                            info.elements_consumed.add(i)

                # Handle individual element measurement (Qubit with reg)
                elif hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                    array_name = qarg.reg.sym
                    if array_name in self.array_info:
                        info = self.array_info[array_name]

                        if hasattr(qarg, "index"):
                            # Individual element measurement
                            index = qarg.index
                            info.element_accesses.add(index)
                            info.elements_consumed.add(index)

                            if index not in info.element_access_positions:
                                info.element_access_positions[index] = []
                            info.element_access_positions[index].append(
                                self.position_counter,
                            )

    def _analyze_quantum_operation(self, op: Any) -> None:
        """Analyze a quantum operation (gate, etc.)."""
        if hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                    array_name = qarg.reg.sym
                    if array_name in self.array_info:
                        info = self.array_info[array_name]

                        if hasattr(qarg, "index"):
                            # Individual element access
                            index = qarg.index
                            info.element_accesses.add(index)

                            if index not in info.element_access_positions:
                                info.element_access_positions[index] = []
                            info.element_access_positions[index].append(
                                self.position_counter,
                            )

                            # Check if there are measurements before this
                            if info.elements_consumed:
                                info.has_operations_between = True

    def _analyze_if_block(self, if_block: Any) -> None:
        """Analyze an if block."""
        prev_conditional = self.in_conditional
        self.in_conditional = True

        # Check condition for array accesses
        if hasattr(if_block, "cond"):
            self._analyze_condition(if_block.cond)

        # Analyze then block
        if hasattr(if_block, "ops"):
            for op in if_block.ops:
                self._analyze_operation(op)

        # Analyze else block
        if (
            hasattr(if_block, "else_block")
            and if_block.else_block
            and hasattr(if_block.else_block, "ops")
        ):
            for op in if_block.else_block.ops:
                self._analyze_operation(op)

        self.in_conditional = prev_conditional

        # Mark arrays used in conditionals
        if self.in_conditional:
            for info in self.array_info.values():
                if info.element_accesses:
                    info.has_conditionals_between = True

    def _analyze_condition(self, cond: Any) -> None:
        """Analyze a condition expression."""
        # Look for array accesses in conditions
        cond_type = type(cond).__name__

        if cond_type == "Bit":
            if hasattr(cond, "reg") and hasattr(cond.reg, "sym"):
                array_name = cond.reg.sym
                if array_name in self.array_info and hasattr(cond, "index"):
                    info = self.array_info[array_name]
                    info.element_accesses.add(cond.index)
                    info.has_conditionals_between = True

        # Handle compound conditions
        elif hasattr(cond, "left"):
            self._analyze_condition(cond.left)
        if hasattr(cond, "right"):
            self._analyze_condition(cond.right)

    def _check_name_conflicts(self, block: SLRBlock, plan: UnpackingPlan) -> None:
        """Check for variable names that conflict with reserved words."""
        if hasattr(block, "vars"):
            for var in block.vars:
                if hasattr(var, "sym") and var.sym in self.reserved_names:
                    # Need to rename this variable
                    new_name = f"{var.sym}_reg"
                    plan.renamed_variables[var.sym] = new_name
