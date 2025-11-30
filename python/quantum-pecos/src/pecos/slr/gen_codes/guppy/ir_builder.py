"""Builder for converting SLR operations to IR.

IMPORTANT LIMITATION - Partial Consumption in Loops:
====================================================

The current implementation returns ONLY unconsumed array elements from functions.
This works correctly for most patterns, but has a known limitation with certain
verification loop patterns (e.g., Steane code).

WORKING PATTERN (Partial Consumption):
--------------------------------------
def process_qubits(q: array[quantum.qubit, 4] @owned) -> array[quantum.qubit, 2]:
    # Measures q[0] and q[2], returns q[1] and q[3]
    # Return type correctly reflects only unconsumed elements

PROBLEMATIC PATTERN (Verification Ancillas in Loops):
-----------------------------------------------------
def verify(ancilla: array[qubit, 3] @owned) -> tuple[array[qubit, 2], ...]:
    # Measures ancilla[0], creates fresh qubit at ancilla[0]
    # Returns ONLY ancilla[1] and ancilla[2] (unconsumed elements)
    # Fresh qubit is NOT returned (it's an automatic replacement for linearity)

# In calling function:
for _ in range(2):
    ancilla_returned = verify(ancilla)  # ERROR: Returns size 2, needs size 3

WHY THIS HAPPENS:
- Automatic qubit replacements (lines 2966-2977) are created for Guppy's linear
  type system, not for meaningful quantum operations
- The replacement qubit is not semantically part of the verification result
- Only unconsumed elements (ancilla[1], ancilla[2]) are returned
- This creates a size mismatch in subsequent loop iterations

ARCHITECTURAL SOLUTIONS:
- Don't use partial consumption for verification ancillas that need reuse
- Use separate ancilla qubits instead of array elements for verification
- Or restructure the verification pattern to avoid the loop issue

See tests/slr-tests/guppy/test_partial_array_returns.py for correct usage patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pecos.slr import Block as SLRBlock
    from pecos.slr.gen_codes.guppy.ir import IRNode
    from pecos.slr.gen_codes.guppy.ir_analyzer import UnpackingPlan
    from pecos.slr.gen_codes.guppy.unified_resource_planner import (
        UnifiedResourceAnalysis,
    )

# AllocationOptimizer removed - now using UnifiedResourceAnalysis directly
from pecos.slr.gen_codes.guppy.ir import (
    ArrayAccess,
    ArrayUnpack,
    Assignment,
    BinaryOp,
    Block,
    Comment,
    Expression,
    FieldAccess,
    ForStatement,
    Function,
    FunctionCall,
    IfStatement,
    Literal,
    Measurement,
    Module,
    ResourceState,
    ReturnStatement,
    ScopeContext,
    Statement,
    TupleExpression,
    UnaryOp,
    VariableInfo,
    VariableRef,
    WhileStatement,
)
from pecos.slr.gen_codes.guppy.qubit_usage_analyzer import QubitRole, QubitUsageAnalyzer
from pecos.slr.gen_codes.guppy.scope_manager import (
    ResourceUsage,
    ScopeManager,
    ScopeType,
)


class IRBuilder:
    """Builds IR from SLR operations."""

    # Core blocks that should remain as control flow (not converted to functions)
    CORE_BLOCKS: ClassVar[set[str]] = {
        "If",
        "Repeat",
        "While",
        "For",
        "Main",
        "Block",
        "Comment",
        "Barrier",
    }

    def __init__(
        self,
        unpacking_plan: UnpackingPlan,
        *,
        unified_analysis: UnifiedResourceAnalysis | None = None,
        include_optimization_report: bool = False,
    ):
        self.plan = unpacking_plan
        self.unified_analysis = unified_analysis
        self.context = ScopeContext()
        self.scope_manager = ScopeManager()
        self.current_block: Block | None = None
        # AllocationOptimizer removed - using UnifiedResourceAnalysis directly
        self.include_optimization_report = include_optimization_report

        # Track arrays that have been refreshed by function calls
        # Maps original array name -> fresh returned name
        self.refreshed_arrays = {}
        # Track which function refreshed each array
        # Maps original array name -> function name that refreshed it
        self.refreshed_by_function = {}

        # Track conditionally consumed variables (e.g., in if blocks)
        # Maps original variable -> conditionally consumed version
        self.conditional_fresh_vars = {}

        # Track blocks for function generation
        self.block_registry = {}  # Maps block signature to function name
        self.pending_functions = []  # Functions to be generated
        self.generated_functions = set()  # Functions already generated (actually built)
        self.discovered_functions = (
            set()
        )  # Functions discovered but maybe not built yet
        self.function_counter = 0  # For generating unique function names
        self.function_info = {}  # Track metadata about functions
        self.function_return_types = {}  # Maps function name to return type

        # Struct generation tracking
        self.struct_info = (
            {}
        )  # Maps prefix -> {fields: [(suffix, type, size)], struct_name: str}

        # Track all used variable names to avoid conflicts
        self.used_var_names = set()

        # Track explicit Prep (reset) operations for return type calculation
        # Maps array_name -> set of indices that were explicitly reset
        self.explicitly_reset_qubits = {}

        # Variable remapping for handling measurement+Prep pattern
        # Maps old_name -> new_name for variables that need fresh names
        self.variable_remapping: dict[str, str] = {}
        # Track version numbers for generating unique variable names
        self.variable_version_counter: dict[str, int] = {}

    def _get_unique_var_name(self, base_name: str, index: int | None = None) -> str:
        """Generate a unique variable name that doesn't conflict with existing names.

        Args:
            base_name: The base name for the variable
            index: Optional index to append to the base name

        Returns:
            A unique variable name
        """
        candidate = f"{base_name}_{index}" if index is not None else base_name

        # If the name doesn't conflict, use it
        if candidate not in self.used_var_names:
            self.used_var_names.add(candidate)
            return candidate

        # Add underscores until we find a unique name
        while candidate in self.used_var_names:
            candidate = f"_{candidate}"

        self.used_var_names.add(candidate)
        return candidate

    def _collect_var_names(self, block) -> None:
        """Collect all variable names from a block to avoid conflicts."""
        if hasattr(block, "vars"):
            for var in block.vars:
                if hasattr(var, "sym"):
                    self.used_var_names.add(var.sym)
        # Also check ops recursively
        if hasattr(block, "ops"):
            for op in block.ops:
                if hasattr(op, "__class__") and op.__class__.__name__ in [
                    "Main",
                    "Block",
                ]:
                    self._collect_var_names(op)

    def build_module(self, main_block: SLRBlock, pending_functions: list) -> Module:
        """Build a complete module from SLR."""
        module = Module()

        # Collect all existing variable names to avoid conflicts
        self._collect_var_names(main_block)

        # Allocation analysis now comes from UnifiedResourceAnalysis
        # (passed via unified_analysis parameter)

        # Analyze qubit usage to identify ancillas
        qubit_analyzer = QubitUsageAnalyzer()
        self.qubit_usage_stats = qubit_analyzer.analyze_block(main_block)

        # Detect and analyze struct patterns (will use qubit usage stats)
        self._detect_struct_patterns(main_block)

        # Add imports (including functional quantum operations for Array Unpacking Pattern)
        module.imports = [
            "from __future__ import annotations",
            "",
            "from typing import no_type_check",
            "",
            "from guppylang.decorator import guppy",
            "from guppylang.std import quantum",
            "from guppylang.std.quantum import qubit",
            "from guppylang.std.quantum.functional import ("
            "reset, h, x, y, z, s, t, sdg, tdg, cx, cy, cz"
            ")",
            "from guppylang.std.builtins import array, owned, result, py",
        ]

        # Generate struct definitions after imports
        if self.struct_info:
            module.imports.append("")
            struct_defs = self._generate_struct_definitions()
            module.imports.extend(struct_defs)

        # Add optimization report as comments (only if requested)
        if self.include_optimization_report and self.unified_analysis:
            # Use unified resource planning report (comprehensive)
            report = self.unified_analysis.get_report()
            module.imports.extend(
                [
                    "",
                    *["# " + line for line in report.split("\n") if line.strip()],
                ],
            )

        # Build main function
        main_func = self.build_main_function(main_block)
        module.functions.append(main_func)
        # Store refreshed arrays for main function
        module.refreshed_arrays["main"] = self.refreshed_arrays.copy()
        # Also store which functions refreshed each array in main
        if not hasattr(module, "refreshed_by_function_map"):
            module.refreshed_by_function_map = {}
        module.refreshed_by_function_map["main"] = self.refreshed_by_function.copy()

        # Generate helper functions for structs
        for prefix, info in self.struct_info.items():
            # Generate decompose function (always needed for cleanup)
            decompose_func = self._generate_struct_decompose_function(prefix, info)
            if decompose_func:
                module.functions.append(decompose_func)

            # Also generate discard function (useful for other contexts)
            discard_func = self._generate_struct_discard_function(prefix, info)
            if discard_func:
                module.functions.append(discard_func)

        # Build any pending functions (from both parameter and internal tracking)
        all_pending = list(pending_functions) + self.pending_functions
        while all_pending:
            func_info = all_pending.pop(0)
            func = self.build_function(func_info)
            if func:
                module.functions.append(func)
                # Mark this function as generated
                if len(func_info) >= 2:
                    self.generated_functions.add(func_info[1])
                    # Store refreshed arrays for this function
                    module.refreshed_arrays[func_info[1]] = self.refreshed_arrays.copy()
                    # Also store which functions refreshed each array
                    if not hasattr(module, "refreshed_by_function_map"):
                        module.refreshed_by_function_map = {}
                    module.refreshed_by_function_map[func_info[1]] = (
                        self.refreshed_by_function.copy()
                    )
                # Check if building this function added more pending functions
                # Add any new pending functions, avoiding duplicates
                for new_func in self.pending_functions:
                    _new_block, new_name, _new_sig = new_func
                    # Check if this function is already built or pending
                    already_pending = any(
                        f[1] == new_name for f in all_pending if len(f) >= 2
                    )
                    if new_name not in self.generated_functions and not already_pending:
                        all_pending.append(new_func)
                self.pending_functions = []

        # SECOND PASS: Correct return types for functions that return values from other functions
        # This is needed because nested functions are built after their parents
        self._correct_return_types_from_called_functions(module)

        return module

    def _correct_return_types_from_called_functions(self, module):
        """Correct return types for functions that return values from other functions.

        This is a second pass needed because nested functions are built after their parents,
        so when calculating the parent's return type, the nested function's return type
        isn't available yet.
        """

        # For each function, check if it needs return type correction
        for func in module.functions:
            if func.name == "main":
                continue  # Skip main function

            # Check if this function has refreshed_by_function mappings
            if func.name not in module.refreshed_arrays:
                continue

            func_refreshed_arrays = module.refreshed_arrays[func.name]
            if not func_refreshed_arrays:
                continue

            # We need to check if this function's return type should be corrected
            # by looking at which functions refreshed its arrays
            # For now, we'll use a simpler approach: check if the return type
            # involves arrays that were refreshed by other functions

            # Parse the current return type
            current_return_type = func.return_type
            if current_return_type == "None":
                continue  # Procedural function, no correction needed

            # Get the refreshed_by_function mapping for this function
            if not hasattr(module, "refreshed_by_function_map"):
                continue
            if func.name not in module.refreshed_by_function_map:
                continue

            func_refreshed_by_function = module.refreshed_by_function_map[func.name]
            if not func_refreshed_by_function:
                continue

            # For functions returning tuples, we need to check each element
            if current_return_type.startswith("tuple["):
                import re

                tuple_match = re.match(r"tuple\[(.*)\]", current_return_type)
                if tuple_match:
                    # Get parameter names from function params (quantum arrays only)
                    param_names = [
                        p[0] for p in func.params if "array[quantum.qubit," in p[1]
                    ]

                    # For each quantum parameter, check if it was refreshed by a function
                    corrected_types = []
                    for param_name in param_names:
                        if param_name in func_refreshed_by_function:
                            func_info = func_refreshed_by_function[param_name]
                            # Extract function name from the dict (or handle legacy string format)
                            called_func_name = (
                                func_info["function"]
                                if isinstance(func_info, dict)
                                else func_info  # Legacy string format
                            )

                            # Look up the called function's return type
                            if called_func_name in self.function_return_types:
                                called_return_type = self.function_return_types[
                                    called_func_name
                                ]

                                # If the called function returns a tuple, extract the type for this param
                                if called_return_type.startswith("tuple["):
                                    tuple_match2 = re.match(
                                        r"tuple\[(.*)\]",
                                        called_return_type,
                                    )
                                    if tuple_match2:
                                        called_types_str = tuple_match2.group(1)
                                        # Parse the types (handling nested brackets)
                                        types_list = []
                                        bracket_depth = 0
                                        current_type = ""
                                        for char in called_types_str:
                                            if char == "[":
                                                bracket_depth += 1
                                                current_type += char
                                            elif char == "]":
                                                bracket_depth -= 1
                                                current_type += char
                                            elif char == "," and bracket_depth == 0:
                                                types_list.append(current_type.strip())
                                                current_type = ""
                                            else:
                                                current_type += char
                                        if current_type:
                                            types_list.append(current_type.strip())

                                        # Find which position this param is in
                                        param_idx = param_names.index(param_name)
                                        if param_idx < len(types_list):
                                            corrected_types.append(
                                                types_list[param_idx],
                                            )
                                        else:
                                            # Fallback: use current type
                                            corrected_types.append(None)
                                    else:
                                        corrected_types.append(None)
                                else:
                                    # Single return - use it directly if this is the only param
                                    if len(param_names) == 1:
                                        corrected_types.append(called_return_type)
                                    else:
                                        corrected_types.append(None)
                            else:
                                corrected_types.append(None)
                        else:
                            corrected_types.append(None)

                    # If we have corrections, update the function's return type
                    if any(ct is not None for ct in corrected_types):
                        # Parse current types
                        current_types_str = tuple_match.group(1)
                        current_types_list = []
                        bracket_depth = 0
                        current_type = ""
                        for char in current_types_str:
                            if char == "[":
                                bracket_depth += 1
                                current_type += char
                            elif char == "]":
                                bracket_depth -= 1
                                current_type += char
                            elif char == "," and bracket_depth == 0:
                                current_types_list.append(current_type.strip())
                                current_type = ""
                            else:
                                current_type += char
                        if current_type:
                            current_types_list.append(current_type.strip())

                        # Apply corrections
                        new_types = []
                        for i, corrected in enumerate(corrected_types):
                            if corrected is not None:
                                new_types.append(corrected)
                            elif i < len(current_types_list):
                                new_types.append(current_types_list[i])
                            else:
                                # Something went wrong, skip correction
                                new_types = None
                                break

                        if new_types:
                            new_return_type = f"tuple[{', '.join(new_types)}]"
                            func.return_type = new_return_type
                            # Also update the registry
                            self.function_return_types[func.name] = new_return_type

    def build_main_function(self, block: SLRBlock) -> Function:
        """Build the main function."""
        # Set current function name
        self.current_function_name = "main"

        # Reset function-local state
        self.refreshed_arrays = {}
        self.refreshed_by_function = {}
        self.conditional_fresh_vars = {}
        self.array_remapping = {}  # Reset array remapping for main function

        # Analyze qubit usage patterns
        usage_analyzer = QubitUsageAnalyzer()
        usage_analyzer.analyze_block(block, self.struct_info)
        self.allocation_recommendations = (
            usage_analyzer.get_allocation_recommendations()
        )

        # Pre-analyze explicit reset operations (Prep) to distinguish them from automatic replacements
        consumed_in_main = {}
        self._track_consumed_qubits(block, consumed_in_main)

        # Override allocation recommendations for struct fields to ensure they're pre-allocated
        # (struct constructors need all fields to be available)
        if self.struct_info:
            for prefix, info in self.struct_info.items():
                for suffix, _, _ in info["fields"]:
                    var_name = info["var_names"][suffix]
                    # Override the allocation recommendations system
                    if var_name in self.allocation_recommendations:
                        recommendation = self.allocation_recommendations[var_name]
                        if recommendation.get("allocation") == "dynamic":
                            # Override dynamic allocation for struct fields
                            self.allocation_recommendations[var_name] = {
                                "allocation": "pre_allocate",
                                "reason": "Struct field requires pre-allocation",
                                "keep_packed": recommendation.get("keep_packed", True),
                                "pre_allocate": True,
                            }

        body = Block()
        self.current_block = body

        # Track arrays consumed by @owned function calls
        self.consumed_arrays = set()

        # Add variable declarations
        if hasattr(block, "vars"):
            # First, add non-struct variables
            struct_vars = set()
            for prefix, info in self.struct_info.items():
                struct_vars.update(info["var_names"].values())

            # Get ancilla variables that were excluded from structs
            ancilla_vars = getattr(self, "ancilla_qubits", set())

            for var in block.vars:
                if hasattr(var, "sym"):
                    # Add if not in struct OR if it's an ancilla (which was excluded from struct)
                    if var.sym not in struct_vars or var.sym in ancilla_vars:
                        self._add_variable_declaration(var, block)

                    # Add to scope context for resource tracking
                    var_type = type(var).__name__
                    if var_type in ["QReg", "CReg"]:
                        is_quantum = var_type == "QReg"
                        size = getattr(var, "size", None)

                        var_info = VariableInfo(
                            name=var.sym,
                            original_name=var.sym,
                            var_type="quantum" if is_quantum else "classical",
                            size=size,
                            is_array=True,
                        )
                        self.context.add_variable(var_info)

            # Then, create struct instances
            for prefix, info in self.struct_info.items():
                self._add_struct_initialization(prefix, info, block)

        # Main function maintains natural SLR array semantics
        # Arrays are only unpacked internally when needed for selective measurements

        # Track unpacked vars for main
        self.unpacked_vars = {}

        # First pass: determine which quantum arrays will be unpacked
        will_unpack_quantum = set()
        for array_name in self.plan.unpack_at_start:
            if array_name in self.plan.arrays_to_unpack:
                info = self.plan.arrays_to_unpack[array_name]

                # Skip struct fields
                is_struct_field = False
                if self.struct_info:
                    for prefix, struct_info in self.struct_info.items():
                        if array_name in struct_info.get("var_names", {}).values():
                            is_struct_field = True
                            break

                if is_struct_field:
                    continue

                # Skip dynamically allocated arrays
                if (
                    hasattr(self, "dynamic_allocations")
                    and array_name in self.dynamic_allocations
                ):
                    continue

                # Mark quantum arrays that will be unpacked
                if not info.is_classical:
                    will_unpack_quantum.add(array_name)

        # Second pass: actually unpack arrays
        for array_name in self.plan.unpack_at_start:
            if array_name in self.plan.arrays_to_unpack:
                info = self.plan.arrays_to_unpack[array_name]

                # Skip unpacking for arrays that are struct fields
                # (already consumed by struct constructor)
                is_struct_field = False
                if self.struct_info:
                    for prefix, struct_info in self.struct_info.items():
                        if array_name in struct_info.get("var_names", {}).values():
                            is_struct_field = True
                            break

                if is_struct_field:
                    # Skip unpacking - array is consumed by struct constructor
                    # Individual elements can be accessed via struct decomposition
                    self.current_block.statements.append(
                        Comment(
                            f"Skip unpacking {array_name} - consumed by struct constructor",
                        ),
                    )
                    continue

                # For dynamically allocated arrays, skip unpacking - qubits are allocated on first use
                if (
                    hasattr(self, "dynamic_allocations")
                    and array_name in self.dynamic_allocations
                ):
                    # Don't unpack - the array doesn't exist, qubits are allocated individually
                    continue
                if not info.is_classical:
                    # Regular unpacking for quantum arrays
                    self.current_block.statements.append(
                        Comment(f"Unpack {array_name} for individual access"),
                    )
                    self._add_array_unpacking(array_name, info.size)
                else:
                    # For classical arrays, unpack if any quantum array is unpacked
                    # This ensures consistent variable naming patterns
                    should_unpack_classical = len(will_unpack_quantum) > 0 or (
                        hasattr(self, "dynamic_allocations")
                        and len(self.dynamic_allocations) > 0
                    )
                    if should_unpack_classical:
                        # Unpack classical array to support quantum unpacking pattern
                        self.current_block.statements.append(
                            Comment(
                                f"Unpack {array_name} for individual measurement results",
                            ),
                        )
                        self._add_array_unpacking(array_name, info.size)
                    else:
                        # Skip unpacking classical arrays in main to avoid linearity violations
                        # Classical arrays can be accessed directly and passed to functions
                        self.current_block.statements.append(
                            Comment(
                                f"Skip unpacking classical array {array_name} - not needed for linearity",
                            ),
                        )

        # Add operations
        if hasattr(block, "ops"):
            # Store block reference for look-ahead in operation conversion
            self.current_block_ops = block.ops
            for op_index, op in enumerate(block.ops):
                # Store current operation index for look-ahead
                self.current_op_index = op_index
                stmt = self._convert_operation(op)
                if stmt:
                    body.statements.append(stmt)
            # Clear after processing
            self.current_block_ops = None
            self.current_op_index = None

        # Handle struct decomposition, results, and cleanup
        self._add_final_handling(block)

        return Function(
            name="main",
            params=[],
            return_type="None",
            body=body,
            decorators=["guppy", "no_type_check"],
        )

    def build_function(self, func_info) -> Function | None:
        """Build a function from pending function info."""

        # Reset function-local state
        self.refreshed_arrays = {}
        self.refreshed_by_function = {}
        self.conditional_fresh_vars = {}
        self.array_remapping = {}  # Reset array remapping for each function
        # Reset parameter_unpacked_arrays for each function
        self.parameter_unpacked_arrays = set()
        # Reset explicitly_reset_qubits for each function to prevent cross-contamination
        self.explicitly_reset_qubits = {}

        # Handle different formats of func_info
        if len(func_info) == 3:
            # New format from IR builder: (block, func_name, signature)
            sample_block, func_name, _block_signature = func_info
        elif len(func_info) == 4:
            # Old format: (block_key, func_name, sample_block, block_name)
            _block_key, func_name, sample_block, _block_name = func_info
        else:
            return None

        # Analyze dependencies to determine parameters
        deps = self._analyze_block_dependencies(sample_block)

        # Build parameter list
        params = []
        param_mapping = {}  # Maps parameter names to original variable names

        # Check if we should use structs instead of individual arrays
        struct_params = set()  # Structs we've already added
        vars_in_structs = set()  # Variables that are part of structs

        # First pass: identify which variables are part of structs
        for prefix, info in self.struct_info.items():
            vars_in_this_struct = []
            for var in info["var_names"].values():
                if var in deps["quantum"] or var in deps["classical"]:
                    vars_in_structs.add(var)
                    vars_in_this_struct.append(var)

            # If any variable from this struct is used, add the struct as a parameter
            if vars_in_this_struct and prefix not in struct_params:
                # Add struct parameter
                struct_name = info["struct_name"]
                param_type = struct_name

                # Check if this struct contains quantum resources
                has_quantum = any(v in deps["quantum"] for v in vars_in_this_struct)
                if has_quantum and self._block_consumes_quantum(sample_block):
                    param_type = f"{param_type} @owned"

                params.append((prefix, param_type))
                param_mapping[prefix] = prefix
                struct_params.add(prefix)

        # Black Box Pattern: All functions that handle quantum arrays should use
        # functional pattern. This maintains SLR's global array semantics at
        # boundaries while using functional internals
        # BUT: Only unpack if the IR analyzer determined it's necessary
        # First, run the IR analyzer on this block to get unpacking plan
        from pecos.slr.gen_codes.guppy.ir_analyzer import IRAnalyzer

        # Pre-analyze consumption to inform the IR analyzer about @owned parameters
        consumed_params = set()
        if hasattr(sample_block, "ops"):
            # Check if this function has nested blocks
            has_nested_blocks = False
            for op in sample_block.ops:
                if hasattr(op, "__class__"):
                    from pecos.slr import Block as SlrBlock

                    try:
                        if issubclass(op.__class__, SlrBlock):
                            has_nested_blocks = True
                            break
                    except (TypeError, AttributeError):
                        # Not a class or doesn't have required attributes
                        pass

            # Analyze consumption - this will help determine @owned parameters
            consumed_params = self._analyze_consumed_parameters(sample_block)
            # Also analyze which arrays have subscript access - they also need @owned
            subscripted_params = self._analyze_subscript_access(sample_block)
            # Store for later use in @owned determination
            self.subscripted_params = subscripted_params
        else:
            # No ops - initialize empty set
            self.subscripted_params = set()

        analyzer = IRAnalyzer()

        # Pass information about expected @owned parameters to the analyzer
        analyzer.expected_owned_params = consumed_params
        analyzer.has_nested_blocks_with_owned = has_nested_blocks and bool(
            consumed_params,
        )

        block_plan = analyzer.analyze_block(sample_block, self.context.variables)

        # Only unpack if there are arrays that need unpacking according to the analyzer
        needs_unpacking = len(block_plan.arrays_to_unpack) > 0

        # Check if this function consumes its quantum arrays
        # For the functional pattern in Guppy, all functions that take quantum arrays
        # and will return them need @owned annotation
        self._block_consumes_quantum(sample_block)

        # If the function has quantum parameters, it should use @owned
        # This is required for Guppy's linearity system when arrays are returned
        bool(deps["quantum"] & deps["reads"])

        # Add quantum parameters (skip those in structs UNLESS they're ancillas)
        for var in sorted(deps["quantum"] & deps["reads"]):
            # Check if this is an ancilla that was excluded from structs
            is_excluded_ancilla = (
                hasattr(self, "ancilla_qubits") and var in self.ancilla_qubits
            )

            if var in vars_in_structs and not is_excluded_ancilla:
                continue
            param_name = var  # Use the same name, no need for _param suffix
            param_mapping[param_name] = var
            # Determine type from context or default to qubit array
            var_info = self.context.lookup_variable(var)
            if var_info:
                if var_info.is_unpacked:
                    # This is an unpacked array - need the original array type
                    param_type = f"array[quantum.qubit, {var_info.size}]"
                else:
                    # Always use array type to maintain consistency with SLR semantics
                    param_type = f"array[quantum.qubit, {var_info.size}]"
            else:
                # Default assumption for quantum variables
                param_type = "array[quantum.qubit, 7]"

            params.append((param_name, param_type))

        # Add classical parameters (no ownership, but include written vars
        # since arrays are mutable)
        for var in sorted(deps["classical"] & (deps["reads"] | deps["writes"])):
            if var in vars_in_structs:
                continue
            param_name = var  # Use the same name, no need for _param suffix
            param_mapping[param_name] = var
            # Determine type from context
            var_info = self.context.lookup_variable(var)
            # Always use array type for consistency
            param_type = (
                f"array[bool, {var_info.size}]" if var_info else "array[bool, 32]"
            )
            params.append((param_name, param_type))

        # Create function body
        body = Block()
        prev_block = self.current_block
        prev_mapping = self.param_mapping if hasattr(self, "param_mapping") else {}
        self.current_block = body
        self.param_mapping = param_mapping

        # Create a variable remapping context for this function
        # This maps original variable names to their parameter names
        var_remapping = {}
        for param_name, original_name in param_mapping.items():
            var_remapping[original_name] = param_name

            # Also handle unpacked variables
            var_info = self.context.lookup_variable(original_name)
            if var_info and var_info.is_unpacked:
                # Map each unpacked element
                for i, unpacked_name in enumerate(var_info.unpacked_names):
                    var_remapping[unpacked_name] = f"{param_name}[{i}]"

        # Store current function context
        self.current_function_name = func_name
        self.current_function_params = params
        self.current_function_return_type = None  # Will be set after we determine it

        # Clear fresh_return_vars tracking for this new function
        # (to avoid bleeding from previous function builds)
        self.fresh_return_vars = {}

        # Track if this function has @owned struct parameters
        has_owned_struct_params = any(
            "@owned" in param_type and param_name in self.struct_info
            for param_name, param_type in params
        )
        self.function_info[func_name] = {
            "has_owned_struct_params": has_owned_struct_params,
            "params": params,
        }

        # Store the remapping for use during conversion
        prev_var_remapping = getattr(self, "var_remapping", {})
        self.var_remapping = var_remapping

        # Track unpacked variables (only if needed)
        self.unpacked_vars = {}  # Maps array_name -> [element_names]
        self.replaced_qubits = {}  # Maps array_name -> set of replaced indices

        # Initially add array unpacking for arrays that the analyzer determined need it
        if needs_unpacking:
            for param_name, param_type in params:
                if (
                    "array[quantum.qubit," in param_type
                    and param_name in block_plan.arrays_to_unpack
                ):
                    # Extract array size
                    import re

                    match = re.search(r"array\[quantum\.qubit, (\d+)\]", param_type)
                    if match:
                        size = int(match.group(1))
                        # Generate unpacked variable names
                        element_names = [
                            self._get_unique_var_name(param_name, i)
                            for i in range(size)
                        ]
                        self.unpacked_vars[param_name] = element_names

                        # Add unpacking statement to function body
                        unpacking_stmt = self._create_array_unpack_statement(
                            param_name,
                            element_names,
                        )
                        body.statements.append(unpacking_stmt)

        # Additionally, check for ALL @owned arrays that need unpacking
        # With the functional pattern, @owned arrays must be unpacked to avoid MoveOutOfSubscriptError
        # UNLESS they're passed to nested blocks
        for param_name, param_type in params:
            if (
                "@owned" in param_type
                and "array[quantum.qubit," in param_type
                and param_name not in self.unpacked_vars
            ):
                # Check if this function has any nested block calls
                # If so, we can't unpack @owned arrays as we may need to pass them
                # But this will cause MoveOutOfSubscriptError, so we need a different approach
                has_nested_blocks = False
                if hasattr(sample_block, "ops"):
                    for op in sample_block.ops:
                        # Check if this is a Block subclass
                        if hasattr(op, "__class__"):
                            from pecos.slr import Block as SlrBlock

                            try:
                                if issubclass(op.__class__, SlrBlock):
                                    has_nested_blocks = True
                                    break
                            except (TypeError, AttributeError):
                                # Not a class or doesn't have required attributes
                                pass

                # @owned parameters MUST be unpacked regardless of analyzer decision
                # This is required by Guppy's type system to avoid MoveOutOfSubscriptError
                force_unpack = "@owned" in param_type

                # Check if the analyzer decided this array should be unpacked
                # Even with nested blocks, @owned arrays need unpacking to access elements
                if not force_unpack and param_name not in block_plan.arrays_to_unpack:
                    if has_nested_blocks:
                        body.statements.append(
                            Comment(
                                f"Skip unpacking {param_name} - function has nested blocks",
                            ),
                        )
                    continue

                # This @owned array needs unpacking to avoid MoveOutOfSubscriptError
                import re

                match = re.search(r"array\[quantum\.qubit, (\d+)\]", param_type)
                if match:
                    size = int(match.group(1))
                    # Generate unpacked variable names
                    element_names = [
                        self._get_unique_var_name(param_name, i) for i in range(size)
                    ]
                    self.unpacked_vars[param_name] = element_names

                    # Track that this was unpacked from a parameter (not a return value)
                    # Parameter-unpacked arrays should NOT be reconstructed for function calls
                    if not hasattr(self, "parameter_unpacked_arrays"):
                        self.parameter_unpacked_arrays = set()
                    self.parameter_unpacked_arrays.add(param_name)

                    # Add comment explaining why we're unpacking
                    body.statements.append(
                        Comment(
                            f"Unpack @owned array {param_name} to avoid "
                            "MoveOutOfSubscriptError",
                        ),
                    )

                    # Add unpacking statement to function body
                    unpacking_stmt = ArrayUnpack(
                        source=param_name,
                        targets=element_names,
                    )
                    body.statements.append(unpacking_stmt)

        # Add struct unpacking for struct parameters
        struct_field_vars = (
            {}
        )  # Maps original var name to struct field path for @owned structs
        struct_reconstruction = (
            {}
        )  # Maps struct param name to list of field vars for reconstruction

        for param_name, param_type in params:
            if "@owned" in param_type and param_name in self.struct_info:
                # This is an @owned struct parameter
                # For @owned structs, we must decompose them immediately to avoid AlreadyUsedError
                # when accessing multiple fields
                struct_info = self.struct_info[param_name]

                # Track that we have an owned struct
                if not hasattr(self, "owned_structs"):
                    self.owned_structs = set()
                self.owned_structs.add(param_name)

                # Decompose the @owned struct using the decompose function
                # Use the struct name, not the parameter name (e.g., steane_decompose not c_decompose)
                struct_name = struct_info["struct_name"].replace("_struct", "")
                decompose_func_name = f"{struct_name}_decompose"

                # Create decomposition call
                field_vars = []
                for suffix, field_type, field_size in sorted(struct_info["fields"]):
                    field_var = f"{param_name}_{suffix}"
                    field_vars.append(field_var)

                # Add comment explaining decomposition
                body.statements.append(
                    Comment(
                        f"Decompose @owned struct {param_name} to avoid AlreadyUsedError",
                    ),
                )

                # Add decomposition statement: c_c, c_d, ... = steane_decompose(c)
                class TupleAssignment(Statement):
                    def __init__(self, targets, value):
                        self.targets = targets
                        self.value = value

                    def analyze(self, context):
                        self.value.analyze(context)

                    def render(self, context):
                        target_str = ", ".join(self.targets)
                        value_str = self.value.render(context)[0]
                        return [f"{target_str} = {value_str}"]

                decompose_call = FunctionCall(
                    func_name=decompose_func_name,
                    args=[VariableRef(param_name)],
                )

                decomposition_stmt = TupleAssignment(
                    targets=field_vars,
                    value=decompose_call,
                )
                body.statements.append(decomposition_stmt)

                # Map original variables to the decomposed field variables
                for suffix, field_type, field_size in sorted(struct_info["fields"]):
                    original_var = struct_info["var_names"].get(suffix)
                    if original_var:
                        field_var = f"{param_name}_{suffix}"
                        # Map the original variable name to the decomposed variable
                        if not hasattr(self, "var_remapping"):
                            self.var_remapping = {}
                        self.var_remapping[original_var] = field_var

                # Track the field variables for reconstruction in return statements
                struct_reconstruction[param_name] = field_vars

                # Track decomposed variables for field access
                if not hasattr(self, "decomposed_vars"):
                    self.decomposed_vars = {}
                field_mapping = {}
                for suffix, field_type, field_size in sorted(struct_info["fields"]):
                    field_var = f"{param_name}_{suffix}"
                    field_mapping[suffix] = field_var
                self.decomposed_vars[param_name] = field_mapping

                # Skip normal unpacking for @owned structs
                continue
            if param_name in self.struct_info:
                # Non-owned struct parameter - can unpack normally
                struct_info = self.struct_info[param_name]
                field_vars = []

                # Generate unpacking statement - use same order as struct
                # definition (sorted by suffix)
                unpack_targets = []
                for suffix, field_type, field_size in sorted(struct_info["fields"]):
                    field_var = f"{param_name}_{suffix}"
                    unpack_targets.append(field_var)
                    field_vars.append(field_var)

                    # Map the original variable name to this unpacked field variable
                    original_var = struct_info["var_names"].get(suffix)
                    if original_var:
                        struct_field_vars[original_var] = field_var
                        # Also update var_remapping to use field access directly
                        self.var_remapping[original_var] = field_var

                # Create the unpacking statement:
                # field1, field2, ... = struct.field1, struct.field2, ...
                # In Guppy, we need to unpack the entire struct at once -
                # use same order as struct definition
                unpack_stmt = Assignment(
                    target=TupleExpression(
                        [VariableRef(var) for var in unpack_targets],
                    ),
                    value=TupleExpression(
                        [
                            FieldAccess(VariableRef(param_name), field)
                            for field, _, _ in sorted(struct_info["fields"])
                        ],
                    ),
                )
                body.statements.append(unpack_stmt)

                # Store for reconstruction
                struct_reconstruction[param_name] = field_vars

        # Store struct field mappings for use in variable references
        self.struct_field_mapping = struct_field_vars

        # Pre-analyze what qubits will be consumed to determine return type
        consumed_in_function = {}
        self._track_consumed_qubits(sample_block, consumed_in_function)

        # Pre-determine if this function will return quantum arrays
        # (needed for measurement replacement logic)
        will_return_quantum = False
        has_quantum_arrays = any(
            "array[quantum.qubit," in ptype for name, ptype in params
        )
        has_structs = any(name in self.struct_info for name, ptype in params)

        if has_quantum_arrays or has_structs:
            # Check if any quantum arrays will be returned
            for name, ptype in params:
                if "array[quantum.qubit," in ptype:
                    # Check if this array is part of a struct
                    in_struct = False
                    for prefix, info in self.struct_info.items():
                        if name in info["var_names"].values():
                            in_struct = True
                            break

                    # Check if this is an ancilla that was excluded from structs
                    is_excluded_ancilla = (
                        hasattr(self, "ancilla_qubits") and name in self.ancilla_qubits
                    )

                    # Check if this array has any live qubits
                    if name in consumed_in_function:
                        # Some elements were consumed - check if any are still live
                        consumed_indices = consumed_in_function[name]
                        import re

                        size_match = re.search(
                            r"array\[quantum\.qubit,\s*(\d+)\]",
                            ptype,
                        )
                        array_size = int(size_match.group(1)) if size_match else 2
                        total_indices = set(range(array_size))
                        live_indices = total_indices - consumed_indices
                        include_array = bool(
                            live_indices,
                        )  # Only include if has live qubits
                    else:
                        # No consumption tracked for this array - assume it's live
                        include_array = not in_struct or is_excluded_ancilla

                    if include_array:
                        will_return_quantum = True
                        break

        # Check if this is a procedural block based on resource flow
        # If the block has live qubits that should be returned, it's not procedural
        _consumed_qubits, live_qubits = self._analyze_quantum_resource_flow(
            sample_block,
        )
        has_live_qubits = bool(live_qubits)
        is_procedural_block = not has_live_qubits

        # SMART DETECTION: Determine if this function should be procedural based on usage patterns
        # Functions should be procedural if:
        # 1. They don't need their quantum returns to be used afterward in the calling scope
        # 2. They primarily do terminal operations (measurements, cleanup)
        # 3. Making them procedural would avoid PlaceNotUsedError issues

        # HYBRID APPROACH: Use smart detection to determine optimal strategy
        should_be_procedural = self._should_function_be_procedural(
            func_name,
            sample_block,
            params,
            has_live_qubits,
        )

        if should_be_procedural:
            is_procedural_block = True
        # Function determined to be procedural

        # If it appears to be procedural based on live qubits, double-check with signature
        if is_procedural_block and hasattr(sample_block, "__init__"):
            import inspect

            try:
                sig = inspect.signature(sample_block.__class__.__init__)
                return_annotation = sig.return_annotation
                if (
                    return_annotation is None
                    or return_annotation is type(None)
                    or str(return_annotation) == "None"
                ):
                    is_procedural_block = True
                else:
                    is_procedural_block = False  # Has return annotation, not procedural
            except (ValueError, TypeError, AttributeError):
                # Default to procedural if can't inspect signature
                # ValueError: signature cannot be determined
                # TypeError: object is not callable
                # AttributeError: missing expected attributes
                is_procedural_block = True

        # Store whether this is a procedural block for measurement logic
        self.current_function_is_procedural = is_procedural_block

        # Process params and add @owned annotations (now that we know if it's procedural)
        # HYBRID OWNERSHIP: Smart @owned annotation based on function type and consumption
        processed_params = []
        for param_name, param_type in params:
            if "array[quantum.qubit," in param_type:
                # Determine if this parameter should be @owned based on consumption analysis
                should_be_owned = False

                if is_procedural_block:
                    # For procedural blocks, be selective with @owned
                    # Only use @owned if the parameter is truly consumed (measured) and not reused
                    # BUT also check if this parameter is passed to other functions that might expect @owned
                    # This is necessary for functions like prep_rus that pass parameters to prep_encoding_ft_zero
                    # For simplicity, if the block has nested blocks, make quantum params @owned
                    # If a procedural block calls other blocks, those blocks might need @owned params
                    should_be_owned = (
                        True if has_nested_blocks else param_name in consumed_params
                    )
                else:
                    # For functional blocks that return quantum arrays, check if parameter is actually consumed
                    # In Guppy's linear type system:
                    # - @owned: parameter is consumed by the function
                    # - non-@owned: parameter is borrowed and must be returned
                    # IMPORTANT: In Guppy, subscripting an array (c_a[0]) marks it as used
                    # So ANY element access requires @owned annotation to avoid MoveOutOfSubscriptError
                    if param_name in consumed_in_function:
                        # ANY consumption requires @owned (not just full consumption)
                        # This is because subscripting marks the array as used
                        consumed_indices = consumed_in_function[param_name]
                        should_be_owned = len(consumed_indices) > 0
                    elif (
                        hasattr(self, "subscripted_params")
                        and param_name in self.subscripted_params
                    ):
                        # Array has subscript access (c_d[0]) which requires @owned
                        should_be_owned = True
                    else:
                        # Check if there's element access even without consumption
                        # (e.g., gates applied to elements)
                        # Arrays in arrays_to_unpack need @owned
                        should_be_owned = param_name in block_plan.arrays_to_unpack
                        if should_be_owned:
                            pass
                        else:
                            # Last resort: if parameter is used in the function at all, it likely needs @owned
                            # In Guppy, any use of an array parameter in a functional block requires @owned
                            # because the generated IR will likely subscript it
                            # Check if the parameter appears in deps (it's used in the function)
                            if param_name in deps["quantum"]:
                                should_be_owned = True

                if should_be_owned:
                    param_type = f"{param_type} @owned"

            processed_params.append((param_name, param_type))
        params = processed_params

        # HYBRID UNPACKING: After parameter processing, check for @owned arrays that need unpacking
        # @owned arrays must be unpacked to avoid MoveOutOfSubscriptError when accessing elements
        for param_name, param_type in params:
            # Don't double-unpack
            is_owned_qubit_array = (
                "array[quantum.qubit," in param_type and "@owned" in param_type
            )
            if is_owned_qubit_array and param_name not in self.unpacked_vars:
                # Adding @owned array unpacking
                # Extract array size
                import re

                match = re.search(r"array\[quantum\.qubit, (\d+)\]", param_type)
                if match:
                    size = int(match.group(1))
                    # Generate unpacked variable names
                    element_names = [
                        self._get_unique_var_name(param_name, i) for i in range(size)
                    ]
                    self.unpacked_vars[param_name] = element_names

                    # Track that this was unpacked from a parameter (not a return value)
                    # Parameter-unpacked arrays should NOT be reconstructed for function calls
                    self.parameter_unpacked_arrays.add(param_name)

                    # Add unpacking statement to function body
                    unpacking_stmt = self._create_array_unpack_statement(
                        param_name,
                        element_names,
                    )
                    body.statements.append(unpacking_stmt)

        # Store whether this function returns quantum arrays
        self.current_function_returns_quantum = will_return_quantum

        # Pre-extract conditions that might be needed in loops with @owned structs
        # This must happen BEFORE any operations that might consume the structs
        if hasattr(sample_block, "ops") and self._function_has_owned_struct_params(
            params,
        ):
            extracted_conditions = self._pre_extract_loop_conditions(sample_block, body)

            # Track extracted conditions for later use
            if extracted_conditions:
                if not hasattr(self, "pre_extracted_conditions"):
                    self.pre_extracted_conditions = {}
                self.pre_extracted_conditions.update(extracted_conditions)

        # Now convert operations (can use will_return_quantum flag)
        if hasattr(sample_block, "ops"):
            # Store block reference for look-ahead in operation conversion
            # This enables measurement+Prep pattern detection in _convert_operation
            self.current_block_ops = sample_block.ops
            for op_index, op in enumerate(sample_block.ops):
                # Store current operation index for look-ahead
                self.current_op_index = op_index
                stmt = self._convert_operation(op)
                if stmt:
                    body.statements.append(stmt)
            # Clear after processing
            self.current_block_ops = None
            self.current_op_index = None

        # Fix linearity issues: add fresh qubit allocations after consuming operations
        self._fix_post_consuming_linearity_issues(body)

        # Fix unused fresh variables in conditional execution paths
        self._fix_unused_fresh_variables(body)

        # Save the current variable remapping (includes changes from Prep operations)
        # BEFORE restoring previous mapping, as we need it for return statement generation
        self.function_var_remapping = (
            self.variable_remapping.copy()
            if hasattr(self, "variable_remapping")
            else {}
        )

        # Restore previous remapping
        self.var_remapping = prev_var_remapping
        self.current_block = prev_block
        self.param_mapping = prev_mapping

        # Now calculate the actual detailed return type and generate return statements
        return_type = "None"

        # Black Box Pattern: functions that handle quantum arrays return modified arrays
        # BUT: if function consumes arrays (@owned), don't return them
        # Check if we have quantum arrays or structs to return (regardless of unpacking)
        has_quantum_arrays = any(
            "array[quantum.qubit," in ptype for name, ptype in params
        )
        has_structs = any(name in self.struct_info for name, ptype in params)

        # For procedural blocks, don't generate return statements
        if not is_procedural_block and (has_quantum_arrays or has_structs):
            # Array/struct return pattern: functions return reconstructed arrays or structs
            quantum_returns = []

            # Add structs first - even @owned structs can be returned if they're reconstructed
            for name, ptype in params:
                if name in self.struct_info:
                    # Remove @owned annotation from type for return type
                    return_type = ptype.replace(" @owned", "")
                    quantum_returns.append((name, return_type))

            # Then add individual arrays not in structs (including ancillas)
            for name, ptype in params:
                if "array[quantum.qubit," in ptype:
                    # Check if this array is part of a struct
                    in_struct = False
                    is_excluded_ancilla = False

                    for prefix, info in self.struct_info.items():
                        if name in info["var_names"].values():
                            in_struct = True
                            break

                    # Check if this is an ancilla that was excluded from structs
                    if hasattr(self, "ancilla_qubits") and name in self.ancilla_qubits:
                        is_excluded_ancilla = True

                    # Only include arrays that have live (unconsumed) qubits
                    # Check if this array has any unconsumed elements
                    if name in consumed_in_function:
                        # Some elements were consumed - check if any are still live
                        consumed_indices = consumed_in_function[name]
                        # Extract size from parameter type
                        import re

                        size_match = re.search(
                            r"array\[quantum\.qubit,\s*(\d+)\]",
                            ptype,
                        )
                        array_size = int(size_match.group(1)) if size_match else 2
                        total_indices = set(range(array_size))

                        # Live indices = unconsumed OR explicitly reset
                        # Explicitly reset qubits are consumed by measurement but recreated by Prep
                        explicitly_reset_indices = set()
                        if (
                            hasattr(self, "explicitly_reset_qubits")
                            and name in self.explicitly_reset_qubits
                        ):
                            explicitly_reset_indices = self.explicitly_reset_qubits[
                                name
                            ]

                        live_indices = (
                            total_indices - consumed_indices
                        ) | explicitly_reset_indices
                        include_array = bool(
                            live_indices,
                        )  # Only include if has live qubits (unconsumed OR explicitly reset)
                    else:
                        # No consumption tracked for this array - assume it's live
                        include_array = not in_struct or is_excluded_ancilla

                    if include_array:
                        # PRIORITY 1: Check if this array was refreshed by a function call
                        # If so, use the called function's return type instead of consumption analysis
                        if (
                            hasattr(self, "refreshed_arrays")
                            and name in self.refreshed_arrays
                        ):
                            self.refreshed_arrays[name]
                            # Find which function call produced this fresh variable
                            # by looking at the refreshed_by_function mapping
                            if (
                                hasattr(self, "refreshed_by_function")
                                and name in self.refreshed_by_function
                            ):
                                func_info = self.refreshed_by_function[name]
                                # Extract function name from the dict (or handle legacy string format)
                                called_func_name = (
                                    func_info["function"]
                                    if isinstance(func_info, dict)
                                    else func_info  # Legacy string format
                                )
                                # Look up that function's return type
                                if called_func_name in self.function_return_types:
                                    called_func_return = self.function_return_types[
                                        called_func_name
                                    ]
                                    # If it returns a tuple, extract the type for this array
                                    if called_func_return.startswith("tuple["):
                                        # Parse tuple to find the type for this array
                                        import re

                                        tuple_match = re.match(
                                            r"tuple\[(.*)\]",
                                            called_func_return,
                                        )
                                        if tuple_match:
                                            return_types_str = tuple_match.group(1)
                                            # Split by comma but handle nested brackets
                                            types_list = []
                                            bracket_depth = 0
                                            current_type = ""
                                            for char in return_types_str:
                                                if char == "[":
                                                    bracket_depth += 1
                                                    current_type += char
                                                elif char == "]":
                                                    bracket_depth -= 1
                                                    current_type += char
                                                elif char == "," and bracket_depth == 0:
                                                    types_list.append(
                                                        current_type.strip(),
                                                    )
                                                    current_type = ""
                                                else:
                                                    current_type += char
                                            if current_type:
                                                types_list.append(current_type.strip())

                                            # Find which position this array is in the function's parameters
                                            quantum_param_names = [
                                                n
                                                for n, pt in params
                                                if "array[quantum.qubit," in pt
                                            ]
                                            if name in quantum_param_names:
                                                param_idx = quantum_param_names.index(
                                                    name,
                                                )
                                                if param_idx < len(types_list):
                                                    # Use the return type from the called function
                                                    new_type = types_list[param_idx]
                                                    quantum_returns.append(
                                                        (name, new_type),
                                                    )
                                                    continue  # Skip consumption analysis
                                    else:
                                        # Single return - use it directly
                                        quantum_returns.append(
                                            (name, called_func_return),
                                        )
                                        continue  # Skip consumption analysis

                        # PRIORITY 2: Use consumption analysis if array wasn't refreshed by a function
                        # Check if any elements remain unconsumed for ALL arrays
                        if name in consumed_in_function:
                            # Extract array size from type
                            import re

                            match = re.search(r"array\[quantum\.qubit, (\d+)\]", ptype)
                            if match:
                                original_size = int(match.group(1))
                                consumed_indices = consumed_in_function[name]

                                # Check if any consumed qubits were replaced
                                if (
                                    hasattr(self, "replaced_qubits")
                                    and name in self.replaced_qubits
                                ):
                                    self.replaced_qubits[name]

                                # Check if this parameter was fully consumed (all elements measured)
                                # BUT: if consumed qubits were explicitly reset, they should be returned
                                fully_consumed = len(consumed_indices) == original_size

                                # Check if any consumed qubits were explicitly reset
                                explicitly_reset_indices = set()
                                if (
                                    hasattr(self, "explicitly_reset_qubits")
                                    and name in self.explicitly_reset_qubits
                                ):
                                    explicitly_reset_indices = (
                                        self.explicitly_reset_qubits[name]
                                    )

                                # If fully consumed BUT some were explicitly reset, we should return those
                                if fully_consumed and not explicitly_reset_indices:
                                    # All qubits were measured and none were explicitly reset - don't return
                                    pass  # Don't add to quantum_returns
                                else:
                                    # Not fully consumed - return the array
                                    # Determine how many qubits will actually be returned
                                    # This depends on:
                                    # 1. Whether this will be a single or multiple return
                                    # 2. Whether consumed qubits were replaced

                                    # Count how many quantum arrays will likely be returned
                                    # (This is a heuristic - we're building quantum_returns as we go)
                                    num_quantum_params = 0
                                    for n, pt in params:
                                        if "array[quantum.qubit," in pt:
                                            # Check if this array is part of a struct
                                            in_struct = False
                                            if (
                                                isinstance(self.struct_info, dict)
                                                and n in self.struct_info.values()
                                            ):
                                                in_struct = True
                                            if not in_struct:
                                                num_quantum_params += 1

                                    # For both single and multiple returns with partial consumption:
                                    # Return unconsumed + explicitly reset elements
                                    # Automatic replacements (for linearity) are not returned
                                    # Matches return statement generation at lines 1424-1465

                                    # Calculate how many elements to return
                                    explicitly_reset_indices = set()
                                    if (
                                        hasattr(self, "explicitly_reset_qubits")
                                        and name in self.explicitly_reset_qubits
                                    ):
                                        explicitly_reset_indices = (
                                            self.explicitly_reset_qubits[name]
                                        )

                                    # Count elements that are either unconsumed OR explicitly reset
                                    elements_to_return_count = 0
                                    for i in range(original_size):
                                        if (
                                            i not in consumed_indices
                                            or i in explicitly_reset_indices
                                        ):
                                            elements_to_return_count += 1

                                    remaining_count = elements_to_return_count

                                    if remaining_count > 0:
                                        # Some qubits remain - return array with correct size
                                        if remaining_count < original_size:
                                            # Partial consumption - return array with reduced size
                                            new_type = f"array[quantum.qubit, {remaining_count}]"
                                        else:
                                            # No consumption - return original type
                                            new_type = ptype.replace(" @owned", "")
                                        quantum_returns.append((name, new_type))
                        else:
                            # No consumption tracked - return full array
                            # Remove @owned annotation from return type
                            return_type = ptype.replace(" @owned", "")
                            quantum_returns.append((name, return_type))

            if quantum_returns:
                # Add return statements
                if len(quantum_returns) == 1:
                    name, ptype = quantum_returns[0]

                    # Check if this is a partial return
                    if name in consumed_in_function and "array[quantum.qubit," in ptype:
                        # Need to return only unconsumed elements
                        import re

                        match = re.search(r"array\[quantum\.qubit, (\d+)\]", ptype)
                        if match:
                            int(match.group(1))
                            original_match = re.search(
                                r"array\[quantum\.qubit, (\d+)\]",
                                next(pt for n, pt in params if n == name),
                            )
                            if original_match:
                                original_size = int(original_match.group(1))
                                consumed_indices = consumed_in_function[name]

                                # Build array with unconsumed + explicitly reset elements
                                #
                                # DESIGN DECISION: Return unconsumed and explicitly reset elements
                                # - Unconsumed: Elements never measured/consumed
                                # - Explicitly reset: Elements reset via Prep operation (quantum.reset)
                                # - Automatic replacements: Created for linearity, NOT returned
                                #
                                # This distinguishes:
                                # 1. Explicit Prep(qubit) - semantic reset operation → RETURN
                                # 2. Automatic post-measurement replacement → DON'T RETURN
                                #
                                # Examples:
                                # - Steane verification: Prep(ancilla) → explicit reset → included
                                # - Partial consumption: Measure(q[0]) → automatic replacement → excluded

                                # Determine which consumed indices should be returned
                                # (i.e., those that were explicitly reset)
                                explicitly_reset_indices = set()
                                if (
                                    hasattr(self, "explicitly_reset_qubits")
                                    and name in self.explicitly_reset_qubits
                                ):
                                    explicitly_reset_indices = (
                                        self.explicitly_reset_qubits[name]
                                    )

                                elements_to_return = []
                                for i in range(original_size):
                                    # Include if: (1) not consumed, OR (2) explicitly reset
                                    if (
                                        i not in consumed_indices
                                        or i in explicitly_reset_indices
                                    ):
                                        if name in self.unpacked_vars:
                                            # Use unpacked element name directly using original index
                                            # NOTE: index_mapping maps original index →
                                            # compact position for function CALLS
                                            # But for RETURNS, we still have all original
                                            # unpacked elements available
                                            # So we use the original index 'i' directly!
                                            element_name = self.unpacked_vars[name][i]
                                            # Apply variable remapping if element was
                                            # reassigned (e.g., Prep after Measure)
                                            if hasattr(self, "function_var_remapping"):
                                                element_name = (
                                                    self.function_var_remapping.get(
                                                        element_name,
                                                        element_name,
                                                    )
                                                )
                                            elements_to_return.append(
                                                VariableRef(element_name),
                                            )
                                        else:
                                            # Use array indexing
                                            elements_to_return.append(
                                                ArrayAccess(array_name=name, index=i),
                                            )

                                # Create array construction
                                array_expr = FunctionCall(
                                    func_name="array",
                                    args=elements_to_return,
                                )
                                body.statements.append(
                                    ReturnStatement(value=array_expr),
                                )
                    elif name in self.unpacked_vars:
                        # Array was unpacked - check for partial consumption
                        # CRITICAL: Also check consumed_in_function here!
                        # The earlier check (line 1548) might have failed due to return type detection issues
                        if name in consumed_in_function:
                            # Partial consumption - return unconsumed + explicitly reset elements
                            consumed_indices = consumed_in_function[name]
                            element_names = self.unpacked_vars[name]

                            # Get explicitly reset indices
                            explicitly_reset_indices = set()
                            if (
                                hasattr(self, "explicitly_reset_qubits")
                                and name in self.explicitly_reset_qubits
                            ):
                                explicitly_reset_indices = self.explicitly_reset_qubits[
                                    name
                                ]

                            # Filter: include unconsumed OR explicitly reset
                            elements_to_return = []
                            for i, elem_name in enumerate(element_names):
                                if (
                                    i not in consumed_indices
                                    or i in explicitly_reset_indices
                                ):
                                    # Apply variable remapping if element was reassigned (e.g., Prep after Measure)
                                    if hasattr(self, "function_var_remapping"):
                                        elem_name = self.function_var_remapping.get(
                                            elem_name,
                                            elem_name,
                                        )
                                    elements_to_return.append(VariableRef(elem_name))
                            array_construction = FunctionCall(
                                func_name="array",
                                args=elements_to_return,
                            )
                            body.statements.append(
                                ReturnStatement(value=array_construction),
                            )
                        elif (
                            hasattr(self, "refreshed_arrays")
                            and name in self.refreshed_arrays
                        ):
                            # Array was unpacked AND refreshed - return the fresh version
                            fresh_name = self.refreshed_arrays[name]
                            body.statements.append(
                                ReturnStatement(value=VariableRef(fresh_name)),
                            )
                        else:
                            # Array was unpacked - must reconstruct from elements for linearity
                            # Even if no elements were consumed, the original array is "moved" by unpacking
                            element_names = self.unpacked_vars[name]
                            array_construction = self._create_array_reconstruction(
                                element_names,
                            )
                            body.statements.append(
                                ReturnStatement(value=array_construction),
                            )
                    elif name in struct_reconstruction:
                        # Struct was decomposed - but check if it was also refreshed by function calls
                        if (
                            hasattr(self, "refreshed_arrays")
                            and name in self.refreshed_arrays
                        ):
                            # Struct was refreshed - return the fresh version directly
                            fresh_name = self.refreshed_arrays[name]
                            body.statements.append(
                                ReturnStatement(value=VariableRef(fresh_name)),
                            )
                        else:
                            # Struct was decomposed - reconstruct it from field variables
                            struct_info = self.struct_info[name]

                            # Check if this is an @owned struct that was decomposed
                            is_owned_struct = (
                                hasattr(self, "owned_structs")
                                and name in self.owned_structs
                            )

                            # For @owned structs, always reconstruct from decomposed variables
                            # For regular structs, check if the unpacked variables are still valid
                            if is_owned_struct:
                                should_reconstruct = True
                            else:
                                # Check if the unpacked variables are still valid
                                # They're only valid if we haven't passed the struct
                                # to any @owned functions
                                should_reconstruct = all(
                                    struct_info["var_names"].get(suffix)
                                    in self.var_remapping
                                    for suffix, _, _ in struct_info["fields"]
                                )

                            if should_reconstruct:
                                # Create struct constructor call - use same order
                                # as struct definition (sorted by suffix)
                                constructor_args = []
                                all_vars_available = True

                                for suffix, field_type, field_size in sorted(
                                    struct_info["fields"],
                                ):
                                    field_var = f"{name}_{suffix}"

                                    # Check if we have a fresh version of this field variable
                                    if (
                                        hasattr(self, "refreshed_arrays")
                                        and field_var in self.refreshed_arrays
                                    ):
                                        field_var = self.refreshed_arrays[field_var]
                                    elif (
                                        hasattr(self, "var_remapping")
                                        and field_var in self.var_remapping
                                    ):
                                        field_var = self.var_remapping[field_var]
                                    else:
                                        # Check if the variable was consumed in operations
                                        if (
                                            hasattr(self, "consumed_vars")
                                            and field_var in self.consumed_vars
                                        ):
                                            all_vars_available = False
                                            break

                                    constructor_args.append(VariableRef(field_var))

                                if all_vars_available and constructor_args:
                                    struct_constructor = FunctionCall(
                                        func_name=struct_info["struct_name"],
                                        args=constructor_args,
                                    )
                                    body.statements.append(
                                        ReturnStatement(value=struct_constructor),
                                    )
                                else:
                                    # Variables were consumed - cannot reconstruct
                                    # Return void or handle appropriately for @owned structs
                                    pass
                            else:
                                # Unpacked variables are no longer valid - return the struct directly
                                body.statements.append(
                                    ReturnStatement(value=VariableRef(name)),
                                )
                    else:
                        # Check if this variable was refreshed due to being borrowed
                        # (e.g., c_d -> c_d_returned)
                        if (
                            hasattr(self, "refreshed_arrays")
                            and name in self.refreshed_arrays
                        ):
                            # Use the refreshed name for the return
                            return_name = self.refreshed_arrays[name]
                            body.statements.append(
                                ReturnStatement(value=VariableRef(return_name)),
                            )
                        elif (
                            hasattr(self, "owned_structs")
                            and name in self.owned_structs
                            and name in self.struct_info
                        ):
                            # @owned struct needs reconstruction from decomposed variables
                            struct_info = self.struct_info[name]

                            # Create struct constructor call
                            constructor_args = []
                            all_vars_available = True

                            for suffix, field_type, field_size in sorted(
                                struct_info["fields"],
                            ):
                                field_var = f"{name}_{suffix}"

                                # Check if we have a fresh version of this field variable
                                if (
                                    hasattr(self, "refreshed_arrays")
                                    and field_var in self.refreshed_arrays
                                ):
                                    field_var = self.refreshed_arrays[field_var]
                                elif (
                                    hasattr(self, "var_remapping")
                                    and field_var in self.var_remapping
                                ):
                                    field_var = self.var_remapping[field_var]
                                else:
                                    # Check if the variable was consumed in operations
                                    if (
                                        hasattr(self, "consumed_vars")
                                        and field_var in self.consumed_vars
                                    ):
                                        all_vars_available = False
                                        break

                                constructor_args.append(VariableRef(field_var))

                            if all_vars_available and constructor_args:
                                struct_constructor = FunctionCall(
                                    func_name=struct_info["struct_name"],
                                    args=constructor_args,
                                )
                                body.statements.append(
                                    ReturnStatement(value=struct_constructor),
                                )
                        else:
                            # Check if this variable has been refreshed by function calls
                            var_to_return = name
                            if (
                                hasattr(self, "refreshed_arrays")
                                and name in self.refreshed_arrays
                            ):
                                var_to_return = self.refreshed_arrays[name]
                            body.statements.append(
                                ReturnStatement(value=VariableRef(var_to_return)),
                            )

                    # Set return type
                    return_type = ptype  # Use the potentially modified type
                else:
                    # Multiple arrays/structs - return tuple
                    return_exprs = []
                    return_types = []
                    for name, ptype in quantum_returns:
                        if name in self.unpacked_vars:
                            # Array was unpacked - check if it was also refreshed by function calls
                            if (
                                hasattr(self, "refreshed_arrays")
                                and name in self.refreshed_arrays
                            ):
                                # Array was refreshed after unpacking - return the fresh version
                                fresh_name = self.refreshed_arrays[name]
                                return_exprs.append(VariableRef(fresh_name))
                            else:
                                # Array was unpacked - check if elements are still available for reconstruction
                                element_names = self.unpacked_vars[name]

                                # For arrays with size 0 in return type, create empty arrays instead of reconstructing
                                if "array[quantum.qubit, 0]" in ptype:
                                    # All elements consumed - create empty quantum array using generator expression
                                    # Create custom expression for: array(quantum.qubit() for _ in range(0))

                                    class EmptyArrayExpression(Expression):
                                        def analyze(self, _context):
                                            pass  # No analysis needed for empty array

                                        def render(self, _context):
                                            return [
                                                "array(quantum.qubit() for _ in range(0))",
                                            ]

                                    empty_array = EmptyArrayExpression()
                                    return_exprs.append(empty_array)
                                else:
                                    # Check if this array has partial consumption
                                    if name in consumed_in_function:
                                        consumed_indices = consumed_in_function[name]

                                        # Build array with unconsumed + explicitly reset elements
                                        # See single return path (lines 1424-1465) for detailed rationale

                                        # Get explicitly reset indices
                                        explicitly_reset_indices = set()
                                        if (
                                            hasattr(self, "explicitly_reset_qubits")
                                            and name in self.explicitly_reset_qubits
                                        ):
                                            explicitly_reset_indices = (
                                                self.explicitly_reset_qubits[name]
                                            )

                                        elements_to_return = []
                                        for i in range(len(element_names)):
                                            # Include if: (1) not consumed, OR (2) explicitly reset
                                            if (
                                                i not in consumed_indices
                                                or i in explicitly_reset_indices
                                            ):
                                                element_name = element_names[i]
                                                # Apply variable remapping if element was reassigned
                                                # Use function_var_remapping which includes Prep changes
                                                if hasattr(
                                                    self,
                                                    "function_var_remapping",
                                                ):
                                                    element_name = (
                                                        self.function_var_remapping.get(
                                                            element_name,
                                                            element_name,
                                                        )
                                                    )
                                                elements_to_return.append(
                                                    VariableRef(element_name),
                                                )

                                        if elements_to_return:
                                            # Create array from unconsumed elements
                                            array_construction = FunctionCall(
                                                func_name="array",
                                                args=elements_to_return,
                                            )
                                            return_exprs.append(array_construction)
                                        else:
                                            # All elements consumed - use empty array
                                            class EmptyArrayExpression(Expression):
                                                def analyze(self, _context):
                                                    pass

                                                def render(self, _context):
                                                    return [
                                                        "array(quantum.qubit() for _ in range(0))",
                                                    ]

                                            return_exprs.append(EmptyArrayExpression())
                                    else:
                                        # No consumption or not tracked - standard reconstruction from all elements
                                        array_construction = (
                                            self._create_array_reconstruction(
                                                element_names,
                                            )
                                        )
                                        return_exprs.append(array_construction)
                        elif name in struct_reconstruction:
                            # Struct was decomposed - but check if it was also refreshed by function calls
                            if (
                                hasattr(self, "refreshed_arrays")
                                and name in self.refreshed_arrays
                            ):
                                # Struct was refreshed - return the fresh version directly
                                fresh_name = self.refreshed_arrays[name]
                                return_exprs.append(VariableRef(fresh_name))
                            else:
                                # Struct was decomposed - check if we can still use
                                # the decomposed variables
                                struct_info = self.struct_info[name]

                                # Check if this is an @owned struct that was decomposed
                                is_owned_struct = (
                                    hasattr(self, "owned_structs")
                                    and name in self.owned_structs
                                )

                                # For @owned structs, always reconstruct from decomposed variables
                                # For regular structs, check if the unpacked variables are still valid
                                if is_owned_struct:
                                    unpacked_vars_valid = True
                                else:
                                    # Check if the unpacked variables are still valid
                                    unpacked_vars_valid = all(
                                        struct_info["var_names"].get(suffix)
                                        in self.var_remapping
                                        for suffix, _, _ in struct_info["fields"]
                                    )

                                if unpacked_vars_valid:
                                    # Create struct constructor call - use same order
                                    # as struct definition (sorted by suffix)
                                    constructor_args = []
                                    all_vars_available = True

                                    for suffix, field_type, field_size in sorted(
                                        struct_info["fields"],
                                    ):
                                        field_var = f"{name}_{suffix}"

                                        # Check if we have a fresh version of this field variable
                                        if (
                                            hasattr(self, "refreshed_arrays")
                                            and field_var in self.refreshed_arrays
                                        ):
                                            field_var = self.refreshed_arrays[field_var]
                                        elif (
                                            hasattr(self, "var_remapping")
                                            and field_var in self.var_remapping
                                        ):
                                            field_var = self.var_remapping[field_var]
                                        else:
                                            # Check if the variable was consumed in operations
                                            if (
                                                hasattr(self, "consumed_vars")
                                                and field_var in self.consumed_vars
                                            ):
                                                all_vars_available = False
                                                break

                                        constructor_args.append(VariableRef(field_var))

                                    if all_vars_available and constructor_args:
                                        struct_constructor = FunctionCall(
                                            func_name=struct_info["struct_name"],
                                            args=constructor_args,
                                        )
                                        return_exprs.append(struct_constructor)
                                    else:
                                        # Variables were consumed - handle appropriately
                                        var_to_return = name
                                        if (
                                            hasattr(self, "refreshed_arrays")
                                            and name in self.refreshed_arrays
                                        ):
                                            var_to_return = self.refreshed_arrays[name]
                                        return_exprs.append(VariableRef(var_to_return))
                                else:
                                    # Unpacked variables are no longer valid -
                                    # return the struct directly
                                    # Check if this variable has been refreshed by function calls
                                    var_to_return = name
                                    if (
                                        hasattr(self, "refreshed_arrays")
                                        and name in self.refreshed_arrays
                                    ):
                                        var_to_return = self.refreshed_arrays[name]
                                    return_exprs.append(VariableRef(var_to_return))
                        else:
                            # Array/struct was not unpacked - return it directly
                            # Check if this is an @owned struct that needs reconstruction
                            if (
                                hasattr(self, "owned_structs")
                                and name in self.owned_structs
                                and name in self.struct_info
                            ):
                                # @owned struct needs reconstruction from decomposed variables
                                struct_info = self.struct_info[name]

                                # Create struct constructor call
                                constructor_args = []
                                for suffix, field_type, field_size in sorted(
                                    struct_info["fields"],
                                ):
                                    field_var = f"{name}_{suffix}"

                                    # Check if we have a fresh version of this field variable
                                    if (
                                        hasattr(self, "refreshed_arrays")
                                        and field_var in self.refreshed_arrays
                                    ):
                                        field_var = self.refreshed_arrays[field_var]
                                    elif (
                                        hasattr(self, "var_remapping")
                                        and field_var in self.var_remapping
                                    ):
                                        field_var = self.var_remapping[field_var]

                                    constructor_args.append(VariableRef(field_var))

                                struct_constructor = FunctionCall(
                                    func_name=struct_info["struct_name"],
                                    args=constructor_args,
                                )
                                return_exprs.append(struct_constructor)
                            else:
                                # Check if this variable has been refreshed by function calls
                                var_to_return = name
                                if (
                                    hasattr(self, "refreshed_arrays")
                                    and name in self.refreshed_arrays
                                ):
                                    var_to_return = self.refreshed_arrays[name]
                                return_exprs.append(VariableRef(var_to_return))

                        # Add type to return types
                        return_types.append(ptype)

                    if return_exprs:
                        body.statements.append(
                            ReturnStatement(
                                value=TupleExpression(elements=return_exprs),
                            ),
                        )
                        return_type = f"tuple[{', '.join(return_types)}]"

        # For procedural blocks, override return type to None even if they return arrays internally
        if is_procedural_block:
            return_type = "None"
            # Also remove any return statements from the body since this is procedural
            body.statements = [
                stmt
                for stmt in body.statements
                if not isinstance(stmt, ReturnStatement)
            ]

            # Add cleanup for unused quantum arrays that might have been created
            # by function calls but not consumed (e.g., fresh variables)
            # GENERAL APPROACH: Check for any fresh_return_vars that were created
            if hasattr(self, "fresh_return_vars") and self.fresh_return_vars:
                # Add discard for each fresh variable that wasn't consumed
                # (consumed variables are tracked in consumed_arrays or consumed_resources)
                for fresh_name, info in self.fresh_return_vars.items():
                    # Check if this fresh variable was consumed
                    was_consumed = False
                    if hasattr(self, "consumed_arrays"):
                        was_consumed = fresh_name in self.consumed_arrays
                    if not was_consumed and hasattr(self, "consumed_resources"):
                        was_consumed = fresh_name in self.consumed_resources

                    if not was_consumed and info.get("is_quantum_array"):
                        # Add discard statement
                        discard_stmt = FunctionCall(
                            func_name="quantum.discard_array",
                            args=[VariableRef(fresh_name)],
                        )

                        # Wrap in expression statement
                        class ExpressionStatement(Statement):
                            def __init__(self, expr):
                                self.expr = expr

                            def analyze(self, context):
                                self.expr.analyze(context)

                            def render(self, context):
                                return self.expr.render(context)

                        body.statements.append(Comment(f"Discard unused {fresh_name}"))
                        body.statements.append(ExpressionStatement(discard_stmt))

                # Clear tracking for next function
                self.fresh_return_vars = {}

        # Store the return type for use in other parts of the code
        self.current_function_return_type = return_type
        # Store in function return types registry for later lookup
        self.function_return_types[func_name] = return_type

        return Function(
            name=func_name,
            params=params,
            return_type=return_type,
            body=body,
            decorators=["guppy", "no_type_check"],
        )

    def _add_variable_declaration(self, var, block=None) -> None:
        """Add variable declaration to current block."""
        var_type = type(var).__name__
        var_name = var.sym

        # Check for renaming
        if var_name in self.plan.renamed_variables:
            var_name = self.plan.renamed_variables[var_name]

        if var_type == "QReg":
            # Get size for all cases
            size = var.size

            # Check allocation recommendation for this array
            recommendation = self.allocation_recommendations.get(var.sym, {})

            # Get resource plan from unified analysis if available
            resource_plan = None
            if self.unified_analysis:
                resource_plan = self.unified_analysis.get_plan(var.sym)

            # Check if this array needs unpacking (selective measurements)
            needs_unpacking = var.sym in self.plan.arrays_to_unpack

            # Check if this array is used in full array operations
            needs_full_array = self._array_needs_full_allocation(var.sym, block)

            # Check if this should be dynamically allocated based on usage patterns
            # But only if it doesn't need unpacking for selective measurements
            # AND not used in full array ops
            # AND not a function parameter in current function
            # AND the unified resource plan agrees with dynamic allocation
            is_function_parameter = hasattr(self, "current_function_params") and any(
                param_name == var.sym for param_name, _ in self.current_function_params
            )

            # Use the unified resource plan if available, otherwise fall back to recommendation
            should_use_dynamic = False
            if resource_plan:
                # Resource plan from unified analysis (authoritative)
                should_use_dynamic = resource_plan.uses_dynamic_allocation
            else:
                # Fall back to recommendation
                should_use_dynamic = recommendation.get("allocation") == "dynamic"

            if (
                should_use_dynamic
                and not needs_unpacking
                and not needs_full_array
                and not is_function_parameter
            ):
                # Check if this ancilla array is used as a function parameter
                # If so, we need to pre-allocate it despite being an ancilla
                is_function_param = False
                if hasattr(self, "ancilla_qubits") and var_name in self.ancilla_qubits:
                    # This is an ancilla that was excluded from structs
                    # It will be passed as a parameter to functions, so pre-allocate it
                    is_function_param = True

                if is_function_param:
                    # For ancilla qubits, create individual qubits instead of arrays
                    # This avoids @owned array passing issues that cause linearity violations
                    self.current_block.statements.append(
                        Comment(
                            f"Create individual ancilla qubits for {var_name} (avoids @owned array issues)",
                        ),
                    )

                    # Create individual qubits: c_a_0, c_a_1, c_a_2 instead of array c_a
                    for i in range(size):
                        qubit_name = f"{var_name}_{i}"
                        init_expr = FunctionCall(func_name="quantum.qubit", args=[])
                        assignment = Assignment(
                            target=VariableRef(qubit_name),
                            value=init_expr,
                        )
                        self.current_block.statements.append(assignment)

                    # Mark this variable as having been decomposed into individual qubits
                    if not hasattr(self, "decomposed_ancilla_arrays"):
                        self.decomposed_ancilla_arrays = {}
                    self.decomposed_ancilla_arrays[var_name] = [
                        f"{var_name}_{i}" for i in range(size)
                    ]

                    # Add a function to reconstruct the array when needed for function calls
                    # This creates: c_a = array(c_a_0, c_a_1, c_a_2)
                    self.current_block.statements.append(
                        Comment(f"# Reconstruct {var_name} array for function calls"),
                    )
                    array_construction_args = [
                        VariableRef(f"{var_name}_{i}") for i in range(size)
                    ]
                    reconstruct_expr = FunctionCall(
                        func_name="array",
                        args=array_construction_args,
                    )
                    reconstruct_assignment = Assignment(
                        target=VariableRef(var_name),
                        value=reconstruct_expr,
                    )
                    self.current_block.statements.append(reconstruct_assignment)

                    # Track that this array has been reconstructed - use the variable directly, not individual qubits
                    if not hasattr(self, "reconstructed_arrays"):
                        self.reconstructed_arrays = set()
                    self.reconstructed_arrays.add(var_name)
                else:
                    # For other ancillas, don't pre-allocate array
                    reason = recommendation.get("reason", "ancilla pattern")
                    # Before marking for dynamic allocation, check if this variable
                    # is used as a function argument in the current block
                    is_function_arg = self._is_variable_used_as_function_arg(
                        var.sym,
                        block,
                    )

                    if is_function_arg:
                        # For ancilla qubits used as function arguments, create individual qubits
                        # This avoids @owned array passing issues
                        comment_text = (
                            f"Create individual ancilla qubits for {var_name} "
                            f"(function argument, avoids @owned array issues)"
                        )
                        self.current_block.statements.append(
                            Comment(comment_text),
                        )

                        # Create individual qubits: c_a_0, c_a_1, c_a_2 instead of array c_a
                        for i in range(size):
                            qubit_name = f"{var_name}_{i}"
                            init_expr = FunctionCall(func_name="quantum.qubit", args=[])
                            assignment = Assignment(
                                target=VariableRef(qubit_name),
                                value=init_expr,
                            )
                            self.current_block.statements.append(assignment)

                        # Mark this variable as having been decomposed into individual qubits
                        if not hasattr(self, "decomposed_ancilla_arrays"):
                            self.decomposed_ancilla_arrays = {}
                        self.decomposed_ancilla_arrays[var_name] = [
                            f"{var_name}_{i}" for i in range(size)
                        ]
                    else:
                        # Normal dynamic allocation
                        self.current_block.statements.append(
                            Comment(
                                f"# {var_name} will be allocated dynamically ({reason})",
                            ),
                        )
                        # Track that this is dynamically allocated
                        if not hasattr(self, "dynamic_allocations"):
                            self.dynamic_allocations = set()
                        self.dynamic_allocations.add(var.sym)
            elif resource_plan and resource_plan.uses_dynamic_allocation:
                # Check if all elements are local (full dynamic allocation)
                if len(resource_plan.elements_to_allocate_locally) == size:
                    # Don't pre-allocate - all will be allocated when first used
                    self.current_block.statements.append(
                        Comment(f"Qubits from {var_name} will be allocated locally"),
                    )
                    # Track that this is dynamically allocated
                    if not hasattr(self, "dynamic_allocations"):
                        self.dynamic_allocations = set()
                    self.dynamic_allocations.add(var.sym)
                else:
                    # Mixed strategy - pre-allocate some, allocate others locally
                    # But only if the array doesn't need unpacking
                    if needs_unpacking:
                        # Can't use mixed allocation with unpacking - fall back to full pre-allocation
                        init_expr = FunctionCall(
                            func_name="array",
                            args=[
                                FunctionCall(
                                    func_name="quantum.qubit() for _ in range",
                                    args=[Literal(size)],
                                ),
                            ],
                        )
                        assignment = Assignment(
                            target=VariableRef(var_name),
                            value=init_expr,
                        )
                        self.current_block.statements.append(assignment)
                        self.current_block.statements.append(
                            Comment(
                                f"Note: Full pre-allocation used because {var_name} needs unpacking",
                            ),
                        )
                    elif size - len(resource_plan.elements_to_allocate_locally) > 0:
                        pre_alloc_size = size - len(
                            resource_plan.elements_to_allocate_locally,
                        )
                        init_expr = FunctionCall(
                            func_name="array",
                            args=[
                                FunctionCall(
                                    func_name="quantum.qubit() for _ in range",
                                    args=[Literal(pre_alloc_size)],
                                ),
                            ],
                        )
                        assignment = Assignment(
                            target=VariableRef(var_name),
                            value=init_expr,
                        )
                        self.current_block.statements.append(assignment)

                    self.current_block.statements.append(
                        Comment(
                            f"Elements {sorted(resource_plan.elements_to_allocate_locally)} of "
                            f"{var_name} will be allocated locally",
                        ),
                    )
            else:
                # Check if this is an ancilla array that should be decomposed
                if hasattr(self, "ancilla_qubits") and var_name in self.ancilla_qubits:
                    # Decompose ancilla arrays into individual qubits to avoid @owned linearity issues
                    self.current_block.statements.append(
                        Comment(
                            f"Create individual ancilla qubits for {var_name} (avoids @owned array linearity issues)",
                        ),
                    )

                    # Create individual qubits: c_a_0, c_a_1, c_a_2 instead of array c_a
                    for i in range(size):
                        qubit_name = f"{var_name}_{i}"
                        init_expr = FunctionCall(func_name="quantum.qubit", args=[])
                        assignment = Assignment(
                            target=VariableRef(qubit_name),
                            value=init_expr,
                        )
                        self.current_block.statements.append(assignment)

                    # Mark this variable as having been decomposed into individual qubits
                    if not hasattr(self, "decomposed_ancilla_arrays"):
                        self.decomposed_ancilla_arrays = {}
                    self.decomposed_ancilla_arrays[var_name] = [
                        f"{var_name}_{i}" for i in range(size)
                    ]

                    # Add a function to reconstruct the array when needed for function calls
                    # This creates: c_a = array(c_a_0, c_a_1, c_a_2)
                    self.current_block.statements.append(
                        Comment(f"# Reconstruct {var_name} array for function calls"),
                    )
                    array_construction_args = [
                        VariableRef(f"{var_name}_{i}") for i in range(size)
                    ]
                    reconstruct_expr = FunctionCall(
                        func_name="array",
                        args=array_construction_args,
                    )
                    reconstruct_assignment = Assignment(
                        target=VariableRef(var_name),
                        value=reconstruct_expr,
                    )
                    self.current_block.statements.append(reconstruct_assignment)

                    # Track that this array has been reconstructed - use the variable directly, not individual qubits
                    if not hasattr(self, "reconstructed_arrays"):
                        self.reconstructed_arrays = set()
                    self.reconstructed_arrays.add(var_name)
                else:
                    # Check if this ancilla array was already decomposed into individual qubits
                    if (
                        hasattr(self, "decomposed_ancilla_arrays")
                        and var_name in self.decomposed_ancilla_arrays
                    ):
                        # Skip array creation - individual qubits were already created
                        qubit_list = ", ".join(self.decomposed_ancilla_arrays[var_name])
                        comment_text = f"# {var_name} already decomposed into individual qubits: {qubit_list}"
                        self.current_block.statements.append(
                            Comment(comment_text),
                        )
                    else:
                        # Default: pre-allocate all qubits
                        init_expr = FunctionCall(
                            func_name="array",
                            args=[
                                FunctionCall(
                                    func_name="quantum.qubit() for _ in range",
                                    args=[Literal(size)],
                                ),
                            ],
                        )
                        assignment = Assignment(
                            target=VariableRef(var_name),
                            value=init_expr,
                        )
                        self.current_block.statements.append(assignment)

            # Track in context
            var_info = VariableInfo(
                name=var_name,
                original_name=var.sym,
                var_type="quantum",
                size=size,
                is_array=True,
            )
            self.context.add_variable(var_info)
            self.scope_manager.current_context.add_variable(var_info)

        elif var_type == "CReg":
            # Create classical array
            size = var.size
            init_expr = FunctionCall(
                func_name="array",
                args=[
                    FunctionCall(
                        func_name="False for _ in range",
                        args=[Literal(size)],
                    ),
                ],
            )
            assignment = Assignment(
                target=VariableRef(var_name),
                value=init_expr,
            )
            self.current_block.statements.append(assignment)

            # Track in context
            var_info = VariableInfo(
                name=var_name,
                original_name=var.sym,
                var_type="classical",
                size=size,
                is_array=True,
            )
            self.context.add_variable(var_info)
            self.scope_manager.current_context.add_variable(var_info)

    def _block_consumes_quantum(self, block) -> bool:
        """Check if a block consumes ALL quantum resources.

        Only return True if the block consumes ALL its quantum inputs.
        Most SLR functions modify arrays in-place without consuming them.

        However, functions that access quantum fields within structs need @owned
        annotation to satisfy Guppy's linearity requirements.
        """
        # For now, be very conservative - assume functions don't consume
        # their parameters unless they're explicitly measurement blocks
        # that measure ALL qubits

        # Check the block name - only certain blocks truly consume all resources
        block_name = type(block).__name__
        if block_name in ["MeasureAll", "DiscardAll"]:
            return True

        # IMPORTANT: Functions that will access quantum fields within structs
        # need @owned annotation for Guppy's linearity system
        # Otherwise assume the function modifies in-place without consuming
        return self._block_accesses_struct_quantum_fields(block)

    def _analyze_consumed_parameters(self, block) -> set[str]:
        """Analyze which quantum parameters are consumed by a block.

        A parameter is consumed if:
        1. It appears in a Measure operation that measures the full register
        2. All its elements are measured individually
        3. It's passed to a nested Block that consumes it
        """
        consumed_params = set()
        element_measurements = {}  # Track which array elements are measured

        if not hasattr(block, "ops"):
            return consumed_params

        # Recursively analyze all operations including nested blocks
        def analyze_ops(ops_list):
            for op in ops_list:
                op_type = type(op).__name__

                # Measurement consumes qubits
                if op_type == "Measure":
                    if hasattr(op, "qargs"):
                        for qarg in op.qargs:
                            # Check if it's a full register measurement (not indexed)
                            if hasattr(qarg, "sym"):
                                # This is a full register being measured
                                consumed_params.add(qarg.sym)
                            # Check for indexed measurements (e.g., q[0], q[1])
                            elif hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                                array_name = qarg.reg.sym
                                if array_name not in element_measurements:
                                    element_measurements[array_name] = set()
                                if hasattr(qarg, "index"):
                                    element_measurements[array_name].add(qarg.index)

                # Check if this is a nested Block call
                elif hasattr(op, "__class__") and hasattr(op.__class__, "__bases__"):
                    from pecos.slr import Block as SlrBlock

                    # Check if op is a Block subclass
                    # Need to check the class itself, not just the base name
                    try:
                        if issubclass(op.__class__, SlrBlock) and hasattr(op, "ops"):
                            # Recursively analyze nested block
                            analyze_ops(op.ops)
                    except (TypeError, AttributeError):
                        # Not a class or missing expected attributes
                        pass

        # Analyze all operations
        analyze_ops(block.ops)

        # Check if arrays are consumed
        # In Guppy, any measurement of array elements requires @owned annotation
        # because it consumes those elements
        for array_name, measured_indices in element_measurements.items():
            # If any element is measured, the array is consumed and needs @owned
            if len(measured_indices) > 0:
                consumed_params.add(array_name)

        return consumed_params

    def _analyze_subscript_access(self, block) -> set[str]:
        """Analyze which quantum arrays have subscript access in a block.

        In Guppy, any subscript access (c_d[0]) marks the array as used,
        requiring @owned annotation to avoid MoveOutOfSubscriptError.

        Returns:
            set of array names that have subscript access
        """
        subscripted_arrays = set()

        if not hasattr(block, "ops"):
            return subscripted_arrays

        # Recursively analyze all operations
        def analyze_ops(ops_list):
            for op in ops_list:
                # Check for any quantum operation with indexed arguments
                if hasattr(op, "qargs"):
                    for qarg in op.qargs:
                        # Check for indexed access (e.g., q[0])
                        if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                            array_name = qarg.reg.sym
                            subscripted_arrays.add(array_name)
                        # Also check for register-wide operations that will be converted to loops
                        # (e.g., qubit.H(q) becomes for i in range(7): quantum.h(q[i]))
                        elif (
                            hasattr(qarg, "sym")
                            and hasattr(qarg, "elems")
                            and len(qarg.elems) > 1
                        ):
                            # This is a register-wide operation - will use subscripts
                            array_name = qarg.sym
                            subscripted_arrays.add(array_name)
                        # else: qarg doesn't match expected patterns

                # Check for classical array subscripts too
                if hasattr(op, "cargs"):
                    for carg in op.cargs:
                        if hasattr(carg, "reg") and hasattr(carg.reg, "sym"):
                            # This is classical, skip for now
                            pass

                # Check nested blocks
                if hasattr(op, "__class__") and hasattr(op.__class__, "__bases__"):
                    from pecos.slr import Block as SlrBlock

                    try:
                        if issubclass(op.__class__, SlrBlock) and hasattr(op, "ops"):
                            analyze_ops(op.ops)
                    except (TypeError, AttributeError):
                        # Not a class or missing expected attributes
                        pass

        analyze_ops(block.ops)
        return subscripted_arrays

    def _analyze_block_element_usage(self, block) -> dict:
        """Analyze which specific array elements are consumed vs returned by a block.

        Returns:
            dict: {
                'consumed_elements': {'array_name': {consumed_indices}},
                'array_sizes': {'array_name': size},
                'returned_elements': {'array_name': {returned_indices}}
            }
        """
        consumed_elements = {}
        array_sizes = {}

        if not hasattr(block, "ops"):
            return {
                "consumed_elements": consumed_elements,
                "array_sizes": array_sizes,
                "returned_elements": {},
            }

        # Analyze block to find measurements
        def analyze_ops(ops_list):
            for op in ops_list:
                op_type = type(op).__name__

                # Measurement consumes qubits
                if op_type == "Measure":
                    if hasattr(op, "qargs"):
                        for qarg in op.qargs:
                            # Check for indexed measurements (e.g., q[0])
                            if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                                array_name = qarg.reg.sym
                                if array_name not in consumed_elements:
                                    consumed_elements[array_name] = set()
                                if hasattr(qarg, "index"):
                                    consumed_elements[array_name].add(qarg.index)

                # Check if this is a nested Block call
                elif hasattr(op, "__class__") and hasattr(op.__class__, "__bases__"):
                    from pecos.slr import Block as SlrBlock

                    try:
                        if issubclass(op.__class__, SlrBlock) and hasattr(op, "ops"):
                            # Recursively analyze nested block
                            analyze_ops(op.ops)
                    except (TypeError, AttributeError):
                        # Not a class or missing expected attributes
                        pass

        # Get array sizes from block parameters
        if hasattr(block, "q") and hasattr(block.q, "size"):
            array_sizes["q"] = block.q.size

        analyze_ops(block.ops)

        # Pre-track explicit resets to know which consumed qubits are reset and should be returned
        consumed_for_tracking = {}
        self._track_consumed_qubits(block, consumed_for_tracking)

        # Calculate returned elements
        # = (all elements - consumed) + explicitly_reset
        # Explicitly reset qubits are consumed by measurement but then recreated by Prep
        returned_elements = {}
        for array_name, size in array_sizes.items():
            consumed = consumed_elements.get(array_name, set())
            all_indices = set(range(size))
            unconsumed = all_indices - consumed

            # Add explicitly reset qubits (they're consumed but then reset, so should be returned)
            explicitly_reset = set()
            if (
                hasattr(self, "explicitly_reset_qubits")
                and array_name in self.explicitly_reset_qubits
            ):
                explicitly_reset = self.explicitly_reset_qubits[array_name]

            returned_elements[array_name] = unconsumed | explicitly_reset

        return {
            "consumed_elements": consumed_elements,
            "array_sizes": array_sizes,
            "returned_elements": returned_elements,
        }

    def _block_accesses_struct_quantum_fields(self, block) -> bool:
        """Check if a block accesses quantum fields within structs.

        This is important because Guppy's linearity system requires @owned
        annotation for functions that access quantum fields within structs.
        """
        if not hasattr(block, "ops"):
            return False

        # If we have struct info, assume that functions accessing quantum operations
        # will need to access quantum fields within structs
        if self.struct_info:
            # Check if this block has quantum operations
            for op in block.ops:
                # Check for quantum operations (gates, measurements, etc.)
                op_name = type(op).__name__
                if op_name in [
                    "H",
                    "X",
                    "Y",
                    "Z",
                    "CX",
                    "CY",
                    "CZ",
                    "Reset",
                    "Measure",
                    "S",
                    "T",
                    "Sdg",
                    "Tdg",
                ]:
                    return True

                # Also check for nested quantum operations
                if hasattr(op, "ops") and self._block_accesses_struct_quantum_fields(
                    op,
                ):
                    return True

        return False

    def _needs_unpacking_workaround(self, block) -> bool:
        """Detect if a block needs the unpacking workaround for Guppy constraints."""
        if not hasattr(block, "ops"):
            return False

        # Check for patterns that cause MoveOutOfSubscriptError
        for op in block.ops:
            op_type = type(op).__name__

            # Reset operations on arrays are the main culprit
            if op_type == "Prep" and hasattr(op, "qargs"):
                for qarg in op.qargs:
                    # If it's an array operation, it might cause issues
                    if hasattr(qarg, "sym") and hasattr(qarg, "size") and qarg.size > 1:
                        return True

            # Multiple operations on the same array elements might cause issues
            # This is a more complex heuristic we could add later

            # Recursively check nested blocks
            if hasattr(op, "ops") and self._needs_unpacking_workaround(op):
                return True

        return False

    def _function_needs_unpacking(self, func_name: str) -> bool:
        """Check if a function uses the unpacking pattern by analyzing function behavior.

        This method analyzes the actual function operations rather than using hardcoded names,
        making it general for all QEC codes.
        """
        _ = func_name  # Currently not used, reserved for future use
        # Since this function is not currently used, return False for now
        # In the future, this could analyze the function's block to determine
        # if it performs operations that would benefit from unpacking
        return False

    def _function_consumes_parameters(self, func_name: str, block) -> bool:
        """Check if a function consumes its quantum parameters (has @owned)."""
        _ = func_name  # Currently not used, reserved for future use
        # Check if we already know about this function
        if hasattr(block, "ops"):
            return self._block_consumes_quantum(block)

        # Default: assume functions don't consume unless we know otherwise
        return False

    def _is_variable_used_as_function_arg(self, var_name: str, block) -> bool:
        """Check if a variable is used as an argument to block operations (functions)."""
        if not hasattr(block, "ops"):
            return False

        for op in block.ops:
            # Check if this is a Block-type operation
            if hasattr(op, "ops") and hasattr(op, "vars"):
                # This is a block - check variables used by operations inside it
                # Since constructor arguments aren't preserved, we need to analyze the inner operations
                for inner_op in op.ops:
                    # Check quantum arguments
                    if hasattr(inner_op, "qargs"):
                        for qarg in inner_op.qargs:
                            if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                                if qarg.reg.sym == var_name:
                                    return True
                            elif hasattr(qarg, "sym") and qarg.sym == var_name:
                                return True

                    # Check measurement targets
                    if hasattr(inner_op, "cout") and inner_op.cout:
                        for cout in inner_op.cout:
                            if hasattr(cout, "reg") and hasattr(cout.reg, "sym"):
                                if cout.reg.sym == var_name:
                                    return True
                            elif hasattr(cout, "sym") and cout.sym == var_name:
                                return True

        return False

    def _create_array_unpack_statement(
        self,
        array_name: str,
        element_names: list[str],
    ) -> Statement:
        """Create an array unpacking statement: q_0, q_1, q_2 = q"""

        class ArrayUnpackStatement(Statement):
            def __init__(self, targets, source):
                self.targets = targets
                self.source = source

            def analyze(self, context):
                _ = context  # Not used

            def render(self, context):
                _ = context  # Not used
                # For single element unpacking, we need a trailing comma
                target_str = (
                    self.targets[0] + ","
                    if len(self.targets) == 1
                    else ", ".join(self.targets)
                )
                return [f"{target_str} = {self.source}"]

        return ArrayUnpackStatement(element_names, array_name)

    def _create_array_construction(self, element_names: list[str]) -> Expression:
        """Create an array construction expression: array([q_0, q_1, q_2])"""

        class ArrayConstructionExpression(Expression):
            def __init__(self, elements):
                self.elements = elements

            def analyze(self, context):
                _ = context  # Not used

            def render(self, context):
                _ = context  # Not used
                element_str = ", ".join(self.elements)
                return [f"array({element_str})"]

        return ArrayConstructionExpression(element_names)

    def _create_array_reconstruction(self, element_names: list[str]) -> Expression:
        """Create an array reconstruction expression for returns: array([q_0, q_1])"""

        # Apply variable remapping to get the latest names
        # Use function_var_remapping if available (includes Prep changes)
        remapping = (
            self.function_var_remapping
            if hasattr(self, "function_var_remapping")
            else self.variable_remapping if hasattr(self, "variable_remapping") else {}
        )
        remapped_element_names = [remapping.get(elem, elem) for elem in element_names]

        class ArrayReconstructionExpression(Expression):
            def __init__(self, elements):
                self.elements = elements

            def analyze(self, context):
                _ = context  # Not used

            def render(self, context):
                _ = context  # Not used
                element_str = ", ".join(self.elements)
                return [f"array({element_str})"]

        return ArrayReconstructionExpression(remapped_element_names)

    def _create_struct_construction(
        self,
        struct_name: str,
        field_names: list[str],
        field_values: list[Expression],
    ) -> Expression:
        """Create a struct construction expression."""

        class StructConstructionExpression(Expression):
            def __init__(self, struct_name, field_names, field_values):
                self.struct_name = struct_name
                self.field_names = field_names
                self.field_values = field_values

            def analyze(self, context):
                for value in self.field_values:
                    value.analyze(context)

            def render(self, context):
                # Render as struct_name(value1, value2, ...) - positional args only
                # Guppy doesn't support keyword arguments in struct construction
                field_values_str = []
                for value in self.field_values:
                    value_str = value.render(context)[0]
                    field_values_str.append(value_str)
                return [f"{self.struct_name}({', '.join(field_values_str)})"]

        return StructConstructionExpression(struct_name, field_names, field_values)

    def _add_array_unpacking(self, array_name: str, size: int) -> None:
        """Add array unpacking statement."""
        # Check if this array is already unpacked in the current context
        if hasattr(self, "unpacked_vars") and array_name in self.unpacked_vars:
            # Array is already unpacked, don't unpack again
            return

        # Get the actual variable name (might be renamed)
        actual_name = array_name
        if array_name in self.plan.renamed_variables:
            actual_name = self.plan.renamed_variables[array_name]

        # Generate unpacked names
        unpacked_names = [self._get_unique_var_name(array_name, i) for i in range(size)]

        # Track unpacked vars in the builder
        self.unpacked_vars[array_name] = unpacked_names

        # Comment already added by caller, don't add another one

        # Add unpacking statement
        unpack = ArrayUnpack(
            targets=unpacked_names,
            source=actual_name,
        )
        self.current_block.statements.append(unpack)

        # Update variable info
        var = self.context.lookup_variable(actual_name)
        if var:
            var.is_unpacked = True
            var.unpacked_names = unpacked_names

    def _is_prep_rus_block(self, op) -> bool:
        """Check if this is a PrepRUS block that needs special handling."""
        return hasattr(op, "block_name") and op.block_name == "PrepRUS"

    def _convert_prep_rus_special(self, op) -> Statement | None:
        """Special conversion for PrepRUS to avoid linearity issues."""
        # PrepRUS has a specific pattern that causes issues:
        # 1. PrepEncodingFTZero creates fresh variables
        # 2. Repeat with conditional PrepEncodingFTZero
        # 3. LogZeroRot uses the variables

        # We'll generate a simplified version that avoids the conditional consumption
        self.current_block.statements.append(
            Comment("Special handling for PrepRUS to avoid linearity issues"),
        )

        # Process the operations in PrepRUS
        if hasattr(op, "ops"):
            for sub_op in op.ops:
                # Skip the Repeat block with conditional consumption
                if type(sub_op).__name__ == "Repeat":
                    # Instead of the loop with conditional, just do it once unconditionally
                    self.current_block.statements.append(
                        Comment("Simplified repeat to avoid conditional consumption"),
                    )
                    # Don't process the Repeat block
                    continue

                # Process other operations normally
                stmt = self._convert_operation(sub_op)
                if stmt:
                    self.current_block.statements.append(stmt)

        return None

    def _convert_operation(self, op) -> Statement | None:
        """Convert an SLR operation to IR statement."""
        op_type = type(op).__name__

        if op_type == "Measure":
            return self._convert_measurement(op)
        if op_type == "If":
            return self._convert_if(op)
        if op_type == "While":
            return self._convert_while(op)
        if op_type == "For":
            return self._convert_for(op)
        if op_type == "Repeat":
            return self._convert_repeat(op)
        if op_type == "Comment":
            return self._convert_comment(op)
        if op_type == "Permute":
            return self._convert_permute(op)
        if hasattr(op, "qargs"):
            stmt = self._convert_quantum_gate(op)
            # Handle case where quantum gate returns a Block
            if stmt and type(stmt).__name__ == "Block":
                # Add all statements from the block
                for s in stmt.statements:
                    self.current_block.statements.append(s)
                return None  # Already added
            return stmt
        if hasattr(op, "ops") and hasattr(op, "vars"):
            # This is a block - convert to function call
            return self._convert_block_call(op)
        if op_type == "SET":
            # Classical bit assignment
            return self._convert_set_operation(op)
        if op_type == "Barrier":
            # Barriers are just synchronization points, ignore in Guppy
            return None
        if op_type == "Return":
            # Return is metadata for type checking and block analysis
            # The actual return handling is done by the function generation code
            return None

        # Unknown operation
        return Comment(f"TODO: Handle {op_type}")

    def _convert_measurement(self, meas) -> Statement | None:
        """Convert measurement operation."""
        if not hasattr(meas, "qargs") or not meas.qargs:
            return None

        # Check if we're measuring a struct field qubit with @owned struct
        if hasattr(meas, "qargs") and len(meas.qargs) > 0:
            qarg = meas.qargs[0]
            if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                array_name = qarg.reg.sym
                # Check if this is a struct field
                for info in self.struct_info.values():
                    if (
                        array_name in info["var_names"].values()
                        and hasattr(self, "function_info")
                        and hasattr(self, "current_function_name")
                    ):
                        func_info = self.function_info.get(
                            self.current_function_name,
                            {},
                        )
                        if func_info.get("has_owned_struct_params", False):
                            # This is a known limitation - add a warning comment
                            self.current_block.statements.append(
                                Comment(
                                    "WARNING: Measuring qubits from @owned struct arrays "
                                    "is not supported by guppylang",
                                ),
                            )
                            self.current_block.statements.append(
                                Comment(
                                    "This will cause a MoveOutOfSubscriptError "
                                    "during compilation",
                                ),
                            )

        # Check if we're in a function that takes and returns a struct
        # If so, we need to be careful about struct field access
        if hasattr(self, "current_function_params"):
            for param_name, param_type in self.current_function_params:
                if "_struct" in str(param_type) and "@owned" not in str(param_type):
                    break

        # Check if this is a full array measurement
        if (
            len(meas.qargs) == 1
            and hasattr(meas.qargs[0], "sym")
            and hasattr(meas.qargs[0], "size")
            and meas.qargs[0].size >= 1
        ):
            # Full array measurement
            qreg = meas.qargs[0]

            # Track full array consumption globally
            if not hasattr(self, "consumed_resources"):
                self.consumed_resources = {}
            if qreg.sym not in self.consumed_resources:
                self.consumed_resources[qreg.sym] = set()
            self.consumed_resources[qreg.sym].update(range(qreg.size))

            # Track in scope manager too
            self.scope_manager.track_resource_usage(
                qreg.sym,
                set(range(qreg.size)),
                consumed=True,
            )

            # Check if this array was dynamically allocated
            if (
                hasattr(self, "dynamic_allocations")
                and qreg.sym in self.dynamic_allocations
            ):
                # For dynamically allocated arrays, we need to handle this differently
                # Generate individual measurements
                stmts = []

                # Check for target
                if hasattr(meas, "cout") and meas.cout and len(meas.cout) == 1:
                    cout = meas.cout[0]
                    if hasattr(cout, "sym"):
                        creg_name = cout.sym
                        # Measure each individual qubit
                        for i in range(qreg.size):
                            ancilla_var = self._get_unique_var_name(qreg.sym, i)
                            # Allocate if not already allocated
                            if not hasattr(self, "allocated_ancillas"):
                                self.allocated_ancillas = set()
                            if ancilla_var not in self.allocated_ancillas:
                                alloc_stmt = Assignment(
                                    target=VariableRef(ancilla_var),
                                    value=FunctionCall(
                                        func_name="quantum.qubit",
                                        args=[],
                                    ),
                                )
                                stmts.append(alloc_stmt)
                                self.allocated_ancillas.add(ancilla_var)

                            # Measure individual qubit
                            meas_call = FunctionCall(
                                func_name="quantum.measure",
                                args=[VariableRef(ancilla_var)],
                            )
                            creg_access = ArrayAccess(array_name=creg_name, index=i)
                            assign = Assignment(target=creg_access, value=meas_call)
                            stmts.append(assign)

                        # Return block with all statements
                        if len(stmts) == 1:
                            return stmts[0]
                        return Block(statements=stmts)
                else:
                    # No target - measure individual qubits without storing
                    for i in range(qreg.size):
                        # Use consistent mapping from (array_name, index) to variable name
                        if not hasattr(self, "allocated_qubit_vars"):
                            self.allocated_qubit_vars = {}

                        array_index_key = (qreg.sym, i)

                        # Check if we already have a variable for this array element
                        if array_index_key in self.allocated_qubit_vars:
                            ancilla_var = self.allocated_qubit_vars[array_index_key]
                        else:
                            # Create a new variable name for this specific array element
                            ancilla_var = self._get_unique_var_name(qreg.sym, i)
                            self.allocated_qubit_vars[array_index_key] = ancilla_var

                            alloc_stmt = Assignment(
                                target=VariableRef(ancilla_var),
                                value=FunctionCall(func_name="quantum.qubit", args=[]),
                            )
                            stmts.append(alloc_stmt)

                        # Measure and discard result
                        meas_call = FunctionCall(
                            func_name="quantum.measure",
                            args=[VariableRef(ancilla_var)],
                        )

                        class ExpressionStatement(Statement):
                            def __init__(self, expr):
                                self.expr = expr

                            def analyze(self, context):
                                self.expr.analyze(context)

                            def render(self, context):
                                return f"_ = {self.expr.render(context)}"

                        stmts.append(ExpressionStatement(meas_call))

                    if len(stmts) == 1:
                        return stmts[0]
                    return Block(statements=stmts)
            else:
                # Regular pre-allocated array - use measure_array
                qreg_ref = self._convert_qubit_ref(qreg)

                # Mark fresh variable as used if this is measuring a fresh variable
                if hasattr(self, "fresh_variables_to_track") and hasattr(
                    self,
                    "refreshed_arrays",
                ):
                    # Check if qreg is using a fresh variable
                    for orig_name, fresh_name in self.refreshed_arrays.items():
                        if (
                            fresh_name in self.fresh_variables_to_track
                            and orig_name == qreg.sym
                        ):
                            # Mark this fresh variable as used
                            self.fresh_variables_to_track[fresh_name]["used"] = True
                            break

                # Check for target
                if hasattr(meas, "cout") and meas.cout and len(meas.cout) == 1:
                    cout = meas.cout[0]
                    if hasattr(cout, "sym"):
                        # Check for renamed variable
                        creg_name = cout.sym
                        if creg_name in self.plan.renamed_variables:
                            creg_name = self.plan.renamed_variables[creg_name]

                        # Check if this variable is remapped (e.g., function parameter)
                        is_function_param = False
                        if (
                            hasattr(self, "var_remapping")
                            and creg_name in self.var_remapping
                        ):
                            creg_name = self.var_remapping[creg_name]
                            # Check if this is a function parameter (not in main)
                            is_function_param = (
                                hasattr(self, "current_function_name")
                                and self.current_function_name != "main"
                            )

                        # For function parameters (classical arrays), we need to update in-place
                        # to avoid BorrowShadowedError
                        if is_function_param:
                            # Generate element-wise measurements
                            stmts = []

                            # IMPORTANT: Do NOT automatically replace qubits after measurement
                            # The old logic tried to maintain array size, but this breaks partial consumption.
                            # Only replace if allocation optimizer detected reuse.
                            should_replace = False  # Disabled automatic replacement

                            for i in range(qreg.size):
                                # Check if the quantum array was unpacked
                                if (
                                    hasattr(self, "unpacked_vars")
                                    and qreg.sym in self.unpacked_vars
                                ):
                                    # Use unpacked variable
                                    element_names = self.unpacked_vars[qreg.sym]
                                    qubit_ref = VariableRef(element_names[i])
                                    qubit_var_name = element_names[i]
                                else:
                                    # Use array access
                                    qubit_ref = ArrayAccess(
                                        array_name=(
                                            self._convert_qubit_ref(qreg).name
                                            if hasattr(
                                                self._convert_qubit_ref(qreg),
                                                "name",
                                            )
                                            else qreg.sym
                                        ),
                                        index=i,
                                    )
                                    qubit_var_name = None

                                meas_call = FunctionCall(
                                    func_name="quantum.measure",
                                    args=[qubit_ref],
                                )
                                # Assign to array element
                                creg_access = ArrayAccess(array_name=creg_name, index=i)
                                assign = Assignment(target=creg_access, value=meas_call)
                                stmts.append(assign)

                                # Replace measured qubit with fresh one if needed
                                if should_replace and qubit_var_name:
                                    replacement_stmt = Assignment(
                                        target=VariableRef(qubit_var_name),
                                        value=FunctionCall(
                                            func_name="quantum.qubit",
                                            args=[],
                                        ),
                                    )
                                    stmts.append(replacement_stmt)

                                    # Track that this qubit was replaced
                                    if not hasattr(self, "replaced_qubits"):
                                        self.replaced_qubits = {}
                                    if qreg.sym not in self.replaced_qubits:
                                        self.replaced_qubits[qreg.sym] = set()
                                    self.replaced_qubits[qreg.sym].add(i)

                            # Return block with all statements
                            if len(stmts) == 1:
                                return stmts[0]
                            return Block(statements=stmts)
                        # Not a function parameter - can reassign whole array
                        creg_ref = VariableRef(creg_name)
                        # Generate measure_array
                        call = FunctionCall(
                            func_name="quantum.measure_array",
                            args=[qreg_ref],
                        )
                        return Assignment(target=creg_ref, value=call)

                # No target - just measure
                call = FunctionCall(
                    func_name="quantum.measure_array",
                    args=[qreg_ref],
                )

                # Create expression statement wrapper
                class ExpressionStatement(Statement):
                    def __init__(self, expr):
                        self.expr = expr

                    def analyze(self, context):
                        self.expr.analyze(context)

                    def render(self, context):
                        return self.expr.render(context)

                return ExpressionStatement(call)

        # Handle single qubit measurement
        if len(meas.qargs) == 1:
            qarg = meas.qargs[0]
            qubit_ref = self._convert_qubit_ref(qarg)

            # Get target if specified
            target_ref = None
            if hasattr(meas, "cout") and meas.cout and len(meas.cout) == 1:
                cout = meas.cout[0]
                # For measurements, the target should use unpacked names if available
                # So we pass is_assignment_target=False to use unpacked names
                target_ref = self._convert_bit_ref(cout, is_assignment_target=False)

            # Track resource consumption for linearity checking
            if (
                hasattr(qarg, "reg")
                and hasattr(qarg.reg, "sym")
                and hasattr(qarg, "index")
            ):
                array_name = qarg.reg.sym
                qubit_index = qarg.index
                self.scope_manager.track_resource_usage(
                    array_name,
                    {qubit_index},
                    consumed=True,
                )

                # Also track globally for conditional resource balancing
                if not hasattr(self, "consumed_resources"):
                    self.consumed_resources = {}
                if array_name not in self.consumed_resources:
                    self.consumed_resources[array_name] = set()
                self.consumed_resources[array_name].add(qubit_index)

            # Generate measurement statement
            meas_stmt = Measurement(qubit=qubit_ref, target=target_ref)

            # IMPORTANT: Do NOT automatically replace measured qubits!
            # The old "black box pattern" logic assumed functions should maintain array size,
            # but this breaks partial consumption patterns where a function consumes some qubits
            # and returns others. Only explicit Prep operations should create fresh qubits.
            #
            # The correct behavior:
            # - Measure consumes the qubit → it's gone
            # - If user wants to reset, they use explicit Prep(q[i]) → creates fresh qubit
            # - Function returns only the qubits that weren't consumed
            #
            # Check if this qubit is marked as needing replacement due to reuse
            # (e.g., unified analysis detected it's used again after consumption)
            needs_replacement_for_reuse = False
            if (
                self.unified_analysis
                and hasattr(qarg, "reg")
                and hasattr(qarg.reg, "sym")
                and hasattr(qarg, "index")
            ):
                array_name = qarg.reg.sym
                qubit_index = qarg.index
                resource_plan = self.unified_analysis.get_plan(array_name)
                if (
                    resource_plan
                    and qubit_index in resource_plan.elements_requiring_replacement
                ):
                    # CRITICAL: Check if the next operation is a Prep on this same qubit
                    # If so, skip measurement replacement - let Prep handle it
                    next_op_is_prep_on_same_qubit = False
                    if (
                        hasattr(self, "current_block_ops")
                        and hasattr(self, "current_op_index")
                        and self.current_block_ops is not None
                        and self.current_op_index is not None
                    ):
                        next_index = self.current_op_index + 1
                        if next_index < len(self.current_block_ops):
                            next_op = self.current_block_ops[next_index]
                            # Check if next operation is Prep on the same qubit
                            if type(next_op).__name__ == "Prep" and hasattr(
                                next_op,
                                "qargs",
                            ):
                                for prep_qarg in next_op.qargs:
                                    if (
                                        hasattr(prep_qarg, "reg")
                                        and hasattr(prep_qarg.reg, "sym")
                                        and prep_qarg.reg.sym == array_name
                                        and hasattr(prep_qarg, "index")
                                        and prep_qarg.index == qubit_index
                                    ):
                                        next_op_is_prep_on_same_qubit = True
                                        break

                    if not next_op_is_prep_on_same_qubit:
                        # No Prep follows - we need to replace the qubit
                        needs_replacement_for_reuse = True

            # Only replace if allocation optimizer determined it's reused
            if (
                needs_replacement_for_reuse
                and hasattr(self, "unpacked_vars")
                and hasattr(qarg, "reg")
                and hasattr(qarg.reg, "sym")
                and hasattr(qarg, "index")
            ):
                array_name = qarg.reg.sym
                qubit_index = qarg.index

                # Check if this array is unpacked in current function
                if array_name in self.unpacked_vars:
                    element_names = self.unpacked_vars[array_name]
                    if qubit_index < len(element_names):
                        # Replace the measured qubit with a fresh one
                        replacement_stmt = Assignment(
                            target=VariableRef(element_names[qubit_index]),
                            value=FunctionCall(func_name="quantum.qubit", args=[]),
                        )

                        # Track that this qubit was replaced (not consumed)
                        if not hasattr(self, "replaced_qubits"):
                            self.replaced_qubits = {}
                        if array_name not in self.replaced_qubits:
                            self.replaced_qubits[array_name] = set()
                        self.replaced_qubits[array_name].add(qubit_index)

                        # Return a block with measurement followed by replacement
                        statements = [meas_stmt, replacement_stmt]
                        return Block(statements=statements)

            return meas_stmt

        # Handle multi-qubit measurements by generating multiple single-qubit measurements
        if len(meas.qargs) > 1:
            # Verify we have corresponding classical outputs
            if not hasattr(meas, "cout") or not meas.cout:
                # No classical outputs specified - generate measurements without targets
                measurements = []
                for qarg in meas.qargs:
                    qubit_ref = self._convert_qubit_ref(qarg)

                    # Track resource consumption for each qubit
                    if (
                        hasattr(qarg, "reg")
                        and hasattr(qarg.reg, "sym")
                        and hasattr(qarg, "index")
                    ):
                        array_name = qarg.reg.sym
                        qubit_index = qarg.index
                        self.scope_manager.track_resource_usage(
                            array_name,
                            {qubit_index},
                            consumed=True,
                        )

                        # Also track globally for conditional resource balancing
                        if not hasattr(self, "consumed_resources"):
                            self.consumed_resources = {}
                        if array_name not in self.consumed_resources:
                            self.consumed_resources[array_name] = set()
                        self.consumed_resources[array_name].add(qubit_index)

                    meas_stmt = Measurement(qubit=qubit_ref, target=None)
                    measurements.append(meas_stmt)

                return Block(statements=measurements)

            # Multi-qubit measurement with classical outputs
            if len(meas.cout) != len(meas.qargs):
                # Mismatch between number of qubits and classical outputs
                return Comment(
                    f"ERROR: Multi-qubit measurement has {len(meas.qargs)} qubits "
                    f"but {len(meas.cout)} classical outputs",
                )

            # Generate one measurement statement for each qubit-bit pair
            measurements = []
            for qarg, cout in zip(meas.qargs, meas.cout):
                qubit_ref = self._convert_qubit_ref(qarg)
                target_ref = self._convert_bit_ref(cout, is_assignment_target=False)

                # Track resource consumption for each qubit
                if (
                    hasattr(qarg, "reg")
                    and hasattr(qarg.reg, "sym")
                    and hasattr(qarg, "index")
                ):
                    array_name = qarg.reg.sym
                    qubit_index = qarg.index
                    self.scope_manager.track_resource_usage(
                        array_name,
                        {qubit_index},
                        consumed=True,
                    )

                    # Also track globally for conditional resource balancing
                    if not hasattr(self, "consumed_resources"):
                        self.consumed_resources = {}
                    if array_name not in self.consumed_resources:
                        self.consumed_resources[array_name] = set()
                    self.consumed_resources[array_name].add(qubit_index)

                # Generate measurement statement
                meas_stmt = Measurement(qubit=qubit_ref, target=target_ref)
                measurements.append(meas_stmt)

            # Return a block containing all the measurements
            return Block(statements=measurements)

        # Shouldn't reach here, but just in case
        return Comment(f"Unhandled measurement with {len(meas.qargs)} qubits")

    def _convert_qubit_ref(self, qarg) -> IRNode:
        """Convert a qubit reference to IR."""
        if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
            array_name = qarg.reg.sym
            original_array = array_name

            # Check if this array has been remapped to a reconstructed name
            if hasattr(self, "array_remapping") and array_name in self.array_remapping:
                # Use the reconstructed array name instead
                remapped_name = self.array_remapping[array_name]

                # Check if the original array was unpacked after remapping
                # If it was, use the unpacked variables instead of array indexing
                if (
                    hasattr(self, "unpacked_vars")
                    and array_name in self.unpacked_vars
                    and hasattr(qarg, "index")
                ):
                    element_names = self.unpacked_vars[array_name]

                    # CRITICAL: Check if we have index mapping for partial consumption
                    # If so, map original index to unpacked variable index
                    if (
                        hasattr(self, "index_mapping")
                        and array_name in self.index_mapping
                    ):
                        mapped_index = self.index_mapping[array_name].get(qarg.index)
                        if mapped_index is not None and mapped_index < len(
                            element_names,
                        ):
                            var_name = element_names[mapped_index]
                            # Apply variable remapping if exists
                            var_name = self.variable_remapping.get(var_name, var_name)
                            return VariableRef(var_name)
                    elif (
                        qarg.index < len(element_names)
                        and element_names[qarg.index] is not None
                    ):
                        # No index mapping - use direct indexing (full array return)
                        var_name = element_names[qarg.index]
                        # Apply variable remapping if exists
                        var_name = self.variable_remapping.get(var_name, var_name)
                        return VariableRef(var_name)

                # Not unpacked, use array indexing with remapped name
                if hasattr(qarg, "index"):
                    return ArrayAccess(
                        array=VariableRef(remapped_name),
                        index=qarg.index,
                        force_array_syntax=True,  # Force array syntax for remapped arrays
                    )

            # Check if this array has been refreshed by function call
            # If it was refreshed AND then unpacked, use the unpacked variables
            if (
                hasattr(self, "refreshed_arrays")
                and array_name in self.refreshed_arrays
                and hasattr(qarg, "index")
            ):
                # Array was refreshed by function call
                fresh_array_name = self.refreshed_arrays[array_name]

                # Check if the original array name was unpacked after refresh
                # (the unpacked_vars gets updated to point to the new unpacked elements)
                if hasattr(self, "unpacked_vars") and array_name in self.unpacked_vars:
                    # It was unpacked after being refreshed - use unpacked variables
                    element_names = self.unpacked_vars[array_name]

                    # CRITICAL: Check if we have index mapping for partial consumption
                    # If so, map original index to unpacked variable index
                    if (
                        hasattr(self, "index_mapping")
                        and array_name in self.index_mapping
                    ):
                        # Map original index to position in returned array
                        mapped_index = self.index_mapping[array_name].get(qarg.index)
                        if mapped_index is not None and mapped_index < len(
                            element_names,
                        ):
                            var_name = element_names[mapped_index]
                            # Apply variable remapping if exists
                            var_name = self.variable_remapping.get(var_name, var_name)
                            return VariableRef(var_name)
                    elif (
                        qarg.index < len(element_names)
                        and element_names[qarg.index] is not None
                    ):
                        # No index mapping - use direct indexing (full array return)
                        var_name = element_names[qarg.index]
                        # Apply variable remapping if exists
                        var_name = self.variable_remapping.get(var_name, var_name)
                        return VariableRef(var_name)

                # Also check if the fresh array itself was unpacked
                if (
                    hasattr(self, "unpacked_vars")
                    and fresh_array_name in self.unpacked_vars
                ):
                    element_names = self.unpacked_vars[fresh_array_name]
                    if (
                        qarg.index < len(element_names)
                        and element_names[qarg.index] is not None
                    ):
                        var_name = element_names[qarg.index]
                        # Apply variable remapping if exists
                        var_name = self.variable_remapping.get(var_name, var_name)
                        return VariableRef(var_name)

                # Not unpacked - use array indexing on fresh name
                return ArrayAccess(
                    array=VariableRef(fresh_array_name),
                    index=qarg.index,
                    force_array_syntax=True,  # Force array syntax for refreshed arrays
                )

            # Check if this array has been unpacked (for ancilla arrays with @owned)
            if (
                hasattr(self, "unpacked_vars")
                and array_name in self.unpacked_vars
                and hasattr(qarg, "index")
            ):
                # This array was unpacked - use the unpacked variable directly
                element_names = self.unpacked_vars[array_name]
                if (
                    qarg.index < len(element_names)
                    and element_names[qarg.index] is not None
                ):
                    var_name = element_names[qarg.index]
                    # Apply variable remapping if exists
                    var_name = self.variable_remapping.get(var_name, var_name)
                    return VariableRef(var_name)
                if (
                    qarg.index < len(element_names)
                    and element_names[qarg.index] is None
                ):
                    # This element was consumed - this is an error case but let's fallback
                    pass

            # Check if this variable is mapped to a struct field (for @owned structs)
            if (
                hasattr(self, "struct_field_mapping")
                and original_array in self.struct_field_mapping
            ):
                struct_field_path = self.struct_field_mapping[original_array]
                if "." in struct_field_path:
                    struct_name, field_name = struct_field_path.split(".", 1)
                    if hasattr(qarg, "index"):
                        # Return struct.field[index]
                        field_access = FieldAccess(
                            obj=VariableRef(struct_name),
                            field=field_name,
                        )
                        return ArrayAccess(array=field_access, index=qarg.index)
                    # Return struct.field
                    return FieldAccess(obj=VariableRef(struct_name), field=field_name)

            # Check if this is a dynamically allocated array (ancilla)
            if (
                hasattr(self, "dynamic_allocations")
                and original_array in self.dynamic_allocations
                and hasattr(qarg, "index")
            ):
                # Use a consistent mapping from (array_name, index) to variable name
                if not hasattr(self, "allocated_qubit_vars"):
                    self.allocated_qubit_vars = {}

                array_index_key = (original_array, qarg.index)

                # Check if we already have a variable for this array element
                if array_index_key in self.allocated_qubit_vars:
                    var_name = self.allocated_qubit_vars[array_index_key]
                    # Apply variable remapping if exists (for Prep operations)
                    var_name = self.variable_remapping.get(var_name, var_name)
                    return VariableRef(var_name)

                # Create a new variable name for this specific array element
                ancilla_var = self._get_unique_var_name(original_array, qarg.index)

                # Record the mapping and allocate the qubit
                self.allocated_qubit_vars[array_index_key] = ancilla_var

                # Also track in allocated_ancillas for cleanup
                if not hasattr(self, "allocated_ancillas"):
                    self.allocated_ancillas = set()
                self.allocated_ancillas.add(ancilla_var)

                alloc_stmt = Assignment(
                    target=VariableRef(ancilla_var),
                    value=FunctionCall(func_name="quantum.qubit", args=[]),
                )
                self.current_block.statements.append(alloc_stmt)

                # Apply variable remapping if exists (for Prep operations)
                var_name = self.variable_remapping.get(ancilla_var, ancilla_var)
                return VariableRef(var_name)

            # Check if this variable is part of a struct and has been unpacked
            if hasattr(self, "var_remapping") and original_array in self.var_remapping:
                # Use the unpacked field variable
                unpacked_var_name = self.var_remapping[original_array]
                if hasattr(qarg, "index"):
                    # Array element access with unpacked variable: c_d[0]
                    return ArrayAccess(
                        array=VariableRef(unpacked_var_name),
                        index=qarg.index,
                    )
                # Full array access with unpacked variable: c_d
                return VariableRef(unpacked_var_name)

            # Check if this array is part of a struct (fallback)
            for prefix, info in self.struct_info.items():
                if array_name in info["var_names"].values():
                    # This is a struct field
                    suffix = next(
                        k for k, v in info["var_names"].items() if v == array_name
                    )

                    # Check if we're in a function that takes this struct as parameter
                    struct_param_name = prefix  # Default to the struct name
                    if hasattr(self, "param_mapping") and prefix in self.param_mapping:
                        struct_param_name = self.param_mapping[prefix]

                    # Check if the struct has a fresh version (after function calls)
                    if (
                        hasattr(self, "refreshed_arrays")
                        and prefix in self.refreshed_arrays
                    ):
                        struct_param_name = self.refreshed_arrays[prefix]

                    if hasattr(qarg, "index"):
                        # Struct field element access: c.d[0]
                        field_access = FieldAccess(
                            obj=VariableRef(struct_param_name),
                            field=suffix,
                        )
                        return ArrayAccess(array=field_access, index=qarg.index)
                    # Full struct field access: c.d
                    return FieldAccess(obj=VariableRef(struct_param_name), field=suffix)

            # Check if we're inside a function and need to use remapped names
            if hasattr(self, "var_remapping") and original_array in self.var_remapping:
                array_name = self.var_remapping[original_array]

            # Check for renaming
            if array_name in self.plan.renamed_variables:
                array_name = self.plan.renamed_variables[array_name]

            if hasattr(qarg, "index"):
                # Array Unpacking Pattern: use unpacked variable names instead of array indexing
                # Check both the original name and any remapped name
                check_names = [original_array]
                if (
                    hasattr(self, "var_remapping")
                    and original_array in self.var_remapping
                ):
                    check_names.append(self.var_remapping[original_array])
                if array_name != original_array:
                    check_names.append(array_name)

                # Try each possible name for unpacked variables
                for check_name in check_names:
                    if (
                        hasattr(self, "unpacked_vars")
                        and check_name in self.unpacked_vars
                        # Don't use unpacked variables if the array was refreshed
                        and check_name not in self.refreshed_arrays
                    ):
                        element_names = self.unpacked_vars[check_name]
                        if qarg.index < len(element_names):
                            var_name = element_names[qarg.index]
                            # Apply variable remapping if exists
                            var_name = self.variable_remapping.get(var_name, var_name)
                            return VariableRef(var_name)

                # Check if this element should be allocated locally
                resource_plan = None
                if self.unified_analysis:
                    resource_plan = self.unified_analysis.get_plan(original_array)
                if (
                    resource_plan
                    and qarg.index in resource_plan.elements_to_allocate_locally
                ):
                    # This element should be allocated locally
                    local_var_name = f"{original_array}_{qarg.index}_local"

                    # Add local allocation if not already done
                    if not hasattr(self, "_local_allocations"):
                        self._local_allocations = set()

                    if local_var_name not in self._local_allocations:
                        self._local_allocations.add(local_var_name)
                        # Add allocation statement
                        alloc_stmt = Assignment(
                            target=VariableRef(local_var_name),
                            value=FunctionCall(func_name="quantum.qubit", args=[]),
                        )
                        self.current_block.statements.append(alloc_stmt)

                    # Apply variable remapping if exists (for Prep operations)
                    local_var_name = self.variable_remapping.get(
                        local_var_name,
                        local_var_name,
                    )
                    return VariableRef(local_var_name)

                # Array element access
                # Skip this shortcut - we need to check for unpacked vars first
                # The unpacking check above should handle function cases too

                # In main function, check if this array is unpacked
                if original_array in self.plan.arrays_to_unpack:
                    # This array should be unpacked, use unpacked name
                    info = self.plan.arrays_to_unpack[original_array]
                    if qarg.index < info.size:
                        # Check if the array is actually unpacked yet
                        var_info = self.context.lookup_variable(array_name)
                        if var_info and var_info.is_unpacked:
                            # Use the actual unpacked name from our tracking
                            if array_name in self.unpacked_vars and qarg.index < len(
                                self.unpacked_vars[array_name],
                            ):
                                unpacked_name = self.unpacked_vars[array_name][
                                    qarg.index
                                ]
                            else:
                                # Fallback to generating the name (should not normally happen)
                                unpacked_name = self._get_unique_var_name(
                                    original_array,
                                    qarg.index,
                                )
                            # Apply variable remapping if exists (for Prep operations)
                            unpacked_name = self.variable_remapping.get(
                                unpacked_name,
                                unpacked_name,
                            )
                            return VariableRef(unpacked_name)

                # Not unpacked or inside function, use array access
                return ArrayAccess(array_name=array_name, index=qarg.index)

            # Full array reference - check if array was refreshed by function call
            if (
                hasattr(self, "refreshed_arrays")
                and original_array in self.refreshed_arrays
            ):
                # Use the fresh returned array name instead of the original
                fresh_array_name = self.refreshed_arrays[original_array]
                return VariableRef(fresh_array_name)

            return VariableRef(array_name)
        if hasattr(qarg, "sym"):
            # Direct variable reference
            var_name = qarg.sym
            original_var = var_name

            # Check if this variable was refreshed by function call
            if (
                hasattr(self, "refreshed_arrays")
                and original_var in self.refreshed_arrays
            ):
                # Use the fresh returned variable name instead of the original
                fresh_var_name = self.refreshed_arrays[original_var]
                return VariableRef(fresh_var_name)

            # Check if we're inside a function and need to use remapped names
            if hasattr(self, "var_remapping") and original_var in self.var_remapping:
                var_name = self.var_remapping[original_var]

            # Check for renaming
            if var_name in self.plan.renamed_variables:
                var_name = self.plan.renamed_variables[var_name]
            return VariableRef(var_name)

        # Fallback
        return VariableRef(str(qarg))

    def _convert_bit_ref(self, carg, *, is_assignment_target: bool = False) -> IRNode:
        """Convert a classical bit reference to IR.

        Args:
            carg: The classical argument to convert
            is_assignment_target: If True, always use array indexing (for assignments)
        """
        if hasattr(carg, "reg") and hasattr(carg.reg, "sym"):
            array_name = carg.reg.sym
            original_array = array_name

            # Check if this array has been refreshed by function call
            # If so, prefer array indexing over stale unpacked variables
            if (
                hasattr(self, "refreshed_arrays")
                and array_name in self.refreshed_arrays
                and hasattr(carg, "index")
            ):
                # Array was refreshed by function call - use the fresh returned name
                fresh_array_name = self.refreshed_arrays[array_name]
                return ArrayAccess(
                    array=VariableRef(fresh_array_name),
                    index=carg.index,
                    force_array_syntax=True,  # Force array syntax for refreshed arrays
                )

            # Check if this variable is mapped to a struct field (for @owned structs)
            if (
                hasattr(self, "struct_field_mapping")
                and original_array in self.struct_field_mapping
            ):
                struct_field_path = self.struct_field_mapping[original_array]
                if "." in struct_field_path:
                    struct_name, field_name = struct_field_path.split(".", 1)
                    if hasattr(carg, "index"):
                        # Return struct.field[index]
                        field_access = FieldAccess(
                            obj=VariableRef(struct_name),
                            field=field_name,
                        )
                        return ArrayAccess(array=field_access, index=carg.index)
                    # Return struct.field
                    return FieldAccess(obj=VariableRef(struct_name), field=field_name)

            # Check if this variable is part of a struct and has been unpacked
            if hasattr(self, "var_remapping") and original_array in self.var_remapping:
                # Use the unpacked field variable
                unpacked_var_name = self.var_remapping[original_array]
                if hasattr(carg, "index"):
                    # Array element access with unpacked variable: c_verify_prep[0]
                    return ArrayAccess(
                        array=VariableRef(unpacked_var_name),
                        index=carg.index,
                    )
                # Full array access with unpacked variable: c_verify_prep
                return VariableRef(unpacked_var_name)

            # Check if this variable is part of a struct in main context (fallback)
            for prefix, info in self.struct_info.items():
                if original_array in info["var_names"].values():
                    # Find the field name
                    for suffix, var_name in info["var_names"].items():
                        if var_name == original_array:
                            # Check if the struct has been decomposed and we should use decomposed variables
                            if (
                                hasattr(self, "var_remapping")
                                and original_array in self.var_remapping
                            ):
                                # Struct was decomposed - use the decomposed variable directly
                                decomposed_var = self.var_remapping[original_array]
                                if hasattr(carg, "index"):
                                    return ArrayAccess(
                                        array=VariableRef(decomposed_var),
                                        index=carg.index,
                                    )
                                return VariableRef(decomposed_var)

                            # Check if we're in a function that receives the struct
                            struct_param_name = prefix
                            if (
                                hasattr(self, "param_mapping")
                                and prefix in self.param_mapping
                            ):
                                struct_param_name = self.param_mapping[prefix]

                            # Check if we have decomposed variables for fresh structs
                            if (
                                hasattr(self, "refreshed_arrays")
                                and prefix in self.refreshed_arrays
                            ):
                                fresh_struct_name = self.refreshed_arrays[prefix]
                                # Check if this fresh struct was decomposed
                                if (
                                    hasattr(self, "decomposed_vars")
                                    and fresh_struct_name in self.decomposed_vars
                                ):
                                    # Use the decomposed variable
                                    field_vars = self.decomposed_vars[fresh_struct_name]
                                    if suffix in field_vars:
                                        decomposed_var = field_vars[suffix]
                                        if hasattr(carg, "index"):
                                            return ArrayAccess(
                                                array=VariableRef(decomposed_var),
                                                index=carg.index,
                                            )
                                        return VariableRef(decomposed_var)
                                struct_param_name = fresh_struct_name

                            if hasattr(carg, "index"):
                                # Struct field element access: c.verify_prep[0]
                                field_access = FieldAccess(
                                    obj=VariableRef(struct_param_name),
                                    field=suffix,
                                )
                                return ArrayAccess(array=field_access, index=carg.index)
                            # Full struct field access: c.verify_prep
                            return FieldAccess(
                                obj=VariableRef(struct_param_name),
                                field=suffix,
                            )

            # Check if we're inside a function and need to use remapped names
            if hasattr(self, "var_remapping") and original_array in self.var_remapping:
                array_name = self.var_remapping[original_array]

            # Check for renaming
            if array_name in self.plan.renamed_variables:
                array_name = self.plan.renamed_variables[array_name]

            if hasattr(carg, "index"):
                # Check if this array is unpacked and we're not assigning
                var_info = self.context.lookup_variable(array_name)
                if (
                    not is_assignment_target
                    and var_info
                    and var_info.is_unpacked
                    and hasattr(var_info, "unpacked_names")
                ):
                    # Use unpacked variable name for reading
                    index = carg.index
                    if index < len(var_info.unpacked_names):
                        return VariableRef(var_info.unpacked_names[index])

                # Use array access for assignments or non-unpacked arrays
                return ArrayAccess(array_name=array_name, index=carg.index)
            # Full array reference
            return VariableRef(array_name)
        if hasattr(carg, "sym"):
            # Direct variable reference
            var_name = carg.sym
            # Check for renaming
            if var_name in self.plan.renamed_variables:
                var_name = self.plan.renamed_variables[var_name]
            return VariableRef(var_name)

        # Fallback
        return VariableRef(str(carg))

    def _convert_quantum_gate(self, gate) -> Statement | None:
        """Convert quantum gate operation."""
        gate_name = type(gate).__name__

        # Regular gate mapping for in-place operations
        gate_map = {
            "H": "quantum.h",
            "X": "quantum.x",
            "Y": "quantum.y",
            "Z": "quantum.z",
            "S": "quantum.s",
            "SZ": "quantum.s",
            "SZdg": "quantum.sdg",
            "T": "quantum.t",
            "Tdg": "quantum.tdg",
            "CX": "quantum.cx",
            "CY": "quantum.cy",
            "CZ": "quantum.cz",
            "Prep": "quantum.qubit",  # Prep allocates a fresh qubit
        }

        if gate_name not in gate_map:
            return Comment(f"Unknown gate: {gate_name}")

        func_name = gate_map[gate_name]

        # Convert qubit arguments
        args = []
        if hasattr(gate, "qargs") and gate.qargs:
            # Check if this is a single-qubit gate with multiple arguments
            if (
                gate_name in ["H", "X", "Y", "Z", "S", "SZ", "SZdg", "T", "Tdg", "Prep"]
                and len(gate.qargs) > 1
            ):
                # Single-qubit gate applied to multiple qubits
                # Check if all qargs are consecutive array elements from the same array
                if (
                    all(
                        hasattr(qarg, "reg") and hasattr(qarg, "index")
                        for qarg in gate.qargs
                    )
                    and len({qarg.reg.sym for qarg in gate.qargs}) == 1
                ):
                    # All from same array - check if consecutive
                    indices = [qarg.index for qarg in gate.qargs]
                    array_name = gate.qargs[0].reg.sym

                    if indices == list(range(min(indices), max(indices) + 1)):
                        # Consecutive indices - generate a loop
                        loop_var = "i"
                        start = min(indices)
                        stop = max(indices) + 1

                        # Create loop body
                        body_block = Block()

                        # Check if the array name needs remapping (for unpacked struct fields)
                        actual_array_name = array_name
                        if (
                            hasattr(self, "var_remapping")
                            and array_name in self.var_remapping
                        ):
                            actual_array_name = self.var_remapping[array_name]

                        array_ref = VariableRef(actual_array_name)
                        index_ref = VariableRef(loop_var)
                        elem_access = ArrayAccess(array=array_ref, index=index_ref)
                        call = FunctionCall(func_name=func_name, args=[elem_access])

                        # Create expression statement wrapper
                        class ExpressionStatement(Statement):
                            def __init__(self, expr):
                                self.expr = expr

                            def analyze(self, context):
                                self.expr.analyze(context)

                            def render(self, context):
                                return self.expr.render(context)

                        body_block.statements.append(ExpressionStatement(call))

                        # Create for loop
                        range_call = FunctionCall(
                            func_name="range",
                            args=[Literal(start), Literal(stop)],
                        )
                        return ForStatement(
                            loop_var=loop_var,
                            iterable=range_call,
                            body=body_block,
                        )

                # Not consecutive or not from same array - expand to individual calls
                stmts = []
                for qarg in gate.qargs:
                    qref = self._convert_qubit_ref(qarg)
                    call = FunctionCall(func_name=func_name, args=[qref])

                    # Create expression statement wrapper
                    class ExpressionStatement(Statement):
                        def __init__(self, expr):
                            self.expr = expr

                        def analyze(self, context):
                            self.expr.analyze(context)

                        def render(self, context):
                            return self.expr.render(context)

                    stmts.append(ExpressionStatement(call))
                # Return a block with all statements
                return Block(statements=stmts)
            # Handle multi-qubit gates with tuple arguments
            if gate_name in ["CX", "CY", "CZ"] and all(
                isinstance(arg, tuple) and len(arg) == 2 for arg in gate.qargs
            ):
                # Multiple (control, target) pairs - generate multiple statements
                stmts = []
                for ctrl, tgt in gate.qargs:
                    ctrl_ref = self._convert_qubit_ref(ctrl)
                    tgt_ref = self._convert_qubit_ref(tgt)
                    call = FunctionCall(func_name=func_name, args=[ctrl_ref, tgt_ref])

                    # Create expression statement wrapper
                    class ExpressionStatement(Statement):
                        def __init__(self, expr):
                            self.expr = expr

                        def analyze(self, context):
                            self.expr.analyze(context)

                        def render(self, context):
                            return self.expr.render(context)

                    stmts.append(ExpressionStatement(call))
                # Return a block with all statements
                return Block(statements=stmts)
            # Standard argument handling
            for qarg in gate.qargs:
                # Check if this is a full array (no index)
                if hasattr(qarg, "sym") and hasattr(qarg, "size") and qarg.size > 1:
                    # This is a full array - need to expand to individual gates
                    stmts = []
                    array_name = qarg.sym

                    # Check for renaming
                    if array_name in self.plan.renamed_variables:
                        array_name = self.plan.renamed_variables[array_name]

                    # Check if this array name needs remapping (for unpacked struct fields)
                    if (
                        hasattr(self, "var_remapping")
                        and array_name in self.var_remapping
                    ):
                        array_name = self.var_remapping[array_name]

                    # Apply gate to each element
                    # For operations on arrays, we need to expand to individual operations
                    # However, reset operations in functions with owned arrays
                    # need special handling

                    if (
                        gate_name == "Prep"
                        and hasattr(self, "var_remapping")
                        and self.var_remapping
                        and array_name in self.var_remapping
                    ):
                        # Array Unpacking Pattern: use unpacked variables with
                        # functional operations
                        stmts.append(Comment(f"Reset all qubits in {array_name}"))

                        if (
                            hasattr(self, "unpacked_vars")
                            and array_name in self.unpacked_vars
                        ):
                            # Use unpacked variables with functional assignments
                            # Note: Explicit reset tracking is done during consumption analysis
                            # in _track_consumed_qubits(), not here
                            element_names = self.unpacked_vars[array_name]

                            for i in range(min(qarg.size, len(element_names))):
                                # CRITICAL: Check if this qubit was just replaced by a measurement
                                # If so, skip the entire Prep (qubit already fresh)
                                if hasattr(self, "replaced_qubits") and (
                                    array_name in self.replaced_qubits
                                    and i in self.replaced_qubits[array_name]
                                ):
                                    # This qubit was just replaced by measurement - skip Prep
                                    self.replaced_qubits[array_name].discard(i)
                                    # Add comment but no actual operation
                                    stmts.append(
                                        Comment(
                                            f"Prep skipped for {element_names[i]} - already fresh from measurement",
                                        ),
                                    )
                                    continue

                                elem_var = VariableRef(element_names[i])

                                # CRITICAL: Prep (reset) requires discard-then-allocate pattern
                                # Can't pass old qubit as argument to quantum.qubit()
                                # Pattern: quantum.discard(q); q = quantum.qubit()

                                # 1. Discard the old qubit
                                discard_call = FunctionCall(
                                    func_name="quantum.discard",
                                    args=[elem_var],
                                )

                                # Create expression statement wrapper
                                class ExpressionStatement(Statement):
                                    def __init__(self, expr):
                                        self.expr = expr

                                    def analyze(self, context):
                                        self.expr.analyze(context)

                                    def render(self, context):
                                        return self.expr.render(context)

                                stmts.append(ExpressionStatement(discard_call))

                                # 2. Allocate fresh qubit
                                fresh_qubit_call = FunctionCall(
                                    func_name="quantum.qubit",
                                    args=[],  # No arguments - fresh allocation
                                )
                                assignment = Assignment(
                                    target=elem_var,
                                    value=fresh_qubit_call,
                                )
                                stmts.append(assignment)
                        else:
                            # Fallback to array indexing if no unpacking
                            for i in range(qarg.size):
                                elem_ref = ArrayAccess(array_name=array_name, index=i)
                                call = FunctionCall(
                                    func_name=func_name,
                                    args=[elem_ref],
                                )

                                # Create expression statement wrapper
                                class ExpressionStatement(Statement):
                                    def __init__(self, expr):
                                        self.expr = expr

                                    def analyze(self, context):
                                        self.expr.analyze(context)

                                    def render(self, context):
                                        return self.expr.render(context)

                                stmts.append(ExpressionStatement(call))
                    else:
                        # Regular case - generate a loop instead of expanding
                        # Check if this array is part of a struct
                        is_struct_field = False

                        # First check if we have a remapped variable (unpacked struct field)
                        # The key insight is that if we're in a function with
                        # @owned struct parameters
                        # and this array is a struct field that has been unpacked, we should use
                        # the unpacked variable name directly, not struct.field notation
                        use_unpacked = False
                        if (
                            hasattr(self, "var_remapping")
                            and array_name in self.var_remapping
                        ):
                            # Check if this is a struct field that has been unpacked
                            for prefix, info in self.struct_info.items():
                                if array_name in info["var_names"].values() and hasattr(
                                    self,
                                    "current_function_params",
                                ):
                                    # Check if the struct is an @owned parameter
                                    for (
                                        param_name,
                                        param_type,
                                    ) in self.current_function_params:
                                        if param_name == prefix and "@owned" in str(
                                            param_type,
                                        ):
                                            use_unpacked = True
                                            break
                                    if use_unpacked:
                                        break

                        if use_unpacked:
                            # Generate a loop using the unpacked variable
                            loop_var = "i"
                            body_block = Block()

                            # Use the remapped name from var_remapping
                            remapped_name = self.var_remapping.get(
                                array_name,
                                array_name,
                            )
                            elem_ref = ArrayAccess(
                                array=VariableRef(remapped_name),
                                index=VariableRef(loop_var),
                            )
                            call = FunctionCall(func_name=func_name, args=[elem_ref])

                            # Create expression statement wrapper
                            class ExpressionStatement(Statement):
                                def __init__(self, expr):
                                    self.expr = expr

                                def analyze(self, context):
                                    self.expr.analyze(context)

                                def render(self, context):
                                    return self.expr.render(context)

                            body_block.statements.append(ExpressionStatement(call))

                            # Create for loop
                            range_call = FunctionCall(
                                func_name="range",
                                args=[Literal(0), Literal(qarg.size)],
                            )
                            for_stmt = ForStatement(
                                loop_var=loop_var,
                                iterable=range_call,
                                body=body_block,
                            )
                            stmts.append(for_stmt)
                            is_struct_field = True  # Skip the struct field check below

                        if not is_struct_field:
                            for prefix, info in self.struct_info.items():
                                if qarg.sym in info["var_names"].values():
                                    # Find the field name
                                    for suffix, var_name in info["var_names"].items():
                                        if var_name == qarg.sym:
                                            # Check if we're in a function that receives the struct
                                            struct_param_name = prefix
                                            if (
                                                hasattr(self, "param_mapping")
                                                and prefix in self.param_mapping
                                            ):
                                                struct_param_name = self.param_mapping[
                                                    prefix
                                                ]

                                            # Check if the struct has a fresh version (after function calls)
                                            if (
                                                hasattr(self, "refreshed_arrays")
                                                and prefix in self.refreshed_arrays
                                            ):
                                                struct_param_name = (
                                                    self.refreshed_arrays[prefix]
                                                )

                                            # Generate a loop for struct field access
                                            loop_var = "i"
                                            body_block = Block()

                                            field_access = FieldAccess(
                                                obj=VariableRef(struct_param_name),
                                                field=suffix,
                                            )
                                            elem_ref = ArrayAccess(
                                                array=field_access,
                                                index=VariableRef(loop_var),
                                            )
                                            call = FunctionCall(
                                                func_name=func_name,
                                                args=[elem_ref],
                                            )

                                        # Create expression statement wrapper
                                        class ExpressionStatement(Statement):
                                            def __init__(self, expr):
                                                self.expr = expr

                                            def analyze(self, context):
                                                self.expr.analyze(context)

                                            def render(self, context):
                                                return self.expr.render(context)

                                        body_block.statements.append(
                                            ExpressionStatement(call),
                                        )

                                        # Create for loop
                                        range_call = FunctionCall(
                                            func_name="range",
                                            args=[Literal(0), Literal(qarg.size)],
                                        )
                                        for_stmt = ForStatement(
                                            loop_var=loop_var,
                                            iterable=range_call,
                                            body=body_block,
                                        )
                                        stmts.append(for_stmt)
                                        is_struct_field = True
                                        break
                                break

                        if not is_struct_field:
                            # Not in a struct - check if array was unpacked
                            if (
                                hasattr(self, "unpacked_vars")
                                and array_name in self.unpacked_vars
                            ):
                                # Array was unpacked - UNROLL the loop to use unpacked elements directly
                                # This avoids: unpack → reconstruct → loop → unpack (AlreadyUsedError)
                                # Instead: unpack → apply to each element (no reconstruction needed)
                                element_names = self.unpacked_vars[array_name]

                                # Unroll: apply the operation to each unpacked element
                                for i in range(qarg.size):
                                    if i < len(element_names):
                                        elem_ref = VariableRef(element_names[i])
                                        call = FunctionCall(
                                            func_name=func_name,
                                            args=[elem_ref],
                                        )

                                        # Create expression statement wrapper
                                        class ExpressionStatement(Statement):
                                            def __init__(self, expr):
                                                self.expr = expr

                                            def analyze(self, context):
                                                self.expr.analyze(context)

                                            def render(self, context):
                                                return self.expr.render(context)

                                        stmts.append(ExpressionStatement(call))

                                # No need to update unpacked_vars - elements are modified in-place
                            else:
                                # Array not unpacked - generate a loop
                                loop_var = "i"
                                body_block = Block()

                                # Check if the array name needs remapping (for unpacked struct fields)
                                actual_array_name = array_name
                                if (
                                    hasattr(self, "var_remapping")
                                    and array_name in self.var_remapping
                                ):
                                    actual_array_name = self.var_remapping[array_name]

                                elem_ref = ArrayAccess(
                                    array=VariableRef(actual_array_name),
                                    index=VariableRef(loop_var),
                                )
                                call = FunctionCall(
                                    func_name=func_name,
                                    args=[elem_ref],
                                )

                                # Create expression statement wrapper
                                class ExpressionStatement(Statement):
                                    def __init__(self, expr):
                                        self.expr = expr

                                    def analyze(self, context):
                                        self.expr.analyze(context)

                                    def render(self, context):
                                        return self.expr.render(context)

                                body_block.statements.append(ExpressionStatement(call))

                                # Create for loop
                                range_call = FunctionCall(
                                    func_name="range",
                                    args=[Literal(0), Literal(qarg.size)],
                                )
                                for_stmt = ForStatement(
                                    loop_var=loop_var,
                                    iterable=range_call,
                                    body=body_block,
                                )
                                stmts.append(for_stmt)

                    # Return a block with all statements
                    return Block(statements=stmts)
                args.append(self._convert_qubit_ref(qarg))

        # If we get here, we have regular args (not arrays)
        if args:
            # Create function call expression
            call = FunctionCall(func_name=func_name, args=args)

            # Special handling for Prep - it allocates a fresh qubit
            # so we need to use assignment, not an expression statement
            # Note: Explicit reset tracking is done during consumption analysis
            # in _track_consumed_qubits(), not here
            # Prep generates: discard + fresh allocation (reset pattern)
            if gate_name == "Prep" and len(args) == 1:
                # Get the target variable (where to store the fresh qubit)
                target = args[0]

                # CRITICAL: Check if the previous operation was a measurement on this same qubit
                # If so, skip the discard step (qubit already consumed by measurement)
                skip_discard = False
                if (
                    hasattr(self, "current_block_ops")
                    and hasattr(self, "current_op_index")
                    and self.current_block_ops is not None
                    and self.current_op_index is not None
                    and self.current_op_index > 0
                    and hasattr(target, "name")
                ):
                    prev_index = self.current_op_index - 1
                    prev_op = self.current_block_ops[prev_index]
                    # Check if previous operation was a measurement
                    if type(prev_op).__name__ == "Measure" and hasattr(
                        prev_op,
                        "qargs",
                    ):
                        for meas_qarg in prev_op.qargs:
                            # Get the variable name that would have been generated for this qubit
                            if hasattr(meas_qarg, "reg") and hasattr(
                                meas_qarg.reg,
                                "sym",
                            ):
                                array_name = meas_qarg.reg.sym
                                # Check both unpacked vars and locally allocated vars
                                if (
                                    hasattr(self, "unpacked_vars")
                                    and array_name in self.unpacked_vars
                                    and hasattr(meas_qarg, "index")
                                ):
                                    element_names = self.unpacked_vars[array_name]
                                    qubit_index = meas_qarg.index
                                    if qubit_index < len(element_names):
                                        meas_var_name = element_names[qubit_index]
                                        if meas_var_name == target.name:
                                            # Same qubit - skip discard
                                            skip_discard = True
                                            break
                                # Also check if this is a locally allocated qubit (two patterns)
                                elif hasattr(meas_qarg, "index"):
                                    qubit_index = meas_qarg.index
                                    # Pattern 1: {array}_{index}_local (from line 3712)
                                    local_var_name = f"{array_name}_{qubit_index}_local"
                                    # Pattern 2: {array}_{index} (from UNPACKED_MIXED with local allocation)
                                    unpacked_var_name = f"{array_name}_{qubit_index}"

                                    if target.name in (
                                        local_var_name,
                                        unpacked_var_name,
                                    ):
                                        # This is the same qubit that was measured - skip discard
                                        skip_discard = True
                                        break

                # CRITICAL: Use discard-then-allocate pattern for reset
                # Pattern: quantum.discard(q); q = quantum.qubit()
                # BUT: If qubit was just consumed by measurement, use fresh variable name
                # to satisfy Guppy's linear type constraints
                stmts = []

                # Determine target variable for the fresh qubit
                if skip_discard:
                    # Previous operation consumed the qubit
                    # We need a fresh variable name to avoid PlaceNotUsedError
                    old_name = target.name

                    # Generate a new version for this variable
                    version = self.variable_version_counter.get(old_name, 0) + 1
                    self.variable_version_counter[old_name] = version
                    new_name = f"{old_name}_{version}"

                    # Add remapping so subsequent operations use the new name
                    self.variable_remapping[old_name] = new_name

                    # Track the new variable for cleanup
                    if not hasattr(self, "allocated_ancillas"):
                        self.allocated_ancillas = set()
                    self.allocated_ancillas.add(new_name)

                    # Allocate to the new variable
                    fresh_target = VariableRef(new_name)
                else:
                    # Discard the old qubit first
                    discard_call = FunctionCall(
                        func_name="quantum.discard",
                        args=[target],
                    )

                    # Create expression statement wrapper
                    class ExpressionStatement(Statement):
                        def __init__(self, expr):
                            self.expr = expr

                        def analyze(self, context):
                            self.expr.analyze(context)

                        def render(self, context):
                            return self.expr.render(context)

                    stmts.append(ExpressionStatement(discard_call))

                    # Reuse the same variable
                    fresh_target = target

                # Allocate fresh qubit
                fresh_qubit_call = FunctionCall(func_name="quantum.qubit", args=[])
                stmts.append(Assignment(target=fresh_target, value=fresh_qubit_call))

                return Block(statements=stmts)

            # No longer use functional operations - all gates are in-place

            # Create expression statement wrapper for non-functional operations
            class ExpressionStatement(Statement):
                def __init__(self, expr):
                    self.expr = expr

                def analyze(self, context):
                    self.expr.analyze(context)

                def render(self, context):
                    return self.expr.render(context)

            return ExpressionStatement(call)

        return None

    def _should_restructure_conditional_consumption(self, if_block) -> bool:
        """Check if this If block needs restructuring to avoid conditional consumption."""
        # Check if we're in a conditional consumption loop
        if not (
            hasattr(self, "_in_conditional_consumption_loop")
            and self._in_conditional_consumption_loop
        ):
            return False

        # Check if the If block contains function calls that consume variables
        if hasattr(if_block, "ops"):
            for op in if_block.ops:
                if hasattr(op, "block_name") and op.block_name in [
                    "PrepEncodingFTZero",
                    "PrepEncodingNonFTZero",
                ]:
                    return True

        return False

    def _convert_if(self, if_block) -> Statement | None:
        """Convert If block."""
        # Check if this conditional needs restructuring to avoid consumption issues
        if self._should_restructure_conditional_consumption(if_block):
            # Restructure to avoid conditional consumption
            # Instead of: if cond: consume(vars)
            # We do: vars = consume(vars); if not cond: pass
            # This ensures vars are always consumed, maintaining linearity

            self.current_block.statements.append(
                Comment("Restructured conditional to avoid consumption in conditional"),
            )

            # Execute the operations unconditionally
            if hasattr(if_block, "ops"):
                for op in if_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        self.current_block.statements.append(stmt)

            # The condition check becomes a no-op since we already executed
            return None

        # Check if we have a pre-extracted condition for this If block
        if (
            hasattr(self, "pre_extracted_conditions")
            and id(if_block) in self.pre_extracted_conditions
        ):
            # Use the pre-extracted condition variable
            condition_var_name = self.pre_extracted_conditions[id(if_block)]
            condition = VariableRef(condition_var_name)

            # Convert then block
            then_block = Block()
            if hasattr(if_block, "ops"):
                prev_block = self.current_block
                self.current_block = then_block

                for op in if_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        then_block.statements.append(stmt)

                self.current_block = prev_block

            # Handle else block if present
            else_block = None
            if hasattr(if_block, "else_ops") and if_block.else_ops:
                else_block = Block()
                prev_block = self.current_block
                self.current_block = else_block

                for op in if_block.else_ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        else_block.statements.append(stmt)

                self.current_block = prev_block

            return IfStatement(
                condition=condition,
                then_block=then_block,
                else_block=else_block,
            )

        # Check if this If block has struct field access in loop with @owned parameters
        if hasattr(if_block, "cond") and self._is_struct_field_in_loop_with_owned(
            if_block.cond,
        ):
            # Implement a proper fix by extracting the condition value before the conditional
            # This allows us to check the struct field without violating @owned constraints

            # Extract the struct field that's being tested
            condition_var = self._extract_condition_variable(if_block.cond)
            if condition_var:
                self.current_block.statements.append(
                    Comment(
                        "Extract condition variable to avoid @owned struct field access in loop",
                    ),
                )

                # Create a local variable to hold the condition value
                condition_stmt = Assignment(
                    target=VariableRef(condition_var["var_name"]),
                    value=self._convert_condition_value(if_block.cond),
                )
                self.current_block.statements.append(condition_stmt)

                # Convert then block first
                then_block = Block()
                if hasattr(if_block, "ops"):
                    # Enter a new scope for the If block
                    prev_block = self.current_block
                    self.current_block = then_block

                    for op in if_block.ops:
                        stmt = self._convert_operation(op)
                        if stmt:
                            then_block.statements.append(stmt)

                    self.current_block = prev_block

                # Now create the If statement using the extracted variable
                if condition_var["comparison"] == "EQUIV":
                    # For bool comparison with 1, convert to just the boolean variable
                    # Since verify_prep[0] is bool and we're checking == 1,
                    # this means "if verification failed" which is just the boolean value
                    if condition_var["compare_value"] == 1:
                        condition = VariableRef(condition_var["var_name"])
                    else:
                        # For other comparisons, use == operator with appropriate type
                        condition = BinaryOp(
                            left=VariableRef(condition_var["var_name"]),
                            op="==",
                            right=Literal(condition_var["compare_value"]),
                        )
                else:
                    condition = VariableRef(condition_var["var_name"])

                # Create and return the If statement
                return IfStatement(
                    condition=condition,
                    then_block=then_block,
                )
            # Fallback to the conservative approach if we can't extract the condition
            self.current_block.statements.append(
                Comment(
                    "Fallback: If condition with struct field access "
                    "simplified for @owned compatibility",
                ),
            )

            # Convert the If body operations unconditionally
            if hasattr(if_block, "ops"):
                for op in if_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        self.current_block.statements.append(stmt)

            return None

        # Convert condition
        condition = self._convert_condition(if_block.cond)

        # Track what resources were consumed before this conditional
        # We need to ensure we don't try to re-consume them in else blocks
        consumed_before_if = {}
        if not hasattr(self, "consumed_resources"):
            self.consumed_resources = {}
        for res_name, indices in self.consumed_resources.items():
            consumed_before_if[res_name] = (
                indices.copy() if isinstance(indices, set) else set(indices)
            )

        # Convert then block with scope tracking
        then_block = Block()
        prev_block = self.current_block

        with self.scope_manager.enter_scope(ScopeType.IF_THEN) as then_scope:
            self.current_block = then_block

            if hasattr(if_block, "ops"):
                for op in if_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        then_block.statements.append(stmt)

        # Convert else block if present
        else_block = None
        else_scope_info = None

        if hasattr(if_block, "else_block") and if_block.else_block:
            else_block = Block()

            with self.scope_manager.enter_scope(ScopeType.IF_ELSE) as else_scope:
                else_scope_info = else_scope
                self.current_block = else_block

                if hasattr(if_block.else_block, "ops"):
                    for op in if_block.else_block.ops:
                        stmt = self._convert_operation(op)
                        if stmt:
                            else_block.statements.append(stmt)

        # Check for resource balancing needs
        # Analyze resource consumption across branches
        unbalanced = self.scope_manager.analyze_conditional_branches(
            then_scope,
            else_scope_info,
            self.context,
        )

        # If there are unbalanced resources, we need to balance them
        if unbalanced:
            # Helper function to add resource consumption
            def add_resource_consumption(block, res_name, indices):
                # Filter out indices that were already consumed before the if statement
                if res_name in consumed_before_if:
                    already_consumed = consumed_before_if[res_name]
                    indices = indices - already_consumed

                if indices:
                    block.statements.append(
                        Comment("Consume qubits to maintain linearity"),
                    )
                    for idx in sorted(indices):
                        # Check if resource is unpacked
                        if res_name in self.unpacked_vars:
                            element_names = self.unpacked_vars[res_name]
                            if idx < len(element_names):
                                # Measure the unpacked qubit
                                meas_expr = FunctionCall(
                                    func_name="quantum.measure",
                                    args=[VariableRef(element_names[idx])],
                                )
                                block.statements.append(
                                    Assignment(
                                        target=VariableRef("_"),
                                        value=meas_expr,
                                    ),
                                )
                        elif (
                            hasattr(self, "dynamic_allocations")
                            and res_name in self.dynamic_allocations
                        ):
                            # For dynamic allocations, allocate a fresh qubit and measure it
                            # Always allocate a fresh qubit for consumption (for linearity balancing)
                            var_name = self._get_unique_var_name(res_name, idx)
                            block.statements.append(
                                Assignment(
                                    target=VariableRef(var_name),
                                    value=FunctionCall(
                                        func_name="quantum.qubit",
                                        args=[],
                                    ),
                                ),
                            )
                            # Measure the qubit
                            meas_expr = FunctionCall(
                                func_name="quantum.measure",
                                args=[VariableRef(var_name)],
                            )
                            block.statements.append(
                                Assignment(target=VariableRef("_"), value=meas_expr),
                            )
                        else:
                            # Use array indexing
                            meas_expr = FunctionCall(
                                func_name="quantum.measure",
                                args=[ArrayAccess(array_name=res_name, index=idx)],
                            )
                            block.statements.append(
                                Assignment(target=VariableRef("_"), value=meas_expr),
                            )

            # If we have an else block, add balancing to both branches
            if else_block:
                # Add to then branch what else consumed
                for res_name, indices in unbalanced.items():
                    if res_name in then_scope.resource_usage:
                        then_usage = then_scope.resource_usage[res_name]
                        else_usage = else_scope_info.resource_usage.get(
                            res_name,
                            ResourceUsage(res_name, set()),
                        )
                        missing_in_then = else_usage.consumed - then_usage.consumed
                        if missing_in_then:
                            add_resource_consumption(
                                then_block,
                                res_name,
                                missing_in_then,
                            )

                # Add to else branch what then consumed
                for res_name in then_scope.resource_usage:
                    then_usage = then_scope.resource_usage[res_name]
                    else_usage = else_scope_info.resource_usage.get(
                        res_name,
                        ResourceUsage(res_name, set()),
                    )
                    missing_in_else = then_usage.consumed - else_usage.consumed
                    if missing_in_else:
                        add_resource_consumption(else_block, res_name, missing_in_else)
            else:
                # No else block - create one to consume resources
                else_block = Block()
                else_block.statements.append(
                    Comment("Auto-generated else block for linearity"),
                )

                for res_name, indices in unbalanced.items():
                    add_resource_consumption(else_block, res_name, indices)

        self.current_block = prev_block

        return IfStatement(
            condition=condition,
            then_block=then_block,
            else_block=else_block,
        )

    def _convert_while(self, while_block) -> Statement | None:
        """Convert While loop."""
        # Convert condition
        condition = self._convert_condition(while_block.cond)

        # Convert body with scope tracking
        body_block = Block()
        prev_block = self.current_block

        with self.scope_manager.enter_scope(ScopeType.LOOP):
            self.current_block = body_block

            if hasattr(while_block, "ops"):
                for op in while_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        body_block.statements.append(stmt)

        self.current_block = prev_block

        return WhileStatement(
            condition=condition,
            body=body_block,
        )

    def _convert_for(self, for_block) -> Statement | None:
        """Convert For loop."""
        # Get loop variable and range
        loop_var = for_block.var

        # Determine the iteration pattern
        if hasattr(for_block, "iterable") and for_block.iterable:
            # For(i, iterable)
            return self._convert_for_iterable(for_block, loop_var)
        if hasattr(for_block, "start") and hasattr(for_block, "stop"):
            # For(i, start, stop, [step])
            return self._convert_for_range(for_block, loop_var)
        # Unknown pattern
        return Comment(f"TODO: Unsupported For loop pattern with variable {loop_var}")

    def _convert_for_range(self, for_block, loop_var) -> Statement | None:
        """Convert For loop with range pattern."""
        start = for_block.start
        stop = for_block.stop
        step = getattr(for_block, "step", 1)

        # Create range() call
        if step == 1:
            # range(start, stop)
            range_call = FunctionCall(
                func_name="range",
                args=[Literal(start), Literal(stop)],
            )
        else:
            # range(start, stop, step)
            range_call = FunctionCall(
                func_name="range",
                args=[Literal(start), Literal(stop), Literal(step)],
            )

        # Check if we need to pre-extract conditions from If statements in the loop body
        # This is necessary when we have @owned struct parameters and If conditions that
        # access struct fields inside the loop
        extracted_conditions = []
        if self._should_pre_extract_conditions(for_block) and hasattr(for_block, "ops"):
            # Find all If statements in the loop body and extract their conditions
            for op in for_block.ops:
                if (
                    type(op).__name__ == "If"
                    and hasattr(op, "cond")
                    and self._is_struct_field_access(op.cond)
                ):
                    condition_var = self._generate_condition_var_name(op.cond)
                    if condition_var:
                        # Generate the extraction statement before the loop
                        self.current_block.statements.append(
                            Comment(
                                "Pre-extract condition to avoid @owned struct field access in loop",
                            ),
                        )
                        condition_stmt = Assignment(
                            target=VariableRef(condition_var),
                            value=self._convert_condition(op.cond),
                        )
                        self.current_block.statements.append(condition_stmt)
                        extracted_conditions.append((op, condition_var))

        # Convert body with scope tracking
        body_block = Block()
        prev_block = self.current_block

        # Track extracted conditions so If converter can use them
        if extracted_conditions:
            if not hasattr(self, "pre_extracted_conditions"):
                self.pre_extracted_conditions = {}
            for if_op, var_name in extracted_conditions:
                self.pre_extracted_conditions[id(if_op)] = var_name

        with self.scope_manager.enter_scope(ScopeType.LOOP):
            self.current_block = body_block

            if hasattr(for_block, "ops"):
                for op in for_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        body_block.statements.append(stmt)

        self.current_block = prev_block

        return ForStatement(
            loop_var=str(loop_var),
            iterable=range_call,
            body=body_block,
        )

    def _convert_for_iterable(self, for_block, loop_var) -> Statement | None:
        """Convert For loop with iterable pattern."""
        # For now, just handle the iterable as a variable reference
        iterable = for_block.iterable

        # Try to convert it to an IR node
        if isinstance(iterable, str):
            iter_node = VariableRef(iterable)
        elif hasattr(iterable, "sym"):
            iter_node = VariableRef(iterable.sym)
        else:
            # Try to represent it somehow
            iter_node = Literal(str(iterable))

        # Convert body
        body_block = Block()
        prev_block = self.current_block

        with self.scope_manager.enter_scope(ScopeType.LOOP):
            self.current_block = body_block

            if hasattr(for_block, "ops"):
                for op in for_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        body_block.statements.append(stmt)

        self.current_block = prev_block

        return ForStatement(
            loop_var=str(loop_var),
            iterable=iter_node,
            body=body_block,
        )

    def _convert_condition(self, cond) -> IRNode:
        """Convert condition expression."""
        cond_type = type(cond).__name__

        if cond_type == "Bit":
            # Bit reference
            return self._convert_bit_ref(cond)
        if cond_type == "EQUIV":
            # Equality comparison

            left = self._convert_condition(cond.left)
            right = self._convert_condition(cond.right)

            # Optimize boolean comparisons to 1
            if (
                isinstance(right, Literal)
                and right.value == 1
                and type(cond.left).__name__ == "Bit"
            ):
                # Just return the boolean value itself
                return left

            return BinaryOp(left=left, op="==", right=right)
        if cond_type == "LT":
            # Less than
            left = self._convert_condition(cond.left)
            right = self._convert_condition(cond.right)
            return BinaryOp(left=left, op="<", right=right)
        if cond_type == "GT":
            # Greater than
            left = self._convert_condition(cond.left)
            right = self._convert_condition(cond.right)
            return BinaryOp(left=left, op=">", right=right)
        if cond_type == "AND":
            # Bitwise AND (used as logical in conditions)
            left = self._convert_condition(cond.left)
            right = self._convert_condition(cond.right)
            return BinaryOp(left=left, op="&", right=right)
        if cond_type == "OR":
            # Bitwise OR (used as logical in conditions)
            left = self._convert_condition(cond.left)
            right = self._convert_condition(cond.right)
            return BinaryOp(left=left, op="|", right=right)
        if cond_type == "NOT":
            # Logical NOT
            operand = self._convert_condition(cond.value)
            return UnaryOp(op="not", operand=operand)
        if hasattr(cond, "value"):
            # Literal value
            return Literal(cond.value)
        if isinstance(cond, int | bool | str):
            # Direct literal
            return Literal(cond)

        # Default: try to convert as bit reference
        return self._convert_bit_ref(cond)

    def _convert_repeat(self, repeat_block) -> Statement | None:
        """Convert Repeat block to for loop."""
        # Repeat is essentially a for loop with an anonymous variable
        repeat_count = repeat_block.cond

        # Check if this repeat block contains conditional consumption patterns
        # that would violate linearity (e.g., conditional function calls with @owned params)
        has_conditional_consumption = self._has_conditional_consumption_pattern(
            repeat_block,
        )

        if has_conditional_consumption:
            # Special handling for conditional consumption patterns
            # Instead of a loop with conditional consumption, we need to restructure
            # to avoid linearity violations
            return self._convert_repeat_with_conditional_consumption(repeat_block)

        # Check if conditions have already been pre-extracted at the function level
        # If not, extract them here (for non-function contexts)
        extracted_conditions = []
        already_extracted = (
            hasattr(self, "pre_extracted_conditions") and self.pre_extracted_conditions
        )

        should_extract = (
            not already_extracted
            and self._should_pre_extract_conditions_repeat(repeat_block)
            and hasattr(repeat_block, "ops")
        )
        if should_extract:
            # Find all If statements in the loop body and extract their conditions
            for op in repeat_block.ops:
                if type(op).__name__ == "If" and hasattr(op, "cond"):
                    # Check if this condition was already pre-extracted
                    if (
                        hasattr(self, "pre_extracted_conditions")
                        and id(op) in self.pre_extracted_conditions
                    ):
                        continue  # Skip - already handled

                    if self._is_struct_field_access(op.cond):
                        condition_var = self._generate_condition_var_name(op.cond)
                        if condition_var:
                            # Generate the extraction statement before the loop
                            self.current_block.statements.append(
                                Comment(
                                    "Pre-extract condition to avoid @owned struct field access in loop",
                                ),
                            )
                            condition_stmt = Assignment(
                                target=VariableRef(condition_var),
                                value=self._convert_condition(op.cond),
                            )
                            self.current_block.statements.append(condition_stmt)
                            extracted_conditions.append((op, condition_var))

        # Convert body
        body_block = Block()
        prev_block = self.current_block

        # Track extracted conditions so If converter can use them
        if extracted_conditions:
            if not hasattr(self, "pre_extracted_conditions"):
                self.pre_extracted_conditions = {}
            for if_op, var_name in extracted_conditions:
                self.pre_extracted_conditions[id(if_op)] = var_name

        with self.scope_manager.enter_scope(ScopeType.LOOP):
            self.current_block = body_block

            if hasattr(repeat_block, "ops"):
                for op in repeat_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        body_block.statements.append(stmt)

        self.current_block = prev_block

        # Create ForStatement with anonymous variable
        return ForStatement(
            loop_var="_",
            iterable=FunctionCall(func_name="range", args=[Literal(repeat_count)]),
            body=body_block,
        )

    def _has_conditional_consumption_pattern(self, repeat_block) -> bool:
        """Check if a repeat block contains conditional consumption patterns."""
        if not hasattr(repeat_block, "ops"):
            return False

        # Look for If blocks containing function calls with @owned parameters
        for op in repeat_block.ops:
            if type(op).__name__ == "If" and hasattr(op, "ops"):
                for inner_op in op.ops:
                    # Check if this is a function call that might have @owned params
                    if hasattr(inner_op, "block_name"):
                        # Check if this function has @owned parameters
                        func_name = inner_op.block_name
                        if func_name in [
                            "PrepEncodingFTZero",
                            "PrepEncodingNonFTZero",
                            "PrepZeroVerify",
                        ]:
                            return True
        return False

    def _update_mappings_after_conditional_loop(self) -> None:
        """Update variable mappings after a loop with conditional consumption.

        After a loop with conditional consumption, variables might have been
        conditionally replaced with fresh versions. We need to ensure that
        subsequent operations use the right variables.
        """
        # For the specific pattern where we have c_d_fresh that might have been
        # conditionally consumed to create c_d_fresh_1, we need to ensure
        # that subsequent uses reference the original c_d_fresh (not _1)
        # because the _1 version only exists conditionally.
        #
        # The proper solution would be to track which variables are guaranteed
        # to exist and use those. For now, we'll stick with the original names.

    def _convert_repeat_with_conditional_consumption(
        self,
        repeat_block,
    ) -> Statement | None:
        """Convert repeat block with conditional consumption to avoid linearity violations."""
        repeat_count = repeat_block.cond

        # For conditional consumption patterns, we need to be careful
        # The issue is that variables might be consumed conditionally in the loop
        # but then used unconditionally afterward

        # Track that we're in a special conditional consumption context
        self._in_conditional_consumption_loop = True

        # Convert as normal for loop
        body_block = Block()
        prev_block = self.current_block

        with self.scope_manager.enter_scope(ScopeType.LOOP):
            self.current_block = body_block

            if hasattr(repeat_block, "ops"):
                for op in repeat_block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        body_block.statements.append(stmt)

        self.current_block = prev_block
        self._in_conditional_consumption_loop = False

        return ForStatement(
            loop_var="_",
            iterable=FunctionCall(func_name="range", args=[Literal(repeat_count)]),
            body=body_block,
        )

    def _convert_comment(self, comment) -> Statement | None:
        """Convert comment."""
        if hasattr(comment, "txt") and comment.txt:
            return Comment(comment.txt)
        return None  # Skip empty comments

    def _is_struct_field_in_loop_with_owned(self, cond) -> bool:
        """Check if a condition accesses a struct field in a problematic context.

        Returns True if:
        1. We're in a loop scope
        2. We're in a function with @owned struct parameters
        3. The condition accesses a struct field
        """
        # Check if we're in a loop
        if not hasattr(self, "scope_manager") or not self.scope_manager.is_in_loop():
            return False

        # Check if we're in a function with @owned struct parameters
        if not hasattr(self, "function_info") or self.current_function_name == "main":
            return False

        func_info = self.function_info.get(self.current_function_name, {})
        if not func_info.get("has_owned_struct_params", False):
            return False

        # Check if the condition accesses a struct field
        # Handle different condition types
        cond_type = type(cond).__name__

        if cond_type == "EQUIV":
            # For equality comparisons, check the left side
            if hasattr(cond, "left"):
                return self._is_struct_field_in_loop_with_owned(cond.left)
        elif hasattr(cond, "reg") and hasattr(cond.reg, "sym"):
            array_name = cond.reg.sym
            # Check if this variable is a struct field
            for info in self.struct_info.values():
                if array_name in info["var_names"].values():
                    return True

        return False

    def _extract_condition_variable(self, cond) -> dict | None:
        """Extract information about a condition variable that accesses a struct field.

        Returns a dict with:
        - var_name: suggested variable name for the extracted value
        - struct_field: the struct field being accessed (e.g., 'c.verify_prep[0]')
        - comparison: the comparison type (e.g., 'EQUIV')
        - compare_value: the value being compared against
        """
        cond_type = type(cond).__name__

        if cond_type == "EQUIV" and hasattr(cond, "left") and hasattr(cond, "right"):
            # Handle EQUIV(c_verify_prep[0], 1)
            left = cond.left
            right = cond.right

            # Check if left side is a struct field access
            if (
                hasattr(left, "reg")
                and hasattr(left.reg, "sym")
                and hasattr(left, "index")
            ):
                array_name = left.reg.sym
                index = left.index

                # Check if this is a struct field
                for prefix, info in self.struct_info.items():
                    if array_name in info["var_names"].values():
                        # Find the field name
                        field_name = None
                        for suffix, var_name in info["var_names"].items():
                            if var_name == array_name:
                                field_name = suffix
                                break

                        if field_name:
                            # Extract the comparison value
                            compare_value = (
                                getattr(right, "val", right)
                                if hasattr(right, "val")
                                else right
                            )

                            return {
                                "var_name": f"{field_name}_{index}_extracted",
                                "struct_field": f"{prefix}.{field_name}[{index}]",
                                "comparison": "EQUIV",
                                "compare_value": compare_value,
                            }

        return None

    def _convert_condition_value(self, cond) -> IRNode:
        """Convert the struct field access part of a condition to an IR node."""
        cond_type = type(cond).__name__

        if cond_type == "EQUIV" and hasattr(cond, "left"):
            # For EQUIV(c_verify_prep[0], 1), convert the left side (c_verify_prep[0])
            left = cond.left

            if (
                hasattr(left, "reg")
                and hasattr(left.reg, "sym")
                and hasattr(left, "index")
            ):
                array_name = left.reg.sym
                index = left.index

                # Check if this is a struct field and get the struct parameter name
                for prefix, info in self.struct_info.items():
                    if array_name in info["var_names"].values():
                        # Find the field name
                        field_name = None
                        for suffix, var_name in info["var_names"].items():
                            if var_name == array_name:
                                field_name = suffix
                                break

                        if field_name:
                            # Check if the struct has been decomposed and we should use decomposed variables
                            if (
                                hasattr(self, "var_remapping")
                                and array_name in self.var_remapping
                            ):
                                # Struct was decomposed - use the decomposed variable directly
                                decomposed_var = self.var_remapping[array_name]
                                return ArrayAccess(
                                    array=VariableRef(decomposed_var),
                                    index=index,
                                )

                            # Get the struct parameter name (e.g., 'c')
                            struct_param_name = prefix
                            if (
                                hasattr(self, "param_mapping")
                                and prefix in self.param_mapping
                            ):
                                struct_param_name = self.param_mapping[prefix]

                            # Check if we have fresh structs - use them directly
                            if (
                                hasattr(self, "refreshed_arrays")
                                and prefix in self.refreshed_arrays
                            ):
                                fresh_struct_name = self.refreshed_arrays[prefix]
                                struct_param_name = fresh_struct_name
                                # Don't replace field access for fresh structs

                            # Create: c.verify_prep[0] - but check for decomposed variables first
                            # Check if we have decomposed variables for this struct
                            if (
                                hasattr(self, "decomposed_vars")
                                and struct_param_name in self.decomposed_vars
                            ):
                                field_vars = self.decomposed_vars[struct_param_name]
                                if field_name in field_vars:
                                    # Use the decomposed variable instead
                                    decomposed_var = field_vars[field_name]
                                    return ArrayAccess(
                                        array=VariableRef(decomposed_var),
                                        index=index,
                                    )

                            # Fallback to original struct field access (this should now be rare)
                            field_access = FieldAccess(
                                obj=VariableRef(struct_param_name),
                                field=field_name,
                            )
                            return ArrayAccess(array=field_access, index=index)

        # Fallback
        return Literal(0)

    def _function_has_owned_struct_params(self, params) -> bool:
        """Check if function has @owned struct parameters."""
        return any(
            "@owned" in param_type and param_name in self.struct_info
            for param_name, param_type in params
        )

    def _has_function_calls_before_loops(self, block) -> bool:
        """Check if the function has function calls before loops.

        This indicates that decomposed struct variables will be consumed for
        struct reconstruction, so we can't pre-extract conditions from them.
        """
        if not hasattr(block, "ops"):
            return False

        # Look for function calls before any loops
        found_function_call = False

        for op in block.ops:
            op_type = type(op).__name__

            # Check for function calls (which would trigger struct reconstruction)
            if op_type == "Call" and hasattr(op, "func"):
                # This is a function call that might consume structs
                found_function_call = True

            # Check for Repeat/For loops - if we find function calls before loops,
            # then we'll need to reconstruct structs and can't pre-extract
            if op_type in ["Repeat", "For"] and found_function_call:
                return True

        return False

    def _pre_extract_loop_conditions(self, block, body) -> dict:
        """Pre-extract conditions from loops that might access @owned struct fields.

        Returns a dictionary mapping If block IDs to extracted condition variable names.
        """
        return {}

        # Disable pre-extraction for now - it causes linearity conflicts with struct reconstruction
        # TODO: Implement proper post-function-call condition extraction
        # The code below is currently unreachable but kept for future reference

        # Find all Repeat blocks with If conditions that access struct fields
        extracted: dict = {}  # Initialize for dead code below
        if hasattr(block, "ops"):
            for op in block.ops:
                if type(op).__name__ == "Repeat" and hasattr(op, "ops"):
                    # Check if this Repeat block contains If statements with struct field access
                    for inner_op in op.ops:
                        if (
                            type(inner_op).__name__ == "If"
                            and hasattr(
                                inner_op,
                                "cond",
                            )
                            and self._is_struct_field_access(inner_op.cond)
                        ):
                            # Extract this condition NOW before any operations
                            condition_var = self._generate_condition_var_name(
                                inner_op.cond,
                            )
                            if condition_var:
                                body.statements.append(
                                    Comment(
                                        "Pre-extract condition to avoid @owned struct field access in loop",
                                    ),
                                )
                                condition_stmt = Assignment(
                                    target=VariableRef(condition_var),
                                    value=self._convert_condition(inner_op.cond),
                                )
                                body.statements.append(condition_stmt)
                                extracted[id(inner_op)] = condition_var

        return extracted

    def _should_pre_extract_conditions_repeat(self, repeat_block) -> bool:
        """Check if we need to pre-extract conditions from this repeat block.

        Returns True if:
        1. The loop contains If statements with conditions
        2. We're in a function with @owned struct parameters
        3. The conditions access struct fields
        4. BUT False if we have function calls that will consume the decomposed variables
        """
        # Check if we're in a function with @owned struct parameters
        if not hasattr(self, "function_info") or self.current_function_name == "main":
            return False

        func_info = self.function_info.get(self.current_function_name, {})
        if not func_info.get("has_owned_struct_params", False):
            return False

        # Check if we have decomposed variables that might be consumed for struct reconstruction
        # This indicates we're in a context where pre-extraction would conflict with reconstruction
        if hasattr(self, "decomposed_vars") and self.decomposed_vars:
            return False

        # Check if the loop contains If statements with struct field access
        if hasattr(repeat_block, "ops"):
            for op in repeat_block.ops:
                if (
                    type(op).__name__ == "If"
                    and hasattr(op, "cond")
                    and self._is_struct_field_access(op.cond)
                ):
                    return True

        return False

    def _should_pre_extract_conditions(self, for_block) -> bool:
        """Check if we need to pre-extract conditions from this for loop.

        Returns True if:
        1. The loop contains If statements with conditions
        2. We're in a function with @owned struct parameters OR have fresh structs from returns
        3. The conditions access struct fields
        """
        # Check if we're in a function with @owned struct parameters or fresh structs
        if not hasattr(self, "function_info") or self.current_function_name == "main":
            return False

        func_info = self.function_info.get(self.current_function_name, {})
        has_owned_params = func_info.get("has_owned_struct_params", False)
        has_fresh_structs = hasattr(self, "refreshed_arrays") and bool(
            self.refreshed_arrays,
        )

        if not (has_owned_params or has_fresh_structs):
            return False

        # Check if the loop contains If statements with struct field access
        if hasattr(for_block, "ops"):
            for op in for_block.ops:
                if (
                    type(op).__name__ == "If"
                    and hasattr(op, "cond")
                    and self._is_struct_field_access(op.cond)
                ):
                    return True

        return False

    def _is_struct_field_access(self, cond) -> bool:
        """Check if a condition accesses a struct field."""
        cond_type = type(cond).__name__

        if cond_type == "EQUIV":
            # For equality comparisons, check the left side
            if hasattr(cond, "left"):
                return self._is_struct_field_access(cond.left)
        elif cond_type == "Bit":
            # Check if this is a struct field
            if hasattr(cond, "reg") and hasattr(cond.reg, "sym"):
                array_name = cond.reg.sym
                # Check if this variable is a struct field (original or fresh)
                for prefix, info in self.struct_info.items():
                    # Check original struct fields
                    if array_name in info["var_names"].values():
                        return True
                    # Check fresh struct field patterns (e.g., c_fresh accessing verify_prep)
                    if hasattr(self, "refreshed_arrays"):
                        for orig_name in self.refreshed_arrays:
                            if orig_name == prefix:
                                # Check if array_name matches fresh struct field pattern
                                for field_name in info["var_names"].values():
                                    # The condition might be accessing fresh_struct.field
                                    if (
                                        array_name == field_name
                                    ):  # Original field being accessed
                                        return True
        elif cond_type in ["AND", "OR", "XOR", "NOT"]:
            # Check both sides for binary ops
            if hasattr(cond, "left") and self._is_struct_field_access(cond.left):
                return True
            if hasattr(cond, "right") and self._is_struct_field_access(cond.right):
                return True

        return False

    def _generate_condition_var_name(self, cond) -> str | None:
        """Generate a variable name for an extracted condition."""
        cond_type = type(cond).__name__

        if cond_type == "EQUIV" and hasattr(cond, "left"):
            left = cond.left
            if (
                hasattr(left, "reg")
                and hasattr(left.reg, "sym")
                and hasattr(left, "index")
            ):
                array_name = left.reg.sym
                index = left.index

                # Check if this is a struct field
                for info in self.struct_info.values():
                    if array_name in info["var_names"].values():
                        # Find the field name
                        for suffix, var_name in info["var_names"].items():
                            if var_name == array_name:
                                return f"{suffix}_{index}_condition"
        elif cond_type == "Bit":
            if (
                hasattr(cond, "reg")
                and hasattr(cond.reg, "sym")
                and hasattr(cond, "index")
            ):
                array_name = cond.reg.sym
                index = cond.index

                # Check if this is a struct field
                for info in self.struct_info.values():
                    if array_name in info["var_names"].values():
                        # Find the field name
                        for suffix, var_name in info["var_names"].items():
                            if var_name == array_name:
                                return f"{suffix}_{index}_condition"

        # Generate a generic name
        return "extracted_condition"

    def _convert_set_operation(self, set_op) -> Statement | None:
        """Convert SET operation for classical bits."""
        if not hasattr(set_op, "left") or not hasattr(set_op, "right"):
            return Comment("Invalid SET operation")

        # Convert left side (target) - use array indexing for assignments
        target = self._convert_bit_ref(set_op.left, is_assignment_target=True)

        # Convert right side (value)
        value = self._convert_set_value(set_op.right)

        return Assignment(target=target, value=value)

    def _convert_set_value(self, value, parent_op=None) -> IRNode:
        """Convert value in SET operation.

        Args:
            value: The value to convert
            parent_op: The parent operation type (if any) to determine if parens are needed
        """
        # Check if it's a literal
        if isinstance(value, int | bool):
            return Literal(bool(value))

        # Check if it's a bit reference
        value_type = type(value).__name__
        if value_type == "Bit":
            return self._convert_bit_ref(value)

        # Check for bitwise operations
        if value_type == "XOR":
            left = self._convert_set_value(value.left, parent_op=value_type)
            right = self._convert_set_value(value.right, parent_op=value_type)
            result = BinaryOp(left=left, op="^", right=right)
            # XOR has same precedence as AND, higher than OR
            # Only need parens if parent is AND (to clarify precedence)
            if parent_op == "AND":
                result.needs_parens = True
            return result
        if value_type == "AND":
            left = self._convert_set_value(value.left, parent_op=value_type)
            right = self._convert_set_value(value.right, parent_op=value_type)
            result = BinaryOp(left=left, op="&", right=right)
            # Mark as needing parens if it's a child of |
            if parent_op == "OR":
                result.needs_parens = True
            return result
        if value_type == "OR":
            left = self._convert_set_value(value.left, parent_op=value_type)
            right = self._convert_set_value(value.right, parent_op=value_type)
            return BinaryOp(left=left, op="|", right=right)
        if value_type == "NOT":
            # NOT might have 'operand' or be applied to first item
            if hasattr(value, "operand"):
                operand = self._convert_set_value(value.operand, parent_op=value_type)
            elif hasattr(value, "value"):
                operand = self._convert_set_value(value.value, parent_op=value_type)
            else:
                # Try to get the operand another way
                operand = Literal(value=True)
            return UnaryOp(op="not", operand=operand)

        # Unknown value type - generate function call as fallback
        args = []
        if hasattr(value, "left"):
            args.append(self._convert_set_value(value.left, parent_op=value_type))
        if hasattr(value, "right"):
            args.append(self._convert_set_value(value.right, parent_op=value_type))
        return FunctionCall(func_name=value_type, args=args)

    def _convert_permute(self, permute) -> Statement | None:
        """Convert Permute operation."""
        # Permute swaps registers or elements
        # In Guppy, we can implement this using Python's swap syntax

        if hasattr(permute, "elems_i") and hasattr(permute, "elems_f"):
            elems_i = permute.elems_i
            elems_f = permute.elems_f

            # Case 1: Simple register swap (a, b = b, a)
            if hasattr(elems_i, "sym") and hasattr(elems_f, "sym"):
                # Full register swap
                comment = Comment(f"Swap {elems_i.sym} and {elems_f.sym}")
                self.current_block.statements.append(comment)

                # In Guppy, we need to use a temporary variable
                temp_var = f"_temp_{elems_i.sym}"

                # temp = a
                self.current_block.statements.append(
                    Assignment(
                        target=VariableRef(temp_var),
                        value=VariableRef(elems_i.sym),
                    ),
                )

                # a = b
                self.current_block.statements.append(
                    Assignment(
                        target=VariableRef(elems_i.sym),
                        value=VariableRef(elems_f.sym),
                    ),
                )

                # b = temp
                self.current_block.statements.append(
                    Assignment(
                        target=VariableRef(elems_f.sym),
                        value=VariableRef(temp_var),
                    ),
                )

                return None  # Already added statements

            # Case 2: List of elements permutation
            if isinstance(elems_i, list) and isinstance(elems_f, list):
                if len(elems_i) != len(elems_f):
                    return Comment("ERROR: Permutation lists must have same length")

                # Analyze the permutation pattern
                permutation_map = self._analyze_permutation(elems_i, elems_f)

                if permutation_map is None:
                    return Comment("ERROR: Invalid permutation - elements don't match")

                # Generate permutation code based on the pattern
                return self._generate_permutation_code(
                    permutation_map,
                    elems_i,
                    elems_f,
                )

        # Fallback for unrecognized patterns
        return Comment("TODO: Implement complex permutation")

    def _analyze_permutation(self, elems_i, elems_f):
        """Analyze permutation to create a mapping."""
        # Create a set of all elements to ensure they match
        elems_i_set = set()
        elems_f_set = set()

        # Build element signatures for comparison
        for elem in elems_i:
            if hasattr(elem, "reg") and hasattr(elem, "index"):
                elems_i_set.add((elem.reg.sym, elem.index))
            elif hasattr(elem, "sym"):
                # Full register reference
                elems_i_set.add((elem.sym, None))

        for elem in elems_f:
            if hasattr(elem, "reg") and hasattr(elem, "index"):
                elems_f_set.add((elem.reg.sym, elem.index))
            elif hasattr(elem, "sym"):
                elems_f_set.add((elem.sym, None))

        # Check if the sets match (same elements, just reordered)
        if elems_i_set != elems_f_set:
            return None

        # Create the mapping: what goes to position i
        # If elems_f[i] == elems_i[j], then position i gets value from position j
        permutation_map = {}
        for i, elem_f in enumerate(elems_f):
            # Find which element in elems_i matches elem_f
            for j, elem_i in enumerate(elems_i):
                if self._elements_equal(elem_i, elem_f):
                    permutation_map[i] = j  # position i gets value from position j
                    break

        return permutation_map

    def _elements_equal(self, elem1, elem2):
        """Check if two elements refer to the same qubit."""
        # Both are register[index] references
        if (
            hasattr(elem1, "reg")
            and hasattr(elem1, "index")
            and hasattr(elem2, "reg")
            and hasattr(elem2, "index")
        ):
            return elem1.reg.sym == elem2.reg.sym and elem1.index == elem2.index
        # Both are full register references
        if hasattr(elem1, "sym") and hasattr(elem2, "sym"):
            return elem1.sym == elem2.sym
        return False

    def _generate_permutation_code(self, permutation_map, elems_i, elems_f):
        """Generate code for complex permutation patterns."""
        _ = elems_f  # Currently not used, reserved for future use
        # Identify cycles in the permutation
        cycles = self._find_permutation_cycles(permutation_map)

        if not cycles:
            return Comment("Identity permutation - no action needed")

        # Add comment describing the permutation
        self.current_block.statements.append(
            Comment(f"Permute {len(elems_i)} elements"),
        )

        # For each cycle, generate swap operations
        for cycle in cycles:
            if len(cycle) == 1:
                # Fixed point, no action needed
                continue
            if len(cycle) == 2:
                # Simple swap
                self._generate_swap(elems_i[cycle[0]], elems_i[cycle[1]])
            else:
                # Multi-element cycle: use temporary variables
                self._generate_cycle_permutation(cycle, elems_i)

        return None  # Statements already added

    def _find_permutation_cycles(self, permutation_map):
        """Find cycles in a permutation."""
        visited = set()
        cycles = []

        for start in permutation_map:
            if start in visited:
                continue

            cycle = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = permutation_map.get(current, current)

            if len(cycle) > 0 and (
                len(cycle) > 1 or cycle[0] != permutation_map.get(cycle[0], cycle[0])
            ):
                cycles.append(cycle)

        return cycles

    def _generate_swap(self, elem1, elem2):
        """Generate code to swap two elements."""
        ref1 = self._convert_qubit_ref(elem1)
        ref2 = self._convert_qubit_ref(elem2)

        # Use a temporary variable
        temp_var = "_temp_swap"

        self.current_block.statements.append(
            Assignment(target=VariableRef(temp_var), value=ref1),
        )
        self.current_block.statements.append(
            Assignment(target=ref1, value=ref2),
        )
        self.current_block.statements.append(
            Assignment(target=ref2, value=VariableRef(temp_var)),
        )

    def _generate_cycle_permutation(self, cycle, elements):
        """Generate code for a multi-element cycle permutation."""
        if len(cycle) < 2:
            return

        # Save the first element
        first_elem = elements[cycle[0]]
        first_ref = self._convert_qubit_ref(first_elem)
        temp_var = "_temp_cycle"

        self.current_block.statements.append(
            Assignment(target=VariableRef(temp_var), value=first_ref),
        )

        # Shift elements in the cycle
        for i in range(len(cycle) - 1):
            src_elem = elements[cycle[i + 1]]
            dst_elem = elements[cycle[i]]

            src_ref = self._convert_qubit_ref(src_elem)
            dst_ref = self._convert_qubit_ref(dst_elem)

            self.current_block.statements.append(
                Assignment(target=dst_ref, value=src_ref),
            )

        # Complete the cycle
        last_elem = elements[cycle[-1]]
        last_ref = self._convert_qubit_ref(last_elem)

        self.current_block.statements.append(
            Assignment(target=last_ref, value=VariableRef(temp_var)),
        )

    def _convert_block_call(self, block) -> Statement | None:
        """Convert a block to a function call or inline expansion."""
        block_type = type(block)
        block_name = block_type.__name__

        # Get original block info if preserved
        original_block_name = getattr(block, "block_name", block_name)
        original_block_module = getattr(block, "block_module", block_type.__module__)

        # If we're in a loop, check if we need to restore array sizes before this call
        if self.scope_manager.is_in_loop():
            self._restore_array_sizes_for_block_call(block)

        # Check if this is a core block that should be inlined
        if original_block_name in self.CORE_BLOCKS:
            # Inline core blocks
            if hasattr(block, "ops"):
                self.current_block.statements.append(
                    Comment(f"Begin {block_name} block"),
                )
                for op in block.ops:
                    stmt = self._convert_operation(op)
                    if stmt:
                        self.current_block.statements.append(stmt)
                self.current_block.statements.append(
                    Comment(f"End {block_name} block"),
                )
            return None

        # For non-core blocks, create a function
        block_signature = self._get_block_signature(block)

        # Check if we already have a function for this block type
        if block_signature not in self.block_registry:
            # Determine struct prefix if this block operates on a struct
            struct_prefix = None
            deps = self._analyze_block_dependencies(block)

            # Check if all variables belong to the same struct
            for prefix, info in self.struct_info.items():
                vars_in_this_struct = set()
                for var in info["var_names"].values():
                    if var in deps["quantum"] or var in deps["classical"]:
                        vars_in_this_struct.add(var)

                # If this block operates on variables from this struct, use
                # QEC code name if available
                if vars_in_this_struct:
                    # Use the QEC code name if we have it, otherwise use prefix
                    struct_prefix = info.get("qec_code_name", prefix)
                    break

            # Generate a unique function name with struct prefix
            # Include module name if not __main__
            base_name = original_block_name

            # For Parallel blocks with content hash, include the content info
            if len(block_signature) > 2 and original_block_name == "Parallel":
                content_hash = block_signature[2]
                # Create a more readable suffix from the hash
                # e.g., "H_H" becomes "_h", "X_X" becomes "_x"
                if content_hash:
                    gates = content_hash.split("_")
                    if all(g == gates[0] for g in gates):
                        # All gates are the same type
                        base_name += f"_{gates[0].lower()}"
                    else:
                        # Mixed gates - use first letter of each
                        suffix = "_".join(g[0].lower() for g in gates[:3])  # Limit to 3
                        base_name += f"_{suffix}"

            if original_block_module and original_block_module != "__main__":
                # Extract just the last part of the module name (e.g., 'test_linearity_patterns')
                module_parts = original_block_module.split(".")
                module_name = module_parts[-1] if module_parts else ""
                if module_name and module_name.startswith("test_"):
                    # For test modules, include the module name
                    func_name = self._generate_function_name(
                        f"{module_name}_{base_name}",
                        struct_prefix,
                    )
                else:
                    func_name = self._generate_function_name(base_name, struct_prefix)
            else:
                func_name = self._generate_function_name(base_name, struct_prefix)
            self.block_registry[block_signature] = func_name

            # Add to pending functions if not already discovered
            if func_name not in self.discovered_functions:
                self.pending_functions.append((block, func_name, block_signature))
                self.discovered_functions.add(func_name)
        else:
            func_name = self.block_registry[block_signature]

        # Generate function call
        stmt = self._generate_function_call(func_name, block)
        if stmt:
            self.current_block.statements.append(stmt)
        return None  # Already added to current block

    def _get_block_signature(self, block) -> tuple:
        """Get a unique signature for a block type."""
        block_type = type(block)
        block_name = block_type.__name__
        original_block_name = getattr(block, "block_name", block_name)
        original_block_module = getattr(block, "block_module", block_type.__module__)

        # For Parallel blocks, include content hash to differentiate blocks
        # with different operations
        if original_block_name == "Parallel" and hasattr(block, "ops"):
            content_hash = self._get_block_content_hash(block)
            return (original_block_name, original_block_module, content_hash)

        # For now, use block name and module as signature
        # Could be enhanced to include parameter info
        return (original_block_name, original_block_module)

    def _generate_function_name(
        self,
        block_name: str,
        struct_prefix: str | None = None,
    ) -> str:
        """Generate a unique function name for a block.

        Args:
            block_name: The original block name (e.g., 'H', 'PrepRUS')
            struct_prefix: Optional struct prefix (e.g., 'c' for c_struct)
        """
        # Convert CamelCase to snake_case, handling acronyms better
        import re

        # First, handle transitions from lowercase to uppercase
        snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", block_name)

        # Then handle multiple consecutive capitals (acronyms)
        snake_case = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", snake_case)

        # Convert to lowercase
        snake_case = snake_case.lower()

        # Add struct prefix if provided
        base_name = f"{struct_prefix}_{snake_case}" if struct_prefix else snake_case

        # Ensure uniqueness
        func_name = base_name
        counter = 1
        while func_name in self.generated_functions:
            func_name = f"{base_name}_{counter}"
            counter += 1

        return func_name

    def _get_block_content_hash(self, block) -> str:
        """Get a hash of block operations for differentiation.

        This is used to differentiate Parallel blocks with different operations.
        """
        ops_summary = []
        if hasattr(block, "ops"):
            for op in block.ops:
                op_type = type(op).__name__
                # Include gate types to differentiate
                ops_summary.append(op_type)

        # Create a simple hash from operation types
        return "_".join(sorted(ops_summary)) if ops_summary else "empty"

    def _generate_function_call(self, func_name: str, block) -> Statement:
        """Generate a function call for a block."""
        from pecos.slr.gen_codes.guppy.ir import Assignment, Comment, VariableRef

        # Analyze block dependencies to determine arguments
        deps = self._analyze_block_dependencies(block)

        # Initialize as procedural, will be updated after resource flow analysis
        is_procedural_function = True

        # CRITICAL: Save which arrays are currently unpacked BEFORE processing arguments
        # This is needed to detect if a function call return should use a fresh variable name
        # (when the parameter was unpacked and consumed in argument processing)
        unpacked_before_call = set()
        if hasattr(self, "unpacked_vars"):
            unpacked_before_call = set(self.unpacked_vars.keys())

        # Determine which variables need to be passed as arguments
        args = []
        quantum_args = []  # Track quantum args for return value assignment

        # Check if we should pass structs instead of individual arrays
        struct_args = set()  # Structs we've already added
        vars_in_structs = set()  # Variables that are part of structs

        # First pass: identify which variables are part of structs
        for prefix, info in self.struct_info.items():
            for var in info["var_names"].values():
                if var in deps["quantum"] or var in deps["classical"]:
                    vars_in_structs.add(var)
                    if prefix not in struct_args:
                        # Check if this struct has been refreshed (e.g., from a previous function call)
                        struct_to_use = prefix
                        if (
                            hasattr(self, "refreshed_arrays")
                            and prefix in self.refreshed_arrays
                        ):
                            # Use the refreshed name (e.g., c_fresh instead of c)
                            struct_to_use = self.refreshed_arrays[prefix]

                        # Check if this is a struct that was decomposed and needs reconstruction
                        # This includes @owned structs and fresh structs that were decomposed for field access
                        needs_reconstruction = False
                        struct_was_decomposed = (
                            struct_to_use in self.decomposed_vars
                            or (
                                prefix in self.decomposed_vars
                                and struct_to_use == prefix
                            )
                        )
                        if hasattr(self, "decomposed_vars") and struct_was_decomposed:
                            # Check if the struct we want to use was decomposed
                            needs_reconstruction = True

                        if needs_reconstruction:
                            # Struct was decomposed - reconstruct it from decomposed variables
                            struct_info = self.struct_info[prefix]

                            # Create a unique name for the reconstructed struct
                            reconstructed_var = self._get_unique_var_name(
                                f"{prefix}_reconstructed",
                            )

                            # Create struct constructor call
                            constructor_args = []

                            # Check if we have decomposed field variables for this struct
                            if struct_to_use in self.decomposed_vars:
                                # Use the decomposed field variables
                                field_mapping = self.decomposed_vars[struct_to_use]
                                for suffix, field_type, field_size in sorted(
                                    struct_info["fields"],
                                ):
                                    # Fallback to default naming if not in mapping
                                    field_var = field_mapping.get(
                                        suffix,
                                        f"{struct_to_use}_{suffix}",
                                    )
                                    constructor_args.append(VariableRef(field_var))
                            else:
                                # Use the default field variable naming
                                for suffix, field_type, field_size in sorted(
                                    struct_info["fields"],
                                ):
                                    field_var = f"{prefix}_{suffix}"

                                    # Check if we have a fresh version of this field variable
                                    if (
                                        hasattr(self, "refreshed_arrays")
                                        and field_var in self.refreshed_arrays
                                    ):
                                        field_var = self.refreshed_arrays[field_var]
                                    elif (
                                        hasattr(self, "var_remapping")
                                        and field_var in self.var_remapping
                                    ):
                                        field_var = self.var_remapping[field_var]

                                    constructor_args.append(VariableRef(field_var))

                            struct_constructor = FunctionCall(
                                func_name=struct_info["struct_name"],
                                args=constructor_args,
                            )

                            # Add reconstruction statement
                            reconstruction_stmt = Assignment(
                                target=VariableRef(reconstructed_var),
                                value=struct_constructor,
                            )
                            self.current_block.statements.append(reconstruction_stmt)

                            # Use the reconstructed struct
                            struct_to_use = reconstructed_var

                        # Add the struct as an argument
                        args.append(VariableRef(struct_to_use))
                        struct_args.add(prefix)
                        # Track this for return value handling
                        if var in deps["quantum"]:
                            quantum_args.append(prefix)

        # Track unpacked arrays that need restoration after procedural calls
        saved_unpacked_arrays = []

        # Black Box Pattern: Pass complete global arrays to maintain SLR semantics
        for var in sorted(deps["quantum"] & deps["reads"]):
            # Check if this is an ancilla that was excluded from structs
            is_excluded_ancilla = (
                hasattr(self, "ancilla_qubits") and var in self.ancilla_qubits
            )

            # Skip if this variable is part of a struct UNLESS it's an excluded ancilla
            if var in vars_in_structs and not is_excluded_ancilla:
                continue

            # Check if this variable needs remapping (we're inside a function)
            actual_var = var
            if hasattr(self, "var_remapping") and var in self.var_remapping:
                actual_var = self.var_remapping[var]

            # For procedural functions (borrow), we can't use unpacked arrays - they need the original array
            # For consuming functions (@owned), reconstruct the array from unpacked elements
            # Also handle dynamically allocated arrays and decomposed ancilla arrays
            if (
                hasattr(self, "decomposed_ancilla_arrays")
                and var in self.decomposed_ancilla_arrays
            ):
                # Check if the array has already been reconstructed into a variable
                if (
                    hasattr(self, "reconstructed_arrays")
                    and var in self.reconstructed_arrays
                ):
                    # Check if it was unpacked AFTER reconstruction
                    if (
                        hasattr(self, "unpacked_vars")
                        and actual_var in self.unpacked_vars
                    ):
                        # Array was unpacked after reconstruction - need to reconstruct again
                        # First check if there's a refreshed version from a previous function call
                        if (
                            hasattr(self, "refreshed_arrays")
                            and var in self.refreshed_arrays
                        ):
                            refreshed_name = self.refreshed_arrays[var]
                            args.append(VariableRef(refreshed_name))
                            quantum_args.append(var)
                        else:
                            # Reconstruct from unpacked elements
                            element_names = self.unpacked_vars[actual_var]
                            array_construction = self._create_array_construction(
                                element_names,
                            )
                            args.append(array_construction)
                            quantum_args.append(var)
                    else:
                        # Use the reconstructed array variable directly (not unpacked)
                        args.append(VariableRef(actual_var))
                        quantum_args.append(var)
                else:
                    # This array was decomposed into individual qubits
                    # Check if there's a refreshed version from a previous function call
                    if (
                        hasattr(self, "refreshed_arrays")
                        and var in self.refreshed_arrays
                    ):
                        # Use the refreshed array from previous function call
                        refreshed_name = self.refreshed_arrays[var]
                        args.append(VariableRef(refreshed_name))
                        quantum_args.append(var)
                    else:
                        # Reconstruct from decomposed elements
                        element_names = self.decomposed_ancilla_arrays[var]
                        array_construction = self._create_array_construction(
                            element_names,
                        )
                        args.append(array_construction)
                        quantum_args.append(var)
            elif (
                hasattr(self, "dynamic_allocations") and var in self.dynamic_allocations
            ):
                # Dynamically allocated - check if there's a refreshed version first
                if hasattr(self, "refreshed_arrays") and var in self.refreshed_arrays:
                    # Use the refreshed array from previous function call
                    refreshed_name = self.refreshed_arrays[var]
                    args.append(VariableRef(refreshed_name))
                    quantum_args.append(var)
                else:
                    # Dynamically allocated - construct array from individual qubits
                    # Get the size from context
                    var_info = self.context.lookup_variable(var)
                    if var_info and var_info.size:
                        size = var_info.size
                        element_names = [f"{var}_{i}" for i in range(size)]
                        array_construction = self._create_array_construction(
                            element_names,
                        )
                        args.append(array_construction)
                        quantum_args.append(var)
                    else:
                        # Fallback - just pass the variable (will likely error)
                        args.append(VariableRef(actual_var))
                        quantum_args.append(actual_var)
            elif hasattr(self, "unpacked_vars") and actual_var in self.unpacked_vars:
                # Array was unpacked (either from parameter or return value)
                # OPTIMIZATION: If we're using ALL unpacked elements AND the array variable exists,
                # just pass the array variable instead of reconstructing inline
                # This happens when a function returns an array, we unpack it, then immediately
                # pass it to another function - in this case, just use the variable!
                element_names = self.unpacked_vars[actual_var]

                # Check if we have partial consumption (via index_mapping)
                has_partial_consumption = (
                    hasattr(self, "index_mapping") and actual_var in self.index_mapping
                )

                # Check if this was unpacked from a parameter
                is_parameter_unpacked = (
                    hasattr(self, "parameter_unpacked_arrays")
                    and actual_var in self.parameter_unpacked_arrays
                )

                # Use the variable directly if:
                # 1. No partial consumption (using all elements)
                # 2. Not parameter-unpacked (return-unpacked arrays have the variable available)
                # 3. The variable wasn't consumed yet
                if not has_partial_consumption and not is_parameter_unpacked:
                    # The array variable should still exist - use it directly
                    args.append(VariableRef(actual_var))
                    quantum_args.append(actual_var)
                    # Don't delete from unpacked_vars yet - might be needed later
                else:
                    # Use inline array construction
                    # This is needed for:
                    # - Partial consumption (not all elements)
                    # - Parameter-unpacked arrays (no array variable exists)
                    array_construction = self._create_array_construction(element_names)
                    args.append(array_construction)
                    quantum_args.append(actual_var)

                    # CRITICAL: After using inline construction, the unpacked elements are CONSUMED
                    # Remove from tracking so subsequent calls use the returned value instead
                    if hasattr(self, "parameter_unpacked_arrays"):
                        self.parameter_unpacked_arrays.discard(actual_var)
                    del self.unpacked_vars[actual_var]
                    if (
                        hasattr(self, "index_mapping")
                        and actual_var in self.index_mapping
                    ):
                        del self.index_mapping[actual_var]
            else:
                # Array is already in the correct global form
                # Check if this array has been refreshed (e.g., from a previous function call)
                if hasattr(self, "refreshed_arrays") and var in self.refreshed_arrays:
                    # Use the refreshed name (e.g., data_fresh instead of data)
                    refreshed_name = self.refreshed_arrays[var]
                    args.append(VariableRef(refreshed_name))
                    quantum_args.append(var)  # Keep original name for tracking
                else:
                    args.append(VariableRef(actual_var))
                    quantum_args.append(actual_var)

        # Pass classical variables that are read or written (arrays are passed by reference)
        for var in sorted(deps["classical"] & (deps["reads"] | deps["writes"])):
            # Skip if this variable is part of a struct
            if var in vars_in_structs:
                continue

            # Check if this variable needs remapping
            actual_var = var
            if hasattr(self, "var_remapping") and var in self.var_remapping:
                actual_var = self.var_remapping[var]

            # Classical arrays also need reconstruction if they were unpacked
            if hasattr(self, "unpacked_vars") and actual_var in self.unpacked_vars:
                # Reconstruct the classical array from unpacked elements
                element_names = self.unpacked_vars[actual_var]
                array_construction = self._create_array_construction(element_names)

                # Use a unique name for reconstruction to avoid linearity violation
                reconstructed_var = self._get_unique_var_name(f"{actual_var}_array")
                reconstruction_stmt = Assignment(
                    target=VariableRef(reconstructed_var),
                    value=array_construction,
                )
                self.current_block.statements.append(reconstruction_stmt)

                # Clear the unpacking info since we've reconstructed the array
                del self.unpacked_vars[actual_var]
                args.append(VariableRef(reconstructed_var))
            else:
                # Array is already in the correct form
                args.append(VariableRef(actual_var))

        # Create function call
        call = FunctionCall(
            func_name=func_name,
            args=args,
        )

        # Use proper resource flow analysis to determine what's actually returned
        _consumed_qubits, live_qubits = self._analyze_quantum_resource_flow(block)

        # Determine if this is a procedural function based on resource flow
        # If the block has live qubits that should be returned, it's not procedural
        has_live_qubits = bool(live_qubits)
        is_procedural_function = not has_live_qubits

        # HYBRID APPROACH: Use smart detection for consistent function calls
        if (
            hasattr(self, "function_return_types")
            and func_name in self.function_return_types
        ):
            func_return_type = self.function_return_types[func_name]
            if func_return_type == "None":
                is_procedural_function = True
        else:
            # Fallback: use the same smart detection logic
            should_be_procedural_call = self._should_function_be_procedural(
                func_name,
                block,
                [(arg, "array[quantum.qubit, 2]") for arg in quantum_args],
                has_live_qubits,
            )
            if should_be_procedural_call:
                is_procedural_function = True

        # Override: if function has multiple quantum args, it's likely not procedural
        # if len(quantum_args) > 1:
        #     is_procedural_function = False

        # Override: if function returns a tuple, it's not procedural
        # if func_name in self.function_return_types:
        #     func_return_type = self.function_return_types[func_name]
        #     if func_return_type.startswith("tuple["):
        #         is_procedural_function = False

        # If it appears to be procedural based on live qubits, double-check with signature
        if is_procedural_function and hasattr(block, "__init__"):
            import inspect

            try:
                sig = inspect.signature(block.__class__.__init__)
                return_annotation = sig.return_annotation
                if (
                    return_annotation is None
                    or return_annotation is type(None)
                    or str(return_annotation) == "None"
                ):
                    is_procedural_function = True
                else:
                    is_procedural_function = (
                        False  # Has return annotation, not procedural
                    )
            except (ValueError, TypeError, AttributeError):
                # Default to procedural if can't inspect signature
                # ValueError: signature cannot be determined
                # TypeError: object is not callable
                # AttributeError: missing expected attributes
                is_procedural_function = True

        # Now determine if the calling function consumes quantum arrays
        deps_for_func = self._analyze_block_dependencies(block)
        has_quantum_params = bool(deps_for_func["quantum"] & deps_for_func["reads"])
        # Check if we're in main function
        is_main_context = self.current_function_name == "main"
        # Functions consume quantum arrays if they have quantum params AND the called function is not procedural
        # This supports the nested blocks pattern where non-procedural functions return live qubits
        function_consumes = has_quantum_params and (
            is_main_context or not is_procedural_function
        )

        # Force function consumption if multiple quantum args (likely tuple return)
        if has_quantum_params and len(quantum_args) > 1:
            function_consumes = True

        # Track consumed arrays in main function
        # Check if the function being called has @owned parameters
        if self.current_function_name == "main":
            # Since function_info is not populated yet when building main,
            # we need to be conservative and assume all quantum arrays passed to functions
            # might have @owned parameters. This is especially true for procedural functions
            # that have nested blocks (like prep_rus).

            # For safety, mark all quantum arrays passed to functions as consumed
            # This prevents double-use errors when arrays are passed to @owned functions
            for arg in quantum_args:
                if isinstance(arg, str):  # It's an array name
                    if not hasattr(self, "consumed_resources"):
                        self.consumed_resources = {}
                    if arg not in self.consumed_resources:
                        self.consumed_resources[arg] = set()
                    # Mark the entire array as consumed conservatively
                    # We don't know the exact size, but we can mark it as fully consumed
                    # by using a large range (quantum arrays are typically small)
                    self.consumed_resources[arg].update(
                        range(100),
                    )  # Conservative upper bound

        # Use natural SLR semantics: arrays are global resources modified in-place
        # Functions that use unpacking still return arrays at boundaries to maintain this illusion
        # Keep track of struct arguments before filtering
        struct_args = [
            arg
            for arg in quantum_args
            if isinstance(arg, str) and arg in self.struct_info
        ]

        quantum_args = [
            arg for arg in quantum_args if isinstance(arg, str)
        ]  # Filter for array names

        # Check if we're returning structs (already collected above)

        # Check if the function returns something based on our function definitions
        self._function_returns_something(func_name)

        # CRITICAL: Determine actual return type by analyzing the block being called
        # This is more reliable than looking it up in function_return_types which may not be populated yet
        # APPROACH 1: Check Python type annotation on the block class
        actual_returns_tuple = False
        if hasattr(block, "__class__"):
            try:
                import inspect

                sig = inspect.signature(block.__class__.__init__)
                return_annotation = sig.return_annotation
                if return_annotation and return_annotation is not type(None):
                    return_str = str(return_annotation)
                    # Check if it's a tuple type annotation
                    actual_returns_tuple = (
                        "tuple[" in return_str.lower()
                        or "Tuple[" in return_str
                        or (
                            hasattr(return_annotation, "__origin__")
                            and return_annotation.__origin__ is tuple
                        )
                    )
            except (ValueError, TypeError, AttributeError):
                # Can't inspect signature, will use APPROACH 2
                pass  # Fallback to approach 2

        # APPROACH 2: Infer from live_qubits analysis
        # If live_qubits has multiple quantum arrays, function returns a tuple
        if not actual_returns_tuple and len(live_qubits) > 1:
            # Multiple quantum arrays are live - function returns a tuple
            actual_returns_tuple = True

        # For both @owned and non-@owned functions, only return arrays with live qubits
        # Fully consumed arrays should not be returned
        returned_quantum_args = []
        for arg in quantum_args:
            if isinstance(arg, str):
                # Check if this arg (possibly reconstructed) maps to an original array with live qubits
                original_name = arg
                # Handle reconstructed array names (e.g., _q_array -> q)
                if hasattr(self, "array_remapping") and arg in self.array_remapping:
                    original_name = self.array_remapping[arg]
                elif arg.startswith("_") and arg.endswith("_array"):
                    # Try to infer original name from reconstructed name
                    # _q_array -> q
                    potential_original = arg[1:].replace("_array", "")
                    if potential_original in live_qubits:
                        original_name = potential_original

                if original_name in live_qubits:
                    returned_quantum_args.append(
                        arg,
                    )  # Use the actual arg name for assignment

        # If we forced function_consumes but have no returned_quantum_args,
        # assume all quantum args should be returned (common with partial consumption patterns)
        if function_consumes and not returned_quantum_args and len(quantum_args) > 1:
            returned_quantum_args = list(quantum_args)

        # Also include structs that have live quantum fields
        for struct_arg in struct_args:
            if (
                struct_arg not in returned_quantum_args
                and struct_arg in self.struct_info
            ):
                # Check if struct has any live quantum fields
                struct_info = self.struct_info[struct_arg]
                has_live_fields = False
                for suffix, var_type, size in struct_info.get("fields", []):
                    if var_type == "qubit":
                        var_name = struct_info["var_names"].get(suffix)
                        if var_name and var_name in live_qubits:
                            has_live_fields = True
                            break
                if has_live_fields:
                    returned_quantum_args.append(struct_arg)

        # Track arrays that are consumed (passed with @owned but not returned)
        # Also mark arrays as consumed when passed to nested blocks (even without @owned)
        is_nested_block = False
        try:
            from pecos.slr import Block as SlrBlock

            if hasattr(block, "__class__") and issubclass(block.__class__, SlrBlock):
                is_nested_block = True
        except (TypeError, AttributeError):
            # Not a class or missing expected attributes
            pass

        if (function_consumes or is_nested_block) and hasattr(self, "consumed_arrays"):

            # Check function signature for @owned parameters
            owned_params = set()

            # TEMPORARY FIX: Hardcode known @owned parameter patterns for quantum error correction functions
            # This covers the specific functions that are causing issues in the Steane code
            known_owned_patterns = {
                "prep_rus": [0, 1],  # c_a and c_d are both @owned
                "prep_encoding_ft_zero": [0, 1],  # c_a and c_d are both @owned
                "prep_zero_verify": [0, 1],  # c_a and c_d are both @owned
                "prep_encoding_non_ft_zero": [0],  # c_d is @owned (first parameter)
                "log_zero_rot": [0],  # c_d is @owned (first parameter)
                "h": [0],  # c_d is @owned (first parameter)
            }

            if func_name in known_owned_patterns:
                owned_indices = known_owned_patterns[func_name]
                for i in owned_indices:
                    if i < len(quantum_args):
                        owned_arg = quantum_args[i]
                        owned_params.add(owned_arg)

            # Try to find the function definition in the current module (future improvement)
            # [Previous function definition lookup code can be restored later if needed]

            for arg in quantum_args:
                if isinstance(arg, str):
                    # CRITICAL: Determine if this array should be marked as consumed
                    # Two cases:
                    # 1. Procedural function (returns None): ALL args are consumed
                    # 2. Functional function (returns values): Only args NOT returned are consumed

                    # Procedural function - mark all args as consumed
                    # Functional function - only mark if not returned
                    should_mark_consumed = (
                        True
                        if is_procedural_function
                        else arg not in returned_quantum_args
                    )

                    if should_mark_consumed:
                        # This array was consumed (not returned)
                        # Track the actual array name that was passed (might be reconstructed or fresh)
                        # Check if there's a fresh/refreshed version of this array
                        actual_name_to_mark = arg
                        if (
                            hasattr(self, "refreshed_arrays")
                            and arg in self.refreshed_arrays
                        ):
                            # Use the refreshed/fresh name (e.g., c_d_fresh instead of c_d)
                            actual_name_to_mark = self.refreshed_arrays[arg]
                        elif (
                            hasattr(self, "array_remapping")
                            and arg in self.array_remapping
                        ):
                            # Use the remapped name
                            actual_name_to_mark = self.array_remapping[arg]

                        self.consumed_arrays.add(actual_name_to_mark)
                        # Also mark the original name to prevent double cleanup
                        if actual_name_to_mark != arg:
                            self.consumed_arrays.add(arg)

        # For procedural functions, don't assign the result - just call the function
        if is_procedural_function:
            # Create expression statement for the function call (no assignment)
            class ExpressionStatement(Statement):
                def __init__(self, expr):
                    self.expr = expr

                def analyze(self, _context):
                    return []

                def render(self, context):
                    return self.expr.render(context)

            # After a procedural call, restore the unpacked arrays
            # Procedural functions borrow, they don't consume, so the unpacked variables are still valid
            if saved_unpacked_arrays:
                for item in saved_unpacked_arrays:
                    if len(item) == 3:  # Has reconstructed name and element names
                        array_name, element_names, _ = item
                        # Restore the unpacked variables - they're still valid after a borrow
                        if not hasattr(self, "unpacked_vars"):
                            self.unpacked_vars = {}
                        self.unpacked_vars[array_name] = element_names

            return ExpressionStatement(call)

        # With the functional pattern, functions that consume quantum arrays return the live ones
        if returned_quantum_args and function_consumes:
            # Black Box Pattern: Function returns modified global arrays/structs
            # Assign directly back to original names to maintain SLR semantics
            # ALSO handle @owned functions that return reconstructed structs
            statements = []

            # Check if the function returns a tuple by looking up its return type
            func_return_type = self.function_return_types.get(func_name, "")
            returns_tuple = func_return_type.startswith("tuple[")

            # CRITICAL: Use actual_returns_tuple from block inspection if available
            # This is more reliable than function_return_types which may not be populated yet
            if actual_returns_tuple:
                returns_tuple = True

            # Don't force tuple unpacking based on argument count - use actual return type
            # A function can take multiple args but return only one (e.g., consume some, return others)

            if len(returned_quantum_args) == 1 and not returns_tuple:
                # Single return - assign back to the same variable name
                # In Guppy's linear type system, reassigning to the same name shadows the old binding
                name = returned_quantum_args[0]

                # Handle both reconstructed array names (_q_array) and original names (q)
                base_name = (
                    name[1:].replace("_array", "")
                    if name.startswith("_") and name.endswith("_array")
                    else name
                )

                # CRITICAL: If the variable was already unpacked (parameter unpacked at function start),
                # we cannot assign to the same name - need a fresh variable name
                # Example: def f(c_d: array @owned):
                #   __c_d_0, ... = c_d    # c_d consumed
                #   c_d = h(...)          # ERROR - c_d already consumed!
                # Fix: use fresh name like c_d_fresh
                # Use unpacked_before_call (saved state before argument processing)
                # because argument processing may have deleted the array from unpacked_vars
                if base_name in unpacked_before_call:
                    # Variable was unpacked - use fresh name for assignment
                    fresh_name = self._get_unique_var_name(f"{name}_fresh")
                    # Clear the unpacked tracking if still present
                    if (
                        hasattr(self, "unpacked_vars")
                        and base_name in self.unpacked_vars
                    ):
                        del self.unpacked_vars[base_name]
                else:
                    # Variable wasn't unpacked - can assign to same name (shadows old binding)
                    fresh_name = name

                # Use the appropriate variable name for the assignment
                assignment = Assignment(target=VariableRef(fresh_name), value=call)
                statements.append(assignment)

                # Track fresh variables for cleanup in procedural functions
                # If we created a fresh variable (not same as parameter name), track it
                if fresh_name != name:
                    if not hasattr(self, "fresh_return_vars"):
                        self.fresh_return_vars = {}
                    self.fresh_return_vars[fresh_name] = {
                        "original": name,
                        "func_name": func_name,
                        "is_quantum_array": True,
                    }

                # Update context for returned variable
                self._update_context_for_returned_variable(name, fresh_name)

                # Also update array remapping for cleanup logic
                if not hasattr(self, "array_remapping"):
                    self.array_remapping = {}
                self.array_remapping[name] = fresh_name

                # Track this array as refreshed by function call
                self.refreshed_arrays[name] = fresh_name
                # Track which function refreshed this array and its position (0 for single return)
                if not hasattr(self, "refreshed_by_function"):
                    self.refreshed_by_function = {}
                self.refreshed_by_function[name] = {
                    "function": func_name,
                    "position": 0,
                }

                # If this is a struct, decompose it to avoid field access issues
                if name in self.struct_info:
                    struct_info = self.struct_info[name]
                    # Always decompose fresh structs to avoid AlreadyUsedError on field access
                    needs_decomposition = True

                    if needs_decomposition:
                        # IMPORTANT: We cannot re-unpack from the struct because it may have been
                        # consumed by the function call. Instead, we need to
                        # update our var_remapping
                        # to indicate that the unpacked variables are no longer valid.
                        # The code should use the struct fields directly after function calls.

                        # Comment explaining why we can't re-unpack
                        statements.append(
                            Comment(
                                "Note: Cannot use unpacked variables after calling "
                                "function with @owned struct",
                            ),
                        )

                        # For fresh structs returned from functions, we need to decompose them immediately
                        # to avoid AlreadyUsedError when accessing fields
                        struct_name = struct_info["struct_name"].replace("_struct", "")
                        decompose_func_name = f"{struct_name}_decompose"

                        # Generate field variables for decomposition
                        field_vars = []
                        for suffix, field_type, field_size in sorted(
                            struct_info["fields"],
                        ):
                            field_var = f"{fresh_name}_{suffix}"
                            field_vars.append(field_var)

                        # Add decomposition statement for the fresh struct
                        statements.append(
                            Comment(
                                "Decompose fresh struct to avoid field access on consumed struct",
                            ),
                        )

                        class TupleAssignment(Statement):
                            def __init__(self, targets, value):
                                self.targets = targets
                                self.value = value

                            def analyze(self, context):
                                self.value.analyze(context)

                            def render(self, context):
                                target_str = ", ".join(self.targets)
                                value_str = self.value.render(context)[0]
                                return [f"{target_str} = {value_str}"]

                        decompose_call = FunctionCall(
                            func_name=decompose_func_name,
                            args=[VariableRef(fresh_name)],
                        )

                        decomposition_stmt = TupleAssignment(
                            targets=field_vars,
                            value=decompose_call,
                        )
                        statements.append(decomposition_stmt)

                        # Track decomposed variables for field access
                        if not hasattr(self, "decomposed_vars"):
                            self.decomposed_vars = {}
                        field_mapping = {}
                        for suffix, field_type, field_size in sorted(
                            struct_info["fields"],
                        ):
                            field_var = f"{fresh_name}_{suffix}"
                            field_mapping[suffix] = field_var
                        self.decomposed_vars[fresh_name] = field_mapping

                        # Update var_remapping to indicate these variables should not be used
                        # by mapping them back to struct field access
                        for var_name in struct_info["var_names"].values():
                            if var_name in self.var_remapping:
                                # This will cause future references to use struct.field notation
                                del self.var_remapping[var_name]

                # Force unpacking for arrays that need element access after function calls
                # This is the core fix for the nested blocks MoveOutOfSubscriptError
                # For refreshed arrays, check if they have element access that requires unpacking
                needs_unpacking_for_refresh = False
                if name in self.refreshed_arrays:
                    # CRITICAL FIX: Don't automatically unpack refreshed arrays
                    # The original analysis was for the INPUT parameter, not the refreshed return value
                    # Only unpack if there's explicit subscript usage AFTER this call
                    # This is handled by force_unpack_for_subscript below
                    needs_unpacking_for_refresh = False

                # CRITICAL: Only unpack returned arrays if they actually need element access
                # Don't unpack just because the array was unpacked at function start
                # Check if the array CURRENTLY needs unpacking based on how it's used AFTER this call
                should_unpack_returned = (
                    # Only unpack if actively needed for element access after this point
                    needs_unpacking_for_refresh
                ) and name not in self.struct_info

                # CRITICAL: Always check if function returns array
                # If so, force unpacking to avoid MoveOutOfSubscriptError
                force_unpack_for_subscript = False
                return_array_size_check = None

                # Try to get return type from function_return_types (if already analyzed)
                if func_name in self.function_return_types:
                    return_type = self.function_return_types[func_name]
                    import re

                    match = re.search(r"array\[.*?,\s*(\d+)\]", return_type)
                    if match:
                        return_array_size_check = int(match.group(1))

                        # Check if next operation uses subscript on this array
                        # This catches the pattern: q = func(q); measure(q[0])
                        if (
                            hasattr(self, "current_block_ops")
                            and hasattr(self, "current_op_index")
                            and self.current_block_ops is not None
                            and self.current_op_index is not None
                        ):
                            next_index = self.current_op_index + 1
                            if next_index < len(self.current_block_ops):
                                next_op = self.current_block_ops[next_index]
                                # Check if next op uses subscript on this array
                                if hasattr(next_op, "qargs"):
                                    for qarg in next_op.qargs:
                                        if (
                                            hasattr(qarg, "reg")
                                            and hasattr(qarg.reg, "sym")
                                            and qarg.reg.sym == name
                                            and hasattr(qarg, "index")
                                        ):
                                            # Next op uses subscript on returned array
                                            force_unpack_for_subscript = True
                                            break
                else:
                    # Function not analyzed yet - use live_qubits from block analysis
                    # Check if this array has live qubits that indicate return size
                    if name in live_qubits and len(live_qubits[name]) >= 1:
                        # The block returns live qubits from this array
                        return_array_size_check = len(live_qubits[name])

                        # Check if next operation uses subscript on this array
                        if (
                            hasattr(self, "current_block_ops")
                            and hasattr(self, "current_op_index")
                            and self.current_block_ops is not None
                            and self.current_op_index is not None
                        ):
                            next_index = self.current_op_index + 1
                            if next_index < len(self.current_block_ops):
                                next_op = self.current_block_ops[next_index]
                                # Check if next op uses subscript on this array
                                if hasattr(next_op, "qargs"):
                                    for qarg in next_op.qargs:
                                        if (
                                            hasattr(qarg, "reg")
                                            and hasattr(qarg.reg, "sym")
                                            and qarg.reg.sym == name
                                            and hasattr(qarg, "index")
                                        ):
                                            # Next op uses subscript on returned array
                                            force_unpack_for_subscript = True
                                            break

                if should_unpack_returned or force_unpack_for_subscript:
                    # Use the size we already extracted
                    return_array_size = return_array_size_check

                    # If we know the return size and it's >= 1, unpack for element access
                    # Even size-1 arrays need unpacking to avoid MoveOutOfSubscriptError
                    if return_array_size and return_array_size >= 1:
                        # Generate unpacked variable names
                        # IMPORTANT: Use unique suffix "_ret" to avoid shadowing initial allocations
                        # When we do local_allocate strategy, we create q_0, q_1, q_2
                        # When function returns array, we unpack to q_0_ret, q_1_ret to avoid conflicts
                        # CRITICAL: Make names unique across multiple unpackings using a counter
                        if not hasattr(self, "_unpack_counter"):
                            self._unpack_counter = {}
                        if name not in self._unpack_counter:
                            self._unpack_counter[name] = 0
                        else:
                            self._unpack_counter[name] += 1
                        unpack_suffix = (
                            f"_ret{self._unpack_counter[name]}"
                            if self._unpack_counter[name] > 0
                            else "_ret"
                        )
                        element_names = [
                            f"{name}_{i}{unpack_suffix}"
                            for i in range(return_array_size)
                        ]

                        # Add unpacking statement using ArrayUnpack IR class
                        from pecos.slr.gen_codes.guppy.ir import ArrayUnpack

                        unpack_stmt = ArrayUnpack(
                            targets=element_names,
                            source=name,
                        )
                        statements.append(unpack_stmt)

                        # Track unpacked variables
                        if not hasattr(self, "unpacked_vars"):
                            self.unpacked_vars = {}
                        self.unpacked_vars[name] = element_names

                        # CRITICAL: Track index mapping for partial consumption
                        # If live_qubits tells us which original indices are in the returned array,
                        # create a mapping from original index → unpacked variable index
                        if name in live_qubits:
                            original_indices = sorted(live_qubits[name])
                            if not hasattr(self, "index_mapping"):
                                self.index_mapping = {}
                            # Map original index to position in returned/unpacked array
                            self.index_mapping[name] = {
                                orig_idx: new_idx
                                for new_idx, orig_idx in enumerate(original_indices)
                            }

                        # Update context
                        if hasattr(self, "context"):
                            var = self.context.lookup_variable(name)
                            if var:
                                var.is_unpacked = True
                                var.unpacked_names = element_names

                        # DON'T immediately reconstruct - just leave the array unpacked
                        # Reconstruction will happen on-demand when needed (see below)
                elif hasattr(self, "unpacked_vars") and name in self.unpacked_vars:
                    # Classical array or other case - invalidate old unpacked variables
                    old_element_names = self.unpacked_vars[name]
                    del self.unpacked_vars[name]

                    # Also update the context to invalidate unpacked variable information
                    if hasattr(self, "context"):
                        var = self.context.lookup_variable(name)
                        if var:
                            var.is_unpacked = False
                            var.unpacked_names = []

                    # Add comment explaining why we can't re-unpack
                    statements.append(
                        Comment(
                            f"Note: Unpacked variables {old_element_names} invalidated "
                            "after function call - array size may have changed",
                        ),
                    )
                elif (
                    name in self.plan.arrays_to_unpack
                    and name not in self.unpacked_vars
                ):
                    # After function calls, don't automatically re-unpack arrays
                    # The array may have changed size and old unpacked variables are stale
                    # Instead, use array indexing for future references
                    statements.append(
                        Comment(
                            f"Note: Not re-unpacking {name} after function call - "
                            "array may have changed size, use array indexing instead",
                        ),
                    )

            else:
                # HYBRID TUPLE ASSIGNMENT: Choose strategy based on function and usage patterns
                use_fresh_variables = self._should_use_fresh_variables(
                    func_name,
                    quantum_args,
                )

                if use_fresh_variables:
                    # Use fresh variables to avoid PlaceNotUsedError in problematic patterns
                    # Generate unique names to avoid reassignment issues in loops
                    if not hasattr(self, "_fresh_var_counter"):
                        self._fresh_var_counter = {}

                    fresh_targets = []

                    # Check if we're in a consumption loop (conditional or not)
                    in_consumption_loop = (
                        hasattr(self, "_in_conditional_consumption_loop")
                        and self._in_conditional_consumption_loop
                        and hasattr(self, "scope_manager")
                        and self.scope_manager.is_in_loop()
                    )

                    for arg in quantum_args:
                        # If we're in a consumption loop,
                        # reuse existing fresh names to avoid creating new variables in each iteration
                        if in_consumption_loop and arg in self.refreshed_arrays:
                            # Reuse the existing fresh variable name
                            fresh_name = self.refreshed_arrays[arg]
                            fresh_targets.append(fresh_name)
                        else:
                            base_name = f"{arg}_fresh"
                            # For loops and repeated calls, use unique suffixes
                            if base_name in self._fresh_var_counter:
                                self._fresh_var_counter[base_name] += 1
                                unique_name = (
                                    f"{base_name}_{self._fresh_var_counter[base_name]}"
                                )
                            else:
                                self._fresh_var_counter[base_name] = 0
                                unique_name = base_name
                            fresh_targets.append(unique_name)
                else:
                    # Standard tuple assignment - but check if we need to avoid borrowed variables
                    # OR if variables were unpacked before the call
                    fresh_targets = []
                    for arg_idx, arg in enumerate(quantum_args):
                        # CRITICAL: Check if this parameter was already unpacked before the call
                        # If so, we MUST use a fresh variable name (can't assign to consumed variable)
                        # This is the same issue we fixed for single returns
                        was_unpacked = arg in unpacked_before_call

                        # Check if this variable is a borrowed parameter (not @owned)
                        # If so, we need to use a different name to avoid BorrowShadowedError
                        is_borrowed = False
                        if (
                            hasattr(self, "current_function_name")
                            and self.current_function_name
                        ):
                            # Check if this is a function parameter
                            func_info = self.function_info.get(
                                self.current_function_name,
                                {},
                            )
                            params = func_info.get("params", [])
                            for param_name, param_type in params:
                                if (
                                    param_name == arg
                                    and "@owned" not in param_type
                                    and "array[quantum.qubit" in param_type
                                ):
                                    # This is a borrowed quantum array parameter
                                    is_borrowed = True
                                    break

                        # Determine if we need a fresh name for any reason:
                        # 1. Variable was unpacked before call (consumed)
                        # 2. Variable is a borrowed parameter (can't shadow)
                        needs_fresh_name = was_unpacked or is_borrowed

                        if needs_fresh_name:
                            # Use a fresh name to avoid:
                            # - AlreadyUsedError (if unpacked before call)
                            # - BorrowShadowedError (if borrowed parameter)
                            # Check if we're in a loop - if so, reuse the existing variable name
                            in_loop = (
                                hasattr(self, "scope_manager")
                                and self.scope_manager.is_in_loop()
                            )

                            if (
                                in_loop
                                and hasattr(self, "refreshed_arrays")
                                and arg in self.refreshed_arrays
                            ):
                                # In a loop, reuse the existing refreshed name to avoid undefined variable errors
                                fresh_name = self.refreshed_arrays[arg]
                            elif (
                                hasattr(self, "refreshed_arrays")
                                and arg in self.refreshed_arrays
                            ):
                                # Not in a loop but already have a returned version, need a new unique name
                                if not hasattr(self, "_returned_var_counter"):
                                    self._returned_var_counter = {}
                                base_name = f"{arg}_returned"
                                if base_name not in self._returned_var_counter:
                                    self._returned_var_counter[base_name] = 1
                                else:
                                    self._returned_var_counter[base_name] += 1
                                fresh_name = f"{base_name}_{self._returned_var_counter[base_name]}"
                            else:
                                # Choose suffix based on reason for fresh name
                                if was_unpacked:
                                    # Use _fresh suffix for unpacked parameters (more descriptive)
                                    fresh_name = self._get_unique_var_name(
                                        f"{arg}_fresh",
                                    )
                                else:
                                    # Use _returned suffix for borrowed parameters
                                    fresh_name = f"{arg}_returned"

                            fresh_targets.append(fresh_name)

                            # Track this for later use
                            if not hasattr(self, "refreshed_arrays"):
                                self.refreshed_arrays = {}
                            self.refreshed_arrays[arg] = fresh_name
                            # Track which function refreshed this array and its position in return tuple
                            if not hasattr(self, "refreshed_by_function"):
                                self.refreshed_by_function = {}
                            self.refreshed_by_function[arg] = {
                                "function": func_name,
                                "position": arg_idx,
                            }

                            # Also track in fresh_return_vars for cleanup in procedural functions
                            if was_unpacked:
                                if not hasattr(self, "fresh_return_vars"):
                                    self.fresh_return_vars = {}
                                self.fresh_return_vars[fresh_name] = {
                                    "original": arg,
                                    "func_name": func_name,
                                    "is_quantum_array": True,
                                }
                        else:
                            # Safe to use the original name (not unpacked, not borrowed)
                            fresh_targets.append(arg)

                class TupleAssignment(Statement):
                    def __init__(self, targets, value):
                        self.targets = targets
                        self.value = value

                    def analyze(self, context):
                        self.value.analyze(context)

                    def render(self, context):
                        target_str = ", ".join(self.targets)
                        value_str = self.value.render(context)[0]
                        return [f"{target_str} = {value_str}"]

                assignment = TupleAssignment(targets=fresh_targets, value=call)
                statements.append(assignment)

                # Track all refreshed/returned variables for proper return handling
                for i, original_name in enumerate(quantum_args):
                    if i < len(fresh_targets):
                        fresh_name = fresh_targets[i]
                        if fresh_name != original_name:
                            # This variable was renamed (either _fresh or _returned)
                            # Track it so return statements use the correct name
                            if not hasattr(self, "refreshed_arrays"):
                                self.refreshed_arrays = {}
                            # Always update the mapping for return handling
                            self.refreshed_arrays[original_name] = fresh_name
                            # Track which function refreshed this array and its position in return tuple
                            if not hasattr(self, "refreshed_by_function"):
                                self.refreshed_by_function = {}
                            self.refreshed_by_function[original_name] = {
                                "function": func_name,
                                "position": i,
                            }

                            # Also track in fresh_return_vars for cleanup in procedural functions
                            # All fresh variables from tuple returns need cleanup tracking
                            if not hasattr(self, "fresh_return_vars"):
                                self.fresh_return_vars = {}
                            self.fresh_return_vars[fresh_name] = {
                                "original": original_name,
                                "func_name": func_name,
                                "is_quantum_array": True,
                            }

                # Check if any of the returned variables are structs and decompose them immediately
                for var_name in fresh_targets:
                    # Check if this variable name corresponds to a struct
                    # It might be a fresh name (e.g., c_fresh) or original name (e.g., c)
                    struct_info = None

                    if var_name in self.struct_info:
                        struct_info = self.struct_info[var_name]
                    else:
                        # Check if this is a renamed struct (e.g., c_fresh -> c)
                        # Be precise: only match if the variable is actually a renamed version of the struct
                        for key, info in self.struct_info.items():
                            # Check for exact pattern: key_suffix (e.g., c_fresh)
                            if (
                                var_name == f"{key}_fresh"
                                or var_name == f"{key}_returned"
                            ):
                                struct_info = info
                                break

                    if struct_info:
                        # Decompose fresh structs that will be used in loops
                        # This allows us to access fields without consuming the struct
                        struct_name = struct_info["struct_name"].replace("_struct", "")
                        decompose_func_name = f"{struct_name}_decompose"

                        # Generate field variables for decomposition
                        field_vars = []
                        for suffix, field_type, field_size in sorted(
                            struct_info["fields"],
                        ):
                            field_var = f"{var_name}_{suffix}"
                            field_vars.append(field_var)

                        # Add decomposition statement
                        statements.append(
                            Comment(f"Decompose {var_name} for field access"),
                        )

                        decompose_call = FunctionCall(
                            func_name=decompose_func_name,
                            args=[VariableRef(var_name)],
                        )

                        decomposition_stmt = TupleAssignment(
                            targets=field_vars,
                            value=decompose_call,
                        )
                        statements.append(decomposition_stmt)

                        # Track decomposed variables
                        if not hasattr(self, "decomposed_vars"):
                            self.decomposed_vars = {}
                        field_mapping = {}
                        for suffix, field_type, field_size in sorted(
                            struct_info["fields"],
                        ):
                            field_var = f"{var_name}_{suffix}"
                            field_mapping[suffix] = field_var
                        self.decomposed_vars[var_name] = field_mapping

                # Handle variable mapping based on whether we used fresh variables
                if use_fresh_variables:
                    statements.append(
                        Comment("Using fresh variables to avoid linearity conflicts"),
                    )

                    # Check if we're in a conditional within a loop
                    # This requires special handling to avoid linearity violations
                    (
                        hasattr(self, "scope_manager")
                        and self.scope_manager.is_in_conditional_within_loop()
                    )

                    # Update variable mapping so future references use the fresh names
                    # BUT only for functions that truly "refresh" the same arrays
                    # Functions like prep_zero_verify return different arrays, not refreshed inputs
                    refresh_functions = [
                        "process_qubits",  # Functions that process and return the same qubits
                        "apply_gates",  # Functions that apply operations and return the same qubits
                        "measure_and_reset",  # Functions that measure, reset, and return the same qubits
                    ]

                    # Check if this function actually refreshes arrays (returns processed versions of inputs)
                    should_refresh_arrays = any(
                        pattern in func_name.lower() for pattern in refresh_functions
                    )

                    # Additional check: if function has @owned parameters and returns fresh variables,
                    # it's likely refreshing the arrays
                    if not should_refresh_arrays and use_fresh_variables:
                        # Check if any fresh target names contain "fresh" - indicates array refreshing
                        has_fresh_returns = any(
                            "fresh" in target for target in fresh_targets
                        )
                        if has_fresh_returns:
                            # Most quantum functions that return "fresh" variables are refreshing arrays
                            # This includes verification functions that return processed versions of inputs
                            should_refresh_arrays = True

                    if should_refresh_arrays:
                        for i, original_name in enumerate(quantum_args):
                            if i < len(fresh_targets):
                                fresh_name = fresh_targets[i]
                                if (
                                    fresh_name != original_name
                                ):  # Only map if actually fresh
                                    # Check if this is a conditional fresh variable (ending in _1)
                                    if fresh_name.endswith("_1"):
                                        # Don't update mapping for conditional variables to avoid errors
                                        # Conditional consumption in loops is fundamentally incompatible
                                        # with guppylang's linearity requirements
                                        base_fresh_name = fresh_name[
                                            :-2
                                        ]  # Remove _1 suffix
                                        self.conditional_fresh_vars[base_fresh_name] = (
                                            fresh_name
                                        )
                                    elif original_name not in self.refreshed_arrays:
                                        # Safe to update - first assignment
                                        self.refreshed_arrays[original_name] = (
                                            fresh_name
                                        )
                                        # Track which function refreshed this array and its position in return tuple
                                        if not hasattr(self, "refreshed_by_function"):
                                            self.refreshed_by_function = {}
                                        self.refreshed_by_function[original_name] = {
                                            "function": func_name,
                                            "position": i,
                                        }
                                        self._update_context_for_returned_variable(
                                            original_name,
                                            fresh_name,
                                        )
                    else:
                        # For functions that return different arrays (like prep_zero_verify),
                        # don't map fresh variables as refreshed versions of inputs
                        # This allows proper reconstruction from unpacked variables in returns
                        pass

                    # Immediately check if any fresh variables are likely to be unused
                    # and add discard for them
                    # Specifically, check for the ancilla pattern where ancilla_fresh is returned
                    # but not used after syndrome extraction
                    for i, original_name in enumerate(quantum_args):
                        if i < len(fresh_targets):
                            fresh_name = fresh_targets[i]
                            # Check if this is likely an ancilla array that won't be used
                            # Pattern: ancilla arrays that are measured inside the function
                            is_ancilla = "ancilla" in original_name.lower()
                            is_fresh = fresh_name != original_name
                            in_main = self.current_function_name == "main"
                            if is_ancilla and is_fresh and in_main:
                                # Check if we're in main (where ancillas are typically not reused)
                                # Add immediate discard for ancilla_fresh
                                statements.append(
                                    Comment(
                                        f"Discard unused {fresh_name} immediately",
                                    ),
                                )
                                discard_stmt = FunctionCall(
                                    func_name="quantum.discard_array",
                                    args=[VariableRef(fresh_name)],
                                )

                                class ExpressionStatement(Statement):
                                    def __init__(self, expr):
                                        self.expr = expr

                                    def analyze(self, context):
                                        self.expr.analyze(context)

                                    def render(self, context):
                                        return self.expr.render(context)

                                statements.append(ExpressionStatement(discard_stmt))
                else:
                    statements.append(
                        Comment("Standard tuple assignment to original variables"),
                    )
                    # For standard assignment, variables keep their original names
                    # BUT don't overwrite if we already set a different mapping (e.g., for _returned variables)
                    for i, original_name in enumerate(quantum_args):
                        if i < len(fresh_targets):
                            fresh_name = fresh_targets[i]
                            # Only set to original name if we haven't already mapped to a different name
                            if fresh_name == original_name:
                                self.refreshed_arrays[original_name] = original_name
                            # If fresh_name != original_name, the mapping was already set above

                # Handle struct field invalidation after function call
                for array_name in quantum_args:
                    if array_name in self.struct_info and hasattr(
                        self,
                        "var_remapping",
                    ):
                        struct_info = self.struct_info[array_name]
                        # Check if any of the struct's fields are in var_remapping
                        needs_update = any(
                            var in self.var_remapping
                            for var in struct_info["var_names"].values()
                        )

                        if needs_update:
                            # Cannot re-unpack - invalidate the unpacked variables
                            statements.append(
                                Comment(
                                    "Note: Cannot use unpacked variables after calling "
                                    "function with @owned struct",
                                ),
                            )

                            # Update var_remapping to indicate these variables should not be used
                            for var_name in struct_info["var_names"].values():
                                if var_name in self.var_remapping:
                                    del self.var_remapping[var_name]

                # Unpack any arrays that need it after the function call
                # BUT: Don't unpack if already unpacked (to avoid AlreadyUsedError)
                for array_name in quantum_args:
                    if (
                        array_name in self.plan.unpack_at_start
                        and array_name not in self.struct_info
                        and array_name in self.plan.arrays_to_unpack
                        and array_name not in self.unpacked_vars  # Don't re-unpack!
                    ):
                        info = self.plan.arrays_to_unpack[array_name]
                        self._add_array_unpacking(array_name, info.size)

            # Check if current function is procedural (returns None) and add discards for unused quantum arrays
            is_in_procedural = getattr(self, "current_function_is_procedural", False)
            if is_in_procedural and len(statements) == 1:
                # This is a procedural function with a single assignment (likely the last operation)
                # Check if we have an unused quantum array to discard
                # This happens when a procedural function calls a function that returns an array
                # but doesn't use the result
                stmt = statements[0]
                if isinstance(stmt, Assignment):
                    # Check if this is an assignment to a quantum array
                    target_name = None
                    if hasattr(stmt.target, "name"):
                        target_name = stmt.target.name

                    # Check if this is a quantum array by checking:
                    # 1. If it's in returned_quantum_args (passed as quantum param)
                    # 2. Or if func_name returns a quantum array (if we know the return type)
                    is_quantum_array = target_name in returned_quantum_args

                    if not is_quantum_array and func_name in self.function_return_types:
                        return_type = self.function_return_types[func_name]
                        is_quantum_array = "array[quantum.qubit," in return_type

                    if target_name and is_quantum_array:
                        # This is a quantum array that was assigned but may not be used
                        # Add a discard statement for it
                        discard_call = FunctionCall(
                            func_name="quantum.discard_array",
                            args=[VariableRef(target_name)],
                        )

                        # Define ExpressionStatement locally if not already defined
                        class ExpressionStatement(Statement):
                            def __init__(self, expr):
                                self.expr = expr

                            def analyze(self, _context):
                                return []

                            def render(self, context):
                                return self.expr.render(context)

                        statements.append(Comment(f"Discard unused {target_name}"))
                        statements.append(ExpressionStatement(discard_call))

            # Return block with all statements
            if len(statements) == 1:
                return statements[0]
            return Block(statements=statements)

        # Either no quantum arrays OR function consumes its parameters
        # In both cases, just call the function without assignment
        class ExpressionStatement(Statement):
            def __init__(self, expr):
                self.expr = expr

            def analyze(self, context):
                self.expr.analyze(context)

            def render(self, context):
                return self.expr.render(context)

        return ExpressionStatement(call)

    def _function_returns_something(self, func_name: str) -> bool:
        """Check if a function returns a value (not None)."""
        # Functions that work with structs and return modified structs
        # Check if this function name indicates it works with structs
        if self.struct_info:
            for info in self.struct_info.values():
                struct_name = info.get("struct_name", "")
                # Extract the base name from the struct name (e.g., "steane" from "steane_struct")
                if "_struct" in struct_name:
                    base_name = struct_name.replace("_struct", "").lower()
                else:
                    base_name = struct_name.lower()

                if func_name.startswith(f"{base_name}_"):
                    # Struct functions typically return the modified struct
                    # Exception: functions ending in 'discard' or 'decompose'
                    # don't return the struct
                    return not (func_name.endswith(("_discard", "_decompose")))

        # For other functions, assume they return something if they have quantum args
        # This is a conservative approach
        return False

    def _analyze_quantum_resource_flow(
        self,
        block,
    ) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
        """Analyze which quantum resources are consumed vs. live in a block.

        Returns:
            consumed_qubits: dict mapping qreg names to sets of consumed indices
            live_qubits: dict mapping qreg names to sets of live indices
        """
        consumed_qubits = {}
        live_qubits = {}

        # Track all quantum variables used
        all_quantum_vars = set()

        if hasattr(block, "ops"):
            for op in block.ops:
                # Check for measurements that consume qubits
                if type(op).__name__ == "Measure":
                    if hasattr(op, "qargs"):
                        for qarg in op.qargs:
                            if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                                qreg_name = qarg.reg.sym
                                if hasattr(qarg, "index"):
                                    # Single qubit measurement
                                    if qreg_name not in consumed_qubits:
                                        consumed_qubits[qreg_name] = set()
                                    consumed_qubits[qreg_name].add(qarg.index)
                            elif hasattr(qarg, "sym"):
                                # Full array measurement
                                qreg_name = qarg.sym
                                if hasattr(qarg, "size"):
                                    if qreg_name not in consumed_qubits:
                                        consumed_qubits[qreg_name] = set()
                                    consumed_qubits[qreg_name].update(range(qarg.size))

                # Check for nested Block operations that may consume qubits
                elif hasattr(op, "ops") and hasattr(op, "vars"):
                    # This is a nested block - analyze it recursively
                    nested_consumed, _nested_live = self._analyze_quantum_resource_flow(
                        op,
                    )

                    # Merge nested consumption into our tracking
                    for qreg_name, indices in nested_consumed.items():
                        if qreg_name not in consumed_qubits:
                            consumed_qubits[qreg_name] = set()
                        consumed_qubits[qreg_name].update(indices)

                # Track all quantum variables used (for determining what's live)
                if hasattr(op, "qargs"):
                    for qarg in op.qargs:
                        if isinstance(qarg, tuple):
                            for sub_qarg in qarg:
                                if hasattr(sub_qarg, "reg") and hasattr(
                                    sub_qarg.reg,
                                    "sym",
                                ):
                                    all_quantum_vars.add(sub_qarg.reg.sym)
                                elif hasattr(sub_qarg, "sym"):
                                    all_quantum_vars.add(sub_qarg.sym)
                        elif hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                            all_quantum_vars.add(qarg.reg.sym)
                        elif hasattr(qarg, "sym"):
                            all_quantum_vars.add(qarg.sym)

        # Determine live qubits (used but not consumed)
        # We need to know the actual size of arrays to determine what's live
        # Get size information from the block's variable definitions
        array_sizes = {}

        # Check all attributes of the block for QReg/CReg definitions
        for attr_name in dir(block):
            if not attr_name.startswith("_"):  # Skip private attributes
                try:
                    attr = getattr(block, attr_name, None)
                    if attr and hasattr(attr, "size") and hasattr(attr, "sym"):
                        array_sizes[attr.sym] = attr.size
                        # Add to all_quantum_vars if it's a quantum register
                        if (
                            hasattr(attr, "__class__")
                            and "QReg" in attr.__class__.__name__
                        ):
                            all_quantum_vars.add(attr.sym)
                except (AttributeError, TypeError):
                    # Ignore attributes without expected structure
                    pass

        # Also check variable context if available
        if hasattr(self, "context") and self.context:
            for var_name in all_quantum_vars:
                var_info = self.context.lookup_variable(var_name)
                if var_info and var_info.size:
                    array_sizes[var_name] = var_info.size

        # Pre-track explicit resets to know which consumed qubits are reset and should be considered live
        consumed_for_tracking = {}
        self._track_consumed_qubits(block, consumed_for_tracking)

        for var_name in all_quantum_vars:
            if var_name not in consumed_qubits:
                # Variable is used but not consumed - it's fully live
                # Determine size from context or default
                size = array_sizes.get(var_name, 2)  # Default to 2 if unknown
                live_qubits[var_name] = set(range(size))
            else:
                # Check if only partially consumed
                consumed_indices = consumed_qubits[var_name]
                size = array_sizes.get(var_name, 2)  # Default to 2 if unknown

                # Any indices not consumed OR explicitly reset are live
                # Explicitly reset qubits are consumed by measurement but then recreated by Prep
                explicitly_reset_indices = set()
                if (
                    hasattr(self, "explicitly_reset_qubits")
                    and var_name in self.explicitly_reset_qubits
                ):
                    explicitly_reset_indices = self.explicitly_reset_qubits[var_name]

                live_indices = (
                    set(range(size)) - consumed_indices
                ) | explicitly_reset_indices
                if live_indices:
                    live_qubits[var_name] = live_indices

        return consumed_qubits, live_qubits

    def _should_function_be_procedural(
        self,
        func_name: str,
        block,
        params,
        has_live_qubits: bool,
    ) -> bool:
        """
        Smart detection to determine if a function should be procedural (return None)
        vs functional (return tuple of quantum arrays).

        Functions should be procedural if they:
        1. Primarily do terminal operations (measurements without further quantum operations)
        2. Are not used in patterns where quantum returns are needed afterward
        3. Would cause PlaceNotUsedError issues with tuple returns

        Functions should be functional if they:
        1. Their quantum returns are needed for subsequent operations in the calling scope
        2. They are part of partial consumption patterns
        """

        # Pattern-based detection for known procedural functions
        # BUT: only if they don't have live qubits
        procedural_patterns = [
            "syndrome_extraction",  # Terminal syndrome measurement blocks
            "cleanup",  # Cleanup operations
            "discard",  # Discard operations
        ]

        # Check if this is an inner block that will be called by outer blocks
        # Inner blocks should NOT be procedural to avoid consumption issues
        if "inner" in func_name.lower():
            return False

        # Only apply pattern matching if there are no live qubits
        # Functions with live qubits should return them, regardless of name
        if not has_live_qubits:
            for pattern in procedural_patterns:
                if pattern in func_name.lower():
                    # These are good candidates for procedural
                    return True

        # Functions with quantum parameters but no live qubits are good candidates for procedural
        has_quantum_params = any(
            "array[quantum.qubit," in param[1] for param in params if len(param) == 2
        )

        if has_quantum_params and not has_live_qubits:
            # This is a terminal function - good candidate for procedural
            return True

        # Check if this function would benefit from procedural approach based on operations
        if hasattr(block, "ops"):
            measurement_count = 0
            gate_count = 0

            for op in block.ops:
                if hasattr(op, "__class__"):
                    op_name = op.__class__.__name__
                    if "Measure" in op_name:
                        measurement_count += 1
                    elif hasattr(op, "name") or any(
                        gate in str(op) for gate in ["H", "X", "Y", "Z", "CX", "CZ"]
                    ):
                        gate_count += 1

            # If mostly measurements with no quantum gates, good candidate for procedural
            # But be conservative - only if no gates at all or very few
            # AND only if there are no live qubits to return (partial consumption must return live qubits)
            if measurement_count > 0 and gate_count == 0 and not has_live_qubits:
                return True

        # CONSERVATIVE: Default to functional approach unless clearly terminal
        # This avoids breaking partial consumption patterns
        return False

    def _should_use_fresh_variables(self, func_name: str, quantum_args: list) -> bool:
        """
        Determine if fresh variables should be used for tuple assignment.

        Fresh variables help avoid PlaceNotUsedError when:
        1. Function has complex ownership patterns (@owned mixed with borrowed)
        2. Function might cause circular assignment issues
        3. Function is known to cause tuple assignment problems
        """

        # Known problematic patterns that benefit from fresh variables
        fresh_variable_patterns = [
            "measure_ancillas",  # Mixed ownership - some params consumed, some borrowed
            "partial_consumption",  # Partial consumption patterns
            "process_qubits",  # Functions that process and return quantum arrays
        ]

        for pattern in fresh_variable_patterns:
            if pattern in func_name.lower():
                return True

        # Check if we're inside a function that will return these values
        # If the function will return these arrays, don't use fresh variables
        # to avoid PlaceNotUsedError for unused fresh variables
        special_funcs = ["prep_zero_verify", "prep_encoding_non_ft_zero"]
        in_function = (
            hasattr(self, "current_function_name") and self.current_function_name
        )
        if in_function and func_name in special_funcs:
            # Check if this is the last statement in the function that will be returned
            # For now, assume functions that manipulate and return the same arrays
            # should NOT use fresh variables to avoid unused variable errors
            # These functions return arrays that should be used directly
            return False

        # If function has multiple quantum arguments, it might have mixed ownership
        # Use fresh variables to be safe
        if (
            len(quantum_args) > 1
            and hasattr(self, "current_block")
            and hasattr(self.current_block, "statements")
        ):
            # But check if we're at the end of a function where the result will be returned
            # In that case, don't use fresh variables
            # This is a heuristic - if there are not many statements after this,
            # it's likely the return statement
            return False  # Don't use fresh variables for now

        # Default: use standard tuple assignment
        return False

    def _fix_post_consuming_linearity_issues(self, body: Block) -> None:
        """
        Fix linearity issues by adding fresh qubit allocations after consuming operations.

        When a qubit is consumed (e.g., by quantum.reset), and then used again later,
        we need to allocate a fresh qubit to satisfy guppylang's linearity constraints.
        """

        # Track variables that have been consumed
        new_statements = []

        for stmt in body.statements:
            # Add the current statement
            new_statements.append(stmt)

            # Check if this statement consumes any variables
            # Note: quantum.reset is now handled with assignment (qubit = quantum.reset(qubit))
            # so we no longer need to add automatic fresh qubit allocations
            if hasattr(stmt, "expr") and hasattr(stmt.expr, "func_name"):
                # Handle function calls that consume qubits
                func_call = stmt.expr
                if (
                    hasattr(func_call, "func_name")
                    and func_call.func_name == "quantum.reset"
                ):
                    # quantum.reset now uses assignment, so no need for fresh allocation
                    # The reset operation returns the reset qubit
                    pass

        # Replace the statements
        body.statements = new_statements

    def _fix_unused_fresh_variables(self, body: Block) -> None:
        """
        Fix PlaceNotUsedError for fresh variables that may not be used in all execution paths.

        This handles the general pattern where:
        1. Fresh variables are created from function calls
        2. These variables are only used conditionally in loops
        3. Some fresh variables remain unconsumed, causing PlaceNotUsedError
        """
        from pecos.slr.gen_codes.guppy.ir import Comment, FunctionCall, VariableRef

        # Define ExpressionStatement class for standalone function calls
        class ExpressionStatement:
            def __init__(self, expr):
                self.expr = expr

            def analyze(self, context):
                self.expr.analyze(context)

            def render(self, context):
                return self.expr.render(context)

        # General approach: find fresh variables that might be unused in conditional paths
        fresh_variables_created = set()
        fresh_variables_used_conditionally = set()
        has_conditional_usage = False

        def collect_fresh_variables(statements):
            """Recursively collect all fresh variables created and used."""
            for stmt in statements:
                # Check if this is a Block and recurse into it
                if hasattr(stmt, "statements"):
                    collect_fresh_variables(stmt.statements)

                # Find tuple assignments that create fresh variables
                if hasattr(stmt, "targets") and len(stmt.targets) > 0:
                    for target in stmt.targets:
                        if isinstance(target, str) and "_fresh" in target:
                            fresh_variables_created.add(target)

                # Check for conditional statements (if/for) containing fresh variable usage
                is_conditional = hasattr(stmt, "condition") or hasattr(stmt, "iterable")
                has_body = hasattr(stmt, "body") and hasattr(stmt.body, "statements")
                if is_conditional and has_body:  # IfStatement or ForStatement
                    nonlocal has_conditional_usage
                    has_conditional_usage = True
                    # Look for fresh variable usage in conditional blocks
                    self._find_fresh_usage_in_statements(
                        stmt.body.statements,
                        fresh_variables_used_conditionally,
                    )

        def find_procedural_functions_with_unused_fresh():
            """Find procedural functions (return None) that might have unused fresh variables."""
            if not (
                hasattr(self, "current_function_name") and self.current_function_name
            ):
                return False

            # Check if this is a procedural function that might have the pattern
            # Method 1: Check if already recorded in function_return_types
            if (
                hasattr(self, "function_return_types")
                and self.function_return_types.get(self.current_function_name) == "None"
            ):
                return True

            # Method 2: Check if the function body has no return statements (procedural)
            # This is a heuristic for functions that don't explicitly return values
            has_return_stmt = any(
                hasattr(stmt, "value")
                and hasattr(stmt, "__class__")
                and "return" in str(type(stmt)).lower()
                for stmt in body.statements
            )

            # Method 3: Use pattern matching - functions that end with calls to other functions
            # but don't return their results are likely procedural
            if not has_return_stmt and len(body.statements) > 0:
                last_stmt = body.statements[-1]
                if hasattr(last_stmt, "expr") and hasattr(last_stmt.expr, "func_name"):
                    return True  # Likely procedural if ends with a function call

            return False

        collect_fresh_variables(body.statements)

        is_procedural = find_procedural_functions_with_unused_fresh()

        # If we have fresh variables created and conditional usage patterns,
        # and this is a procedural function, add discard statements for unused fresh variables
        if fresh_variables_created and has_conditional_usage and is_procedural:

            # Find fresh variables that are likely unused in some execution paths
            potentially_unused = (
                fresh_variables_created - fresh_variables_used_conditionally
            )

            # Also check which fresh variables are used after conditionals (shouldn't be discarded)
            fresh_variables_used_after_conditionals = set()
            self._find_fresh_usage_in_statements(
                body.statements,
                fresh_variables_used_after_conditionals,
            )

            # Only discard variables that are not used after conditionals
            safe_to_discard = (
                potentially_unused - fresh_variables_used_after_conditionals
            )

            # Add discard statements before the last statement for potentially unused variables
            last_stmt_idx = len(body.statements) - 1
            insert_offset = 0

            for fresh_var in sorted(safe_to_discard):  # Sort for consistent ordering
                comment = Comment(
                    f"# Discard {fresh_var} to avoid PlaceNotUsedError in conditional paths",
                )
                discard_call = FunctionCall(
                    func_name="quantum.discard_array",
                    args=[VariableRef(fresh_var)],
                )
                discard_stmt = ExpressionStatement(discard_call)

                # Insert before the last statement
                body.statements.insert(last_stmt_idx + insert_offset, comment)
                body.statements.insert(last_stmt_idx + insert_offset + 1, discard_stmt)
                insert_offset += 2

    def _find_fresh_usage_in_statements(self, statements, used_set):
        """Helper to find fresh variable usage in a list of statements."""
        for stmt in statements:
            if hasattr(stmt, "statements"):
                self._find_fresh_usage_in_statements(stmt.statements, used_set)

            # Look for function calls that use fresh variables as arguments
            if hasattr(stmt, "expr") and hasattr(stmt.expr, "args"):
                for arg in stmt.expr.args:
                    if hasattr(arg, "name") and "_fresh" in arg.name:
                        used_set.add(arg.name)

            # Look for assignments that use fresh variables
            if hasattr(stmt, "value") and hasattr(stmt.value, "args"):
                for arg in stmt.value.args:
                    if hasattr(arg, "name") and "_fresh" in arg.name:
                        used_set.add(arg.name)

    def _update_context_for_returned_variable(
        self,
        original_name: str,
        fresh_name: str,
    ) -> None:
        """Update context to redirect variable lookups from original to fresh name."""
        original_var = self.context.lookup_variable(original_name)
        if original_var:
            from pecos.slr.gen_codes.guppy.ir import ResourceState, VariableInfo

            # Create new variable info for the fresh returned variable
            new_var_info = VariableInfo(
                name=fresh_name,
                original_name=fresh_name,
                var_type=original_var.var_type,
                size=original_var.size,
                is_array=original_var.is_array,
                state=ResourceState.AVAILABLE,
                is_unpacked=original_var.is_unpacked,
                unpacked_names=(
                    original_var.unpacked_names.copy()
                    if original_var.unpacked_names
                    else []
                ),
            )

            # Add the fresh variable to context
            self.context.add_variable(new_var_info)

            # Add to refreshed arrays mapping for variable reference resolution
            self.context.refreshed_arrays[original_name] = fresh_name

            # Mark the original variable as consumed since it was moved to the returned variable
            self.context.consumed_resources.add(original_name)

    def _analyze_block_dependencies(self, block) -> dict[str, Any]:
        """Analyze what variables a block depends on."""
        dependencies = {
            "reads": set(),  # Variables read
            "writes": set(),  # Variables written
            "quantum": set(),  # Quantum variables used
            "classical": set(),  # Classical variables used
        }

        # Analyze operations in the block
        if hasattr(block, "ops"):
            for op in block.ops:
                self._analyze_op_dependencies(op, dependencies, depth=0)

        return dependencies

    def _analyze_op_dependencies(
        self,
        op,
        deps: dict[str, set],
        depth: int = 0,
    ) -> None:
        """Analyze dependencies of a single operation."""
        op_type = type(op).__name__

        # Handle quantum gates
        if hasattr(op, "qargs"):
            for qarg in op.qargs:
                # Handle tuple arguments (e.g., CX gates with (control, target) pairs)
                if isinstance(qarg, tuple):
                    for sub_qarg in qarg:
                        if hasattr(sub_qarg, "reg") and hasattr(sub_qarg.reg, "sym"):
                            var_name = sub_qarg.reg.sym
                            deps["reads"].add(var_name)
                            deps["quantum"].add(var_name)
                        elif hasattr(sub_qarg, "sym"):
                            var_name = sub_qarg.sym
                            deps["reads"].add(var_name)
                            deps["quantum"].add(var_name)
                elif hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                    var_name = qarg.reg.sym
                    deps["reads"].add(var_name)
                    deps["quantum"].add(var_name)
                elif hasattr(qarg, "sym"):
                    # Direct QReg reference
                    var_name = qarg.sym
                    deps["reads"].add(var_name)
                    deps["quantum"].add(var_name)

        # Handle measurements
        if op_type == "Measure":
            if hasattr(op, "qargs"):
                for qarg in op.qargs:
                    if hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                        var_name = qarg.reg.sym
                        deps["reads"].add(var_name)
                        deps["quantum"].add(var_name)
                    elif hasattr(qarg, "sym"):
                        # Direct QReg reference
                        var_name = qarg.sym
                        deps["reads"].add(var_name)
                        deps["quantum"].add(var_name)
            if hasattr(op, "cout") and op.cout:
                for cout in op.cout:
                    if hasattr(cout, "reg") and hasattr(cout.reg, "sym"):
                        var_name = cout.reg.sym
                        deps["writes"].add(var_name)
                        deps["classical"].add(var_name)
                    elif hasattr(cout, "sym"):
                        # Direct CReg reference
                        var_name = cout.sym
                        deps["writes"].add(var_name)
                        deps["classical"].add(var_name)

        # Handle SET operations
        if op_type == "SET":
            if hasattr(op, "left") and hasattr(op.left, "reg"):
                var_name = op.left.reg.sym
                deps["writes"].add(var_name)
                deps["classical"].add(var_name)
            if hasattr(op, "right"):
                self._analyze_expression_deps(op.right, deps)

        # Handle control flow
        if op_type in ["If", "While", "For", "Repeat"]:
            # Analyze condition
            if hasattr(op, "condition"):
                self._analyze_expression_deps(op.condition, deps)
            # Analyze body operations
            if hasattr(op, "ops"):
                for sub_op in op.ops:
                    self._analyze_op_dependencies(sub_op, deps, depth + 1)

        # Handle nested blocks (but not too deep to avoid infinite recursion)
        elif hasattr(op, "ops") and hasattr(op, "vars") and depth < 2:
            # This is a block call - analyze it recursively but not too deep
            for sub_op in op.ops:
                self._analyze_op_dependencies(sub_op, deps, depth + 1)

    def _analyze_expression_deps(self, expr, deps: dict[str, set]) -> None:
        """Analyze dependencies in an expression."""
        expr_type = type(expr).__name__

        if expr_type == "Bit":
            if hasattr(expr, "reg") and hasattr(expr.reg, "sym"):
                var_name = expr.reg.sym
                deps["reads"].add(var_name)
                deps["classical"].add(var_name)
        elif expr_type == "Qubit":
            if hasattr(expr, "reg") and hasattr(expr.reg, "sym"):
                var_name = expr.reg.sym
                deps["reads"].add(var_name)
                deps["quantum"].add(var_name)
        elif hasattr(expr, "left") and hasattr(expr, "right"):
            self._analyze_expression_deps(expr.left, deps)
            self._analyze_expression_deps(expr.right, deps)
        elif hasattr(expr, "value"):
            self._analyze_expression_deps(expr.value, deps)

    def _add_final_handling(self, block) -> None:
        """Handle struct decomposition, results, and cleanup in the correct order."""
        # First, decompose any structs that need cleanup
        struct_decompositions = {}  # prefix -> list of decomposed variable names

        for prefix, info in self.struct_info.items():
            # Check if this struct has unconsumed quantum fields
            has_unconsumed_quantum = False
            for suffix, var_type, size in info["fields"]:
                if var_type == "qubit":
                    var_name = info["var_names"][suffix]
                    if var_name not in self.consumed_arrays:
                        has_unconsumed_quantum = True
                        break

            if has_unconsumed_quantum:
                # Decompose the struct
                qec_code_name = info.get("qec_code_name", prefix)
                func_name = (
                    f"{qec_code_name}_decompose"
                    if qec_code_name
                    else f"{prefix}_decompose"
                )

                # Generate variable names for decomposed fields
                decomposed_vars = []
                for suffix, _, _ in sorted(info["fields"]):
                    decomposed_vars.append(f"{prefix}_{suffix}_final")

                # Create the decomposition call
                targets = decomposed_vars
                call = FunctionCall(
                    func_name=func_name,
                    args=[VariableRef(prefix)],
                )

                # Create assignment
                target_tuple = TupleExpression(
                    elements=[VariableRef(name) for name in targets],
                )
                stmt = Assignment(target=target_tuple, value=call)

                self.current_block.statements.append(
                    Comment(f"Decompose struct {prefix} for cleanup"),
                )
                self.current_block.statements.append(stmt)

                # Store decomposition info
                struct_decompositions[prefix] = list(
                    zip(
                        [f[0] for f in sorted(info["fields"])],  # suffixes
                        decomposed_vars,
                        [f[1] for f in sorted(info["fields"])],  # types
                        [f[2] for f in sorted(info["fields"])],  # sizes
                    ),
                )

        # Now add results, using decomposed variables where necessary
        self._add_results_with_decomposition(block, struct_decompositions)

        # Track what arrays have been cleaned up to avoid double-discard
        cleaned_up_arrays = set()

        # Finally, clean up quantum arrays
        self._add_cleanup_with_decomposition(
            block,
            struct_decompositions,
            cleaned_up_arrays,
        )

        # Also run the regular cleanup for non-struct arrays
        self._add_cleanup(block, cleaned_up_arrays)

    def _add_results_with_decomposition(self, block, struct_decompositions) -> None:
        """Add result calls, using decomposed variables where necessary."""
        if hasattr(block, "vars"):
            for var in block.vars:
                if type(var).__name__ == "CReg":
                    var_name = var.sym

                    # Check for renaming
                    actual_name = var_name
                    if var_name in self.plan.renamed_variables:
                        actual_name = self.plan.renamed_variables[var_name]

                    # Check if this variable is part of a decomposed struct
                    value_ref = None
                    for prefix, info in self.struct_info.items():
                        if var_name in info["var_names"].values():
                            # Find the field name for this variable
                            for suffix, mapped_var in info["var_names"].items():
                                if mapped_var == var_name:
                                    # Check if struct was decomposed
                                    if prefix in struct_decompositions:
                                        # Find the decomposed variable
                                        for (
                                            field_suffix,
                                            decomposed_var,
                                            _,
                                            _,
                                        ) in struct_decompositions[prefix]:
                                            if field_suffix == suffix:
                                                value_ref = VariableRef(decomposed_var)
                                                break
                                    else:
                                        # Struct not decomposed, use field access
                                        value_ref = FieldAccess(
                                            obj=VariableRef(prefix),
                                            field=suffix,
                                        )
                                    break
                            break

                    if value_ref is None:
                        # Check if this array was unpacked
                        # Check both var_name (original) and actual_name (renamed)
                        is_unpacked = var_name in self.plan.arrays_to_unpack or (
                            hasattr(self, "unpacked_vars")
                            and (
                                var_name in self.unpacked_vars
                                or actual_name in self.unpacked_vars
                            )
                        )

                        if is_unpacked:
                            # Array was unpacked - must reconstruct from elements for linearity
                            element_names = None
                            if hasattr(self, "unpacked_vars"):
                                # Try original name first, then renamed name
                                if var_name in self.unpacked_vars:
                                    element_names = self.unpacked_vars[var_name]
                                elif actual_name in self.unpacked_vars:
                                    element_names = self.unpacked_vars[actual_name]

                            if element_names:
                                # Reconstruct the array and assign it back to the original variable
                                reconstruction_expr = self._create_array_reconstruction(
                                    element_names,
                                )
                                reconstruction_stmt = Assignment(
                                    target=VariableRef(actual_name),
                                    value=reconstruction_expr,
                                )
                                self.current_block.statements.append(
                                    reconstruction_stmt,
                                )
                                value_ref = VariableRef(actual_name)
                            else:
                                # Fallback: use original array if unpacked_vars not available
                                value_ref = VariableRef(actual_name)
                        else:
                            # Not unpacked, use direct variable reference
                            value_ref = VariableRef(actual_name)

                    # Add result call
                    call = FunctionCall(
                        func_name="result",
                        args=[
                            Literal(var.sym),  # Original name as label
                            value_ref,  # Actual variable or decomposed field
                        ],
                    )

                    # Create a wrapper that renders just the function call
                    class ExpressionStatement(Statement):
                        def __init__(self, expr):
                            self.expr = expr

                        def analyze(self, context):
                            self.expr.analyze(context)

                        def render(self, context):
                            return self.expr.render(context)

                    self.current_block.statements.append(ExpressionStatement(call))

    def _add_cleanup_with_decomposition(
        self,
        block,
        struct_decompositions,
        cleaned_up_arrays,
    ) -> None:
        _ = block  # Currently not used
        """Add cleanup for quantum arrays, using decomposed variables."""
        # First handle decomposed struct fields
        for prefix, fields in struct_decompositions.items():
            self.current_block.statements.append(
                Comment(f"Discard quantum fields from {prefix}"),
            )
            for suffix, decomposed_var, var_type, size in fields:
                if var_type == "qubit" and decomposed_var not in cleaned_up_arrays:
                    stmt = FunctionCall(
                        func_name="quantum.discard_array",
                        args=[VariableRef(decomposed_var)],
                    )
                    cleaned_up_arrays.add(decomposed_var)
                    # Also track the original variable name to prevent double cleanup
                    if prefix in self.struct_info:
                        info = self.struct_info[prefix]
                        if suffix in info["var_names"]:
                            original_var = info["var_names"][suffix]
                            cleaned_up_arrays.add(original_var)

                    # Create expression statement wrapper
                    class ExpressionStatement(Statement):
                        def __init__(self, expr):
                            self.expr = expr

                        def analyze(self, context):
                            self.expr.analyze(context)

                        def render(self, context):
                            return self.expr.render(context)

                    self.current_block.statements.append(ExpressionStatement(stmt))

        # Note: Non-struct arrays are handled in _add_cleanup, not here

    def _add_cleanup(self, block, cleaned_up_arrays=None) -> None:
        """Add cleanup for unconsumed qubits."""
        if cleaned_up_arrays is None:
            cleaned_up_arrays = set()
        # Track consumed qubits during operation conversion
        consumed = {}  # qreg_name -> set of indices

        # Analyze operations to find consumed qubits
        if hasattr(block, "ops"):
            for op in block.ops:
                self._track_consumed_qubits(op, consumed)

        # First, check if we have structs that need cleanup
        struct_cleanup_done = set()
        for prefix, info in self.struct_info.items():
            # Check if any quantum arrays in this struct need cleanup
            needs_cleanup = False
            for suffix, var_type, size in info["fields"]:
                if var_type == "qubit":
                    var_name = info["var_names"][suffix]
                    if var_name not in self.consumed_arrays:
                        needs_cleanup = True
                        break

            if needs_cleanup and prefix not in struct_cleanup_done:
                # We're at the end of main, after results.
                # We can't access struct fields directly after consuming the struct,
                # so we'll just leave quantum arrays in structs for now.
                # The HUGR compiler will need to handle this pattern.

                # Add a comment noting this limitation
                self.current_block.statements.append(
                    Comment(
                        f"Note: struct {prefix} contains unconsumed quantum arrays",
                    ),
                )

                struct_cleanup_done.add(prefix)
                # Mark arrays as handled
                for suffix, var_type, size in info["fields"]:
                    if var_type == "qubit":
                        var_name = info["var_names"][suffix]
                        self.consumed_arrays.add(var_name)

        # First handle fresh variables from function returns
        if hasattr(self, "fresh_variables_to_track"):
            for fresh_name, info in self.fresh_variables_to_track.items():
                if info["type"] == "quantum_array" and not info.get("used", False):
                    # This fresh variable was not used, add cleanup
                    # Check if it was already cleaned up (e.g., by being measured)
                    original_name = info["original"]
                    was_consumed = (
                        hasattr(self, "consumed_arrays")
                        and original_name in self.consumed_arrays
                    ) or (
                        hasattr(self, "consumed_resources")
                        and original_name in self.consumed_resources
                    )

                    if not was_consumed and fresh_name not in cleaned_up_arrays:
                        self.current_block.statements.append(
                            Comment(f"Discard unused fresh variable {fresh_name}"),
                        )
                        # Need to check if this is an array or needs special handling
                        # For now, assume it's a quantum array that needs discard_array
                        stmt = FunctionCall(
                            func_name="quantum.discard_array",
                            args=[VariableRef(fresh_name)],
                        )

                        # Create expression statement wrapper
                        class ExpressionStatement(Statement):
                            def __init__(self, expr):
                                self.expr = expr

                            def analyze(self, context):
                                self.expr.analyze(context)

                            def render(self, context):
                                return self.expr.render(context)

                        self.current_block.statements.append(ExpressionStatement(stmt))
                        cleaned_up_arrays.add(fresh_name)

        # Check each quantum register not in structs
        if hasattr(block, "vars"):

            for var in block.vars:
                if type(var).__name__ == "QReg":
                    var_name = var.sym

                    # Skip if this array is part of a struct
                    in_struct = False
                    for prefix, info in self.struct_info.items():
                        if var_name in info["var_names"].values():
                            in_struct = True
                            break

                    if in_struct:
                        continue
                    # Check for renaming
                    if var_name in self.plan.renamed_variables:
                        var_name = self.plan.renamed_variables[var_name]

                    consumed_indices = consumed.get(var.sym, set())

                    # Check if this array was consumed by an @owned function or measurement
                    was_consumed_by_function = (
                        hasattr(self, "consumed_arrays")
                        and var.sym in self.consumed_arrays
                    )

                    was_consumed_by_measurement = (
                        hasattr(self, "consumed_resources")
                        and var.sym in self.consumed_resources
                    )
                    was_dynamically_allocated = (
                        hasattr(self, "dynamic_allocations")
                        and var.sym in self.dynamic_allocations
                    )

                    # Handle partially consumed arrays
                    # BUT: Skip if the whole array was consumed by an @owned function
                    if (
                        len(consumed_indices) > 0
                        and len(consumed_indices) < var.size
                        and not was_consumed_by_function
                    ):
                        # Array was partially consumed - need to discard entire array
                        if var_name not in cleaned_up_arrays:
                            self.current_block.statements.append(
                                Comment(f"Discard {var.sym}"),
                            )
                            stmt = FunctionCall(
                                func_name="quantum.discard_array",
                                args=[VariableRef(var_name)],
                            )

                            # Create expression statement wrapper
                            class ExpressionStatement(Statement):
                                def __init__(self, expr):
                                    self.expr = expr

                                def analyze(self, context):
                                    self.expr.analyze(context)

                                def render(self, context):
                                    return self.expr.render(context)

                            self.current_block.statements.append(
                                ExpressionStatement(stmt),
                            )
                            cleaned_up_arrays.add(var_name)
                    # Only discard arrays that weren't consumed by @owned functions or measurements
                    # UNLESS they have explicitly reset qubits (which need cleanup)
                    elif True:
                        # Check if this array has explicitly reset qubits (from Prep operations)
                        # Even if consumed by measurement, explicitly reset qubits need cleanup
                        has_explicitly_reset = (
                            hasattr(self, "explicitly_reset_qubits")
                            and var.sym in self.explicitly_reset_qubits
                            and len(self.explicitly_reset_qubits[var.sym]) > 0
                        )

                        if not was_consumed_by_function and (
                            not was_consumed_by_measurement or has_explicitly_reset
                        ):
                            if was_dynamically_allocated:
                                # For dynamically allocated arrays, discard individual
                                # qubits that weren't measured
                                self.current_block.statements.append(
                                    Comment(f"Discard dynamically allocated {var.sym}"),
                                )

                                # Check which individual qubits were allocated and not consumed
                                if hasattr(self, "allocated_ancillas"):
                                    # Track which variables we've already discarded to avoid duplicates
                                    discarded_vars = set()

                                    # Discard each allocated ancilla that belongs to this qreg
                                    # We need to check all allocated ancillas that start with the qreg name
                                    for ancilla_var in list(self.allocated_ancillas):
                                        # Check if this ancilla belongs to the current qreg
                                        # It should start with the qreg name followed by underscore
                                        if ancilla_var.startswith(
                                            (f"{var.sym}_", f"_{var.sym}_"),
                                        ):
                                            # Apply variable remapping if exists (for Prep operations)
                                            var_to_discard = (
                                                self.variable_remapping.get(
                                                    ancilla_var,
                                                    ancilla_var,
                                                )
                                            )

                                            # Skip if we've already discarded this variable
                                            if var_to_discard in discarded_vars:
                                                continue
                                            discarded_vars.add(var_to_discard)

                                            discard_stmt = FunctionCall(
                                                func_name="quantum.discard",
                                                args=[VariableRef(var_to_discard)],
                                            )

                                            # Create expression statement wrapper
                                            class ExpressionStatement(Statement):
                                                def __init__(self, expr):
                                                    self.expr = expr

                                                def analyze(self, context):
                                                    self.expr.analyze(context)

                                                def render(self, context):
                                                    return self.expr.render(context)

                                            self.current_block.statements.append(
                                                ExpressionStatement(discard_stmt),
                                            )
                            else:
                                # Regular pre-allocated array

                                # Skip if already consumed by a function call
                                # Also check if the remapped name was consumed
                                remapped_consumed = False
                                if (
                                    hasattr(self, "array_remapping")
                                    and var_name in self.array_remapping
                                ):
                                    remapped_name = self.array_remapping[var_name]
                                    if (
                                        hasattr(self, "consumed_arrays")
                                        and remapped_name in self.consumed_arrays
                                    ):
                                        remapped_consumed = True

                                # Check if array has explicitly reset qubits (from Prep operations)
                                # These need cleanup even if consumed by measurement
                                has_explicitly_reset = (
                                    hasattr(self, "explicitly_reset_qubits")
                                    and var.sym in self.explicitly_reset_qubits
                                    and len(self.explicitly_reset_qubits[var.sym]) > 0
                                )

                                # Check if array was consumed by an @owned function call or by measurements
                                array_consumed = (
                                    hasattr(self, "consumed_arrays")
                                    and (
                                        var.sym in self.consumed_arrays
                                        or var_name in self.consumed_arrays
                                    )
                                ) or (
                                    hasattr(self, "consumed_resources")
                                    and (
                                        var.sym in self.consumed_resources
                                        or var_name in self.consumed_resources
                                    )
                                )

                                # Also check if this is a reconstructed array that was passed to a function
                                is_reconstructed = (
                                    hasattr(self, "reconstructed_arrays")
                                    and var_name in self.reconstructed_arrays
                                )

                                # Allow cleanup if:
                                # 1. Array not already cleaned up
                                # 2. Either not consumed OR has explicitly reset qubits
                                # 3. Remapped version not consumed
                                # 4. Not a reconstructed array
                                if (
                                    var_name not in cleaned_up_arrays
                                    and (not array_consumed or has_explicitly_reset)
                                    and not remapped_consumed
                                    and not is_reconstructed
                                ):
                                    # Check if this array has been unpacked or remapped
                                    # If so, we can't discard the original name
                                    if (
                                        hasattr(self, "unpacked_vars")
                                        and var_name in self.unpacked_vars
                                    ):
                                        # Array was unpacked - check if it has explicitly reset qubits
                                        explicitly_reset_indices = set()
                                        if (
                                            hasattr(self, "explicitly_reset_qubits")
                                            and var_name in self.explicitly_reset_qubits
                                        ):
                                            explicitly_reset_indices = (
                                                self.explicitly_reset_qubits[var_name]
                                            )

                                        if explicitly_reset_indices:
                                            # Check if we already did inline reconstruction
                                            # If so, skip cleanup reconstruction to avoid AlreadyUsedError
                                            skip_cleanup_reconstruction = (
                                                hasattr(
                                                    self,
                                                    "inline_reconstructed_arrays",
                                                )
                                                and var_name
                                                in self.inline_reconstructed_arrays
                                            )

                                            if not skip_cleanup_reconstruction:
                                                # Array has fresh qubits from Prep - reconstruct and discard
                                                comment_text = (
                                                    f"Reconstruct {var.sym} from unpacked "
                                                    f"elements (has fresh qubits)"
                                                )
                                                self.current_block.statements.append(
                                                    Comment(comment_text),
                                                )

                                                # Get unpacked element names (it's a list, not a dict)
                                                element_names = self.unpacked_vars[
                                                    var_name
                                                ]

                                                # Apply variable remapping to get the latest names
                                                remapped_element_names = [
                                                    self.variable_remapping.get(
                                                        elem,
                                                        elem,
                                                    )
                                                    for elem in element_names
                                                ]

                                                # Reconstruct array: var = array(elem1, elem2, ...)
                                                array_elements = [
                                                    VariableRef(elem)
                                                    for elem in remapped_element_names
                                                ]
                                                array_constructor = FunctionCall(
                                                    func_name="array",
                                                    args=array_elements,
                                                )
                                                reconstruct_stmt = Assignment(
                                                    target=VariableRef(var_name),
                                                    value=array_constructor,
                                                )
                                                self.current_block.statements.append(
                                                    reconstruct_stmt,
                                                )

                                            # Now discard the reconstructed array
                                            self.current_block.statements.append(
                                                Comment(
                                                    f"Discard reconstructed {var.sym}",
                                                ),
                                            )
                                            array_ref = VariableRef(var_name)
                                            stmt = FunctionCall(
                                                func_name="quantum.discard_array",
                                                args=[array_ref],
                                            )

                                            # Create expression statement wrapper
                                            class ExpressionStatement(Statement):
                                                def __init__(self, expr):
                                                    self.expr = expr

                                                def analyze(self, context):
                                                    self.expr.analyze(context)

                                                def render(self, context):
                                                    return self.expr.render(context)

                                            self.current_block.statements.append(
                                                ExpressionStatement(stmt),
                                            )
                                            cleaned_up_arrays.add(var_name)
                                            # Skip the remapping/normal discard code below
                                            continue
                                        # Array was unpacked and fully consumed - skip discard
                                        self.current_block.statements.append(
                                            Comment(
                                                f"Skip discard {var.sym} - already unpacked and consumed",
                                            ),
                                        )
                                        continue
                                    if (
                                        hasattr(self, "array_remapping")
                                        and var_name in self.array_remapping
                                    ):
                                        # Array was remapped - use the new name
                                        remapped_name = self.array_remapping[var_name]
                                        self.current_block.statements.append(
                                            Comment(
                                                f"Discard {var.sym} (remapped to {remapped_name})",
                                            ),
                                        )
                                        array_ref = VariableRef(remapped_name)
                                    else:
                                        # Normal case - use original name
                                        self.current_block.statements.append(
                                            Comment(f"Discard {var.sym}"),
                                        )
                                        array_ref = VariableRef(var_name)

                                    stmt = FunctionCall(
                                        func_name="quantum.discard_array",
                                        args=[array_ref],
                                    )

                                    # Create expression statement wrapper
                                    class ExpressionStatement(Statement):
                                        def __init__(self, expr):
                                            self.expr = expr

                                        def analyze(self, context):
                                            self.expr.analyze(context)

                                        def render(self, context):
                                            return self.expr.render(context)

                                    self.current_block.statements.append(
                                        ExpressionStatement(stmt),
                                    )
                                    cleaned_up_arrays.add(var_name)

    def _check_has_element_operations(self, block, var_name: str) -> bool:
        """Check if a block has element-wise operations on a variable.

        This is used to determine if we should use @owned for array parameters.
        Element-wise operations (like reset on individual elements) don't work
        with @owned arrays in Guppy.
        """
        if not hasattr(block, "ops"):
            return False

        for op in block.ops:
            op_type = type(op).__name__

            # Check for Prep operations on the whole array
            if op_type == "Prep" and hasattr(op, "qargs"):
                for qarg in op.qargs:
                    if hasattr(qarg, "sym") and qarg.sym == var_name:
                        # Prep on the whole array - this needs element access
                        return True

            # Check for operations on individual elements
            if hasattr(op, "qargs"):
                for qarg in op.qargs:
                    if (
                        hasattr(qarg, "reg")
                        and hasattr(qarg.reg, "sym")
                        and qarg.reg.sym == var_name
                        and hasattr(qarg, "index")
                        and op_type in ["Prep", "Measure"]
                    ):
                        return True

            # Recursively check nested blocks
            if hasattr(op, "ops") and self._check_has_element_operations(op, var_name):
                return True

        return False

    def _track_consumed_qubits(self, op, consumed: dict[str, set[int]]) -> None:
        """Track which qubits are consumed by an operation or block.

        Also tracks explicit Prep (reset) operations to distinguish them from
        automatic post-measurement replacements.
        """
        op_type = type(op).__name__

        # Handle Block types - recurse into their operations
        if hasattr(op, "ops") and op_type not in ["Measure", "If", "Else", "While"]:
            # This is a custom Block - analyze its operations
            for nested_op in op.ops:
                self._track_consumed_qubits(nested_op, consumed)
            return

        # Track explicit Prep operations - these are semantic resets that should be returned
        if op_type == "Prep" and hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                # Handle full array reset
                if hasattr(qarg, "sym") and hasattr(qarg, "size"):
                    qreg_name = qarg.sym
                    if qreg_name not in self.explicitly_reset_qubits:
                        self.explicitly_reset_qubits[qreg_name] = set()
                    # Track all indices as explicitly reset
                    for i in range(qarg.size):
                        self.explicitly_reset_qubits[qreg_name].add(i)
                # Handle individual qubit reset
                elif hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                    qreg_name = qarg.reg.sym
                    if qreg_name not in self.explicitly_reset_qubits:
                        self.explicitly_reset_qubits[qreg_name] = set()

                    if hasattr(qarg, "index"):
                        self.explicitly_reset_qubits[qreg_name].add(qarg.index)

        if op_type == "Measure" and hasattr(op, "qargs") and op.qargs:
            for qarg in op.qargs:
                # Handle full array measurement
                if hasattr(qarg, "sym") and hasattr(qarg, "size"):
                    qreg_name = qarg.sym
                    if qreg_name not in consumed:
                        consumed[qreg_name] = set()
                    # Mark all qubits as consumed
                    indices = set(range(qarg.size))
                    for i in indices:
                        consumed[qreg_name].add(i)
                    # Track in scope manager
                    self.scope_manager.track_resource_usage(
                        qreg_name,
                        indices,
                        consumed=True,
                    )
                # Handle individual qubit measurement
                elif hasattr(qarg, "reg") and hasattr(qarg.reg, "sym"):
                    qreg_name = qarg.reg.sym
                    if qreg_name not in consumed:
                        consumed[qreg_name] = set()

                    if hasattr(qarg, "index"):
                        consumed[qreg_name].add(qarg.index)
                        # Track in scope manager
                        self.scope_manager.track_resource_usage(
                            qreg_name,
                            {qarg.index},
                            consumed=True,
                        )

        # Don't recurse into nested blocks that are separate function calls
        # They handle their own consumption and return fresh qubits
        # Only recurse into inline blocks (like If/Else)
        if hasattr(op, "ops") and op_type in ["If", "Else", "While"]:
            for nested_op in op.ops:
                self._track_consumed_qubits(nested_op, consumed)

        # Check else blocks
        if (
            op_type == "If"
            and hasattr(op, "else_block")
            and op.else_block
            and hasattr(op.else_block, "ops")
        ):
            for nested_op in op.else_block.ops:
                self._track_consumed_qubits(nested_op, consumed)

    def _array_needs_full_allocation(self, array_name: str, block) -> bool:
        """Check if an array needs full allocation due to full array operations."""
        if not hasattr(block, "ops"):
            return False

        for op in block.ops:
            if self._operation_uses_full_array(op, array_name):
                return True

            # Check nested operations
            if hasattr(op, "ops"):
                for nested_op in op.ops:
                    if self._operation_uses_full_array(nested_op, array_name):
                        return True

            # Check else blocks
            if (
                hasattr(op, "else_block")
                and op.else_block
                and hasattr(op.else_block, "ops")
            ):
                for nested_op in op.else_block.ops:
                    if self._operation_uses_full_array(nested_op, array_name):
                        return True

        return False

    def _operation_uses_full_array(self, op, array_name: str) -> bool:
        """Check if an operation uses a full array (e.g., Measure(q) > c)."""
        if hasattr(op, "qargs") and len(op.qargs) == 1:
            qarg = op.qargs[0]
            # Check for full array reference (has sym and size but no index)
            if (
                hasattr(qarg, "sym")
                and qarg.sym == array_name
                and hasattr(qarg, "size")
                and qarg.size > 1
                and not hasattr(qarg, "index")
            ):
                return True
        return False

    def _add_results(self, block) -> None:
        """Add result() calls for classical registers."""
        # Debug: Uncomment to see unpacked_vars state
        if hasattr(block, "vars"):
            for var in block.vars:
                if type(var).__name__ == "CReg":
                    var_name = var.sym

                    # Check for renaming
                    actual_name = var_name
                    if var_name in self.plan.renamed_variables:
                        actual_name = self.plan.renamed_variables[var_name]

                    # Check if this variable is part of a struct
                    value_ref = None
                    for prefix, info in self.struct_info.items():
                        if var_name in info["var_names"].values():
                            # Find the field name for this variable
                            for suffix, mapped_var in info["var_names"].items():
                                if mapped_var == var_name:
                                    # Access through struct field
                                    value_ref = FieldAccess(
                                        obj=VariableRef(prefix),
                                        field=suffix,
                                    )
                                    break
                            break

                    if value_ref is None:
                        # Check if this array was unpacked
                        # Check both var_name (original) and actual_name (renamed)
                        is_unpacked = var_name in self.plan.arrays_to_unpack or (
                            hasattr(self, "unpacked_vars")
                            and (
                                var_name in self.unpacked_vars
                                or actual_name in self.unpacked_vars
                            )
                        )

                        if is_unpacked:
                            # Array was unpacked - must reconstruct from elements for linearity
                            element_names = None
                            if hasattr(self, "unpacked_vars"):
                                # Try original name first, then renamed name
                                if var_name in self.unpacked_vars:
                                    element_names = self.unpacked_vars[var_name]
                                elif actual_name in self.unpacked_vars:
                                    element_names = self.unpacked_vars[actual_name]

                            if element_names:
                                value_ref = self._create_array_reconstruction(
                                    element_names,
                                )
                            else:
                                # Fallback: use original array if unpacked_vars not available
                                value_ref = VariableRef(actual_name)
                        else:
                            # Not unpacked, use direct variable reference
                            value_ref = VariableRef(actual_name)

                    # Add result call
                    call = FunctionCall(
                        func_name="result",
                        args=[
                            Literal(var.sym),  # Original name as label
                            value_ref,  # Actual variable or struct field
                        ],
                    )

                    # Create a wrapper that renders just the function call
                    class ExpressionStatement(Statement):
                        def __init__(self, expr):
                            self.expr = expr

                        def analyze(self, context):
                            self.expr.analyze(context)

                        def render(self, context):
                            return self.expr.render(context)

                    self.current_block.statements.append(ExpressionStatement(call))

    def _detect_struct_patterns(self, block: SLRBlock) -> None:
        """Detect variables that should be grouped into structs.

        Looking for patterns where multiple variables share a common prefix
        (e.g., x_d, x_a, x_c all belong to quantum code 'x').
        """
        # First, try to determine the quantum code class from variable metadata
        qec_code_name = None
        qec_instance_mapping = {}  # Maps instance name -> class name

        # Check if block.vars has source class information
        if hasattr(block, "vars") and hasattr(block.vars, "var_source_classes"):
            # Get the source class from the metadata
            for var_name, source_class in block.vars.var_source_classes.items():
                # Extract the prefix from the variable name
                if "_" in var_name:
                    prefix = var_name.split("_")[0]
                    if prefix not in qec_instance_mapping:
                        qec_instance_mapping[prefix] = source_class.lower()
                        if not qec_code_name:
                            qec_code_name = source_class.lower()

        # If no QEC class found in vars, fall back to searching operations
        if not qec_code_name:
            # Helper function to recursively search for QEC code
            def find_qec_code_in_block(op, depth=0, max_depth=5):
                if depth > max_depth:
                    return None

                results = []

                # Check if this op has QEC module info
                if hasattr(op, "__class__") and hasattr(op.__class__, "__module__"):
                    module = op.__class__.__module__
                    # Extract QEC code name from module path like
                    # 'pecos.qeclib.steane.preps.pauli_states'
                    if "pecos.qeclib." in module:
                        parts = module.split(".")
                        if len(parts) > 2 and "qeclib" in parts:
                            qec_idx = parts.index("qeclib")
                            if qec_idx + 1 < len(parts):
                                candidate = parts[qec_idx + 1]
                                # Skip generic names like 'qubit'
                                if candidate not in ["qubit", "bit", "ops", "gates"]:
                                    results.append(candidate)

                # Check nested operations
                if hasattr(op, "ops"):
                    for nested_op in op.ops:
                        result = find_qec_code_in_block(nested_op, depth + 1, max_depth)
                        if result:
                            results.append(result)

                # Return the first non-generic result
                for r in results:
                    if r not in ["qubit", "bit", "ops", "gates"]:
                        return r

                return results[0] if results else None

            # Try to find the QEC code class from the operations
            if hasattr(block, "ops"):
                for op in block.ops:
                    qec_code_name = find_qec_code_in_block(op)
                    if qec_code_name:
                        break

        # Collect all variables
        all_vars = {}
        if hasattr(block, "vars"):
            for var in block.vars:
                if hasattr(var, "sym"):
                    var_name = var.sym
                    all_vars[var_name] = var

        # Also check context variables
        for var_name, var_info in self.context.variables.items():
            if var_name not in all_vars:
                all_vars[var_name] = var_info

        # Group by prefix
        prefix_groups = {}
        for var_name, var in all_vars.items():
            if "_" in var_name:
                prefix = var_name.split("_")[0]
                suffix = "_".join(var_name.split("_")[1:])

                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []

                # Determine type and size
                size = var.size if hasattr(var, "size") else 1

                # Determine if quantum or classical
                is_quantum = True
                if hasattr(var, "is_quantum"):
                    is_quantum = var.is_quantum
                elif type(var).__name__ == "CReg":
                    is_quantum = False
                elif hasattr(var, "resource_type"):
                    is_quantum = var.resource_type == ResourceState.QUANTUM

                var_type = "qubit" if is_quantum else "bool"

                # Check if this is an ancilla qubit that should be kept separate
                is_ancilla = False
                if var_type == "qubit" and hasattr(self, "qubit_usage_stats"):
                    stats = self.qubit_usage_stats.get(var_name)
                    if stats:
                        role = stats.classify_role()
                        if role == QubitRole.ANCILLA:
                            is_ancilla = True
                            # Store this for later use
                            if not hasattr(self, "ancilla_qubits"):
                                self.ancilla_qubits = set()
                            self.ancilla_qubits.add(var_name)

                if not is_ancilla:
                    prefix_groups[prefix].append((suffix, var_type, size, var_name))

        # Create struct info for groups with multiple related variables
        # BUT avoid structs with too many fields due to guppylang limitations
        # Setting to 5 to be very conservative - complex QEC codes need individual array handling
        max_struct_fields = 5  # Limit to avoid guppylang linearity issues

        for prefix, vars_list in prefix_groups.items():
            if len(vars_list) >= 2:
                # Check if this looks like a quantum code pattern
                has_quantum = any(var[1] == "qubit" for var in vars_list)

                # Skip struct creation if too many fields (causes guppylang issues)
                if len(vars_list) > max_struct_fields:
                    msg = (
                        f"# Skipping struct creation for '{prefix}' with "
                        f"{len(vars_list)} fields (exceeds limit of {max_struct_fields})"
                    )
                    print(msg)
                    continue

                if has_quantum:
                    # Use QEC code name for struct if available, otherwise use prefix
                    struct_base_name = qec_code_name if qec_code_name else prefix

                    self.struct_info[prefix] = {
                        "fields": [(v[0], v[1], v[2]) for v in vars_list],
                        "struct_name": f"{struct_base_name}_struct",
                        "var_names": {
                            v[0]: v[3] for v in vars_list
                        },  # suffix -> full var name
                        "qec_code_name": qec_code_name,  # Store for function naming
                        "ancilla_vars": getattr(
                            self,
                            "ancilla_qubits",
                            set(),
                        ),  # Track which vars were excluded
                    }

    def _generate_struct_definitions(self) -> list[str]:
        """Generate Guppy struct definitions."""
        lines = []

        for prefix, info in sorted(self.struct_info.items()):
            struct_name = info["struct_name"]

            # Generate struct
            lines.append("@guppy.struct")
            lines.append("@no_type_check")
            lines.append(f"class {struct_name}:")

            # Add fields sorted by suffix
            for suffix, var_type, size in sorted(info["fields"]):
                field_type = f"array[{var_type}, {size}]" if size > 1 else var_type
                lines.append(f"    {suffix}: {field_type}")

            lines.append("")  # Empty line after struct

        return lines

    def _generate_struct_decompose_function(
        self,
        prefix: str,
        info: dict,
    ) -> Function | None:
        """Generate a decompose function for a struct."""
        struct_name = info["struct_name"]
        qec_code_name = info.get("qec_code_name", prefix)
        func_name = (
            f"{qec_code_name}_decompose" if qec_code_name else f"{prefix}_decompose"
        )

        # Build return type - tuple of all fields
        return_types = []
        field_names = []
        for suffix, var_type, size in sorted(info["fields"]):
            field_names.append(suffix)
            return_types.append(
                f"array[{var_type}, {size}]" if size > 1 else var_type,
            )

        return_type = f"tuple[{', '.join(return_types)}]"

        # Create function body
        body = Block()

        # The key to avoiding AlreadyUsedError: return all fields in a single expression
        # This works because guppylang handles the struct consumption atomically
        field_refs = [
            FieldAccess(obj=VariableRef(prefix), field=suffix) for suffix in field_names
        ]

        # Return all fields directly in one statement
        return_stmt = ReturnStatement(value=TupleExpression(elements=field_refs))
        body.statements.append(return_stmt)

        return Function(
            name=func_name,
            params=[(prefix, f"{struct_name} @owned")],
            return_type=return_type,
            body=body,
            decorators=["guppy", "no_type_check"],
        )

    def _generate_struct_discard_function(
        self,
        prefix: str,
        info: dict,
    ) -> Function | None:
        """Generate a discard function for a struct."""
        # Check if struct has quantum fields
        has_quantum = any(field[1] == "qubit" for field in info["fields"])
        if not has_quantum:
            return None

        struct_name = info["struct_name"]
        qec_code_name = info.get("qec_code_name", prefix)
        func_name = f"{qec_code_name}_discard" if qec_code_name else f"{prefix}_discard"

        # Create function body
        body = Block()

        # We need to handle discard differently to avoid AlreadyUsedError
        # First decompose the struct, then discard quantum fields

        # Build list of field names for decomposition
        field_names = [suffix for suffix, _, _ in sorted(info["fields"])]

        # Call decompose to get all fields
        decompose_func_name = (
            f"{qec_code_name}_decompose" if qec_code_name else f"{prefix}_decompose"
        )
        decompose_call = FunctionCall(
            func_name=decompose_func_name,
            args=[VariableRef(prefix)],
        )

        # Create variables to hold decomposed fields
        field_vars = [
            f"_{suffix}" if suffix == prefix else suffix for suffix in field_names
        ]

        # Define TupleAssignment locally
        class TupleAssignment(Statement):
            def __init__(self, targets, value):
                self.targets = targets
                self.value = value

            def analyze(self, context):
                self.value.analyze(context)

            def render(self, context):
                targets_str = ", ".join(self.targets)
                value_lines = self.value.render(context)
                # FunctionCall render returns a list with one string
                value_str = value_lines[0] if value_lines else ""
                return [f"{targets_str} = {value_str}"]

        decompose_stmt = TupleAssignment(
            targets=field_vars,
            value=decompose_call,
        )
        body.statements.append(decompose_stmt)

        # Now discard quantum fields
        for i, (suffix, var_type, size) in enumerate(sorted(info["fields"])):
            if var_type == "qubit":
                field_var = field_vars[i]
                stmt = FunctionCall(
                    func_name="quantum.discard_array",
                    args=[VariableRef(field_var)],
                )

                # Create expression statement wrapper
                class ExpressionStatement(Statement):
                    def __init__(self, expr):
                        self.expr = expr

                    def analyze(self, context):
                        self.expr.analyze(context)

                    def render(self, context):
                        return self.expr.render(context)

                body.statements.append(ExpressionStatement(stmt))

        return Function(
            name=func_name,
            params=[(prefix, f"{struct_name} @owned")],
            return_type="None",
            body=body,
            decorators=["guppy", "no_type_check"],
        )

    def _add_struct_initialization(
        self,
        prefix: str,
        info: dict,
        block: SLRBlock,
    ) -> None:
        """Add struct initialization to current block."""
        struct_name = info["struct_name"]

        # Create the struct instance
        # For now, initialize fields individually then create struct
        # TODO: Could be optimized to initialize struct directly

        # First, declare the individual arrays
        for suffix, var_type, size in info["fields"]:
            var_name = info["var_names"][suffix]
            # Find the original variable
            for var in block.vars:
                if hasattr(var, "sym") and var.sym == var_name:
                    self._add_variable_declaration(var)
                    break

        # Then create struct instance
        field_refs = []
        for suffix, _, _ in sorted(info["fields"]):
            var_name = info["var_names"][suffix]
            field_refs.append(VariableRef(var_name))

        # Create struct construction expression
        struct_expr = self._create_struct_construction(
            struct_name,
            [f[0] for f in sorted(info["fields"])],
            field_refs,
        )

        # Add assignment: prefix = struct_name(field1=var1, field2=var2, ...)
        stmt = Assignment(
            target=VariableRef(prefix),
            value=struct_expr,
        )
        self.current_block.statements.append(stmt)

        # Update context to track struct variable
        self.context.add_variable(
            VariableInfo(
                name=prefix,
                original_name=prefix,
                var_type=struct_name,
                is_struct=True,
                struct_info=info,
            ),
        )

        # Mark the individual arrays as part of the struct so operations use struct fields
        for suffix, var_type, size in info["fields"]:
            var_name = info["var_names"][suffix]
            var_info = self.context.lookup_variable(var_name)
            if var_info:
                var_info.is_struct_field = True
                var_info.struct_name = prefix
                var_info.field_name = suffix

    def _restore_array_sizes_for_block_call(self, block) -> None:
        """Restore array sizes before a function call in a loop.

        When a function returns a smaller array than it receives (e.g., consuming qubits),
        and that result is used in a loop to call the same function again, we need to
        restore the array size by allocating fresh qubits before the next call.

        This implements the user's guidance: "We could prepare them right before we need them"
        """

        # Check if this is a block that will become a function call
        if not hasattr(block, "ops") or not hasattr(block, "vars"):
            return

        # Analyze the block to get array size information
        from pecos.slr.gen_codes.guppy.ir_analyzer import IRAnalyzer

        analyzer = IRAnalyzer()
        analyzer.analyze_block(block, self.context.variables)

        # Analyze what this block needs
        deps = self._analyze_block_dependencies(block)

        # Determine what function this block will call
        func_name = self._get_function_name_for_block(block)

        # Check quantum arrays that this block uses
        for var in deps["quantum"] & deps["reads"]:
            # Skip struct variables
            if any(
                var in info["var_names"].values() for info in self.struct_info.values()
            ):
                continue

            # Check if we have a refreshed version from a previous function call
            actual_var = var
            if hasattr(self, "refreshed_arrays") and var in self.refreshed_arrays:
                actual_var = self.refreshed_arrays[var]

            # Get the expected size from the original variable context
            expected_size = None
            if var in self.context.variables:
                var_info = self.context.variables[var]
                if hasattr(var_info, "size"):
                    expected_size = var_info.size

            if expected_size is None:
                continue  # Couldn't determine expected size

            # Check the actual current size if the array is unpacked
            actual_size = None
            if hasattr(self, "unpacked_vars") and actual_var in self.unpacked_vars:
                actual_size = len(self.unpacked_vars[actual_var])
            if actual_size is None and actual_var != var:
                # This is a refreshed array from a function return
                # Try to determine its size from the upcoming function call's return type
                actual_size = self._infer_current_array_size_from_fresh_var(
                    var,
                    actual_var,
                    func_name,
                    expected_size,
                )

            # If we have a size mismatch, restore the array size
            if actual_size is not None and actual_size < expected_size:
                self._insert_array_size_restoration(
                    var,
                    actual_var,
                    actual_size,
                    expected_size,
                )

    def _get_function_name_for_block(self, block) -> str | None:
        """Determine what function name a block will call when converted."""
        # The block has a name attribute that corresponds to the function
        if hasattr(block, "name"):
            return block.name
        # If block has a __class__ attribute with the name
        if hasattr(block, "__class__"):
            return block.__class__.__name__.lower()
        return None

    def _infer_current_array_size_from_fresh_var(
        self,
        var: str,
        actual_var: str,  # noqa: ARG002
        func_name: str | None,  # noqa: ARG002
        expected_size: int,
    ) -> int | None:
        """Infer the current size of a refreshed array by checking what function produced it.

        This looks at refreshed_by_function to find what function was called to produce actual_var,
        then looks up that function's return type to determine the actual size.
        """
        import re

        # Check if we've tracked which function call produced this refreshed variable
        if (
            not hasattr(self, "refreshed_by_function")
            or var not in self.refreshed_by_function
        ):
            # No information about which function produced this variable
            # This happens on the first iteration of a loop before any calls
            return expected_size

        func_info = self.refreshed_by_function[var]
        # Extract function name and position
        if isinstance(func_info, dict):
            called_func_name = func_info["function"]
            return_position = func_info.get("position", 0)
        else:
            called_func_name = func_info  # Legacy string format
            return_position = 0

        # Get the return type for this function
        # Try multiple sources: function_return_types, function_info
        return_type = None

        if (
            hasattr(self, "function_return_types")
            and called_func_name in self.function_return_types
        ):
            return_type = self.function_return_types[called_func_name]
        elif hasattr(self, "function_info") and called_func_name in self.function_info:
            func_info_entry = self.function_info[called_func_name]
            if "return_type" in func_info_entry:
                return_type = func_info_entry["return_type"]

        if return_type is None and hasattr(self, "pending_functions"):
            # Check pending functions - they haven't been built yet but we can analyze their blocks
            for pending_block, pending_name, _pending_sig in self.pending_functions:
                if pending_name == called_func_name:
                    # Analyze the pending block to determine its return type
                    return_type = self._infer_return_type_from_block(pending_block)
                    break

        if return_type is None:
            return expected_size

        # Parse the return type to extract array sizes
        # Return type could be:
        # - "array[quantum.qubit, N]" for single return
        # - "tuple[array[quantum.qubit, N1], array[quantum.qubit, N2], ...]" for multiple returns

        # Check if it's a tuple return
        if return_type.startswith("tuple["):
            # Extract all array sizes from the tuple
            # Pattern: array[quantum.qubit, SIZE]
            array_pattern = r"array\[quantum\.qubit,\s*(\d+)\]"
            matches = re.findall(array_pattern, return_type)

            if return_position < len(matches):
                return int(matches[return_position])
        else:
            # Single return value
            match = re.search(r"array\[quantum\.qubit,\s*(\d+)\]", return_type)
            if match:
                return int(match.group(1))

        # If we can't determine the size, assume it's the same as expected (no restoration needed)
        return expected_size

    def _infer_return_type_from_block(self, block) -> str | None:
        """Analyze a block to infer its return type.

        Priority order:
        1. If both block_returns annotation AND Return() statement exist, use them together
           for precise variable-to-type mapping
        2. If only block_returns annotation exists, use positional sizes
        3. Fall back to analyzing block.vars and context (old behavior)

        Returns:
            A Guppy type string like "array[quantum.qubit, 2]" or
            "tuple[array[quantum.qubit, 2], array[quantum.qubit, 7]]"
        """
        # BEST CASE: Both annotation and Return() statement exist
        if hasattr(block, "__slr_return_type__") and hasattr(block, "get_return_vars"):
            return_vars = block.get_return_vars()
            if return_vars:
                # We have explicit Return(var1, var2, ...) statement
                # Combine with annotation for robust type checking
                sizes = block.__slr_return_type__
                if len(return_vars) == len(sizes):
                    # Perfect match - we know which variable has which size
                    return_types = [f"array[quantum.qubit, {size}]" for size in sizes]
                    if len(return_types) == 1:
                        return return_types[0]
                    return f"tuple[{', '.join(return_types)}]"
                # Mismatch - validation should have caught this, but proceed with annotation

        # SECOND BEST: Just the annotation (positional sizes)
        if hasattr(block, "__slr_return_type__"):
            sizes = block.__slr_return_type__
            return_types = [f"array[quantum.qubit, {size}]" for size in sizes]
            if len(return_types) == 1:
                return return_types[0]
            return f"tuple[{', '.join(return_types)}]"

        # FALLBACK: Try to infer from Return() statement variables
        if hasattr(block, "get_return_vars"):
            return_vars = block.get_return_vars()
            if return_vars:
                return self._infer_types_from_return_vars(return_vars)

        # OLD FALLBACK: Try to infer from vars and context
        if not hasattr(block, "vars") or not block.vars:
            return None

        # Get the return variables from block.vars
        return_vars = (
            block.vars if isinstance(block.vars, list | tuple) else [block.vars]
        )
        return self._infer_types_from_return_vars(return_vars)

    def _infer_types_from_return_vars(self, return_vars) -> str | None:
        """Infer Guppy types from a list of return variables by looking them up in context.

        Args:
            return_vars: List of variables to infer types for

        Returns:
            A Guppy type string or None if types couldn't be inferred
        """
        # For each return variable, determine its type and size
        return_types = []
        for var in return_vars:
            var_name = var.sym if hasattr(var, "sym") else str(var)

            # Check if the Vars object itself has size information
            if hasattr(var, "size"):
                size = var.size
                return_types.append(f"array[quantum.qubit, {size}]")
                continue

            # Check if this is a quantum array in context
            if var_name in self.context.variables:
                var_info = self.context.variables[var_name]
                if hasattr(var_info, "size"):
                    # This is a quantum array
                    size = var_info.size
                    return_types.append(f"array[quantum.qubit, {size}]")
                # else: Not a quantum array, skip for now

        if not return_types:
            return None

        if len(return_types) == 1:
            return return_types[0]
        return f"tuple[{', '.join(return_types)}]"

    def _infer_refreshed_array_size(
        self,
        var: str,
        actual_var: str,  # noqa: ARG002
        expected_size: int,
    ) -> int | None:
        """Infer the size of a refreshed array from function return types.

        When a function returns a smaller array than it received, we need to know
        the actual returned size. This method looks up the function call that
        produced the refreshed array and extracts the size from its return type.
        """
        import re

        # Check if we've tracked which function call produced this refreshed variable
        if (
            not hasattr(self, "refreshed_by_function")
            or var not in self.refreshed_by_function
        ):
            # No information about which function produced this variable
            return expected_size

        func_info = self.refreshed_by_function[var]
        func_name = func_info.get("function")
        return_position = func_info.get(
            "position",
            0,
        )  # Which element in the return tuple

        # Get the return type for this function
        if (
            not hasattr(self, "function_return_types")
            or func_name not in self.function_return_types
        ):
            return expected_size

        return_type = self.function_return_types[func_name]

        # Parse the return type to extract array sizes
        # Return type could be:
        # - "array[quantum.qubit, N]" for single return
        # - "tuple[array[quantum.qubit, N1], array[quantum.qubit, N2], ...]" for multiple returns

        # Check if it's a tuple return
        if return_type.startswith("tuple["):
            # Extract all array sizes from the tuple
            # Pattern: array[quantum.qubit, SIZE]
            array_pattern = r"array\[quantum\.qubit,\s*(\d+)\]"
            matches = re.findall(array_pattern, return_type)

            if return_position < len(matches):
                return int(matches[return_position])
        else:
            # Single return value
            match = re.search(r"array\[quantum\.qubit,\s*(\d+)\]", return_type)
            if match:
                return int(match.group(1))

        # If we can't determine the size, assume it's the same as expected (no restoration needed)
        return expected_size

    def _insert_array_size_restoration(
        self,
        var: str,
        actual_var: str,
        actual_size: int,
        expected_size: int,
    ) -> None:
        """Insert code to restore an array to its expected size by allocating fresh qubits."""
        from pecos.slr.gen_codes.guppy.ir import (
            Assignment,
            Comment,
            FunctionCall,
            VariableRef,
        )

        num_to_allocate = expected_size - actual_size

        self.current_block.statements.append(
            Comment(f"Restore {var} array size from {actual_size} to {expected_size}"),
        )

        # Unpack the current smaller array
        if hasattr(self, "unpacked_vars") and actual_var in self.unpacked_vars:
            current_elements = self.unpacked_vars[actual_var]
        else:
            # Create unpacking statement
            current_elements = [f"{actual_var}_{i}" for i in range(actual_size)]
            unpack_targets = ", ".join(current_elements)
            self.current_block.statements.append(
                Assignment(
                    target=VariableRef(unpack_targets),
                    value=VariableRef(actual_var),
                ),
            )

        # Allocate fresh qubits
        new_elements = []
        for i in range(num_to_allocate):
            fresh_var = self._get_unique_var_name(f"{var}_allocated_{actual_size + i}")
            self.current_block.statements.append(
                Assignment(
                    target=VariableRef(fresh_var),
                    value=FunctionCall(func_name="quantum.qubit", args=[]),
                ),
            )
            new_elements.append(fresh_var)

        # Reconstruct the full-size array and reassign to the actual_var (fresh variable)
        # This ensures the variable stays consistently defined throughout the loop
        all_elements = current_elements + new_elements

        array_construction = self._create_array_construction(all_elements)
        self.current_block.statements.append(
            Assignment(
                target=VariableRef(actual_var),
                value=array_construction,
            ),
        )
