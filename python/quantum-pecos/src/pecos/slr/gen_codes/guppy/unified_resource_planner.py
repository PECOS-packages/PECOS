"""Unified resource planning framework for Guppy code generation.

This module provides a holistic approach to resource management by combining:
1. Array unpacking decisions (rule-based from unpacking_rules.py)
2. Local allocation analysis (computed directly from usage patterns)
3. Data flow analysis (precise element-level tracking from data_flow.py)

The unified planner makes coordinated decisions about BOTH unpacking and allocation,
eliminating conflicts and enabling cross-cutting optimizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr import Block as SLRBlock
    from pecos.slr.gen_codes.guppy.data_flow import DataFlowAnalysis
    from pecos.slr.gen_codes.guppy.ir_analyzer import ArrayAccessInfo, UnpackingPlan


class ResourceStrategy(Enum):
    """Unified strategy for how to manage a quantum/classical register.

    This combines both unpacking and allocation decisions into a coherent plan.
    """

    # Keep as packed array, pre-allocate all elements
    PACKED_PREALLOCATED = auto()

    # Keep as packed array, allocate elements dynamically as needed
    PACKED_DYNAMIC = auto()

    # Unpack into individual variables, pre-allocate all
    UNPACKED_PREALLOCATED = auto()

    # Unpack into individual variables, allocate some locally
    UNPACKED_MIXED = auto()

    # Unpack completely, all elements allocated locally when first used
    UNPACKED_LOCAL = auto()


class DecisionPriority(Enum):
    """Priority level for resource planning decisions."""

    REQUIRED = auto()  # Semantic necessity (would fail otherwise)
    RECOMMENDED = auto()  # Strong evidence for this approach
    OPTIONAL = auto()  # Minor benefit
    DISCOURAGED = auto()  # Minor drawback
    FORBIDDEN = auto()  # Would cause errors


@dataclass
class ResourcePlan:
    """Unified plan for managing a single register (array or unpacked).

    This combines unpacking and allocation decisions into a coherent strategy.
    """

    array_name: str
    size: int
    is_classical: bool

    # Unified strategy
    strategy: ResourceStrategy

    # Fine-grained control
    elements_to_unpack: set[int] = field(default_factory=set)  # Which to unpack
    elements_to_allocate_locally: set[int] = field(
        default_factory=set,
    )  # Which to allocate locally
    elements_requiring_replacement: set[int] = field(
        default_factory=set,
    )  # Which need Prep after Measure

    # Decision reasoning
    priority: DecisionPriority = DecisionPriority.OPTIONAL
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, any] = field(default_factory=dict)

    @property
    def needs_unpacking(self) -> bool:
        """Check if this register needs to be unpacked."""
        return self.strategy in (
            ResourceStrategy.UNPACKED_PREALLOCATED,
            ResourceStrategy.UNPACKED_MIXED,
            ResourceStrategy.UNPACKED_LOCAL,
        )

    @property
    def uses_dynamic_allocation(self) -> bool:
        """Check if this register uses any dynamic allocation."""
        return self.strategy in (
            ResourceStrategy.PACKED_DYNAMIC,
            ResourceStrategy.UNPACKED_MIXED,
            ResourceStrategy.UNPACKED_LOCAL,
        )

    def get_explanation(self) -> str:
        """Get human-readable explanation of the plan."""
        lines = [
            f"Resource Plan for '{self.array_name}' (size={self.size}, "
            f"{'classical' if self.is_classical else 'quantum'}):",
            f"  Strategy: {self.strategy.name}",
            f"  Priority: {self.priority.name}",
        ]

        if self.elements_to_unpack:
            lines.append(f"  Elements to unpack: {sorted(self.elements_to_unpack)}")

        if self.elements_to_allocate_locally:
            lines.append(
                f"  Local allocation: {sorted(self.elements_to_allocate_locally)}",
            )

        if self.elements_requiring_replacement:
            lines.append(
                f"  Need replacement: {sorted(self.elements_requiring_replacement)}",
            )

        if self.reasons:
            lines.append("  Reasons:")
            lines.extend(f"    - {reason}" for reason in self.reasons)

        return "\n".join(lines)


@dataclass
class UnifiedResourceAnalysis:
    """Complete resource analysis for a block.

    Contains coordinated plans for all registers.
    """

    plans: dict[str, ResourcePlan] = field(default_factory=dict)
    global_recommendations: list[str] = field(default_factory=list)
    _original_unpacking_plan: UnpackingPlan | None = field(default=None, repr=False)

    def get_plan(self, array_name: str) -> ResourcePlan | None:
        """Get the resource plan for a specific array."""
        return self.plans.get(array_name)

    def set_original_unpacking_plan(self, plan: UnpackingPlan) -> None:
        """Store the original UnpackingPlan from IRAnalyzer for backward compatibility."""
        self._original_unpacking_plan = plan

    def get_report(self) -> str:
        """Generate comprehensive resource planning report."""
        lines = [
            "=" * 70,
            "UNIFIED RESOURCE PLANNING REPORT",
            "=" * 70,
            "",
        ]

        if self.global_recommendations:
            lines.append("Global Recommendations:")
            lines.extend(f"  - {rec}" for rec in self.global_recommendations)
            lines.append("")

        for array_name, plan in sorted(self.plans.items()):
            lines.append(plan.get_explanation())
            lines.append("")

        lines.extend(
            [
                "=" * 70,
                f"Total registers analyzed: {len(self.plans)}",
                f"Unpacking recommended: {sum(1 for p in self.plans.values() if p.needs_unpacking)}",
                f"Dynamic allocation: {sum(1 for p in self.plans.values() if p.uses_dynamic_allocation)}",
                "=" * 70,
            ],
        )

        return "\n".join(lines)

    def get_unpacking_plan(self) -> UnpackingPlan:
        """Get the UnpackingPlan from IRAnalyzer.

        The UnifiedResourcePlanner internally runs IRAnalyzer, so we always
        have the original unpacking plan available.

        Returns:
            UnpackingPlan from IRAnalyzer with all detailed state preserved
        """
        # We always have the original plan because UnifiedResourcePlanner
        # runs IRAnalyzer internally during analyze()
        if self._original_unpacking_plan is None:
            msg = "get_unpacking_plan() called but no original plan available"
            raise RuntimeError(msg)

        return self._original_unpacking_plan


class UnifiedResourcePlanner:
    """Unified planner that coordinates unpacking and allocation decisions.

    This planner integrates:
    1. Data flow analysis (precise element-level tracking)
    2. Unpacking rules (semantic requirements from usage patterns)
    3. Local allocation analysis (computed from consumption & reuse patterns)

    The result is a coordinated ResourcePlan for each register that makes
    coherent decisions about both unpacking and allocation.
    """

    def __init__(self):
        self.analysis: UnifiedResourceAnalysis | None = None
        self.original_unpacking_plan: UnpackingPlan | None = None

    def analyze(
        self,
        block: SLRBlock,
        variable_context: dict[str, any],
        *,
        array_access_info: dict[str, ArrayAccessInfo] | None = None,
        data_flow_analysis: DataFlowAnalysis | None = None,
    ) -> UnifiedResourceAnalysis:
        """Perform unified resource planning for a block.

        Args:
            block: The SLR block to analyze
            variable_context: Context of variables in the block
            array_access_info: Optional pre-computed array access info from IRAnalyzer
            data_flow_analysis: Optional pre-computed data flow analysis

        Returns:
            UnifiedResourceAnalysis with coordinated plans for all registers
        """
        self.analysis = UnifiedResourceAnalysis()

        # If we don't have the required analyses, compute them now
        if array_access_info is None:
            from pecos.slr.gen_codes.guppy.ir_analyzer import IRAnalyzer

            analyzer = IRAnalyzer()
            plan = analyzer.analyze_block(block, variable_context)
            array_access_info = plan.all_analyzed_arrays
            # Store the original unpacking plan
            self.original_unpacking_plan = plan

        if data_flow_analysis is None:
            from pecos.slr.gen_codes.guppy.data_flow import DataFlowAnalyzer

            dfa = DataFlowAnalyzer()
            data_flow_analysis = dfa.analyze(block, variable_context)

        # Now perform unified planning for each array
        for array_name, access_info in array_access_info.items():
            plan = self._create_unified_plan(
                array_name,
                access_info,
                data_flow_analysis,
            )
            self.analysis.plans[array_name] = plan

        # Add global recommendations
        self._add_global_recommendations()

        # Store the original unpacking plan in the analysis for get_unpacking_plan()
        if self.original_unpacking_plan:
            self.analysis.set_original_unpacking_plan(self.original_unpacking_plan)

        return self.analysis

    def _create_unified_plan(
        self,
        array_name: str,
        access_info: ArrayAccessInfo,
        data_flow: DataFlowAnalysis,
    ) -> ResourcePlan:
        """Create a unified resource plan for a single array.

        This is the core decision logic that coordinates unpacking and allocation.
        """
        plan = ResourcePlan(
            array_name=array_name,
            size=access_info.size,
            is_classical=access_info.is_classical,
            strategy=ResourceStrategy.PACKED_PREALLOCATED,  # Default
        )

        # Collect evidence from different analyses
        self._collect_evidence(plan, access_info, data_flow)

        # Determine which elements can be allocated locally
        self._determine_local_allocation(plan, access_info, data_flow)

        # Make coordinated decision based on all evidence
        self._decide_strategy(plan, access_info, data_flow)

        return plan

    def _collect_evidence(
        self,
        plan: ResourcePlan,
        access_info: ArrayAccessInfo,
        data_flow: DataFlowAnalysis,
    ) -> None:
        """Collect evidence from all analyses."""
        evidence = plan.evidence

        # Evidence from array access patterns (counts for decisions)
        evidence["has_individual_access"] = access_info.has_individual_access
        evidence["all_elements_accessed"] = access_info.all_elements_accessed
        evidence["has_full_array_access"] = bool(access_info.full_array_accesses)
        evidence["elements_accessed"] = len(access_info.element_accesses)
        evidence["elements_consumed"] = len(access_info.elements_consumed)
        evidence["has_operations_between"] = access_info.has_operations_between
        evidence["has_conditionals"] = access_info.has_conditionals_between

        # Copy element-level information for get_unpacking_plan()
        evidence["element_accesses"] = access_info.element_accesses
        evidence["elements_consumed_set"] = access_info.elements_consumed

        # Evidence from data flow analysis (element-level precision)
        for (arr_name, idx), flow_info in data_flow.element_flows.items():
            if arr_name == plan.array_name and flow_info.has_use_after_consumption():
                plan.elements_requiring_replacement.add(idx)

        # Evidence from conditional tracking (element-level)
        conditionally_accessed = set()
        for arr_name, idx in data_flow.conditional_accesses:
            if arr_name == plan.array_name:
                conditionally_accessed.add(idx)
        evidence["conditionally_accessed_elements"] = conditionally_accessed

    def _determine_local_allocation(
        self,
        plan: ResourcePlan,
        access_info: ArrayAccessInfo,
        _data_flow: DataFlowAnalysis,
    ) -> None:
        """Determine which elements can be allocated locally.

        Elements can be allocated locally if they are:
        - Quantum qubits (classical arrays don't use local allocation)
        - Consumed (measured) and not reused
        - Not in conditional scopes or loops (single-scope usage)
        """
        if plan.is_classical:
            return  # Classical arrays don't use local allocation

        # Find elements that are consumed and not reused
        for idx in access_info.elements_consumed:
            # Check if this element is reused after consumption
            if idx in plan.elements_requiring_replacement:
                continue  # This element is reused, can't allocate locally

            # Check if used in conditionals (prevents local allocation)
            if idx in access_info.conditionally_accessed_elements:
                continue  # Conditional usage prevents local allocation

            # This element is a good candidate for local allocation
            plan.elements_to_allocate_locally.add(idx)

    def _decide_strategy(
        self,
        plan: ResourcePlan,
        access_info: ArrayAccessInfo,
        _data_flow: DataFlowAnalysis,
    ) -> None:
        """Make unified strategy decision based on collected evidence.

        Decision tree (in priority order):
        1. Check for REQUIRED unpacking (semantic necessity)
        2. Check for FORBIDDEN unpacking (would cause errors)
        3. Check for allocation optimization opportunities
        4. Make quality-based decisions

        Note: Local allocation candidates are already determined in
        _determine_local_allocation() and stored in plan.elements_to_allocate_locally
        """
        ev = plan.evidence

        # Rule 1: Full array operations FORBID unpacking
        if ev["has_full_array_access"]:
            plan.strategy = ResourceStrategy.PACKED_PREALLOCATED
            plan.priority = DecisionPriority.FORBIDDEN
            plan.reasons.append(
                "Full array operations require packed representation",
            )
            # Clear local allocation - packed arrays don't use it
            plan.elements_to_allocate_locally.clear()
            return

        # Rule 2: No individual access = no unpacking needed
        if not ev["has_individual_access"]:
            # Check if allocation optimizer suggests dynamic allocation
            if plan.elements_to_allocate_locally:
                plan.strategy = ResourceStrategy.PACKED_DYNAMIC
                plan.priority = DecisionPriority.RECOMMENDED
                plan.reasons.append("Dynamic allocation recommended by optimizer")
            else:
                plan.strategy = ResourceStrategy.PACKED_PREALLOCATED
                plan.priority = DecisionPriority.OPTIONAL
                plan.reasons.append("No individual element access detected")
            # Clear local allocation - packed arrays don't use it
            plan.elements_to_allocate_locally.clear()
            return

        # Rule 3: Quantum arrays with operations after measurement REQUIRE unpacking
        if not plan.is_classical and ev["has_operations_between"]:
            # Check if we can use local allocation
            if plan.elements_to_allocate_locally:
                plan.strategy = ResourceStrategy.UNPACKED_MIXED
                plan.elements_to_unpack = set(range(plan.size))
                # Local elements already determined in _determine_local_allocation()
                plan.priority = DecisionPriority.REQUIRED
                plan.reasons.append(
                    "Operations after measurement require unpacking (with local allocation)",
                )
            else:
                plan.strategy = ResourceStrategy.UNPACKED_PREALLOCATED
                plan.elements_to_unpack = set(range(plan.size))
                plan.priority = DecisionPriority.REQUIRED
                plan.reasons.append(
                    "Operations after measurement require unpacking",
                )
            return

        # Rule 4: Individual quantum measurements REQUIRE unpacking
        if not plan.is_classical and ev["elements_consumed"] > 0:
            # Determine unpacking strategy based on allocation
            if plan.elements_to_allocate_locally:
                # Some elements can be allocated locally
                plan.strategy = ResourceStrategy.UNPACKED_MIXED
                plan.elements_to_unpack = set(range(plan.size))
                # Local elements already determined in _determine_local_allocation()
                plan.priority = DecisionPriority.REQUIRED
                plan.reasons.append(
                    f"Individual quantum measurements require unpacking "
                    f"({len(plan.elements_to_allocate_locally)} elements local)",
                )
            else:
                plan.strategy = ResourceStrategy.UNPACKED_PREALLOCATED
                plan.elements_to_unpack = set(range(plan.size))
                plan.priority = DecisionPriority.REQUIRED
                plan.reasons.append(
                    "Individual quantum measurements require unpacking",
                )
            return

        # Rule 5: Conditional element access REQUIRES unpacking
        conditional_elements = ev.get("conditionally_accessed_elements", set())
        if conditional_elements:
            # Only unpack elements that are actually accessed (not just in conditionals)
            elements_needing_unpack = (
                conditional_elements & access_info.element_accesses
            )

            if elements_needing_unpack:
                # Check allocation strategy
                if plan.elements_to_allocate_locally:
                    plan.strategy = ResourceStrategy.UNPACKED_MIXED
                    # Local elements already determined in _determine_local_allocation()
                else:
                    plan.strategy = ResourceStrategy.UNPACKED_PREALLOCATED

                plan.elements_to_unpack = set(range(plan.size))
                plan.priority = DecisionPriority.REQUIRED
                plan.reasons.append(
                    f"Conditional access to elements {sorted(elements_needing_unpack)} requires unpacking",
                )
                return

        # Rule 6: Single element access - prefer direct indexing
        if ev["elements_accessed"] == 1:
            plan.strategy = ResourceStrategy.PACKED_PREALLOCATED
            plan.priority = DecisionPriority.RECOMMENDED
            plan.reasons.append(
                "Single element access - direct indexing preferred",
            )
            return

        # Rule 7: Classical arrays with multiple accesses benefit from unpacking
        if plan.is_classical and ev["elements_accessed"] > 1:
            plan.strategy = ResourceStrategy.UNPACKED_PREALLOCATED
            plan.elements_to_unpack = set(range(plan.size))
            plan.priority = DecisionPriority.RECOMMENDED
            plan.reasons.append(
                f"Classical array with {ev['elements_accessed']} accesses - unpacking improves readability",
            )
            return

        # Rule 9: Partial array usage
        if ev["elements_accessed"] > 0 and not ev["all_elements_accessed"]:
            access_ratio = ev["elements_accessed"] / plan.size
            if access_ratio > 0.5:
                plan.strategy = ResourceStrategy.UNPACKED_PREALLOCATED
                plan.elements_to_unpack = set(range(plan.size))
                plan.priority = DecisionPriority.OPTIONAL
                plan.reasons.append(
                    f"Partial array usage ({access_ratio:.0%}) - unpacking for clarity",
                )
                return

            # Low access ratio - keep as array
            plan.strategy = ResourceStrategy.PACKED_PREALLOCATED
            plan.priority = DecisionPriority.OPTIONAL
            plan.reasons.append(
                f"Low access ratio ({access_ratio:.0%}) - keeping as array",
            )
            return

        # Default: Keep as packed, pre-allocated (simplest approach)
        plan.strategy = ResourceStrategy.PACKED_PREALLOCATED
        plan.priority = DecisionPriority.OPTIONAL
        plan.reasons.append("Default strategy - no strong evidence for alternatives")

    def _add_global_recommendations(self) -> None:
        """Add global recommendations based on overall analysis."""
        if not self.analysis:
            return

        # Count strategies
        strategy_counts = {}
        for plan in self.analysis.plans.values():
            strategy = plan.strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Recommend patterns
        total = len(self.analysis.plans)
        if total == 0:
            return

        unpacked_count = sum(
            1 for p in self.analysis.plans.values() if p.needs_unpacking
        )
        dynamic_count = sum(
            1 for p in self.analysis.plans.values() if p.uses_dynamic_allocation
        )

        if unpacked_count > total * 0.7:
            self.analysis.global_recommendations.append(
                f"High unpacking ratio ({unpacked_count}/{total}) - "
                "consider if element-level APIs would be more natural",
            )

        if dynamic_count > 0:
            self.analysis.global_recommendations.append(
                f"Dynamic allocation used for {dynamic_count}/{total} registers - "
                "ensure proper lifetime management",
            )

        # Check for potential conflicts
        required_plans = [
            p
            for p in self.analysis.plans.values()
            if p.priority == DecisionPriority.REQUIRED
        ]
        if len(required_plans) == total and total > 1:
            self.analysis.global_recommendations.append(
                "All registers require unpacking - this may indicate complex control flow",
            )
