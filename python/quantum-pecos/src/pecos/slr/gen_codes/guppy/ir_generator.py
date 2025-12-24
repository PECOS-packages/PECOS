"""IR-based Guppy generator that uses two-pass architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.gen_codes.generator import Generator
from pecos.slr.gen_codes.guppy.dependency_analyzer import DependencyAnalyzer
from pecos.slr.gen_codes.guppy.ir import ScopeContext
from pecos.slr.gen_codes.guppy.ir_builder import IRBuilder
from pecos.slr.gen_codes.guppy.ir_postprocessor import IRPostProcessor
from pecos.slr.gen_codes.guppy.unified_resource_planner import UnifiedResourcePlanner

if TYPE_CHECKING:
    from pecos.slr import Block


class IRGuppyGenerator(Generator):
    """Generator that uses IR for two-pass Guppy code generation."""

    def __init__(self):
        """Initialize the IR-based generator."""
        self.output = []
        self.variable_context = {}
        self.dependency_analyzer = DependencyAnalyzer()

    def generate_block(self, block: Block) -> None:
        """Generate Guppy code for a block using IR."""
        # Build variable context from block
        self._build_variable_context(block)

        # First pass: Analyze the block with unified resource planning
        # This coordinates unpacking decisions with allocation strategies
        planner = UnifiedResourcePlanner()
        unified_analysis = planner.analyze(block, self.variable_context)

        # Convert unified analysis to UnpackingPlan
        # The unified planner internally runs IRAnalyzer, so we don't need to run it again
        unpacking_plan = unified_analysis.get_unpacking_plan()

        # Second pass: Build IR with both unpacking plan and unified analysis
        builder = IRBuilder(
            unpacking_plan,
            unified_analysis=unified_analysis,
            include_optimization_report=True,
        )
        module = builder.build_module(block, [])  # No pending functions for now

        # Post-processing pass: Fix array accesses after unpacking
        context = ScopeContext()
        postprocessor = IRPostProcessor()
        postprocessor.process_module(module, context)

        # Third pass: Render to Guppy code
        lines = module.render(context)

        self.output = lines

    def _build_variable_context(self, block: Block) -> None:
        """Build variable context from block declarations."""
        if hasattr(block, "vars"):
            for var in block.vars:
                if hasattr(var, "sym"):
                    self.variable_context[var.sym] = var

    def get_output(self) -> str:
        """Get the generated Guppy code."""
        return "\n".join(self.output)
