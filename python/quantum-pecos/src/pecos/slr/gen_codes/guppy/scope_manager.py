"""Enhanced scope management for IR-based code generation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from pecos.slr.gen_codes.guppy.ir import ResourceState, ScopeContext

# Maximum array size fallback when actual size cannot be determined from context
# This is a conservative upper bound to ensure all indices are covered
_MAX_ARRAY_SIZE_FALLBACK = 1000


class ScopeType(Enum):
    """Type of scope."""

    MODULE = "module"
    FUNCTION = "function"
    BLOCK = "block"
    IF_THEN = "if_then"
    IF_ELSE = "if_else"
    LOOP = "loop"


@dataclass
class ResourceUsage:
    """Track resource usage in a scope."""

    qreg_name: str
    indices: set[int]
    is_consumed: bool = False
    is_borrowed: bool = False


@dataclass
class ScopeInfo:
    """Enhanced scope information."""

    scope_type: ScopeType
    context: ScopeContext
    resource_usage: dict[str, ResourceUsage] = field(default_factory=dict)
    borrowed_resources: set[str] = field(default_factory=set)
    returned_resources: set[str] = field(default_factory=set)


class ScopeManager:
    """Manages scope contexts for IR generation."""

    def __init__(self):
        self.scope_stack: list[ScopeInfo] = []
        self.global_context = ScopeContext()

    @property
    def current_scope(self) -> ScopeInfo | None:
        """Get the current scope."""
        return self.scope_stack[-1] if self.scope_stack else None

    @property
    def current_context(self) -> ScopeContext:
        """Get the current scope context."""
        if self.current_scope:
            return self.current_scope.context
        return self.global_context

    @contextmanager
    def enter_scope(self, scope_type: ScopeType):
        """Enter a new scope."""
        parent_context = self.current_context
        new_context = ScopeContext(parent=parent_context)
        new_scope = ScopeInfo(scope_type=scope_type, context=new_context)

        self.scope_stack.append(new_scope)
        try:
            yield new_scope
        finally:
            # Analyze resource flow when exiting scope
            self._analyze_scope_exit(new_scope)
            self.scope_stack.pop()

    def _analyze_scope_exit(self, scope: ScopeInfo) -> None:
        """Analyze resource usage when exiting a scope."""
        # For conditional scopes, propagate resource usage to parent
        if (
            scope.scope_type in [ScopeType.IF_THEN, ScopeType.IF_ELSE]
            and self.current_scope
        ):
            for res_name, usage in scope.resource_usage.items():
                if res_name not in self.current_scope.resource_usage:
                    self.current_scope.resource_usage[res_name] = ResourceUsage(
                        qreg_name=usage.qreg_name,
                        indices=set(),
                    )
                # Merge usage
                parent_usage = self.current_scope.resource_usage[res_name]
                parent_usage.indices.update(usage.indices)
                if usage.is_consumed:
                    parent_usage.is_consumed = True

    def track_resource_usage(
        self,
        qreg_name: str,
        indices: set[int],
        *,
        consumed: bool = False,
    ) -> None:
        """Track usage of a quantum resource in current scope."""
        if not self.current_scope:
            return

        if qreg_name not in self.current_scope.resource_usage:
            self.current_scope.resource_usage[qreg_name] = ResourceUsage(
                qreg_name=qreg_name,
                indices=set(),
            )

        usage = self.current_scope.resource_usage[qreg_name]
        usage.indices.update(indices)
        if consumed:
            usage.is_consumed = True

    def mark_resource_borrowed(self, qreg_name: str) -> None:
        """Mark a resource as borrowed in current scope."""
        if self.current_scope:
            self.current_scope.borrowed_resources.add(qreg_name)
            if qreg_name in self.current_scope.resource_usage:
                self.current_scope.resource_usage[qreg_name].is_borrowed = True

    def is_in_loop(self) -> bool:
        """Check if currently inside a loop scope."""
        return any(scope.scope_type == ScopeType.LOOP for scope in self.scope_stack)

    def is_in_conditional_within_loop(self) -> bool:
        """Check if currently inside a conditional (if) within a loop."""
        in_loop = False
        in_conditional = False

        for scope in self.scope_stack:
            if scope.scope_type == ScopeType.LOOP:
                in_loop = True
            elif scope.scope_type in (ScopeType.IF_THEN, ScopeType.IF_ELSE) and in_loop:
                in_conditional = True

        return in_loop and in_conditional

    def mark_resource_returned(self, qreg_name: str) -> None:
        """Mark a resource as returned from current scope."""
        if self.current_scope:
            self.current_scope.returned_resources.add(qreg_name)

    def get_unconsumed_resources(self) -> dict[str, set[int]]:
        """Get all unconsumed quantum resources in current scope."""
        unconsumed = {}

        # Look through all variables in current context
        context = self.current_context
        for var_name, var_info in context.variables.items():
            if (
                (
                    var_info.var_type == "quantum"
                    and var_info.state != ResourceState.CONSUMED
                )
                and var_info.is_array
                and var_info.size
            ):
                # Check which indices are consumed
                consumed_indices = set()
                if self.current_scope and var_name in self.current_scope.resource_usage:
                    usage = self.current_scope.resource_usage[var_name]
                    consumed_indices = (
                        set(range(var_info.size))
                        if usage.is_consumed
                        else usage.indices
                    )

                # Find unconsumed indices
                all_indices = set(range(var_info.size))
                unconsumed_indices = all_indices - consumed_indices

                if unconsumed_indices:
                    unconsumed[var_name] = unconsumed_indices

        return unconsumed

    def analyze_conditional_branches(
        self,
        then_scope: ScopeInfo,
        else_scope: ScopeInfo | None,
        context: ScopeContext | None = None,
    ) -> dict[str, set[int]]:
        """Analyze resource consumption across conditional branches."""
        # Get resources consumed in then branch
        then_consumed = {}
        for res_name, usage in then_scope.resource_usage.items():
            if usage.is_consumed:
                # Get actual array size from context if available
                if context:
                    var_info = context.lookup_variable(res_name)
                    if var_info and var_info.size:
                        then_consumed[res_name] = set(range(var_info.size))
                    else:
                        then_consumed[res_name] = set(range(_MAX_ARRAY_SIZE_FALLBACK))
                else:
                    then_consumed[res_name] = set(range(_MAX_ARRAY_SIZE_FALLBACK))
            elif usage.indices:
                then_consumed[res_name] = usage.indices

        # Get resources consumed in else branch (if exists)
        else_consumed = {}
        if else_scope:
            for res_name, usage in else_scope.resource_usage.items():
                if usage.is_consumed:
                    # Get actual array size from context if available
                    if context:
                        var_info = context.lookup_variable(res_name)
                        if var_info and var_info.size:
                            else_consumed[res_name] = set(range(var_info.size))
                        else:
                            else_consumed[res_name] = set(
                                range(_MAX_ARRAY_SIZE_FALLBACK),
                            )
                    else:
                        else_consumed[res_name] = set(range(_MAX_ARRAY_SIZE_FALLBACK))
                elif usage.indices:
                    else_consumed[res_name] = usage.indices

        # Find resources that need to be balanced
        all_resources = set(then_consumed.keys()) | set(else_consumed.keys())
        unbalanced = {}

        for res_name in all_resources:
            then_indices = then_consumed.get(res_name, set())
            else_indices = else_consumed.get(res_name, set())

            if then_indices != else_indices:
                # Find indices consumed in one branch but not the other
                else_indices - then_indices
                missing_in_else = then_indices - else_indices

                # For now, we track indices missing in else branch
                # (consumed in then but not else)
                if missing_in_else:
                    unbalanced[res_name] = missing_in_else

        return unbalanced
