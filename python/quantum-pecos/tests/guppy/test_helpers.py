"""Test helpers for Guppy tests."""

from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def needs_state_vector_desc(func: F) -> F:
    """Decorator to indicate test needs state vector engine for non-Clifford gates."""
    func._needs_state_vector = True
    return func
