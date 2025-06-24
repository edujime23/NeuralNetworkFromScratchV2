from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from network.types.tensor import Tensor


# Internal-facing types. Users defining gradients won't need these.
@dataclass
class OpNode:
    """Internal: Represents a recorded operation in the computation graph."""

    op_name: str
    inputs: tuple[Tensor, ...]
    kwargs: dict[str, Any]
    result: Tensor
    parents: list[OpNode] = field(default_factory=list)


@dataclass
class Gradient:
    """Internal: Stores holomorphic and anti-holomorphic gradient components."""

    h: Tensor  # Holomorphic ∂f/∂z
    ah: Tensor  # Anti-holomorphic ∂f/∂∂z̄

    @property
    def total(self):
        """
        Returns the sum of Gradient.h and Gradient.ah.
        Used when var is real as summing them returns the classic gradient.
        For optimizing using 2 * Gradient.ah also works.

        Returns:
            Tensor: The sum of Gradient.h and Gradient.ah.
        """
        return self.h + self.ah
