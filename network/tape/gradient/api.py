from __future__ import annotations

from network.queues.tapes import tapes
from network.types.tensor import Tensor
from network.types.variable import Variable

from .core.tape import GradientTapeCore
from .types import Gradient


class GradientTape(GradientTapeCore):
    """A high-level interface for complex-aware automatic differentiation."""

    def __enter__(self) -> GradientTape:
        tapes.append(self)
        return self

    def __exit__(self, *args):
        tapes.pop()

    def watch(self, *tensors: Tensor | Variable):
        """Explicitly tracks gradients for the given Tensors or Variables."""
        for t in tensors:
            tensor = t.value if isinstance(t, Variable) else t
            super()._watch(tensor)

    def gradient(
        self,
        target: Tensor,
        sources: list[Tensor | Variable],
        output_gradients: Gradient | None = None,
    ) -> list[Gradient | None]:
        """
        Computes the gradient of 'target' with respect to 'sources'.
        Sources can be Tensors or Variables.
        """
        self._is_used = True

        if isinstance(sources, (Tensor, Variable)):
            sources = [sources]

        # Convert Variables to Tensors
        source_tensors = [s.value if isinstance(s, Variable) else s for s in sources]

        # Call the parent's gradient method which handles real variables properly
        gradients = super().gradient(target, source_tensors)

        # Clear state only if not persistent and not inside a `with` block
        if not self.persistent and not tapes:
            self._clear_state()

        return gradients
