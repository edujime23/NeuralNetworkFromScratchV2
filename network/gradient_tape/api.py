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

    def watch(self, *tensors: list[Tensor | Variable]):
        """Explicitly tracks gradients for the given Tensors or Variables."""
        if isinstance(tensors, (Tensor, Variable)):
            tensors = [tensors]
        for t in tensors:
            tensor = t.value if isinstance(t, Variable) else t
            super()._watch(tensor)

    def gradient(
        self,
        target: Tensor,
        sources: list[Tensor | Variable],  # Allow both Tensor and Variable
        output_gradients: Gradient | None = None,
    ) -> list[Gradient | None]:
        """
        Computes the gradient of 'target' with respect to 'sources'.
        Sources can be Tensors or Variables.
        """
        self._is_used = True

        if isinstance(sources, (Tensor, Variable)):
            sources = [sources]

        self._backpropagate(target, output_gradients)

        results: list[Gradient | None] = []
        for s in sources:
            source_tensor = s.value if isinstance(s, Variable) else s
            if grad_pair := self._grads.get(id(source_tensor)):
                results.append(grad_pair)
            else:
                results.append(None)

        # Clear state only if not persistent and not inside a `with` block
        if not self.persistent and not tapes:
            self._clear_state()
        return results
