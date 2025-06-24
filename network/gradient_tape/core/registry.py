from __future__ import annotations

from functools import wraps
from typing import Protocol, runtime_checkable

from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


@runtime_checkable
class GradientFunction(Protocol):
    def __call__(
        self, grad: Gradient, tensor: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]: ...


class GradientRegistry:
    """Manages the mapping of operation names to their gradient functions."""

    def __init__(self):
        self._registry: dict[str, GradientFunction] = {}

    def register(self, op_name: str) -> GradientFunction:
        """
        Returns a decorator to register a gradient function for an operation.

        The decorated function MUST have the following signature:
        `def grad_func(upstream: Gradient, result: Tensor, *inputs, **kwargs) -> list[Gradient | None]:`

        - `upstream`: A `Gradient` object containing the upstream gradients (`.h` and `.ah`).
        - `result`: The `Tensor` produced by the forward operation.
        - `*inputs`, `**kwargs`: The original arguments to the forward operation.

        It MUST return a list containing one `Gradient` object (or `None`) for each input.
        """

        def decorator(
            grad_func: GradientFunction,
        ) -> GradientFunction:
            @wraps(grad_func)
            def wrapper(upstream: Gradient, result: Tensor, *args, **kwargs):
                return grad_func(upstream, result, *args, **kwargs)

            self._registry[op_name] = grad_func
            return wrapper

        return decorator

    def get(self, op_name: str) -> GradientFunction | None:
        return self._registry.get(op_name)


registry = GradientRegistry()
