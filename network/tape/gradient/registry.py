# autodiff/tape/registry.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Assuming GRADIENTS is a global dictionary defined elsewhere,
# or imported from a central 'funcs' module if it holds primitive functions.
# For now, let's assume it's defined here and imported by funcs, or is
# a global registry for all gradients.
# For simplicity, we'll keep it here as a central point for registering gradients.
GRADIENTS: dict[str, Callable[..., Any]] = {}
_PRIMITIVE_REGISTRY: dict[str, None] = {}


def primitive(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to register a function as a differentiable primitive operation.

    Functions decorated with `@primitive` are recognized by the
    tape as atomic operations for which gradients can be defined. The function's
    `__name__` is used as the key for registration.

    Args:
        func: The function to register as a primitive.

    Returns:
        The original function, unmodified.
    """
    name = func.__name__
    _PRIMITIVE_REGISTRY[name] = None  # Value doesn't matter, just key presence
    return func


def def_grad(grad_func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to define an analytic gradient function for a registered primitive.

    The decorated function must implement the reverse-mode gradient calculation
    for a corresponding primitive. Its name *must* be in the format
    `<primitive_func_name>_grad`.

    Args:
        grad_func: The gradient function to register. It should typically
                   accept `(upstream_gradients, inputs, **kwargs)` and
                   return a tuple of gradients for each input.

    Returns:
        The original gradient function, unmodified.

    Raises:
        ValueError: If the corresponding primitive function has not been
                    registered using `@primitive` first.
    """
    target_name = grad_func.__name__.replace("_grad", "")
    if target_name not in _PRIMITIVE_REGISTRY:
        raise ValueError(
            f"Primitive '{target_name}' not registered before defining its gradient. "
            "Ensure you decorate the primitive function with `@primitive` first."
        )
    GRADIENTS[target_name] = grad_func
    return grad_func
