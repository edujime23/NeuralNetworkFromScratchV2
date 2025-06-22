# autodiff/tape/registry.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

gradients: dict[str, Callable[..., Any]] = {}


def def_grad(grad_func: Callable[..., Any]) -> Callable[..., Any]:
    r"""
    Decorator to define an analytic gradient function (Jacobian components)
    for a registered primitive.

    The decorated function must implement the reverse-mode local gradient
    calculation for a corresponding primitive.

    It should typically accept `(inputs, **kwargs)` where `inputs` is a tuple
    of Tensors and `kwargs` are the keyword arguments passed to the primitive.
    It must return a list of tuples `[(J_h_i, J_ah_i), ...]` where each `(J_h_i, J_ah_i)`
    is a pair of holomorphic and anti-holomorphic Jacobian components for `inputs[i]`
    with respect to the *output* of the primitive.

    Args:
        grad_func: The gradient function to register.

    Returns:
        The original gradient function, unmodified.
    """
    target_name = grad_func.__name__
    gradients[target_name] = grad_func
    return grad_func
