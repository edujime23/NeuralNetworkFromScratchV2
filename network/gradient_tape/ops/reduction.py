import numpy as np

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor
from .util import _broadcast_reduction_result


@registry.register("sum")
def _sum_grad(
    upstream: Gradient, result: Tensor, a: Tensor, *args, **kwargs
) -> list[Gradient]:
    grad_h = upstream.h * np.ones_like(a.data)
    grad_ah = upstream.ah * np.ones_like(a.data)
    return [Gradient(h=grad_h, ah=grad_ah)]


@registry.register("mean")
def _mean_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    n = a.size
    grad_h = upstream.h * np.ones_like(a) / n

    grad_ah = upstream.ah * np.ones_like(np.conj(a)) / n

    return [Gradient(h=grad_h, ah=grad_ah)]


@registry.register("prod")
def _prod_grad(
    upstream: Gradient, result: Tensor, a: Tensor, *args, **kwargs
) -> list[Gradient]:
    data = a.data
    zero_mask = data == 0
    safe_data = np.where(zero_mask, 1, data)
    basic = result.data / safe_data
    if np.any(zero_mask):
        if zero_mask.sum() == 1:
            basic = np.where(zero_mask, np.prod(data[~zero_mask]), 0)
        else:
            basic = np.zeros_like(data)
    grad_h = upstream.h * basic
    grad_ah = upstream.ah * basic
    return [Gradient(h=grad_h, ah=grad_ah)]


@registry.register("max")
@registry.register("maximum")
def _max_grad(
    upstream: Gradient, result: Tensor, a: Tensor, *args, **kwargs
) -> list[Gradient]:
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    data = a.data
    full_result = _broadcast_reduction_result(data, result.data, axis, keepdims)
    mask = data == full_result
    num_max = np.sum(mask, axis=axis, keepdims=keepdims)
    divisor = _broadcast_reduction_result(data, num_max, axis, keepdims)
    grad_mask = mask / divisor

    grad_h = upstream.h * grad_mask
    grad_ah = upstream.ah * grad_mask
    return [Gradient(h=grad_h, ah=grad_ah)]


@registry.register("min")
@registry.register("minimum")
def _min_grad(
    upstream: Gradient, result: Tensor, a: Tensor, *args, **kwargs
) -> list[Gradient]:
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    data = a.data
    full_result = _broadcast_reduction_result(data, result.data, axis, keepdims)
    mask = data == full_result
    num_min = np.sum(mask, axis=axis, keepdims=keepdims)
    divisor = _broadcast_reduction_result(data, num_min, axis, keepdims)
    grad_mask = mask / divisor

    grad_h = upstream.h * grad_mask
    grad_ah = upstream.ah * grad_mask
    return [Gradient(h=grad_h, ah=grad_ah)]
