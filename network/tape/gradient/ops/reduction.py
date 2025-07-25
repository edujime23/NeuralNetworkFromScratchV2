import numpy as np

from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor
from .util import _broadcast_reduction_result


@registry.register("sum")
def _sum_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> list[Gradient]:
    if isinstance(axis, int):
        axis = (axis,)

    if not keepdims and axis is not None:
        shape = list(a.shape)
        for ax in axis:
            shape[ax] = 1
        upstream_h = upstream.h.reshape(shape)
        upstream_ah = upstream.ah.reshape(shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    ones = np.ones_like(a)
    grad_h = upstream_h * ones
    grad_ah = upstream_ah * ones
    return [Gradient(h=grad_h, ah=grad_ah)]


@registry.register("mean")
def _mean_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False
) -> list[Gradient]:
    n = np.prod(a.shape) if axis is None else np.prod(np.array(a.shape)[axis])

    if isinstance(axis, int):
        axis = (axis,)

    if not keepdims and axis is not None:
        upstream_h = np.expand_dims(upstream.h, axis=axis)
        upstream_ah = np.expand_dims(upstream.ah, axis=axis)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    grad_h = np.ones_like(a) * (upstream_h / n)
    grad_ah = np.ones_like(a) * (upstream_ah / n)
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


@registry.register("clip")
def _clip_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    a_min: float | Tensor | None = None,
    a_max: float | Tensor | None = None,
    *args,
    **kwargs
) -> list[Gradient]:
    data = a.data

    grad_mask = np.ones_like(data, dtype=bool)
    if a_min is not None:
        grad_mask = np.logical_and(grad_mask, data >= a_min)
    if a_max is not None:
        grad_mask = np.logical_and(grad_mask, data <= a_max)

    grad_mask = grad_mask.astype(data.dtype)

    grad_h = upstream.h * grad_mask
    grad_ah = upstream.ah * grad_mask
    return [Gradient(h=grad_h, ah=grad_ah)]
