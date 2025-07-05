from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor
import numpy as np


@registry.register("reshape")
def _reshape_grad(
    upstream: Gradient, result: Tensor, a: Tensor, newshape: tuple
) -> list[Gradient]:
    grad_h = upstream.h.reshape(a.shape)
    grad_ah = upstream.ah.reshape(a.shape)
    return [Gradient(h=grad_h, ah=grad_ah)]


def complex_log(x: Tensor):
    if np.any(np.isclose(x, 0.0, atol=1e-8, rtol=0.0, equal_nan=True)):
        log_base = np.log(np.abs(1e-8))
    else:
        log_base = np.log(np.abs(x))

    if not np.iscomplexobj(x) and not np.any(x < 0):
        return log_base

    log_base = log_base.astype(np.complex64) + 1j * np.angle(x)
    return log_base


def _broadcast_reduction_result(a: Tensor, result: Tensor, axis, keepdims):
    """
    If keepdims=False, we need to reintroduce the reduced axes as singleton
    dims in `result` so that it broadcasts back to `a.shape`.
    """
    if axis is None or keepdims:
        return result
    # allow axis to be int or tuple
    axes = axis if isinstance(axis, tuple) else (axis,)
    # for negative axes, convert to positive
    axes = tuple(a.ndim + ax if ax < 0 else ax for ax in axes)
    # expand each reduced axis
    for ax in sorted(axes):
        result = np.expand_dims(result, axis=ax)
    return result
