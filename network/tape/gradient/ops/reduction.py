# reduction.py
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
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = sum(a, axis) is holomorphic.
    df/da_i = 1 for all elements i that are being summed.
    """
    # If not keepdims, we need to reshape upstream for broadcasting
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    # Sum is holomorphic: dsum/da_i = 1, dsum/da_i_conj = 0
    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h,  # * 1
            ah=upstream_ah  # * conj(1) = 1
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = 0.5  # (1/2) * 1
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None]


@registry.register("mean")
def _mean_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = mean(a, axis) = sum(a, axis) / n is holomorphic.
    df/da_i = 1/n for all elements i that are being averaged.
    """
    # Calculate the number of elements being averaged
    if axis is None:
        n = np.prod(a.shape)
    else:
        axes = [axis] if isinstance(axis, int) else axis
        n = np.prod([a.shape[ax] for ax in axes])

    # Reshape upstream if needed
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    # Mean is holomorphic: dmean/da_i = 1/n
    local_grad = 1.0 / n

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h * local_grad,
            ah=upstream_ah * local_grad  # conj(1/n) = 1/n
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad * 0.5
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None]


@registry.register("prod")
def _prod_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = prod(a, axis) is holomorphic.
    df/da_i = product of all elements except a_i = result / a_i
    """
    zero_mask = a == 0

    # Handle zeros in the input
    if np.any(zero_mask):
        if np.sum(zero_mask) == 1:
            # If there's exactly one zero, calculate gradient properly
            grad_data = np.zeros_like(a)
            non_zero_data = a[~zero_mask]

            if axis is None:
                prod_non_zero = np.prod(non_zero_data)
                grad_data[zero_mask] = prod_non_zero
            else:
                raise NotImplementedError(
                    "Gradient of prod with zeros and axis reduction not implemented"
                )
        else:
            # Multiple zeros: gradient is zero everywhere
            grad_data = np.zeros_like(a)
    else:
        # No zeros: gradient is product / element
        result_broadcast = _broadcast_reduction_result(a, result, axis, keepdims)
        grad_data = result_broadcast / a

    # Reshape upstream if needed
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h * grad_data,
            ah=upstream_ah * np.conj(grad_data)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_data * 0.5
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None]


@registry.register("max")
def _max_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = max(a, axis) - gradient distributed equally among maximum elements.
    """
    # Broadcast result to input shape for comparison
    result_broadcast = _broadcast_reduction_result(a, result, axis, keepdims)

    # Identify elements that match the maximum
    mask = a == result_broadcast

    # Count and distribute gradient equally
    if axis is None:
        num_max = np.sum(mask)
        grad_mask = mask / num_max
    else:
        axes = [axis] if isinstance(axis, int) else axis
        num_max = np.sum(mask, axis=axes, keepdims=True)
        grad_mask = mask / num_max

    # Reshape upstream if needed
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h * grad_mask,
            ah=upstream_ah * np.conj(grad_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_mask * 0.5
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None]


@registry.register("min")
def _min_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = min(a, axis) - gradient distributed equally among minimum elements.
    """
    # Broadcast result to input shape for comparison
    result_broadcast = _broadcast_reduction_result(a, result, axis, keepdims)

    # Identify elements that match the minimum
    mask = a == result_broadcast

    # Count and distribute gradient equally
    if axis is None:
        num_min = np.sum(mask)
        grad_mask = mask / num_min
    else:
        axes = [axis] if isinstance(axis, int) else axis
        num_min = np.sum(mask, axis=axes, keepdims=True)
        grad_mask = mask / num_min

    # Reshape upstream if needed
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h * grad_mask,
            ah=upstream_ah * np.conj(grad_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_mask * 0.5
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None]


@registry.register("var")
def _var_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = var(a, axis) is holomorphic.
    dvar/da_i = 2(a_i - mean(a))/(n-ddof)
    """
    # Calculate mean along the specified axis
    mean = np.mean(a, axis=axis, keepdims=True)

    # Calculate n based on ddof
    if axis is None:
        n = np.prod(a.shape) - ddof
    else:
        axes = [axis] if isinstance(axis, int) else axis
        n = np.prod([a.shape[ax] for ax in axes]) - ddof

    # Calculate centered data and local gradient
    centered = a - mean
    local_grad = 2 * centered / n

    # Reshape upstream if needed
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h * local_grad,
            ah=upstream_ah * np.conj(local_grad)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad * 0.5
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None, None]


@registry.register("std")
def _std_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    *,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    **kwargs
) -> list[Gradient]:
    """
    f(a) = std(a, axis) = sqrt(var(a, axis)) is holomorphic.
    dstd/da_i = (a_i - mean(a))/((n-ddof)*std(a))
    """
    # Calculate mean along the specified axis
    mean = np.mean(a, axis=axis, keepdims=True)

    # Calculate n based on ddof
    if axis is None:
        n = np.prod(a.shape) - ddof
    else:
        axes = [axis] if isinstance(axis, int) else axis
        n = np.prod([a.shape[ax] for ax in axes]) - ddof

    # Calculate centered data
    centered = a - mean

    # Avoid division by zero
    safe_result = np.where(result == 0, 1e-10, result)
    result_broadcast = _broadcast_reduction_result(a, safe_result, axis, keepdims)

    # Local gradient: (x - mean)/(n*std)
    local_grad = centered / (n * result_broadcast)

    # Reshape upstream if needed
    if not keepdims and axis is not None:
        target_shape = list(a.shape)
        axes = [axis] if isinstance(axis, int) else axis
        for ax in axes:
            target_shape[ax] = 1

        upstream_h = upstream.h.reshape(target_shape)
        upstream_ah = upstream.ah.reshape(target_shape)
    else:
        upstream_h = upstream.h
        upstream_ah = upstream.ah

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream_h * local_grad,
            ah=upstream_ah * np.conj(local_grad)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad * 0.5
        grad = Gradient(
            h=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv,
            ah=upstream_h * wirtinger_deriv + upstream_ah * wirtinger_deriv
        )

    return [grad, None, None, None]


# POTENTIAL DUPLICATES - These might conflict with element-wise versions in floating_point.py
@registry.register("maximum")  # DUPLICATE WARNING: Also in floating_point.py
def _maximum_reduction_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    b: Tensor
) -> list[Gradient]:
    """
    WARNING: This might be a DUPLICATE of the element-wise maximum in floating_point.py

    f(a, b) = maximum(a, b) element-wise maximum.
    """
    # Create masks for the different conditions
    a_gt_b = a > b
    a_eq_b = np.isclose(a, b, atol=1e-12)

    grad_a_mask = np.where(a_gt_b, 1.0, np.where(a_eq_b, 0.5, 0.0))
    grad_b_mask = np.where(a_gt_b, 0.0, np.where(a_eq_b, 0.5, 1.0))

    if np.iscomplexobj(a):
        grad_a = Gradient(
            h=upstream.h * grad_a_mask,
            ah=upstream.ah * np.conj(grad_a_mask)
        )
    else:
        wirtinger_deriv = grad_a_mask * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        grad_b = Gradient(
            h=upstream.h * grad_b_mask,
            ah=upstream.ah * np.conj(grad_b_mask)
        )
    else:
        wirtinger_deriv = grad_b_mask * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("minimum")  # DUPLICATE WARNING: Also in floating_point.py
def _minimum_reduction_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    b: Tensor
) -> list[Gradient]:
    """
    WARNING: This might be a DUPLICATE of the element-wise minimum in floating_point.py

    f(a, b) = minimum(a, b) element-wise minimum.
    """
    # Create masks for the different conditions
    a_lt_b = a < b
    a_eq_b = np.isclose(a, b, atol=1e-12)

    grad_a_mask = np.where(a_lt_b, 1.0, np.where(a_eq_b, 0.5, 0.0))
    grad_b_mask = np.where(a_lt_b, 0.0, np.where(a_eq_b, 0.5, 1.0))

    if np.iscomplexobj(a):
        grad_a = Gradient(
            h=upstream.h * grad_a_mask,
            ah=upstream.ah * np.conj(grad_a_mask)
        )
    else:
        wirtinger_deriv = grad_a_mask * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        grad_b = Gradient(
            h=upstream.h * grad_b_mask,
            ah=upstream.ah * np.conj(grad_b_mask)
        )
    else:
        wirtinger_deriv = grad_b_mask * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("clip")  # POTENTIAL DUPLICATE: Also in floating_point.py
def _clip_reduction_grad(
    upstream: Gradient,
    result: Tensor,
    a: Tensor,
    a_min: float | Tensor | None = None,
    a_max: float | Tensor | None = None,
    **kwargs
) -> list[Gradient]:
    """
    WARNING: This might be a DUPLICATE of clip in floating_point.py

    f(a) = clip(a, a_min, a_max) clips values to range.
    """
    # Create mask for where gradient should be 1 (within range)
    mask = np.ones_like(a, dtype=bool)

    if a_min is not None:
        mask = np.logical_and(mask, a > a_min)
    if a_max is not None:
        mask = np.logical_and(mask, a < a_max)

    grad_mask = mask.astype(a.dtype)

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h * grad_mask,
            ah=upstream.ah * np.conj(grad_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_mask * 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad, None, None]