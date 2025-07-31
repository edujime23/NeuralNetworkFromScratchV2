# util.py (utilities)
from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor
import numpy as np
from typing import Union, Tuple, Optional
import warnings


@registry.register("reshape")
def _reshape_grad(
    upstream: Gradient, result: Tensor, a: Tensor, newshape: tuple
) -> list[Gradient]:
    """
    Gradient for reshape operation.

    Reshaping doesn't change values, only their arrangement, so the gradient
    is simply reshaped back to the original shape.

    This is correct for Wirtinger calculus: reshape is a linear operation that
    doesn't affect the mathematical relationship between variables.
    """
    try:
        grad_h = upstream.h.reshape(a.shape)
        grad_ah = upstream.ah.reshape(a.shape)
    except ValueError as e:
        raise ValueError(
            f"Cannot reshape gradient from {upstream.h.shape} to {a.shape}: {e}"
        )

    # None for newshape parameter
    return [Gradient(h=grad_h, ah=grad_ah), None]


def complex_log(
    x: Tensor,
    k: int = 0
) -> Tensor:
    """
    Compute the complex logarithm with robust handling of edge cases.

    Computes log(z) = log|z| + i(arg(z) + 2*pi*k) where k selects the branch.

    Special handling for:
    - Zero/near-zero values (replaced with machine epsilon)
    - Negative real values (returns complex result)
    - NaN/Inf values (propagated according to numpy rules)
    - Real positive values (returns real result to preserve type)

    Args:
        x: Input tensor
        k: Branch index for multivalued logarithm (default 0 for principal branch)

    Returns:
        Complex logarithm of x. Returns real type for positive real inputs,
        complex type otherwise.

    Raises:
        ValueError: If k is not an integer
    """
    if not isinstance(k, (int, np.integer)):
        raise ValueError(f"Branch index k must be an integer, got {type(k).__name__}")

    # Use machine epsilon relative to the data type for better numerical stability
    if x.dtype in [np.float32, np.complex64]:
        eps = np.finfo(np.float32).eps
        complex_dtype = np.complex64
    else:
        eps = np.finfo(np.float64).eps
        complex_dtype = np.complex128

    # Handle zero/near-zero values
    abs_x = np.abs(x)
    zero_mask = abs_x < eps

    # Safe computation of log|x|
    with np.errstate(divide='ignore', invalid='ignore'):
        # This will produce -inf for true zeros, which is mathematically correct
        safe_x = np.where(zero_mask, eps, abs_x)
        log_abs = np.log(safe_x)

    # Check if we can return a real result
    if not np.iscomplexobj(x):
        # For real input, check if all values are positive
        if np.all(x > 0) or (np.all(x >= 0) and not np.any(zero_mask)):
            # Pure positive real - return real result as tensor
            return Tensor(log_abs)
        elif np.all(x >= 0) and np.any(zero_mask):
            # Has zeros - warn about replacement
            warnings.warn(
                f"Replaced {np.sum(zero_mask)} zero value(s) with epsilon={eps:.2e} "
                "in logarithm computation",
                RuntimeWarning,
                stacklevel=2
            )
            return Tensor(log_abs)

    # Need complex result
    # Compute angle (argument)
    angle = np.angle(x)

    # Add branch correction
    if k != 0:
        angle = angle + 2 * k * np.pi

    # Combine magnitude and phase
    result_data = log_abs.astype(complex_dtype) + 1j * angle

    # Warn about zero replacements for complex case
    if np.any(zero_mask):
        warnings.warn(
            f"Replaced {np.sum(zero_mask)} near-zero value(s) with epsilon={eps:.2e} "
            "in complex logarithm computation",
            RuntimeWarning,
            stacklevel=2
        )

    return result_data


def _broadcast_reduction_result(
    a: Tensor,
    result: Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
) -> Tensor:
    """
    Broadcast a reduction result back to the original tensor shape.

    When a reduction operation (like sum, mean, max) is performed with
    keepdims=False, the reduced dimensions are removed. This function
    reinserts those dimensions as size-1 dimensions so the result can
    be broadcast against the original tensor.

    Args:
        a: Original tensor that was reduced
        result: Result of the reduction operation
        axis: Axis or axes along which reduction was performed.
              Can be None (full reduction), int, or tuple of ints.
              Negative values are supported.
        keepdims: Whether the reduction preserved dimensions

    Returns:
        The result with singleton dimensions inserted at the reduced axes,
        ready for broadcasting against the original tensor shape.

    Raises:
        ValueError: If axes are out of bounds or result shape is incompatible
    """
    # Fast path: if keepdims=True or axis=None, no reshaping needed
    if axis is None or keepdims:
        return result

    ndim = len(a.shape)

    # Normalize axis to a tuple of positive integers
    if axis is None:
        # Should not reach here due to fast path above
        return result
    elif isinstance(axis, (int, np.integer)):
        axes = (int(axis),)
    elif isinstance(axis, tuple):
        axes = tuple(int(ax) for ax in axis)
    elif isinstance(axis, list):
        axes = tuple(int(ax) for ax in axis)
    else:
        raise TypeError(
            f"axis must be None, int, or tuple/list of ints, got {type(axis).__name__}"
        )

    # Validate and normalize negative axes
    normalized_axes = []
    for ax in axes:
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise ValueError(
                f"axis {ax - ndim if ax >= 0 else ax} is out of bounds "
                f"for tensor of dimension {ndim}"
            )
        normalized_axes.append(ax)

    # Remove duplicates and sort
    normalized_axes = sorted(set(normalized_axes))

    # Verify result shape matches expected shape after reduction
    expected_shape = [a.shape[i] for i in range(ndim) if i not in normalized_axes]
    if list(result.shape) != expected_shape:
        raise ValueError(
            f"Result shape {result.shape} is incompatible with "
            f"reducing axes {tuple(axes)} from shape {a.shape}. "
            f"Expected shape: {tuple(expected_shape)}"
        )

    # Build the target shape with singleton dimensions
    target_shape = []
    result_idx = 0
    for i in range(ndim):
        if i in normalized_axes:
            target_shape.append(1)
        else:
            target_shape.append(result.shape[result_idx])
            result_idx += 1

    # Reshape with singleton dimensions
    return result.reshape(target_shape)