from typing import Callable, List, Optional, Tuple, Any
import numpy as np
from numba import vectorize, complex128, float64, int64
from functools import wraps

def alias(aliases: List[str]):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        func.__aliases__ = tuple(aliases)
        return wrapper
    return decorator

@alias(['log'])
@vectorize([complex128(complex128)], cache=True, nopython=True, fastmath=True, target='parallel')
def complex_log_complex(z):
    x = z.real
    y = z.imag
    r = np.sqrt(x*x + y*y)
    if r == 0.0:
        return complex(-np.inf, 0.0)
    theta = np.arctan2(y, x)
    return complex(np.log(r), theta)

@alias(['log'])
@vectorize([float64(float64), int64(int64)], cache=True, nopython=True, fastmath=True, target='parallel')
def complex_log_real(z):
    # Corrected real log handling:
    if z == 0:
        return -np.inf
    elif z < 0:
        # Real log undefined for negative inputs, 
        # raise error or return nan:
        # Here we return np.nan to avoid crashing, but you can choose to raise.
        return np.nan
    else:
        return np.log(z)

def complex_log(z: Any) -> Any:
    """
    Computes the complex logarithm for complex or real inputs.
    Uses complex log if input is complex or contains negative values,
    else uses real log.
    """
    z_arr = np.asarray(z)
    if np.iscomplexobj(z_arr) or np.any(z_arr < 0):
        return complex_log_complex(z_arr)
    else:
        return complex_log_real(z_arr)

def ensure_shape(
    x: np.typing.NDArray[Any], 
    shape: Optional[Tuple[int, ...]] = None
) -> np.typing.NDArray[Any]:
    """
    Ensures that the input array x has the specified shape.
    If x has more dimensions, sums over leading dims.
    Sums over broadcasted dims to match target shape.
    Finally broadcasts to target shape.
    """
    if shape is None:
        _target_shape: Tuple[int, ...] = ()
    elif isinstance(shape, int):
        _target_shape = (shape,)
    else:
        _target_shape = tuple(shape)

    x_shape: Tuple[int, ...] = x.shape

    if x_shape == _target_shape:
        return x

    # Handle scalar x broadcasting to non-scalar target
    if not x_shape and _target_shape:  # x is scalar, target is not
        try:
            return np.broadcast_to(x, _target_shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot broadcast scalar input (shape {x_shape}) to target shape {_target_shape}."
            ) from e

    x_current = x
    current_x_shape = x_shape

    # Sum over leading dimensions if x has more dimensions than target.
    rank_diff = len(current_x_shape) - len(_target_shape)
    if rank_diff > 0:
        axes_to_sum_leading = tuple(range(rank_diff))
        x_current = np.sum(x_current, axis=axes_to_sum_leading, keepdims=False)
        current_x_shape = x_current.shape

    # Sum over dimensions that were broadcasted
    axes_to_sum_broadcast: List[int] = []
    effective_x_rank = len(current_x_shape)

    for i in range(len(_target_shape)):
        x_dim_idx_aligned = effective_x_rank - len(_target_shape) + i

        if 0 <= x_dim_idx_aligned < effective_x_rank:
            target_dim_size = _target_shape[i]
            current_x_dim_size = current_x_shape[x_dim_idx_aligned]

            if target_dim_size == 1 and current_x_dim_size > 1:
                axes_to_sum_broadcast.append(x_dim_idx_aligned)

    if axes_to_sum_broadcast:
        # sum along these dims keeping dims for broadcast compatibility
        x_current = np.sum(x_current, axis=tuple(axes_to_sum_broadcast), keepdims=True)
        current_x_shape = x_current.shape

    # Pad shape with leading 1s if needed before broadcasting
    pad_dims = len(_target_shape) - len(current_x_shape)
    if pad_dims > 0:
        x_current = x_current.reshape((1,) * pad_dims + current_x_shape)

    # Final broadcast if shapes still don't match
    if x_current.shape != _target_shape:
        try:
            return np.broadcast_to(x_current, _target_shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot ensure shape: input_x shape {x.shape} -> after reduction became {x_current.shape}, "
                f"which is not compatible with target shape {_target_shape} for broadcasting."
            ) from e

    return x_current