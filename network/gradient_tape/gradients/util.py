from typing import Optional, Tuple, Any
import numpy as np
from numba import vectorize, complex128, float64

@vectorize([complex128(complex128)], cache=True)
def complex_log_complex(z):
    x = z.real
    y = z.imag
    r = np.sqrt(x*x + y*y)
    if r == 0.0:
        return complex(-np.inf, 0.0)
    theta = np.arctan2(y, x)
    return complex(np.log(r), theta)

@vectorize([float64(float64)], cache=True)
def complex_log_real(z):
    if z <= 0.0:
        return -np.inf if z == 0 else np.log(z)
    else:
        return np.log(z)

def complex_log(z):
    return complex_log_complex(z) if np.iscomplexobj(z) or np.any(z < 0) else complex_log_real(z)

def ensure_shape(x: np.typing.NDArray[Any], shape: Optional[Tuple[(int, ...)]] = None) -> np.typing.NDArray[Any]:
    if shape is None:
        _target_shape: Tuple[(int, ...)] = ()
    elif isinstance(shape, int):
        _target_shape = (shape,)
    else:
        _target_shape = tuple(shape)

    x_shape: Tuple[(int, ...)] = x.shape

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
    axes_to_sum_broadcast: list[int] = []
    effective_x_rank = len(current_x_shape)

    for i in range(len(_target_shape)):
        x_dim_idx_aligned = effective_x_rank - len(_target_shape) + i

        if 0 <= x_dim_idx_aligned < effective_x_rank:
            target_dim_size = _target_shape[i]
            current_x_dim_size = current_x_shape[x_dim_idx_aligned]

            if target_dim_size == 1 and current_x_dim_size > 1:
                axes_to_sum_broadcast.append(x_dim_idx_aligned)

    if axes_to_sum_broadcast:
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