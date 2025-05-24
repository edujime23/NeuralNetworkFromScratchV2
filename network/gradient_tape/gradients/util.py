from typing import Callable, List, Optional, Tuple, Any
import numpy as np
from numba import vectorize, complex128, float64, int64
from functools import wraps

def alias(aliases: List[str]):
    """Decorator to assign aliases to functions."""
    def decorator(func: Callable):
        func.__aliases__ = tuple(aliases)
        return func
    return decorator

@alias(['log'])
@vectorize([complex128(complex128)], cache=True, nopython=True, fastmath=True, target='parallel')
def complex_log_complex(z: complex) -> complex:
    """Compute element-wise complex logarithm for complex inputs."""
    x, y = z.real, z.imag
    r = np.sqrt(x**2 + y**2)
    if r == 0.0:
        return complex(-np.inf, 0.0)
    theta = np.arctan2(y, x)
    return complex(np.log(r), theta)

@alias(['log'])
@vectorize([float64(float64), float64(int64)], cache=True, nopython=True, fastmath=True, target='parallel')
def complex_log_real(z: float) -> float:
    """Compute element-wise natural log for real inputs, with special cases."""
    if z == 0:
        return -np.inf
    return np.nan if z < 0 else np.log(z)

def complex_log(z: Any) -> Any:
    """
    Compute logarithm, dispatching between real and complex versions.
    Uses complex log if input is complex or contains negatives.
    """
    z_arr = np.asarray(z)
    if np.iscomplexobj(z_arr) or np.any(z_arr < 0):
        return complex_log_complex(z_arr)
    return complex_log_real(z_arr)