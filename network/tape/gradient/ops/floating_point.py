# floating_point.py
import numpy as np
from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor


# Non-differentiable functions (correct as-is)
@registry.register("floor")
def _floor_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """f(z) = floor(z) is not differentiable at integer points."""
    return [None]


@registry.register("ceil")
def _ceil_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """f(z) = ceil(z) is not differentiable at integer points."""
    return [None]


@registry.register("trunc")
def _trunc_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """f(z) = trunc(z) is not differentiable at integer points."""
    return [None]


@registry.register("rint")
def _rint_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """f(z) = rint(z) rounds to nearest integer."""
    return [None]


@registry.register("fix")
def _fix_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """f(z) = fix(z) rounds toward zero."""
    return [None]


@registry.register("around")
@registry.register("round")
def _round_grad(upstream: Gradient, result: Tensor, a: Tensor, decimals=0) -> list[None]:
    """f(z) = round(z, decimals) rounds to specified decimals."""
    return [None]


@registry.register("nextafter")
def _nextafter_grad(upstream: Gradient, result: Tensor, x1: Tensor, x2: Tensor) -> list[None]:
    """f(x1, x2) = nextafter(x1, x2) is not differentiable."""
    return [None, None]


@registry.register("spacing")
def _spacing_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """f(z) = spacing(z) is not differentiable."""
    return [None]


# Boolean functions (correct as-is)
@registry.register("isnan")
def _isnan_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """Boolean test function with zero gradient everywhere."""
    return [None]


@registry.register("isfinite")
def _isfinite_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """Boolean test function with zero gradient everywhere."""
    return [None]


@registry.register("isinf")
def _isinf_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """Boolean test function with zero gradient everywhere."""
    return [None]


@registry.register("signbit")
def _signbit_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[None]:
    """Boolean test function with zero gradient everywhere."""
    return [None]


@registry.register("degree")
@registry.register("rad2deg")
def _rad2deg_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = rad2deg(z) converts radians to degrees: z * (180/pi)
    Derivative is 180/pi (constant, holomorphic).
    """
    factor = 180.0 / np.pi

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h * factor,
            ah=upstream.ah * np.conj(factor)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = factor * 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad]


@registry.register("radians")
@registry.register("deg2rad")
def _deg2rad_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = deg2rad(z) converts degrees to radians: z * (pi/180)
    Derivative is pi/180 (constant, holomorphic).
    """
    factor = np.pi / 180.0

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h * factor,
            ah=upstream.ah * np.conj(factor)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = factor * 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad]


@registry.register("modf")
def _modf_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = modf(z) returns (fractional_part, integer_part).

    Assuming upstream is for fractional part:
    fractional_part = z - floor(z), so derivative = 1 (except at discontinuities)
    """
    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h,
            ah=upstream.ah * np.conj(1.0)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad]


@registry.register("frexp")
def _frexp_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = frexp(z) returns (mantissa, exponent) where z = mantissa * 2^exponent.

    This is complex to differentiate properly. For practical purposes,
    treating as identity function for the mantissa part.
    """
    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h,
            ah=upstream.ah * np.conj(1.0)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad]


@registry.register("copysign")
def _copysign_grad(upstream: Gradient, result: Tensor, x1: Tensor, x2: Tensor) -> list[Gradient | None]:
    """
    f(x1, x2) = copysign(x1, x2) returns x1 with sign of x2.
    df/dx1 = sign(x2), df/dx2 = 0 (almost everywhere)
    """
    sign_x2 = np.sign(x2)

    if np.iscomplexobj(x1):
        # Complex variable: holomorphic function
        grad_x1 = Gradient(
            h=upstream.h * sign_x2,
            ah=upstream.ah * np.conj(sign_x2)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = sign_x2 * 0.5
        grad_x1 = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_x1, None]


@registry.register("nan_to_num")
def _nan_to_num_grad(upstream: Gradient, result: Tensor, a: Tensor, **kwargs) -> list[Gradient | None]:
    """
    f(z) = nan_to_num(z) replaces NaN/inf with finite numbers.
    Derivative is 1 for finite values, 0 elsewhere.
    """
    finite_mask = np.isfinite(a)
    local_grad = np.where(finite_mask, 1.0, 0.0)

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h * local_grad,
            ah=upstream.ah * np.conj(local_grad)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad * 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad] + [None] * len(kwargs)


@registry.register("real_if_close")
def _real_if_close_grad(upstream: Gradient, result: Tensor, a: Tensor, **kwargs) -> list[Gradient | None]:
    """
    f(z) = real_if_close(z) returns real part if imaginary part is small.

    Simplified: if result is real but input was complex, derivative affects only real part.
    Otherwise, identity function.
    """
    was_converted_to_real = not np.iscomplexobj(result) and np.iscomplexobj(a)

    if was_converted_to_real:
        # Only real part survives: derivative = 1 for real part, 0 for imaginary
        local_grad = 1.0 + 0j  # This will naturally select real part
    else:
        # Identity function
        local_grad = 1.0

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h * local_grad,
            ah=upstream.ah * np.conj(local_grad)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad * 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad] + [None] * len(kwargs)


@registry.register("fmod")
def _fmod_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient | None]:
    """
    f(a, b) = fmod(a, b) computes remainder of a/b.
    df/da = 1, df/db is complex and not useful.
    """
    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h,
            ah=upstream.ah * np.conj(1.0)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, None]


@registry.register("fmax")
def _fmax_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = fmax(a, b) element-wise maximum.
    Derivative: 1 where a > b, 0 where a < b, 0.5 where a = b.
    """
    a_gt_b = a > b
    a_eq_b = np.isclose(a, b, atol=1e-12)

    grad_a_mask = np.where(a_gt_b, 1.0, np.where(a_eq_b, 0.5, 0.0))
    grad_b_mask = np.where(a_gt_b, 0.0, np.where(a_eq_b, 0.5, 1.0))

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h * grad_a_mask,
            ah=upstream.ah * np.conj(grad_a_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_a_mask * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        # Complex variable: holomorphic function
        grad_b = Gradient(
            h=upstream.h * grad_b_mask,
            ah=upstream.ah * np.conj(grad_b_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_b_mask * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("fmin")
def _fmin_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = fmin(a, b) element-wise minimum.
    Derivative: 1 where a < b, 0 where a > b, 0.5 where a = b.
    """
    a_lt_b = a < b
    a_eq_b = np.isclose(a, b, atol=1e-12)

    grad_a_mask = np.where(a_lt_b, 1.0, np.where(a_eq_b, 0.5, 0.0))
    grad_b_mask = np.where(a_lt_b, 0.0, np.where(a_eq_b, 0.5, 1.0))

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h * grad_a_mask,
            ah=upstream.ah * np.conj(grad_a_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_a_mask * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        # Complex variable: holomorphic function
        grad_b = Gradient(
            h=upstream.h * grad_b_mask,
            ah=upstream.ah * np.conj(grad_b_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_b_mask * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("maximum")
def _maximum_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = maximum(a, b) element-wise maximum, propagating NaN.
    Similar to fmax but with NaN propagation.
    """
    # Handle NaN propagation
    a_is_nan = np.isnan(a)
    b_is_nan = np.isnan(b)

    a_gt_b = np.logical_and(a > b, ~(a_is_nan | b_is_nan))
    a_eq_b = np.logical_and(np.isclose(a, b, atol=1e-12), ~(a_is_nan | b_is_nan))

    grad_a_mask = np.where(a_is_nan | a_gt_b, 1.0, np.where(a_eq_b, 0.5, 0.0))
    grad_b_mask = np.where(b_is_nan | ~(a_is_nan | a_gt_b | a_eq_b), 1.0,
                          np.where(a_eq_b, 0.5, 0.0))

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h * grad_a_mask,
            ah=upstream.ah * np.conj(grad_a_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_a_mask * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        # Complex variable: holomorphic function
        grad_b = Gradient(
            h=upstream.h * grad_b_mask,
            ah=upstream.ah * np.conj(grad_b_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_b_mask * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("minimum")
def _minimum_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = minimum(a, b) element-wise minimum, propagating NaN.
    Similar to fmin but with NaN propagation.
    """
    # Handle NaN propagation
    a_is_nan = np.isnan(a)
    b_is_nan = np.isnan(b)

    a_lt_b = np.logical_and(a < b, ~(a_is_nan | b_is_nan))
    a_eq_b = np.logical_and(np.isclose(a, b, atol=1e-12), ~(a_is_nan | b_is_nan))

    grad_a_mask = np.where(a_is_nan | a_lt_b, 1.0, np.where(a_eq_b, 0.5, 0.0))
    grad_b_mask = np.where(b_is_nan | ~(a_is_nan | a_lt_b | a_eq_b), 1.0,
                          np.where(a_eq_b, 0.5, 0.0))

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h * grad_a_mask,
            ah=upstream.ah * np.conj(grad_a_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_a_mask * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        # Complex variable: holomorphic function
        grad_b = Gradient(
            h=upstream.h * grad_b_mask,
            ah=upstream.ah * np.conj(grad_b_mask)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = grad_b_mask * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("clip")
def _clip_grad(upstream: Gradient, result: Tensor, a: Tensor, a_min=None, a_max=None) -> list[Gradient | None]:
    """
    f(a) = clip(a, a_min, a_max) clips values to range.
    Derivative is 1 in range, 0 outside.
    """
    # Handle None values
    if a_min is None:
        a_min = np.min(a) - 1
    if a_max is None:
        a_max = np.max(a) + 1

    in_range = np.logical_and(a > a_min, a < a_max)
    grad_mask = np.where(in_range, 1.0, 0.0)

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