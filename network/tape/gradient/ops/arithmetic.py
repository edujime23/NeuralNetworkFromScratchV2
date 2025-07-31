# arithmetic.py
import numpy as np
from network.tape.gradient.core.registry import registry
from network.tape.gradient.ops.util import complex_log
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor


def _safe_divide(numerator, denominator, epsilon=1e-12):
    """Safe division avoiding division by zero."""
    safe_denom = np.where(
        np.abs(denominator) < epsilon,
        epsilon + 0j if np.iscomplexobj(denominator) else epsilon,
        denominator
    )
    return numerator / safe_denom


def _safe_complex_log(x, epsilon=1e-12):
    """Safe logarithm using complex_log, avoiding log(0)."""
    safe_x = np.where(
        np.abs(x) < epsilon,
        epsilon + 0j if np.iscomplexobj(x) else epsilon,
        x
    )
    return complex_log(safe_x)


@registry.register("add")
def _add_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a,b) = a + b is holomorphic in both variables.
    df/da = 1, df/da_conj = 0
    df/db = 1, df/db_conj = 0
    """
    grad_a = Gradient(h=upstream.h, ah=upstream.ah)
    grad_b = Gradient(h=upstream.h, ah=upstream.ah)
    return [grad_a, grad_b]


@registry.register("subtract")
def _subtract_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a,b) = a - b is holomorphic in both variables.
    df/da = 1, df/da_conj = 0
    df/db = -1, df/db_conj = 0
    """
    grad_a = Gradient(h=upstream.h, ah=upstream.ah)
    grad_b = Gradient(h=-upstream.h, ah=-upstream.ah)
    return [grad_a, grad_b]


@registry.register("multiply")
def _multiply_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a,b) = a*b is holomorphic in both variables.
    df/da = b, df/da_conj = 0
    df/db = a, df/db_conj = 0

    Chain rule: For holomorphic f,
    dL/da = upstream.h * df/da + upstream.ah * df_conj/da
          = upstream.h * b + upstream.ah * 0
    dL/da_conj = upstream.h * df/da_conj + upstream.ah * df_conj/da_conj
               = upstream.h * 0 + upstream.ah * b_conj
    """
    grad_a = Gradient(
        h=upstream.h * b,
        ah=upstream.ah * np.conj(b)
    )

    grad_b = Gradient(
        h=upstream.h * a,
        ah=upstream.ah * np.conj(a)
    )

    return [grad_a, grad_b]


@registry.register("power")
def _power_grad(upstream: Gradient, result: Tensor, base: Tensor, exp: Tensor) -> list[Gradient]:
    """
    f(z,w) = z^w using pure Wirtinger calculus.

    For holomorphic functions:
    df/dz = w * z^(w-1), df/dz_conj = 0
    df/dw = z^w * ln(z), df/dw_conj = 0

    For real variables: both Wirtinger derivatives equal (1/2) * real_derivative
    """
    # Base gradient: d(z^w)/dz = w * z^(w-1) = w * result / z
    base_deriv = _safe_divide(exp * result, base)

    if np.iscomplexobj(base):
        # Complex base: holomorphic function
        grad_base = Gradient(
            h=upstream.h * base_deriv,
            ah=upstream.ah * np.conj(base_deriv)
        )
    else:
        # Real base: convert to Wirtinger derivatives
        wirtinger_deriv = base_deriv * 0.5
        grad_base = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    # Exponent gradient: d(z^w)/dw = z^w * ln(z) = result * ln(z)
    exp_deriv = result * _safe_complex_log(base)

    if np.iscomplexobj(exp):
        # Complex exponent: holomorphic function
        grad_exp = Gradient(
            h=upstream.h * exp_deriv,
            ah=upstream.ah * np.conj(exp_deriv)
        )
    else:
        # Real exponent: convert to Wirtinger derivatives
        wirtinger_deriv = exp_deriv * 0.5
        grad_exp = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_base, grad_exp]


@registry.register("abs")
@registry.register("fabs")
@registry.register("absolute")
def _abs_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = |z| is NOT holomorphic.

    Wirtinger derivatives:
    d|z|/dz = z_conj/(2|z|)
    d|z|/dz_conj = z/(2|z|)

    Since |z| is real-valued: f_conj(z,z_conj) = f(z_conj,z), so:
    df_conj/dz = df/dz_conj and df_conj/dz_conj = df/dz

    Chain rule:
    dL/dz = upstream.h * df/dz + upstream.ah * df_conj/dz
          = upstream.h * df/dz + upstream.ah * df/dz_conj
    dL/dz_conj = upstream.h * df/dz_conj + upstream.ah * df_conj/dz_conj
               = upstream.h * df/dz_conj + upstream.ah * df/dz
    """
    # Avoid division by zero
    safe_result = np.where(result == 0, 1e-12, result)

    if np.iscomplexobj(a):
        # Complex input: use standard complex Wirtinger derivatives
        # d|z|/dz = z_conj/(2|z|)
        dfdz = np.conj(a) / (2 * safe_result)
        # d|z|/dz_conj = z/(2|z|)
        dfdzbar = a / (2 * safe_result)
    else:
        # Real input: |x| = sign(x) * x, so d|x|/dx = sign(x)
        # For Wirtinger: d|x|/dz = d|x|/dz_conj = (1/2) * sign(x)
        sign_a = np.sign(a)
        wirtinger_deriv = sign_a / 2
        dfdz = wirtinger_deriv
        dfdzbar = wirtinger_deriv

    # Apply chain rule for real-valued function
    grad_h = upstream.h * dfdz + upstream.ah * dfdzbar
    grad_ah = upstream.h * dfdzbar + upstream.ah * dfdz

    return [Gradient(h=grad_h, ah=grad_ah)]