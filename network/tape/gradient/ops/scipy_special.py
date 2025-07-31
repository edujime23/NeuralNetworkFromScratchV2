# special.py
import numpy as np
from scipy.special import digamma

from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor


@registry.register("gamma")
def _gamma_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = gamma(z) is holomorphic for Re(z) > 0.
    dgamma(z)/dz = gamma(z) * digamma(z)
    """
    local_grad = result * digamma(a)

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

    return [grad]


@registry.register("loggamma")
def _loggamma_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = loggamma(z) is holomorphic for Re(z) > 0.
    dloggamma(z)/dz = digamma(z)
    """
    local_grad = digamma(a)

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

    return [grad]


@registry.register("erf")
def _erf_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = erf(z) is holomorphic (entire function).
    derf(z)/dz = (2/sqrt(pi)) * exp(-z^2)
    """
    local_grad = (2 / np.sqrt(np.pi)) * np.exp(-(a**2))

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

    return [grad]


@registry.register("erfc")
def _erfc_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = erfc(z) = 1 - erf(z) is holomorphic (entire function).
    derfc(z)/dz = -(2/sqrt(pi)) * exp(-z^2)
    """
    local_grad = (-2 / np.sqrt(np.pi)) * np.exp(-(a**2))

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

    return [grad]