# exponential.py
import numpy as np
from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor


@registry.register("exp")
def _exp_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = exp(z) is holomorphic.
    df/dz = exp(z) = result
    df/dz_conj = 0
    """
    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad = Gradient(
            h=upstream.h * result,
            ah=upstream.ah * np.conj(result)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = result * 0.5
        grad = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad]


@registry.register("expm1")
def _expm1_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = exp(z) - 1 is holomorphic.
    df/dz = exp(z) = result + 1
    df/dz_conj = 0
    """
    # exp(z) = result + 1
    local_grad = result + 1

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


@registry.register("exp2")
def _exp2_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = 2^z = exp(z * ln(2)) is holomorphic.
    df/dz = 2^z * ln(2) = result * ln(2)
    df/dz_conj = 0
    """
    ln2 = np.log(2)
    local_grad = result * ln2

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


@registry.register("log")
def _log_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = log(z) is holomorphic (except at z=0 and negative real axis).
    df/dz = 1/z
    df/dz_conj = 0
    """
    # Avoid division by zero
    safe_a = np.where(a == 0, 1e-12, a)
    local_grad = 1 / safe_a

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


@registry.register("log10")
def _log10_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = log10(z) = log(z) / log(10) is holomorphic.
    df/dz = 1 / (z * ln(10))
    df/dz_conj = 0
    """
    # Avoid division by zero
    safe_a = np.where(a == 0, 1e-12, a)
    ln10 = np.log(10)
    local_grad = 1 / (safe_a * ln10)

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


@registry.register("log2")
def _log2_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = log2(z) = log(z) / log(2) is holomorphic.
    df/dz = 1 / (z * ln(2))
    df/dz_conj = 0
    """
    # Avoid division by zero
    safe_a = np.where(a == 0, 1e-12, a)
    ln2 = np.log(2)
    local_grad = 1 / (safe_a * ln2)

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


@registry.register("log1p")
def _log1p_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = log(1 + z) is holomorphic.
    df/dz = 1 / (1 + z)
    df/dz_conj = 0
    """
    local_grad = 1 / (1 + a)

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


@registry.register("logaddexp")
def _logaddexp_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    """
    f(a,b) = log(exp(a) + exp(b)) is holomorphic in both variables.

    Using the chain rule:
    df/da = exp(a) / (exp(a) + exp(b)) = exp(a - f)
    df/db = exp(b) / (exp(a) + exp(b)) = exp(b - f)
    """
    # For numerical stability, use exp(x - result) instead of exp(x)/exp(result)
    local_grad_a = np.exp(a - result)
    local_grad_b = np.exp(b - result)

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h * local_grad_a,
            ah=upstream.ah * np.conj(local_grad_a)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad_a * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        # Complex variable: holomorphic function
        grad_b = Gradient(
            h=upstream.h * local_grad_b,
            ah=upstream.ah * np.conj(local_grad_b)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad_b * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("logaddexp2")
def _logaddexp2_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    """
    f(a,b) = log2(2^a + 2^b) is holomorphic in both variables.

    Similar to logaddexp but with base 2:
    df/da = 2^a / (2^a + 2^b) = 2^(a - f)
    df/db = 2^b / (2^a + 2^b) = 2^(b - f)
    """
    # For numerical stability
    local_grad_a = 2 ** (a - result)
    local_grad_b = 2 ** (b - result)

    if np.iscomplexobj(a):
        # Complex variable: holomorphic function
        grad_a = Gradient(
            h=upstream.h * local_grad_a,
            ah=upstream.ah * np.conj(local_grad_a)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad_a * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    if np.iscomplexobj(b):
        # Complex variable: holomorphic function
        grad_b = Gradient(
            h=upstream.h * local_grad_b,
            ah=upstream.ah * np.conj(local_grad_b)
        )
    else:
        # Real variable: both Wirtinger derivatives contribute
        wirtinger_deriv = local_grad_b * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]