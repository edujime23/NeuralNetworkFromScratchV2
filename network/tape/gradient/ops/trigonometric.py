# trigonometric.py
import numpy as np
from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor


@registry.register("sin")
def _sin_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = sin(z) is holomorphic.
    df/dz = cos(z), df/dz_conj = 0
    """
    local_grad = np.cos(a)

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


@registry.register("cos")
def _cos_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = cos(z) is holomorphic.
    df/dz = -sin(z), df/dz_conj = 0
    """
    local_grad = -np.sin(a)

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


@registry.register("tan")
def _tan_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = tan(z) is holomorphic except at z = (n+1/2)pi.
    df/dz = sec^2(z) = 1/cos^2(z), df/dz_conj = 0
    """
    cos_a = np.cos(a)
    local_grad = 1 / (cos_a * cos_a)

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


@registry.register("arcsin")
def _arcsin_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = arcsin(z) is holomorphic except on (-inf,-1] and [1,inf).
    df/dz = 1/sqrt(1-z^2), df/dz_conj = 0
    """
    # Avoid numerical issues near branch cuts
    safe_a = np.where(np.abs(a) >= 1, np.sign(a) * (1 - 1e-10), a)
    local_grad = 1 / np.sqrt(1 - safe_a * safe_a)

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


@registry.register("arccos")
def _arccos_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = arccos(z) is holomorphic except on (-inf,-1] and [1,inf).
    df/dz = -1/sqrt(1-z^2), df/dz_conj = 0
    """
    # Avoid numerical issues near branch cuts
    safe_a = np.where(np.abs(a) >= 1, np.sign(a) * (1 - 1e-10), a)
    local_grad = -1 / np.sqrt(1 - safe_a * safe_a)

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


@registry.register("arctan")
def _arctan_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = arctan(z) is holomorphic except on (-i*inf,-i] and [i,i*inf).
    df/dz = 1/(1+z^2), df/dz_conj = 0
    """
    local_grad = 1 / (1 + a * a)

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


@registry.register("arctan2")
def _arctan2_grad(upstream: Gradient, result: Tensor, y: Tensor, x: Tensor) -> list[Gradient]:
    """
    f(y, x) = arctan2(y, x) = arctan(y/x) with appropriate quadrant

    df/dy = x / (x^2 + y^2)
    df/dx = -y / (x^2 + y^2)
    """
    # Compute the common denominator
    denom = x * x + y * y
    safe_denom = np.where(denom == 0, 1e-10, denom)

    # Gradient with respect to y
    local_grad_y = x / safe_denom

    if np.iscomplexobj(y):
        grad_y = Gradient(
            h=upstream.h * local_grad_y,
            ah=upstream.ah * np.conj(local_grad_y)
        )
    else:
        wirtinger_deriv = local_grad_y * 0.5
        grad_y = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    # Gradient with respect to x
    local_grad_x = -y / safe_denom

    if np.iscomplexobj(x):
        grad_x = Gradient(
            h=upstream.h * local_grad_x,
            ah=upstream.ah * np.conj(local_grad_x)
        )
    else:
        wirtinger_deriv = local_grad_x * 0.5
        grad_x = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_y, grad_x]


@registry.register("sinh")
def _sinh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = sinh(z) is holomorphic.
    df/dz = cosh(z), df/dz_conj = 0
    """
    local_grad = np.cosh(a)

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


@registry.register("cosh")
def _cosh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = cosh(z) is holomorphic.
    df/dz = sinh(z), df/dz_conj = 0
    """
    local_grad = np.sinh(a)

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


@registry.register("tanh")
def _tanh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = tanh(z) is holomorphic.
    df/dz = sech^2(z) = 1 - tanh^2(z), df/dz_conj = 0
    """
    # More numerically stable: 1 - tanh^2(z)
    local_grad = 1 - result * result

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


@registry.register("arcsinh")
def _arcsinh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = arcsinh(z) is holomorphic.
    df/dz = 1/sqrt(z^2 + 1), df/dz_conj = 0
    """
    local_grad = 1 / np.sqrt(a * a + 1)

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


@registry.register("arccosh")
def _arccosh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = arccosh(z) is holomorphic except on (-inf,1].
    df/dz = 1/sqrt(z^2 - 1), df/dz_conj = 0
    """
    # Avoid numerical issues near the branch cut
    safe_a = np.where(a <= 1, 1 + 1e-10, a)
    local_grad = 1 / np.sqrt(safe_a * safe_a - 1)

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


@registry.register("arctanh")
def _arctanh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = arctanh(z) is holomorphic except on (-inf,-1] and [1,inf).
    df/dz = 1/(1-z^2), df/dz_conj = 0
    """
    # Avoid numerical issues near branch cuts
    safe_a = np.where(np.abs(a) >= 1, np.sign(a) * (1 - 1e-10), a)
    local_grad = 1 / (1 - safe_a * safe_a)

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


@registry.register("hypot")
def _hypot_grad(upstream: Gradient, result: Tensor, *args) -> list[Gradient]:
    """
    f(x1, x2, ...) = hypot(x1, x2, ...) = sqrt(x1^2 + x2^2 + ...)
    df/dxi = xi / sqrt(x1^2 + x2^2 + ...) = xi / result
    """
    grads = []

    for arg in args:
        # dhypot/dxi = xi / hypot
        local_grad = arg / result

        if np.iscomplexobj(arg):
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

        grads.append(grad)

    return grads


@registry.register("sinc")
def _sinc_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    """
    f(z) = sinc(z) = sin(pi*z)/(pi*z) is holomorphic.
    df/dz = (pi*z*cos(pi*z) - sin(pi*z))/((pi*z)^2)
    """
    pi = np.pi
    pi_a = pi * a

    # Handle the special case at z=0
    if np.isscalar(a):
        zeros = (a == 0)
    else:
        zeros = (a == 0)

    if np.any(zeros):
        cos_pi_a = np.cos(pi_a)
        sin_pi_a = np.sin(pi_a)

        numerator = pi_a * cos_pi_a - sin_pi_a
        denominator = pi_a * pi_a

        # Avoid division by zero - derivative at z=0 is 0
        safe_denom = np.where(zeros, 1.0, denominator)
        local_grad = np.where(zeros, 0.0, numerator / safe_denom)
    else:
        cos_pi_a = np.cos(pi_a)
        sin_pi_a = np.sin(pi_a)

        numerator = pi_a * cos_pi_a - sin_pi_a
        local_grad = numerator / (pi_a * pi_a)

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