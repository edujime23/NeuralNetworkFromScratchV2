import numpy as np

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


@registry.register("add")
def _add_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    return [
        Gradient(h=upstream.h, ah=upstream.ah),
        Gradient(h=upstream.h, ah=upstream.ah),
    ]


@registry.register("subtract")
def _subtract_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    return [
        Gradient(h=upstream.h, ah=upstream.ah),
        Gradient(h=-upstream.h, ah=-upstream.ah),
    ]


@registry.register("negative")
def _negative_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    return [Gradient(h=-upstream.h, ah=-upstream.ah)]


@registry.register("multiply")
def _multiply_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    grad_a_h = upstream.h * b
    grad_b_h = upstream.h * a
    grad_a_ah = upstream.ah * np.conj(b)
    grad_b_ah = upstream.ah * np.conj(a)

    return [Gradient(h=grad_a_h, ah=grad_a_ah), Gradient(h=grad_b_h, ah=grad_b_ah)]


@registry.register("divide")  # Or "true_divide"
def _divide_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    grad_a_h = upstream.h / b
    grad_a_ah = upstream.ah / np.conj(b)
    grad_b_h = upstream.h * (-a / (b**2))
    grad_b_ah = upstream.ah * (-np.conj(a) / (np.conj(b) ** 2))
    return [
        Gradient(h=grad_a_h, ah=grad_a_ah),
        Gradient(h=grad_b_h, ah=grad_b_ah),
    ]


@registry.register("power")
def _power_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    grad_a_h = upstream.h * b * (a ** (b - 1))
    grad_a_ah = upstream.ah * np.conj(b) * (np.conj(a) ** (np.conj(b) - 1))
    grad_b_h = upstream.h * result * np.log(a)
    grad_b_ah = upstream.ah * np.conj(result) * np.conj(np.log(a))

    return [
        Gradient(h=grad_a_h, ah=grad_a_ah),
        Gradient(h=grad_b_h, ah=grad_b_ah),
    ]


@registry.register("floor_divide")
def _floor_divide_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient | None]:
    # Hint: Floor division is non-differentiable.
    return [None, None]


@registry.register("remainder")  # or "mod"
def _remainder_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient | None]:
    # Hint: Remainder operation is non-differentiable.
    return [None, None]


@registry.register("reciprocal")
def _reciprocal_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (-1 / (a**2))
    grad_a_ah = upstream.ah * (-1 / (np.conj(a) ** 2))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("mean")
def _mean_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    n = a.size
    grad_h = upstream.h * np.ones_like(a) / n

    grad_ah = upstream.ah * np.ones_like(np.conj(a)) / n

    return [Gradient(h=grad_h, ah=grad_ah)]


@registry.register("matmul")
def _matmul_grad(
    upstream: Gradient, result: Tensor, A: Tensor, B: Tensor
) -> list[Gradient]:
    # Gradient with respect to A: dL/dA = dL/dZ @ B.T
    grad_A_h = upstream.h @ B.T
    grad_A_ah = upstream.ah @ np.conj(B.T)

    # Gradient with respect to B: dL/dB = A.T @ dL/dZ
    grad_B_h = A.T @ upstream.h
    grad_B_ah = np.conj(A.T) @ upstream.ah

    return [
        Gradient(h=grad_A_h, ah=grad_A_ah),
        Gradient(h=grad_B_h, ah=grad_B_ah),
    ]
