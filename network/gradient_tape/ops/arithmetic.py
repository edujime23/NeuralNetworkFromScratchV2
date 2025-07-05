import numpy as np

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor
from .util import complex_log


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
    grad_b_h = upstream.h * result * complex_log(a)
    grad_b_ah = upstream.ah * np.conj(result) * np.conj(complex_log(a))

    return [
        Gradient(h=grad_a_h, ah=grad_a_ah),
        Gradient(h=grad_b_h, ah=grad_b_ah),
    ]


@registry.register("floor_divide")
def _floor_divide_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient | None]:
    return [None, None]


@registry.register("remainder")  # or "mod"
def _remainder_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient | None]:
    return [None, None]


@registry.register("reciprocal")
def _reciprocal_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (-1 / (a**2))
    grad_a_ah = upstream.ah * (-1 / (np.conj(a) ** 2))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]
