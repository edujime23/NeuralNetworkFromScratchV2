import numpy as np

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


@registry.register("fabs")
@registry.register("abs")
@registry.register("absolute")
def _fabs_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    a_abs = np.abs(a)
    grad_a_h = np.where(a_abs != 0, upstream.h * (np.conj(a) / (2 * a_abs + 1e-8)), 0.0)
    grad_a_ah = np.where(a_abs != 0, upstream.ah * (a / (2 * a_abs + 1e-8)), 0.0)
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("sqrt")
def _sqrt_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = np.where(result != 0, upstream.h * (0.5 / result), 0.0)
    grad_a_ah = np.where(
        np.conj(result) != 0, upstream.ah * (0.5 / np.conj(result)), 0.0
    )
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("cbrt")
def _cbrt_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = np.where(a != 0, upstream.h * (1 / 3) * (a ** (-2 / 3)), 0.0)
    grad_a_ah = np.where(
        np.conj(a) != 0, upstream.ah * (1 / 3) * (np.conj(a) ** (-2 / 3)), 0.0
    )
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("ldexp")
def _ldexp_grad(
    upstream: Gradient, result: Tensor, x1: Tensor, x2: Tensor
) -> list[Gradient]:
    grad_x1_h = upstream.h * np.exp2(x2)
    grad_x1_ah = upstream.ah * np.conj(np.exp2(x2))

    grad_x2_h = upstream.h * result * np.log(2)
    grad_x2_ah = upstream.ah * np.conj(result) * np.log(2)

    return [
        Gradient(h=grad_x1_h, ah=grad_x1_ah),
        Gradient(h=grad_x2_h, ah=grad_x2_ah),
    ]
