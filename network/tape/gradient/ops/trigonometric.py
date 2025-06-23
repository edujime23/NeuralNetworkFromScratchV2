import numpy as np

from ....types.tensor import Tensor
from ..core.registry import registry
from ..types import Gradient


@registry.register("sin")
def _sin_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * np.cos(a)
    grad_a_ah = upstream.ah * np.conj(np.cos(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("cos")
def _cos_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (-np.sin(a))
    grad_a_ah = upstream.ah * np.conj(-np.sin(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("tan")
def _tan_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / np.cos(a)**2)
    grad_a_ah = upstream.ah * np.conj(1 / np.cos(a)**2)
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arcsin")
def _arcsin_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / np.sqrt(1 - a**2))
    grad_a_ah = upstream.ah * np.conj(1 / np.sqrt(1 - a**2))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arccos")
def _arccos_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (-1 / np.sqrt(1 - a**2))
    grad_a_ah = upstream.ah * np.conj(-1 / np.sqrt(1 - a**2))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arctan")
def _arctan_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / (1 + a**2))
    grad_a_ah = upstream.ah * np.conj(1 / (1 + a**2))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arctan2")
def _arctan2_grad(upstream: Gradient, result: Tensor, y: Tensor, x: Tensor) -> list[Gradient]:
    # Hint: arctan2 is a real-valued function, NumPy's implementation expects real inputs.
    # Gradients propagate through real parts of inputs if Tensor abstraction handles this.
    denom = x**2 + y**2

    grad_y_h = upstream.h * (x / denom)
    grad_y_ah = upstream.ah * (x / denom)

    grad_x_h = upstream.h * (-y / denom)
    grad_x_ah = upstream.ah * (-y / denom)

    return [
        Gradient(h=grad_y_h, ah=grad_y_ah),
        Gradient(h=grad_x_h, ah=grad_x_ah),
    ]


@registry.register("hypot")
def _hypot_grad(upstream: Gradient, result: Tensor, x1: Tensor, x2: Tensor) -> list[Gradient]:
    # Hint: hypot is a real-valued function, NumPy's implementation expects real inputs.
    grad_x1_h = upstream.h * (x1 / result)
    grad_x1_ah = upstream.ah * (x1 / result)

    grad_x2_h = upstream.h * (x2 / result)
    grad_x2_ah = upstream.ah * (x2 / result)

    return [
        Gradient(h=grad_x1_h, ah=grad_x1_ah),
        Gradient(h=grad_x2_h, ah=grad_x2_ah),
    ]


@registry.register("degrees")
def _degrees_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    scale_factor = 180 / np.pi
    return [
        Gradient(h=upstream.h * scale_factor, ah=upstream.ah * scale_factor)
    ]


@registry.register("radians")
def _radians_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    scale_factor = np.pi / 180
    return [
        Gradient(h=upstream.h * scale_factor, ah=upstream.ah * scale_factor)
    ]