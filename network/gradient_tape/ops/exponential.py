import numpy as np

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


@registry.register("exp")
def _exp_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * result
    grad_a_ah = upstream.ah * np.conj(result)
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("expm1")
def _expm1_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (result + 1)
    grad_a_ah = upstream.ah * np.conj(result + 1)
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("exp2")
def _exp2_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * result * np.log(2)
    grad_a_ah = upstream.ah * np.conj(result) * np.log(2)
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("log")
def _log_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / a)
    grad_a_ah = upstream.ah * (1 / np.conj(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("log10")
def _log10_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / (a * np.log(10)))
    grad_a_ah = upstream.ah * (1 / (np.conj(a) * np.log(10)))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("log2")
def _log2_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / (a * np.log(2)))
    grad_a_ah = upstream.ah * (1 / (np.conj(a) * np.log(2)))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("log1p")
def _log1p_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / (1 + a))
    grad_a_ah = upstream.ah * (1 / (1 + np.conj(a)))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("logaddexp")
def _logaddexp_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    grad_a_h = upstream.h * np.exp(a - result)
    grad_a_ah = upstream.ah * np.conj(np.exp(a - result))

    grad_b_h = upstream.h * np.exp(b - result)
    grad_b_ah = upstream.ah * np.conj(np.exp(b - result))

    return [
        Gradient(h=grad_a_h, ah=grad_a_ah),
        Gradient(h=grad_b_h, ah=grad_b_ah),
    ]


@registry.register("logaddexp2")
def _logaddexp2_grad(
    upstream: Gradient, result: Tensor, a: Tensor, b: Tensor
) -> list[Gradient]:
    grad_a_h = upstream.h * (2 ** (a - result))
    grad_a_ah = upstream.ah * np.conj(2 ** (a - result))

    grad_b_h = upstream.h * (2 ** (b - result))
    grad_b_ah = upstream.ah * np.conj(2 ** (b - result))

    return [
        Gradient(h=grad_a_h, ah=grad_a_ah),
        Gradient(h=grad_b_h, ah=grad_b_ah),
    ]
