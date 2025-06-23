import numpy as np

from ....types.tensor import Tensor
from ..core.registry import registry
from ..types import Gradient


@registry.register("sinh")
def _sinh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * np.cosh(a)
    grad_a_ah = upstream.ah * np.conj(np.cosh(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("cosh")
def _cosh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * np.sinh(a)
    grad_a_ah = upstream.ah * np.conj(np.sinh(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("tanh")
def _tanh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / np.cosh(a)**2)
    grad_a_ah = upstream.ah * np.conj(1 / np.cosh(a)**2)
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arcsinh")
def _arcsinh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / np.sqrt(a**2 + 1))
    grad_a_ah = upstream.ah * np.conj(1 / np.sqrt(a**2 + 1))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arccosh")
def _arccosh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / np.sqrt(a**2 - 1))
    grad_a_ah = upstream.ah * np.conj(1 / np.sqrt(a**2 - 1))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("arctanh")
def _arctanh_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (1 / (1 - a**2))
    grad_a_ah = upstream.ah * np.conj(1 / (1 - a**2))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]