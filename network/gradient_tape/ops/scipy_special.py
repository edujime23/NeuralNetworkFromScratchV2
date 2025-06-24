import numpy as np
from scipy.special import digamma

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


@registry.register("gamma")
def _gamma_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * result * digamma(a)
    grad_a_ah = upstream.ah * np.conj(result * digamma(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("loggamma")
def _loggamma_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * digamma(a)
    grad_a_ah = upstream.ah * np.conj(digamma(a))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("erf")
def _erf_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (2 / np.sqrt(np.pi)) * np.exp(-(a**2))
    grad_a_ah = upstream.ah * np.conj((2 / np.sqrt(np.pi)) * np.exp(-(a**2)))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]


@registry.register("erfc")
def _erfc_grad(upstream: Gradient, result: Tensor, a: Tensor) -> list[Gradient]:
    grad_a_h = upstream.h * (-2 / np.sqrt(np.pi)) * np.exp(-(a**2))
    grad_a_ah = upstream.ah * np.conj((-2 / np.sqrt(np.pi)) * np.exp(-(a**2)))
    return [Gradient(h=grad_a_h, ah=grad_a_ah)]
