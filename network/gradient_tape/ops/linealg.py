import numpy as np

from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


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
