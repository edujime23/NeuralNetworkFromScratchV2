from network.gradient_tape.core.registry import registry
from network.gradient_tape.types import Gradient
from network.types.tensor import Tensor


@registry.register("reshape")
def _reshape_grad(
    upstream: Gradient, result: Tensor, a: Tensor, newshape: tuple
) -> list[Gradient]:
    grad_h = upstream.h.reshape(a.shape)
    grad_ah = upstream.ah.reshape(a.shape)
    return [Gradient(h=grad_h, ah=grad_ah)]
