import numpy as np

from ....types import Tensor


class ExponentialGradients:
    @staticmethod
    def exp(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        """
        f(z) = exp(z) is holomorphic:
          ∂f/∂z = exp(z),   ∂f/∂z̄ = 0
        But f̄ = conj(exp(z)) depends on z̄ too:
          ∂f̄/∂z̄ = conj(exp(z))
        """
        (inp,) = inputs

        # unpack upstream Wirtinger pair
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(gh)

        # holomorphic path: gh * ∂f/∂z
        grad_h = gh * np.exp(inp)
        # anti-holomorphic path: gah * ∂f̄/∂z̄
        grad_ah = gah * np.conj(np.exp(inp))

        return [(grad_h, grad_ah)]

    @staticmethod
    def exp2(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        """
        f(z) = 2**z = exp(z ln 2):
          ∂f/∂z = ln2 * 2**z,   ∂f/∂z̄ = 0
          ∂f̄/∂z̄ = conj(ln2 * 2**z)
        """
        (inp,) = inputs

        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(gh)

        base = np.power(2, inp)
        ln2 = np.log(2)

        grad_h = gh * ln2 * base
        grad_ah = gah * np.conj(ln2 * base)

        return [(grad_h, grad_ah)]

    @staticmethod
    def expm1(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        # f(z) = exp(z) - 1 has the *same* derivatives as exp(z)
        return ExponentialGradients.exp(grad_output, inputs)
