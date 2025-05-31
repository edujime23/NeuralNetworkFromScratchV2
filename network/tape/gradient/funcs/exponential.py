import numpy as np

from ....types import Tensor


class ExponentialGradients:
    @staticmethod
    def exp(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = grad_output_h * np.exp(np.conjugate(inp))
        grad_ah = grad_output_ah * np.exp(inp)

        return [(grad_h, grad_ah)]

    @staticmethod
    def exp2(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = grad_output_h * np.log(2) * 2 ** np.conjugate(inp)
        grad_ah = grad_output_ah * np.log(2) * 2**inp

        return [(grad_h, grad_ah)]

    @staticmethod
    def expm1(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = grad_output_h * np.exp(np.conjugate(inp))
        grad_ah = grad_output_ah * np.exp(inp)

        return [(grad_h, grad_ah)]
