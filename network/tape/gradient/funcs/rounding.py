import numpy as np

from ....types import Tensor


class RoundingGradients:
    @staticmethod
    def floor(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(grad_h, grad_ah)]

    @staticmethod
    def ceil(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(grad_h, grad_ah)]

    @staticmethod
    def round(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(grad_h, grad_ah)]

    @staticmethod
    def trunc(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(grad_h, grad_ah)]
