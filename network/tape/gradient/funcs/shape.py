import numpy as np
from ....types import Tensor

class ShapeGradients:
    @staticmethod
    def reshape(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        grad_h = grad_output_h.reshape(np.shape(x))
        grad_ah = grad_output_ah.reshape(np.shape(x))

        return [(grad_h, grad_ah)]