import numpy as np

from ....types import Tensor


class ShapeGradients:
    @staticmethod
    def reshape(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        newshape=None,
        order="C",
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        # Reshape gradients back to original tensor shape
        grad_h = grad_output_h.reshape(x.shape)
        grad_ah = grad_output_ah.reshape(x.shape)

        return [(grad_h, grad_ah)]
