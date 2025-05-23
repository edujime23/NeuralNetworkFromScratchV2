from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class ShapeGradients:
    @staticmethod
    def reshape(
        grad_output: np.ndarray,
        inputs: Tuple[np.ndarray, ...]
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        grad_h = grad_output_h.reshape(np.shape(x))
        grad_ah = grad_output_ah.reshape(np.shape(x))

        return [(ensure_shape(grad_h, np.shape(x)), ensure_shape(grad_ah, np.shape(x)))]