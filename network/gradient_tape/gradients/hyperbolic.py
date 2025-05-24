from typing import Tuple, Any
import numpy as np

class HyperbolicGradients:
    @staticmethod
    def sinh(grad_output: Any, inputs: Tuple[np.ndarray, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = grad_output_h * np.cosh(np.conjugate(inp))
        grad_ah = grad_output_ah * np.cosh(inp)

        return [(grad_h, grad_ah)]

    @staticmethod
    def cosh(grad_output: Any, inputs: Tuple[np.ndarray, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = grad_output_h * np.sinh(np.conjugate(inp))
        grad_ah = grad_output_ah * np.sinh(inp)

        return [(grad_h, grad_ah)]

    @staticmethod
    def tanh(grad_output: Any, inputs: Tuple[np.ndarray, ...]):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        tanh_conj = np.tanh(np.conjugate(inp))
        tanh_val = np.tanh(inp)

        grad_h = grad_output_h * (1 - tanh_conj ** 2)
        grad_ah = grad_output_ah * (1 - tanh_val ** 2)

        return [(grad_h, grad_ah)]

    @staticmethod
    def arcsinh(grad_output: Any, inputs: Tuple[np.ndarray, ...]):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        denom_h = np.sqrt(np.conjugate(x)**2 + 1)
        denom_ah = np.sqrt(x**2 + 1)

        grad_h = grad_output_h / denom_h
        grad_ah = grad_output_ah / denom_ah

        return [(grad_h, grad_ah)]

    @staticmethod
    def arccosh(grad_output: Any, inputs: Tuple[np.ndarray, ...]):
        x = inputs[0]

        if np.isrealobj(x) and np.any(np.abs(x) < 1):
            raise ValueError("Input to arccosh must be >= 1 for real values.")

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        denom_h = np.sqrt(np.conjugate(x)**2 - 1)
        denom_ah = np.sqrt(x**2 - 1)

        grad_h = grad_output_h / denom_h
        grad_ah = grad_output_ah / denom_ah

        return [(grad_h, grad_ah)]

    @staticmethod
    def arctanh(grad_output: Any, inputs: Tuple[np.ndarray, ...]):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        denom_h = 1 - np.conjugate(x)**2
        denom_ah = 1 - x**2

        grad_h = grad_output_h / denom_h
        grad_ah = grad_output_ah / denom_ah

        return [(grad_h, grad_ah)]
