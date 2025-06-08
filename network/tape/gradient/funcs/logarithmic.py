import warnings
import numpy as np
from ....types import Tensor
from .util import epsilon

class LogarithmicGradients:
    @staticmethod
    def log(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_inp_h = grad_output_h / (np.conjugate(inp) + epsilon)
        grad_inp_ah = grad_output_ah / (inp + epsilon)

        return [(grad_inp_h, grad_inp_ah)]

    @staticmethod
    def log2(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        denom_h = np.conjugate(inp) * np.log(2) + epsilon
        denom_ah = inp * np.log(2) + epsilon

        grad_inp_h = grad_output_h / denom_h
        grad_inp_ah = grad_output_ah / denom_ah

        return [(grad_inp_h, grad_inp_ah)]

    @staticmethod
    def log10(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        denom_h = np.conjugate(inp) * np.log(10) + epsilon
        denom_ah = inp * np.log(10) + epsilon

        grad_inp_h = grad_output_h / denom_h
        grad_inp_ah = grad_output_ah / denom_ah

        return [(grad_inp_h, grad_inp_ah)]

    @staticmethod
    def log1p(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_inp_h = grad_output_h / (np.conjugate(inp) + 1)
        grad_inp_ah = grad_output_ah / (inp + 1)

        return [(grad_inp_h, grad_inp_ah)]

    @staticmethod
    def logaddexp(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        a, b = inputs

        dtype_h = np.result_type(grad_output[0]) if isinstance(grad_output, tuple) else np.result_type(grad_output)
        dtype_ah = np.result_type(grad_output[1]) if isinstance(grad_output, tuple) else dtype_h

        if np.iscomplexobj(a) or np.iscomplexobj(b):
            warnings.warn("Gradient of logaddexp is not well-defined for complex inputs. Returning zero gradients.", stacklevel=2)
            zero_a = np.zeros_like(a, dtype=dtype_h)
            zero_b = np.zeros_like(b, dtype=dtype_h)
            if not isinstance(grad_output, tuple):
                return [zero_a, zero_b]

            zero_a_ah = np.zeros_like(a, dtype=dtype_ah)
            zero_b_ah = np.zeros_like(b, dtype=dtype_ah)
            return [(zero_a, zero_a_ah), (zero_b, zero_b_ah)]

        exp_a, exp_b = np.exp(a), np.exp(b)
        denom = exp_a + exp_b + epsilon

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
            zero_a_ah = np.zeros_like(a)
            zero_b_ah = np.zeros_like(b)
            grad_a_h = grad_output_h * exp_a / denom
            grad_b_h = grad_output_h * exp_b / denom

            return [
                (grad_a_h, zero_a_ah),
                (grad_b_h, zero_b_ah)
            ]
        else:
            grad_a = grad_output * exp_a / denom
            grad_b = grad_output * exp_b / denom
            return [grad_a, grad_b]

    @staticmethod
    def logaddexp2(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...]
    ):
        a, b = inputs

        dtype_h = np.result_type(grad_output[0]) if isinstance(grad_output, tuple) else np.result_type(grad_output)
        dtype_ah = np.result_type(grad_output[1]) if isinstance(grad_output, tuple) else dtype_h

        if np.iscomplexobj(a) or np.iscomplexobj(b):
            warnings.warn("Gradient of logaddexp2 is not well-defined for complex inputs. Returning zero gradients.", stacklevel=2)
            zero_a = np.zeros_like(a, dtype=dtype_h)
            zero_b = np.zeros_like(b, dtype=dtype_h)
            if not isinstance(grad_output, tuple):
                return [zero_a, zero_b]

            zero_a_ah = np.zeros_like(a, dtype=dtype_ah)
            zero_b_ah = np.zeros_like(b, dtype=dtype_ah)
            return [(zero_a, zero_a_ah), (zero_b, zero_b_ah)]

        exp2_a, exp2_b = 2 ** a, 2 ** b
        denom = exp2_a + exp2_b + epsilon

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
            zero_a_ah = np.zeros_like(a)
            zero_b_ah = np.zeros_like(b)

            grad_a_h = grad_output_h * exp2_a / denom * np.log(2)
            grad_b_h = grad_output_h * exp2_b / denom * np.log(2)

            return [
                (grad_a_h, zero_a_ah),
                (grad_b_h, zero_b_ah)
            ]
        else:
            grad_a = grad_output * exp2_a / denom * np.log(2)
            grad_b = grad_output * exp2_b / denom * np.log(2)
            return [grad_a, grad_b]
