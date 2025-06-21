import numpy as np

from ....types import Tensor
from .util import complex_log, epsilon


class ArithmeticGradients:
    @staticmethod
    def add(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        a, b = inputs
        # ∂(a + b)/∂a = 1, ∂(a + b)/∂b = 1
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)
        return [(gh, gah), (gh, gah)]

    @staticmethod
    def subtract(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        a, b = inputs
        # ∂(a - b)/∂a = 1, ∂(a - b)/∂b = -1
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)
        return [(gh, gah), (-gh, -gah)]

    @staticmethod
    def multiply(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        a, b = inputs
        # Product rule in Wirtinger calc
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)

        # ∂(ab)/∂a = b, ∂(ab)/∂ā = b̄
        grad_a_h = gh * b + gah * np.conj(b)
        grad_a_ah = gah * b + gh * np.conj(b)
        # ∂(ab)/∂b = a, ∂(ab)/∂b̄ = ā
        grad_b_h = gh * a + gah * np.conj(a)
        grad_b_ah = gah * a + gh * np.conj(a)
        return [(grad_a_h, grad_a_ah), (grad_b_h, grad_b_ah)]

    @staticmethod
    def divide(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        a, b = inputs
        # Quotient rule in Wirtinger calc
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)

        b_conj = np.conj(b)
        denom = np.abs(b) ** 2 + epsilon

        # ∂(a/b)/∂a = 1/b, ∂/∂ā = 0
        grad_a_h = gh / b
        grad_a_ah = gah / b_conj
        # ∂(a/b)/∂b = -a/b², ∂/∂b̄ = -ā/|b|²
        grad_b_h = -gh * a / (b * b) - gah * np.conj(a) * b / (denom * b_conj)
        grad_b_ah = -gah * a / (b_conj * b_conj) - gh * np.conj(a) * b_conj / (
            denom * b
        )
        return [(grad_a_h, grad_a_ah), (grad_b_h, grad_b_ah)]

    @staticmethod
    def floor_divide(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        a, b = inputs
        # Not differentiable → zero gradient
        zero_a = (np.zeros_like(a), np.zeros_like(a))
        zero_b = (np.zeros_like(b), np.zeros_like(b))
        return [zero_a, zero_b]

    @staticmethod
    def remainder(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        a, b = inputs
        # ∂(a % b)/∂a = 1, ∂/∂b = -floor(a/b)
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)

        floor_div = np.floor(a / (b + epsilon))
        return [(gh, gah), (-gh * floor_div, -gah * floor_div)]

    @staticmethod
    def power(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        base, exp = inputs
        # ∂(base^exp) via Wirtinger
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)

        result = np.power(base, exp)
        safe_base = base + epsilon

        if np.iscomplexobj(base):
            if not np.iscomplexobj(exp):
                # complex base, real exponent
                grad_base_h = gh * exp * np.power(base, exp - 1)
                grad_base_ah = np.zeros_like(gah)
                grad_exp = gh * result * np.log(safe_base)
                return [(grad_base_h, grad_base_ah), grad_exp]
            else:
                # both complex
                logb = complex_log(base)
                grad_base_h = gh * exp * result / safe_base
                grad_base_ah = np.zeros_like(gah)
                grad_exp_h = gh * logb * result
                grad_exp_ah = np.zeros_like(gah)
                return [(grad_base_h, grad_base_ah), (grad_exp_h, grad_exp_ah)]

        # real base
        factor = exp * np.power(base, exp - 1)
        grad_base = gh * factor
        if np.iscomplexobj(exp):
            logb = np.log(safe_base)
            grad_exp_h = gh * result * logb
            grad_exp_ah = np.zeros_like(gah)
            return [grad_base, (grad_exp_h, grad_exp_ah)]
        else:
            grad_exp = gh * result * np.log(safe_base)
            return [grad_base, grad_exp]

    float_power = power

    @staticmethod
    def fmod(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        a, b = inputs
        # fmod: ∂/∂a = 1, ∂/∂b = -trunc(a/b)
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)

        with np.errstate(divide="ignore", invalid="ignore"):
            trunc_div = np.trunc(a / (b + epsilon))
        return [(gh, gah), (-gh * trunc_div, -gah * trunc_div)]

    @staticmethod
    def negative(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        # ∂(-x)/∂x = -1
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)
        return [(-gh, -gah)]

    @staticmethod
    def absolute(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        (z,) = inputs
        # ∂|z|/∂z = z̄/(2|z|), ∂/∂z̄ = z/(2|z|)
        if isinstance(grad_output, tuple):
            gh, gah = grad_output
        else:
            gh = grad_output
            gah = np.zeros_like(grad_output)

        if np.iscomplexobj(z):
            abs_z = np.abs(z) + epsilon
            grad_h = gh * np.conj(z) / (2 * abs_z) + gah * z / (2 * abs_z)
            grad_ah = gah * np.conj(z) / (2 * abs_z) + gh * z / (2 * abs_z)
            return [(grad_h, grad_ah)]
        else:
            sign = np.sign(z)
            return [(gh * sign, gah * sign)]
