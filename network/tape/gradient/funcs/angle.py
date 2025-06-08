import numpy as np
from .util import epsilon

from ....types import Tensor


class AngleGradients:
    @staticmethod
    def degrees(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        scale = 180 / np.pi
        grad_h = grad_output_h * scale
        grad_ah = grad_output_ah * scale

        return [(grad_h, grad_ah)]

    @staticmethod
    def radians(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        scale = np.pi / 180
        grad_h = grad_output_h * scale
        grad_ah = grad_output_ah * scale

        return [(grad_h, grad_ah)]

    @staticmethod
    def deg2rad(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        scale = np.pi / 180
        grad_h = grad_output_h * scale
        grad_ah = grad_output_ah * scale

        return [(grad_h, grad_ah)]

    @staticmethod
    def rad2deg(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        scale = 180 / np.pi
        grad_h = grad_output_h * scale
        grad_ah = grad_output_ah * scale

        return [(grad_h, grad_ah)]

    @staticmethod
    def angle(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(x)

        abs_x_squared = np.conj(x) * x + epsilon  # avoid div by zero
        grad_h = grad_output_h * (1j * x / abs_x_squared)
        grad_ah = grad_output_ah * (-1j * np.conj(x) / abs_x_squared)

        return [(grad_h, grad_ah)]
