from typing import Tuple
import numpy as np
from .util import ensure_shape

class RoundingGradients:
    @staticmethod
    def floor(
        grad_output: np.ndarray,
        inputs: Tuple[np.ndarray, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(ensure_shape(grad_h, np.shape(inp)), ensure_shape(grad_ah, np.shape(inp)))]

    @staticmethod
    def ceil(
        grad_output: np.ndarray,
        inputs: Tuple[np.ndarray, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(ensure_shape(grad_h, np.shape(inp)), ensure_shape(grad_ah, np.shape(inp)))]

    @staticmethod
    def round(
        grad_output: np.ndarray,
        inputs: Tuple[np.ndarray, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(ensure_shape(grad_h, np.shape(inp)), ensure_shape(grad_ah, np.shape(inp)))]

    @staticmethod
    def trunc(
        grad_output: np.ndarray,
        inputs: Tuple[np.ndarray, ...]
    ):
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(inp)

        grad_h = np.zeros_like(inp, dtype=grad_output_h.dtype)
        grad_ah = np.zeros_like(inp, dtype=grad_output_ah.dtype)

        return [(ensure_shape(grad_h, np.shape(inp)), ensure_shape(grad_ah, np.shape(inp)))]
