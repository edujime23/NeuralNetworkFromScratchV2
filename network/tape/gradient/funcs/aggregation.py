import warnings
from typing import Any

import numpy as np

from ....types import Tensor
from .util import epsilon


class AggregationGradients:
    @staticmethod
    def sum(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool | None = False,
    ) -> list[tuple[Tensor, Tensor]]:
        inp = inputs[0]
        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        grad_h = np.broadcast_to(grad_output_h, inp.shape)
        grad_ah = np.broadcast_to(grad_output_ah, inp.shape)

        return [(grad_h, grad_ah)]

    @staticmethod
    def mean(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        inp = inputs[0]
        shape = inp.shape if hasattr(inp, "shape") else ()

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        if axis is None:
            count = np.prod(shape) if shape else 1
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = np.prod([shape[a] for a in axes])
            if not keepdims:
                grad_output_h = np.expand_dims(grad_output_h, axes)
                grad_output_ah = np.expand_dims(grad_output_ah, axes)

        grad_h = grad_output_h / count
        grad_ah = grad_output_ah / count

        return [(grad_h, grad_ah)]

    @staticmethod
    def nanmean(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        x = inputs[0]
        mask = ~np.isnan(x)

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        if axis is not None:
            mask_sum = np.sum(mask, axis=axis, keepdims=keepdims)
        else:
            mask_sum = np.sum(mask)

        grad_h = grad_output_h * mask / (mask_sum + epsilon)
        grad_ah = (
            grad_output_ah * mask / (mask_sum + epsilon)
        )

        if np.iscomplexobj(x):
            grad_h = np.conj(grad_h)
            grad_ah = np.conj(grad_ah)

        return [(grad_h, grad_ah)]

    @staticmethod
    def prod(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool | None = False,
    ) -> list[tuple[Tensor, Tensor]]:
        inp = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        prod_conj_val = np.prod(np.conjugate(inp), axis=axis, keepdims=True)
        prod_conj_broadcasted = np.broadcast_to(prod_conj_val, inp.shape)

        grad_h = grad_output_h * prod_conj_broadcasted / (np.conjugate(inp) + epsilon)
        grad_ah = grad_output_ah * prod_conj_broadcasted / (np.conjugate(inp) + epsilon)

        return [(grad_h, grad_ah)]

    @staticmethod
    def max(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        inp = inputs[0]

        if np.iscomplexobj(inp):
            warnings.warn(
                "Gradient of max is not well-defined for complex inputs. Returning zero gradients.",
                stacklevel=2,
            )
            zero_h = np.zeros_like(
                inp,
                dtype=(
                    grad_output[0].dtype
                    if isinstance(grad_output, tuple)
                    else grad_output.dtype
                ),
            )
            zero_ah = np.zeros_like(inp, dtype=zero_h.dtype)
            return [(zero_h, zero_ah)]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        max_val = np.max(inp, axis=axis, keepdims=True)
        mask = inp == max_val
        num_max = np.sum(mask, axis=axis, keepdims=True)

        grad_h = grad_output_h * mask / (num_max + epsilon)
        grad_ah = grad_output_ah * mask / (num_max + epsilon)

        return [(grad_h, grad_ah)]

    @staticmethod
    def maximum(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, Tensor],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        a, b = inputs

        if np.iscomplexobj(a) or np.iscomplexobj(b):
            zero_a = np.zeros_like(
                a,
                dtype=(
                    grad_output[0].dtype
                    if isinstance(grad_output, tuple)
                    else grad_output.dtype
                ),
            )
            zero_b = np.zeros_like(b, dtype=zero_a.dtype)
            zero_ah_a = np.zeros_like(zero_a)
            zero_ah_b = np.zeros_like(zero_b)
            return [(zero_a, zero_ah_a), (zero_b, zero_ah_b)]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        if axis is None:
            grad_a_h = grad_output_h * (a >= b)
            grad_b_h = grad_output_h * (b > a)
            grad_a_ah = grad_output_ah * (a >= b)
            grad_b_ah = grad_output_ah * (b > a)
        else:
            max_a = a >= b
            max_b = b > a

            grad_a_h = np.where(max_a, grad_output_h, 0)
            grad_b_h = np.where(max_b, grad_output_h, 0)
            grad_a_ah = np.where(max_a, grad_output_ah, 0)
            grad_b_ah = np.where(max_b, grad_output_ah, 0)

            if keepdims:
                grad_a_h = np.sum(grad_a_h, axis=axis, keepdims=True)
                grad_b_h = np.sum(grad_b_h, axis=axis, keepdims=True)
                grad_a_ah = np.sum(grad_a_ah, axis=axis, keepdims=True)
                grad_b_ah = np.sum(grad_b_ah, axis=axis, keepdims=True)
            else:
                grad_a_h = np.sum(grad_a_h, axis=axis)
                grad_b_h = np.sum(grad_b_h, axis=axis)
                grad_a_ah = np.sum(grad_a_ah, axis=axis)
                grad_b_ah = np.sum(grad_b_ah, axis=axis)

        return [(grad_a_h, grad_a_ah), (grad_b_h, grad_b_ah)]

    @staticmethod
    def min(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        inp = inputs[0]

        if np.iscomplexobj(inp):
            warnings.warn(
                "Gradient of min is not well-defined for complex inputs. Returning zero gradients.",
                stacklevel=2,
            )
            zero_h = np.zeros_like(
                inp,
                dtype=(
                    grad_output[0].dtype
                    if isinstance(grad_output, tuple)
                    else grad_output.dtype
                ),
            )
            zero_ah = np.zeros_like(inp, dtype=zero_h.dtype)
            return [(zero_h, zero_ah)]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        min_val = np.min(inp, axis=axis, keepdims=True)
        mask = inp == min_val
        num_min = np.sum(mask, axis=axis, keepdims=True)

        grad_h = grad_output_h * mask / (num_min + epsilon)
        grad_ah = grad_output_ah * mask / (num_min + epsilon)

        return [(grad_h, grad_ah)]

    @staticmethod
    def minimum(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, Tensor],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        a, b = inputs

        if np.iscomplexobj(a) or np.iscomplexobj(b):
            zero_a = np.zeros_like(
                a,
                dtype=(
                    grad_output[0].dtype
                    if isinstance(grad_output, tuple)
                    else grad_output.dtype
                ),
            )
            zero_b = np.zeros_like(b, dtype=zero_a.dtype)
            zero_ah_a = np.zeros_like(zero_a)
            zero_ah_b = np.zeros_like(zero_b)
            return [(zero_a, zero_ah_a), (zero_b, zero_ah_b)]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        if axis is None:
            grad_a_h = grad_output_h * (a <= b)
            grad_b_h = grad_output_h * (b < a)
            grad_a_ah = grad_output_ah * (a <= b)
            grad_b_ah = grad_output_ah * (b < a)
        else:
            min_a = a <= b
            min_b = b < a

            grad_a_h = np.where(min_a, grad_output_h, 0)
            grad_b_h = np.where(min_b, grad_output_h, 0)
            grad_a_ah = np.where(min_a, grad_output_ah, 0)
            grad_b_ah = np.where(min_b, grad_output_ah, 0)

            if keepdims:
                grad_a_h = np.sum(grad_a_h, axis=axis, keepdims=True)
                grad_b_h = np.sum(grad_b_h, axis=axis, keepdims=True)
                grad_a_ah = np.sum(grad_a_ah, axis=axis, keepdims=True)
                grad_b_ah = np.sum(grad_b_ah, axis=axis, keepdims=True)
            else:
                grad_a_h = np.sum(grad_a_h, axis=axis)
                grad_b_h = np.sum(grad_b_h, axis=axis)
                grad_a_ah = np.sum(grad_a_ah, axis=axis)
                grad_b_ah = np.sum(grad_b_ah, axis=axis)

        return [(grad_a_h, grad_a_ah), (grad_b_h, grad_b_ah)]

    @staticmethod
    def std(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        x = inputs[0]
        n = x.size if axis is None else x.shape[axis]
        xm = np.mean(x, axis=axis, keepdims=keepdims)
        std_x = np.std(x, axis=axis, keepdims=keepdims)
        std_x_safe = std_x + epsilon

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        grad_h = grad_output_h * (x - xm) / (std_x_safe * n)
        grad_ah = grad_output_ah * (x - xm) / (std_x_safe * n)

        if np.iscomplexobj(x):
            grad_h = np.conj(grad_h)
            grad_ah = np.conj(grad_ah)

        return [(grad_h, grad_ah)]

    @staticmethod
    def nanstd(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        x = inputs[0]
        mask = ~np.isnan(x)

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        if axis is not None:
            mask_sum = np.sum(mask, axis=axis, keepdims=keepdims)
            xm = np.nansum(x, axis=axis, keepdims=keepdims) / mask_sum
            std_x = np.nanstd(x, axis=axis, keepdims=keepdims)
        else:
            mask_sum = np.sum(mask)
            xm = np.nansum(x) / mask_sum
            std_x = np.nanstd(x)

        std_x_safe = std_x + epsilon

        grad_h = grad_output_h * (x - xm) * mask / (mask_sum * std_x_safe)
        grad_ah = grad_output_ah * (x - xm) * mask / (mask_sum * std_x_safe)

        if np.iscomplexobj(x):
            grad_h = np.conj(grad_h)
            grad_ah = np.conj(grad_ah)

        return [(grad_h, grad_ah)]

    @staticmethod
    def var(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        x = inputs[0]
        n = x.size if axis is None else x.shape[axis]
        xm = np.mean(x, axis=axis, keepdims=keepdims)

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        grad_h = grad_output_h * 2 * (x - xm) / n
        grad_ah = grad_output_ah * 2 * (x - xm) / n

        if np.iscomplexobj(x):
            grad_h = np.conj(grad_h)
            grad_ah = np.conj(grad_ah)

        return [(grad_h, grad_ah)]

    @staticmethod
    def nanvar(
        grad_output: Tensor | tuple[Tensor, Tensor],
        inputs: tuple[Tensor, ...],
        **kwargs: dict[str, Any]
    ) -> list[tuple[Tensor, Tensor]]:
        x = inputs[0]
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        mask = ~np.isnan(x)

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output_h)

        n = np.sum(mask, axis=axis, keepdims=bool(keepdims))
        xm = np.nansum(x, axis=axis, keepdims=bool(keepdims)) / n

        grad_h = grad_output_h * 2 * (x - xm) * mask / n
        grad_ah = grad_output_ah * 2 * (x - xm) * mask / n

        if np.iscomplexobj(x):
            grad_h = np.conj(grad_h)
            grad_ah = np.conj(grad_ah)

        return [(grad_h, grad_ah)]
