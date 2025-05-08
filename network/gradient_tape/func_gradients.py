import numpy as np
from typing import Tuple
import warnings

class FuncGradients:
    @staticmethod
    def ensure_shape(x, shape):
        x_shape = np.shape(x)
        if shape == () and x_shape != ():

            return np.sum(x)
        elif x_shape != shape:
            if axes_to_sum := tuple(
                i
                for i in range(len(x_shape))
                if i >= len(shape) or x_shape[i] != shape[i]
            ):
                x = np.sum(x, axis=axes_to_sum, keepdims=True)

            reshape_dims = tuple(d if d != 1 else -1 for d in shape)
            x = x.reshape(reshape_dims)

            if squeeze_axes := tuple(
                i
                for i in range(len(x.shape))
                if i < len(shape) and shape[i] == 1 and x.shape[i] > 1
            ):
                x = np.squeeze(x, axis=squeeze_axes)

            return np.broadcast_to(x, shape)

        return x

    @staticmethod
    def add(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):


        a, b = inputs
        grad_a = grad_output
        grad_b = grad_output

        return [FuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def sum(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike], axis=None, keepdims=False):

        inp = inputs[0]
        shape = inp.shape if hasattr(inp, 'shape') else ()
        grad = np.broadcast_to(grad_output, shape)
        return [grad]

    @staticmethod
    def subtract(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        a, b = inputs
        grad_a = grad_output
        grad_b = -grad_output

        return [FuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def multiply(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):

        a, b = inputs
        grad_a = grad_output * np.conjugate(b)
        grad_b = grad_output * np.conjugate(a)

        return [FuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def matmul(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Complex gradient for matmul might not align with standard 2*dL/d(conj(X)) definition.")
        a, b = inputs

        a_val = np.array(a)
        b_val = np.array(b)
        grad_out_val = np.array(grad_output)

        a_orig_shape = a.shape if hasattr(a, 'shape') else ()
        b_orig_shape = b.shape if hasattr(b, 'shape') else ()

        reshape_a = False
        if a_val.ndim == 1:
            a_val = a_val.reshape(1, -1)

            if grad_out_val.ndim == 1:
                   grad_out_val = grad_out_val.reshape(-1, 1)
            elif grad_out_val.ndim > 1:
                   grad_out_val = np.expand_dims(grad_out_val, axis=-2)
            reshape_a = True

        reshape_b = False
        if b_val.ndim == 1:
            b_val = b_val.reshape(-1, 1)

            if grad_out_val.ndim == 1:
                   grad_out_val = grad_out_val.reshape(1, -1)
            elif grad_out_val.ndim > 1:
                   grad_out_val = np.expand_dims(grad_out_val, axis=-1)
            reshape_b = True

        if np.iscomplexobj(a_val) or np.iscomplexobj(b_val) or np.iscomplexobj(grad_out_val):
    
             grad_a = np.matmul(grad_out_val, np.conjugate(np.swapaxes(b_val, -1, -2)))
             grad_b = np.matmul(np.conjugate(np.swapaxes(a_val, -1, -2)), grad_out_val)
        else:

            grad_a = np.matmul(grad_out_val, np.swapaxes(b_val, -1, -2))
            grad_b = np.matmul(np.swapaxes(a_val, -1, -2), grad_out_val)

        if reshape_a:
            grad_a = np.squeeze(grad_a, axis=-2)
        if reshape_b:
             grad_b = np.squeeze(grad_b, axis=-1)

        return [FuncGradients.ensure_shape(grad_a, a_orig_shape),
                FuncGradients.ensure_shape(grad_b, b_orig_shape)]


    @staticmethod
    def divide(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        a, b = inputs
        grad_a = grad_output / np.conjugate(b)
        grad_b = -grad_output * np.conjugate(a) / (np.conjugate(b) * np.conjugate(b))

        return [FuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def floor_divide(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of floor_divide is zero almost everywhere and undefined at discontinuities.")
        a, b = inputs
        return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def power(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Complex gradient for power might not align with standard 2*dL/d(conj(X)) definition.")
        base, exp = inputs
        base_val = base if hasattr(base, 'shape') else np.array(base)
        exp_val = exp if hasattr(exp, 'shape') else np.array(exp)
        base_shape = base.shape if hasattr(base, 'shape') else ()
        exp_shape = exp.shape if hasattr(exp, 'shape') else ()

        grad_base = grad_output * exp_val * np.power(base_val, exp_val - 1)
        grad_exp = grad_output * np.log(base_val + np.finfo(inputs[0].dtype).eps) * np.power(base_val, exp_val)

        if exp_shape == ():
             grad_exp = np.sum(grad_exp)
        if base_shape == ():
             grad_base = np.sum(grad_base)

        return [FuncGradients.ensure_shape(grad_base, base_shape), FuncGradients.ensure_shape(grad_exp, exp_shape)]

    @staticmethod
    def square(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp, = inputs
        grad_inp = grad_output * 2 * np.conjugate(inp)

        if hasattr(inp, 'shape') and inp.shape == ():
            grad_inp = np.sum(grad_inp)
        return [FuncGradients.ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def sin(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * np.cos(np.conjugate(inp)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def cos(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(-grad_output * np.sin(np.conjugate(inp)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def exp(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * np.exp(np.conjugate(inp)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def log(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output / (np.conjugate(inp) + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def negative(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(-grad_output, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def maximum(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        a, b = inputs
        if np.iscomplexobj(a) or np.iscomplexobj(b):
             warnings.warn("Gradient of maximum is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

        grad_a = grad_output * (a >= b)
        grad_b = grad_output * (b > a)
        return [FuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def minimum(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        a, b = inputs
        if np.iscomplexobj(a) or np.iscomplexobj(b):
             warnings.warn("Gradient of minimum is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

        grad_a = grad_output * (a <= b)
        grad_b = grad_output * (b < a)
        return [FuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def tanh(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        tanh_conj_inp = np.tanh(np.conjugate(inp))
        return [FuncGradients.ensure_shape(grad_output * (1 - tanh_conj_inp ** 2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def absolute(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):

        inp = inputs[0]
        inp_shape = inp.shape if hasattr(inp, 'shape') else ()

        if np.iscomplexobj(inp):
            abs_inp = np.abs(inp)

            grad_inp = grad_output * inp / (abs_inp + np.finfo(inputs[0].dtype).eps)
        else:

            grad_inp = grad_output * np.sign(inp)

        return [FuncGradients.ensure_shape(grad_inp, inp_shape)]

    @staticmethod
    def floor(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of floor is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [FuncGradients.ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def ceil(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of ceil is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [FuncGradients.ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def round(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of round is zero almost everywhere and undefined at .5 boundaries.")
        inp = inputs[0]
        return [FuncGradients.ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def mod(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of mod is not well defined at discontinuities.")
        a, b = inputs
        return [FuncGradients.ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), a.shape if hasattr(a, 'shape') else ()),
                np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def remainder(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of remainder is not well defined at discontinuities.")
        a, b = inputs
        return [FuncGradients.ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), a.shape if hasattr(a, 'shape') else ()),
                np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def fmod(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of fmod is not well defined at discontinuities.")
        a, b = inputs
        return [FuncGradients.ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), a.shape if hasattr(a, 'shape') else ()),
                np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def clip(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp, min_val, max_val = inputs
        if np.iscomplexobj(inp) or np.iscomplexobj(min_val) or np.iscomplexobj(max_val):
             warnings.warn("Gradient of clip is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(inp, dtype=grad_output.dtype),
                     np.zeros_like(min_val, dtype=grad_output.dtype),
                     np.zeros_like(max_val, dtype=grad_output.dtype)]

        mask = (inp >= min_val) & (inp <= max_val)
        return [FuncGradients.ensure_shape(grad_output * mask, inp.shape if hasattr(inp, 'shape') else ()),
                np.zeros_like(min_val, dtype=grad_output.dtype),
                np.zeros_like(max_val, dtype=grad_output.dtype)]

    @staticmethod
    def arcsin(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output
                / np.sqrt(1 - np.conjugate(inp) ** 2 + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def arccos(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                -grad_output
                / np.sqrt(1 - np.conjugate(inp) ** 2 + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def arctan(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output / (1 + np.conjugate(inp)**2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def sqrt(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output
                / (2 * np.sqrt(np.conjugate(inp) + np.finfo(inp.dtype).eps)),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def sinh(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * np.cosh(np.conjugate(inp)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def cosh(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * np.sinh(np.conjugate(inp)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def tan(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output
                / (np.cos(np.conjugate(inp)) ** 2 + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def log1p(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output / (np.conjugate(inp) + 1 + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def expm1(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * np.exp(np.conjugate(inp)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def reciprocal(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                -grad_output / (np.conjugate(inp) ** 2 + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def log2(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output
                / (np.conjugate(inp) * np.log(2) + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def log10(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output
                / (np.conjugate(inp) * np.log(10) + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def arctan2(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        y, x = inputs
        if np.iscomplexobj(y) or np.iscomplexobj(x):
             warnings.warn("Gradient of arctan2 is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(y, dtype=grad_output.dtype), np.zeros_like(x, dtype=grad_output.dtype)]

        denom = x ** 2 + y ** 2 + np.finfo(inputs[0].dtype).eps
        grad_y = grad_output * x / denom
        grad_x = -grad_output * y / denom
        return [FuncGradients.ensure_shape(grad_y, y.shape if hasattr(y, 'shape') else ()),
                FuncGradients.ensure_shape(grad_x, x.shape if hasattr(x, 'shape') else ())]

    @staticmethod
    def hypot(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        x, y = inputs
        if np.iscomplexobj(x) or np.iscomplexobj(y):
             warnings.warn("Gradient of hypot is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(x, dtype=grad_output.dtype), np.zeros_like(y, dtype=grad_output.dtype)]

        denom = np.sqrt(x ** 2 + y ** 2 + np.finfo(inputs[0].dtype).eps)
        grad_x = grad_output * x / denom
        grad_y = grad_output * y / denom
        return [FuncGradients.ensure_shape(grad_x, x.shape if hasattr(x, 'shape') else ()),
                FuncGradients.ensure_shape(grad_y, y.shape if hasattr(y, 'shape') else ())]

    @staticmethod
    def mean(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike], axis=None, keepdims=False):
        inp = inputs[0]
        if hasattr(inp, 'shape'):
            shape = inp.shape

            if axis is None:
                count = np.prod(np.array(shape))
            else:
                axes_to_mean = (axis,) if isinstance(axis, int) else tuple(axis)
                count = np.prod([shape[a] for a in axes_to_mean])


            grad = np.broadcast_to(grad_output / count, shape)
        else:

            grad = grad_output
        return [grad]

    @staticmethod
    def prod(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike], axis=None, keepdims=False):
        inp = inputs[0]
        prod_conj_val = np.prod(np.conjugate(inp), axis=axis, keepdims=True)
        prod_conj_broadcasted = np.broadcast_to(prod_conj_val, inp.shape)
        grad = grad_output * prod_conj_broadcasted / (np.conjugate(inp) + np.finfo(inputs[0].dtype).eps)

        return [FuncGradients.ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]


    @staticmethod
    def max(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike], axis=None, keepdims=False):
        inp = inputs[0]
        if np.iscomplexobj(inp):
             warnings.warn("Gradient of max is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(inp, dtype=grad_output.dtype)]

        max_val = np.max(inp, axis=axis, keepdims=True)
        mask = (inp == max_val)
        num_max = np.sum(mask, axis=axis, keepdims=True)
        grad_output_broadcasted = np.broadcast_to(grad_output, num_max.shape)
        grad = grad_output_broadcasted * mask / (num_max + np.finfo(inputs[0].dtype).eps)

        return [FuncGradients.ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def min(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike], axis=None, keepdims=False):
        inp = inputs[0]
        if np.iscomplexobj(inp):
             warnings.warn("Gradient of min is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(inp, dtype=grad_output.dtype)]

        min_val = np.min(inp, axis=axis, keepdims=True)
        mask = (inp == min_val)
        num_min = np.sum(mask, axis=axis, keepdims=True)
        grad_output_broadcasted = np.broadcast_to(grad_output, num_min.shape)
        grad = grad_output_broadcasted * mask / (num_min + np.finfo(inputs[0].dtype).eps)

        return [FuncGradients.ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def erf(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def erfc(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(-grad_output * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def exp2(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [FuncGradients.ensure_shape(grad_output * np.log(2) * 2**np.conjugate(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def logaddexp(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        a, b = inputs
        if np.iscomplexobj(a) or np.iscomplexobj(b):
             warnings.warn("Gradient of logaddexp is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

        exp_a, exp_b = np.exp(a), np.exp(b)
        denom = exp_a + exp_b + np.finfo(inputs[0].dtype).eps
        return [FuncGradients.ensure_shape(grad_output * exp_a / denom, a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_output * exp_b / denom, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def logaddexp2(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        a, b = inputs
        if np.iscomplexobj(a) or np.iscomplexobj(b):
             warnings.warn("Gradient of logaddexp2 is not well-defined for complex inputs. Returning zero gradients.")
             return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

        exp2_a, exp2_b = 2**a, 2**b
        denom = exp2_a + exp2_b + np.finfo(inputs[0].dtype).eps
        return [FuncGradients.ensure_shape(grad_output * exp2_a / denom * np.log(2), a.shape if hasattr(a, 'shape') else ()),
                FuncGradients.ensure_shape(grad_output * exp2_b / denom * np.log(2), b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def cbrt(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        inp = inputs[0]
        return [
            FuncGradients.ensure_shape(
                grad_output
                / (3 * np.cbrt(np.conjugate(inp) ** 2 + np.finfo(inp.dtype).eps)),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def deg2rad(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        return [FuncGradients.ensure_shape(grad_output * np.pi / 180, inputs[0].shape if hasattr(inputs[0], 'shape') else ())]

    @staticmethod
    def rad2deg(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        return [FuncGradients.ensure_shape(grad_output * 180 / np.pi, inputs[0].shape if hasattr(inputs[0], 'shape') else ())]

    @staticmethod
    def heaviside(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Heaviside has undefined derivative at zero.")
        return [np.zeros_like(inputs[0], dtype=grad_output.dtype), np.zeros_like(inputs[1], dtype=grad_output.dtype)]

    @staticmethod
    def sign(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of sign is zero almost everywhere and undefined at zero.")
        inp = inputs[0]
        return [FuncGradients.ensure_shape(np.zeros_like(inp, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def conjugate(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        return [np.conjugate(grad_output)]

    @staticmethod
    def real(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        return [grad_output * 0.5]

    @staticmethod
    def imag(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        return [grad_output * 0.5j]

    @staticmethod
    def bitwise_and(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of bitwise_and is zero almost everywhere.")
        a, b = inputs
        return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def bitwise_or(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of bitwise_or is zero almost everywhere.")
        a, b = inputs
        return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def bitwise_xor(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of bitwise_xor is zero almost everywhere.")
        a, b = inputs
        return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

    @staticmethod
    def invert(grad_output: np.typing.ArrayLike, inputs: Tuple[np.typing.ArrayLike]):
        warnings.warn("Gradient of invert is zero almost everywhere.")
        inp = inputs[0]
        return [np.zeros_like(inp, dtype=grad_output.dtype)]