import numpy as np
import warnings
class UfuncGradients:
    @staticmethod
    def ensure_shape(x, shape):
        x_shape = np.shape(x)
        if shape == () and x_shape != ():
            return np.sum(x)
        elif x_shape != shape:
            return np.broadcast_to(x, shape)
        return x

    @staticmethod
    def add(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output
        grad_b = grad_output
        
        if hasattr(a, 'shape') and a.shape == ():
            grad_a = np.sum(grad_a)
        if hasattr(b, 'shape') and b.shape == ():
            grad_b = np.sum(grad_b)

        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def sum(grad_output, inputs, axis=None, keepdims=False):
        inp = inputs[0]
        shape = inp.shape if hasattr(inp, 'shape') else ()
        grad = np.broadcast_to(grad_output, shape)
        return [grad]



    @staticmethod
    def subtract(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output
        grad_b = -grad_output
        if hasattr(a, 'shape') and a.shape == ():
            grad_a = np.sum(grad_a)
        if hasattr(b, 'shape') and b.shape == ():
            grad_b = np.sum(grad_b)
        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def multiply(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output * b
        grad_b = grad_output * a
        if hasattr(a, 'shape') and a.shape == ():
            grad_a = np.sum(grad_a)
        if hasattr(b, 'shape') and b.shape == ():
            grad_b = np.sum(grad_b)
        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]
    
    @staticmethod
    def matmul(grad_output, inputs):
        a, b = inputs

        a_val = np.array(a)
        b_val = np.array(b)
        grad_out_val = np.array(grad_output)

        if a_val.ndim == 1:
            a_val = a_val.reshape(1, -1)
            grad_out_val = grad_out_val.reshape(1, -1)
            reshape_a = True
        else:
            reshape_a = False

        if b_val.ndim == 1:
            b_val = b_val.reshape(-1, 1)
            grad_out_val = grad_out_val.reshape(-1, 1)
            reshape_b = True
        else:
            reshape_b = False

        grad_a = np.matmul(grad_out_val, np.swapaxes(b_val, -1, -2))
        grad_b = np.matmul(np.swapaxes(a_val, -1, -2), grad_out_val)

        if reshape_a:
            grad_a = grad_a.reshape(-1)
        if reshape_b:
            grad_b = grad_b.reshape(-1)

        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def divide(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output / b
        grad_b = -grad_output * a / (b * b)
        a_is_scalar = not hasattr(a, 'shape') or a.shape == ()
        b_is_scalar = not hasattr(b, 'shape') or b.shape == ()
        if a_is_scalar:
            grad_a = np.sum(grad_a)
        if b_is_scalar:
            grad_b = np.sum(grad_b)
        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def power(grad_output, inputs):
        base, exp = inputs
        base_val = base if hasattr(base, 'shape') else np.array(base)
        exp_val = exp if hasattr(exp, 'shape') else np.array(exp)
        base_shape = base.shape if hasattr(base, 'shape') else ()
        exp_shape = exp.shape if hasattr(exp, 'shape') else ()
        grad_base = grad_output * exp_val * np.power(base_val, exp_val - 1)
        grad_exp = grad_output * np.log(base_val + np.finfo(np.float32).eps) * np.power(base_val, exp_val)
        if exp_shape == ():
            grad_exp = np.sum(grad_exp)
        if base_shape == ():
            grad_base = np.sum(grad_base)
        return [UfuncGradients.ensure_shape(grad_base, base_shape), UfuncGradients.ensure_shape(grad_exp, exp_shape)]

    @staticmethod
    def square(grad_output, inputs):
        inp, = inputs
        grad_inp = grad_output * 2 * inp
        if hasattr(inp, 'shape') and inp.shape == ():
            grad_inp = np.sum(grad_inp)
        return [UfuncGradients.ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def sin(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.cos(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def cos(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(-grad_output * np.sin(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def exp(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.exp(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def log(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (inp + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def negative(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(-grad_output, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def maximum(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output * (a >= b)
        grad_b = grad_output * (b > a)
        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def minimum(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output * (a <= b)
        grad_b = grad_output * (b < a)
        return [UfuncGradients.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), UfuncGradients.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def tanh(grad_output, inputs):
        inp = inputs[0]
        tanh_val = np.tanh(inp)
        return [UfuncGradients.ensure_shape(grad_output * (1 - tanh_val ** 2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def abs(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.sign(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def floor(grad_output, inputs):
        warnings.warn("Gradient of floor is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(np.zeros_like(grad_output), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def ceil(grad_output, inputs):
        warnings.warn("Gradient of ceil is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(np.zeros_like(grad_output), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def round(grad_output, inputs):
        warnings.warn("Gradient of round is zero almost everywhere and undefined at .5 boundaries.")
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(np.zeros_like(grad_output), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def mod(grad_output, inputs):
        warnings.warn("Gradient of mod is not well defined at discontinuities.")
        a, b = inputs
        return [UfuncGradients.ensure_shape(grad_output, a.shape if hasattr(a, 'shape') else ()), np.zeros_like(b)]

    @staticmethod
    def remainder(grad_output, inputs):
        warnings.warn("Gradient of remainder is not well defined at discontinuities.")
        a, b = inputs
        return [UfuncGradients.ensure_shape(grad_output, a.shape if hasattr(a, 'shape') else ()), np.zeros_like(b)]

    @staticmethod
    def fmod(grad_output, inputs):
        warnings.warn("Gradient of fmod is not well defined at discontinuities.")
        a, b = inputs
        return [UfuncGradients.ensure_shape(grad_output, a.shape if hasattr(a, 'shape') else ()), np.zeros_like(b)]

    @staticmethod
    def clip(grad_output, inputs):
        inp, min_val, max_val = inputs
        mask = (inp >= min_val) & (inp <= max_val)
        return [UfuncGradients.ensure_shape(grad_output * mask, inp.shape if hasattr(inp, 'shape') else ()), np.zeros_like(min_val), np.zeros_like(max_val)]

    @staticmethod
    def arcsin(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / np.sqrt(1 - inp**2 + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def arccos(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(-grad_output / np.sqrt(1 - inp**2 + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def arctan(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (1 + inp**2), inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def sqrt(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (2 * np.sqrt(inp + np.finfo(np.float32).eps)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def sinh(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.cosh(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def cosh(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.sinh(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def tan(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (np.cos(inp)**2 + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def log1p(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (inp + 1 + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def expm1(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.exp(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def reciprocal(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(-grad_output / (inp ** 2 + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def log2(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (inp * np.log(2) + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def log10(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (inp * np.log(10) + np.finfo(np.float32).eps), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def arctan2(grad_output, inputs):
        y, x = inputs
        denom = x ** 2 + y ** 2 + np.finfo(np.float32).eps
        grad_y = grad_output * x / denom
        grad_x = -grad_output * y / denom
        return [UfuncGradients.ensure_shape(grad_y, y.shape if hasattr(y, 'shape') else ()),
                UfuncGradients.ensure_shape(grad_x, x.shape if hasattr(x, 'shape') else ())]

    @staticmethod
    def hypot(grad_output, inputs):
        x, y = inputs
        denom = np.sqrt(x ** 2 + y ** 2 + np.finfo(np.float32).eps)
        grad_x = grad_output * x / denom
        grad_y = grad_output * y / denom
        return [UfuncGradients.ensure_shape(grad_x, x.shape if hasattr(x, 'shape') else ()),
                UfuncGradients.ensure_shape(grad_y, y.shape if hasattr(y, 'shape') else ())]
        
    @staticmethod
    def mean(grad_output, inputs, axis=None, keepdims=False):
        inp = inputs[0]
        if hasattr(inp, 'shape'):
            shape = inp.shape
            count = np.prod(np.array(shape)) if axis is None else np.prod(np.array(inp).shape if isinstance(axis, int) else [inp.shape[a] for a in axis])
            grad = np.broadcast_to(grad_output / count, shape)
        else:
            grad = grad_output
        return [grad]

    @staticmethod
    def prod(grad_output, inputs, axis=None, keepdims=False):
        inp = inputs[0]
        prod_val = np.prod(inp, axis=axis, keepdims=True)
        grad = grad_output * prod_val / (inp + np.finfo(np.float32).eps)
        return [UfuncGradients.ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def max(grad_output, inputs, axis=None, keepdims=False):
        inp = inputs[0]
        max_val = np.max(inp, axis=axis, keepdims=True)
        mask = (inp == max_val)
        num = np.sum(mask, axis=axis, keepdims=True)
        grad = grad_output * mask / num
        return [UfuncGradients.ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def min(grad_output, inputs, axis=None, keepdims=False):
        inp = inputs[0]
        min_val = np.min(inp, axis=axis, keepdims=True)
        mask = (inp == min_val)
        num = np.sum(mask, axis=axis, keepdims=True)
        grad = grad_output * mask / num
        return [UfuncGradients.ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def erf(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * (2 / np.sqrt(np.pi)) * np.exp(-inp**2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def erfc(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(-grad_output * (2 / np.sqrt(np.pi)) * np.exp(-inp**2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def exp2(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output * np.log(2) * 2**inp, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def logaddexp(grad_output, inputs):
        a, b = inputs
        exp_a, exp_b = np.exp(a), np.exp(b)
        denom = exp_a + exp_b + np.finfo(np.float32).eps
        return [UfuncGradients.ensure_shape(grad_output * exp_a / denom, a.shape if hasattr(a, 'shape') else ()),
                UfuncGradients.ensure_shape(grad_output * exp_b / denom, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def logaddexp2(grad_output, inputs):
        a, b = inputs
        exp2_a, exp2_b = 2**a, 2**b
        denom = exp2_a + exp2_b + np.finfo(np.float32).eps
        return [UfuncGradients.ensure_shape(grad_output * exp2_a / denom * np.log(2), a.shape if hasattr(a, 'shape') else ()),
                UfuncGradients.ensure_shape(grad_output * exp2_b / denom * np.log(2), b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def cbrt(grad_output, inputs):
        inp = inputs[0]
        return [UfuncGradients.ensure_shape(grad_output / (3 * np.cbrt(inp ** 2 + np.finfo(np.float32).eps)), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def deg2rad(grad_output, inputs):
        return [UfuncGradients.ensure_shape(grad_output * np.pi / 180, inputs[0].shape if hasattr(inputs[0], 'shape') else ())]

    @staticmethod
    def rad2deg(grad_output, inputs):
        return [UfuncGradients.ensure_shape(grad_output * 180 / np.pi, inputs[0].shape if hasattr(inputs[0], 'shape') else ())]

    @staticmethod
    def heaviside(grad_output, inputs):
        warnings.warn("Heaviside has undefined derivative at zero.")
        return [np.zeros_like(inputs[0]), np.zeros_like(inputs[1])]

    @staticmethod
    def sign(grad_output, inputs):
        warnings.warn("Gradient of sign is zero almost everywhere and undefined at zero.")
        return [np.zeros_like(inputs[0])]