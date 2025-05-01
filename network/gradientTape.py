# network/gradientTape.py

import numpy as np
import warnings

class Tape:
    _CURRENT_TAPE = None

    def __init__(self, persistent=False):
        self.operations = []
        self.watched = set()
        self.persistent = persistent

    def __enter__(self):
        Tape._CURRENT_TAPE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.persistent:
            self.operations.clear()
            self.watched.clear()
        Tape._CURRENT_TAPE = None

    def watch(self, tensor):
        self.watched.add(id(tensor))

    def record(self, ufunc, method, inputs, kwargs, result):
        if any(id(i) in self.watched for i in inputs):
            self.operations.append({
                'ufunc': ufunc,
                'method': method,
                'inputs': inputs,
                'kwargs': kwargs,
                'result': result,
            })
            self.watched.add(id(result))

    def gradient(self, target, sources):
        grads = { id(target): np.ones_like(target) }

        for op in reversed(self.operations):
            ufunc = op['ufunc']
            inputs = op['inputs']
            result = op['result']

            print(ufunc.__name__)

            if id(result) not in grads:
                continue

            grad_output = grads[id(result)]
            if hasattr(Tape, ufunc.__name__):
                grads_list = getattr(Tape, ufunc.__name__)(grad_output, inputs)
                for inp, g in zip(inputs, grads_list):
                    # Ensure correct shape for gradients
                    grads[id(inp)] = grads.get(id(inp), 0.0) + g

        return [grads.get(id(src), np.zeros_like(src)) for src in sources]


    @staticmethod
    def ensure_shape(x, shape):
        x = x if isinstance(x, np.ndarray) else np.array(x)
        # Handle scalar conversion explicitly
        if np.shape(x) == ():
            x = np.broadcast_to(x, shape)
        elif np.shape(x) != shape:
            try:
                x = np.broadcast_to(x, shape)
            except ValueError:
                # This case occurs when x and shape are not compatible for broadcasting
                raise ValueError(f"Cannot broadcast shape {x.shape} to {shape}")
        return x

    @staticmethod
    def add(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output
        grad_b = grad_output
        if hasattr(a, 'shape') and a.shape == ():
            grad_a = np.sum(grad_output)  # Scalar handling
        if hasattr(b, 'shape') and b.shape == ():
            grad_b = np.sum(grad_output)  # Scalar handling
        return [Tape.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                Tape.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]


    @staticmethod
    def subtract(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output
        grad_b = -grad_output
        if hasattr(a, 'shape') and a.shape == ():
            grad_a = np.sum(grad_a)
        if hasattr(b, 'shape') and b.shape == ():
            grad_b = np.sum(grad_b)
        return [Tape.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), Tape.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def multiply(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output * b
        grad_b = grad_output * a
        if hasattr(a, 'shape') and a.shape == ():
            grad_a = np.sum(grad_a)
        if hasattr(b, 'shape') and b.shape == ():
            grad_b = np.sum(grad_b)
        return [Tape.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), Tape.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

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
        return [Tape.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), Tape.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def power(grad_output, inputs):
        base, exp = inputs
        base_val = base if hasattr(base, 'shape') else np.array(base)
        exp_val = exp if hasattr(exp, 'shape') else np.array(exp)
        base_shape = base.shape if hasattr(base, 'shape') else ()
        exp_shape = exp.shape if hasattr(exp, 'shape') else ()
        grad_base = grad_output * exp_val * np.power(base_val, exp_val - 1)
        grad_exp = grad_output * np.log(base_val + 1e-10) * np.power(base_val, exp_val)
        if exp_shape == ():
            grad_exp = np.sum(grad_exp)
        if base_shape == ():
            grad_base = np.sum(grad_base)
        return [Tape.ensure_shape(grad_base, base_shape), Tape.ensure_shape(grad_exp, exp_shape)]

    @staticmethod
    def square(grad_output, inputs):
        inp, = inputs
        grad_inp = grad_output * 2 * inp
        if hasattr(inp, 'shape') and inp.shape == ():
            grad_inp = np.sum(grad_inp)
        return [Tape.ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def sin(grad_output, inputs):
        inp = inputs[0]
        return [Tape.ensure_shape(grad_output * np.cos(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def cos(grad_output, inputs):
        inp = inputs[0]
        return [Tape.ensure_shape(-grad_output * np.sin(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def exp(grad_output, inputs):
        inp = inputs[0]
        return [Tape.ensure_shape(grad_output * np.exp(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def log(grad_output, inputs):
        inp = inputs[0]
        return [Tape.ensure_shape(grad_output / (inp + 1e-10), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def negative(grad_output, inputs):
        inp = inputs[0]
        return [Tape.ensure_shape(-grad_output, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def maximum(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output * (a >= b)
        grad_b = grad_output * (b > a)
        return [Tape.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), Tape.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def minimum(grad_output, inputs):
        a, b = inputs
        grad_a = grad_output * (a <= b)
        grad_b = grad_output * (b < a)
        return [Tape.ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()), Tape.ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def tanh(grad_output, inputs):
        inp = inputs[0]
        tanh_val = np.tanh(inp)
        return [Tape.ensure_shape(grad_output * (1 - tanh_val ** 2), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def abs(grad_output, inputs):
        inp = inputs[0]
        return [Tape.ensure_shape(grad_output * np.sign(inp), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def floor(grad_output, inputs):
        warnings.warn("Gradient of floor is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [Tape.ensure_shape(np.zeros_like(grad_output), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def ceil(grad_output, inputs):
        warnings.warn("Gradient of ceil is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [Tape.ensure_shape(np.zeros_like(grad_output), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def round(grad_output, inputs):
        warnings.warn("Gradient of round is zero almost everywhere and undefined at .5 boundaries.")
        inp = inputs[0]
        return [Tape.ensure_shape(np.zeros_like(grad_output), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def mod(grad_output, inputs):
        warnings.warn("Gradient of mod is not well defined at discontinuities.")
        a, b = inputs
        return [Tape.ensure_shape(grad_output, a.shape if hasattr(a, 'shape') else ()), np.zeros_like(b)]

    @staticmethod
    def remainder(grad_output, inputs):
        warnings.warn("Gradient of remainder is not well defined at discontinuities.")
        a, b = inputs
        return [Tape.ensure_shape(grad_output, a.shape if hasattr(a, 'shape') else ()), np.zeros_like(b)]

    @staticmethod
    def fmod(grad_output, inputs):
        warnings.warn("Gradient of fmod is not well defined at discontinuities.")
        a, b = inputs
        return [Tape.ensure_shape(grad_output, a.shape if hasattr(a, 'shape') else ()), np.zeros_like(b)]

    @staticmethod
    def clip(grad_output, inputs):
        inp, min_val, max_val = inputs
        mask = (inp >= min_val) & (inp <= max_val)
        return [Tape.ensure_shape(grad_output * mask, inp.shape if hasattr(inp, 'shape') else ()), np.zeros_like(min_val), np.zeros_like(max_val)]