import numpy as np
from .ufunc_gradients import UfuncGradients

class GradientTape:
    _CURRENT_GRADIENT_TAPE = None

    def __init__(self, persistent=False):
        self.operations = []
        self.watched = set()
        self.persistent = persistent

    def __enter__(self):
        GradientTape._CURRENT_GRADIENT_TAPE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.persistent:
            self.operations.clear()
            self.watched.clear()
        GradientTape._CURRENT_GRADIENT_TAPE = None

    def watch(self, tensor):
        self.watched.add(id(tensor))

    def record(self, func, method, inputs, kwargs, result):
        if any(id(i) in self.watched for i in inputs):  
            self.operations.append({
                'func': func,
                'method': method,
                'inputs': inputs,
                'kwargs': kwargs,
                'result': result,
            })
            
            self.watched.add(id(result))

    def gradient(self, target, sources):
        grads = { id(target): np.ones_like(target) }
        for op in reversed(self.operations):
            func = op['func']
            inputs = op['inputs']
            result = op['result']

            if id(result) not in grads:
                continue

            grad_output = grads[id(result)]
            if hasattr(UfuncGradients, func.__name__):
                grads_list = getattr(UfuncGradients, func.__name__)(grad_output, inputs)
                for inp, g in zip(inputs, grads_list):
                    grads[id(inp)] = grads.get(id(inp), 0.0) + g

        return [grads.get(id(src)) for src in sources]
