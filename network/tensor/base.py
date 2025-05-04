import numpy as np
from typing import Tuple, Union
from ..util import function
from ..gradient_tape.gradient_tape import GradientTape

class BaseType(np.ndarray):
    __name = None
    def __new__(cls, value: Union[np.typing.NDArray, np.number], shape: Tuple[int], dtype: np.typing.DTypeLike, name: str):
        base_array = np.zeros(shape=shape, dtype=dtype)
        obj = np.asarray(base_array).view(cls)
        obj.__name = name

        if not value:
            pass
        elif np.isscalar(value):
            obj.fill(value)
        else:
            value_array = np.copy(value).astype(dtype)
            if value_array.shape != shape:
                raise ValueError(f"Value shape {value_array.shape} does not match expected shape {shape}")
            obj[...] = value_array

        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if getattr(ufunc, 'flags', {}).get("supress_tape_recording", False):
            return ufunc(*inputs, **kwargs)
        out = kwargs.pop('out', None)

        raw_inputs = [i.view(np.ndarray) if isinstance(i, BaseType) else i for i in inputs]

        raw_out = None
        if out:
            raw_out = tuple(o.view(np.ndarray) if isinstance(o, BaseType) else o for o in out)
            raw_kwargs = {**kwargs, 'out': raw_out}
        else:
            raw_kwargs = kwargs

        raw_result = getattr(ufunc, method)(*raw_inputs, **raw_kwargs)

        result = self if out else raw_result.view(BaseType)
        for tape in GradientTape._GRADIENTS_TAPES:
            tape.record(ufunc, method, inputs, kwargs, result)

        return result

    def __array_function__(self, func, types, inputs, kwargs):
        if getattr(func, 'flags', {}).get("supress_tape_recording", False):
            return func(*inputs, **kwargs)

        raw_inputs = tuple(x.view(np.ndarray) if isinstance(x, BaseType) else x for x in inputs)
        raw_out = func(*raw_inputs, **kwargs)
        if isinstance(raw_out, np.ndarray):
            out = raw_out.view(BaseType)
            for tape in GradientTape._GRADIENTS_TAPES:
                tape.record(func, '__call__', inputs, kwargs, out)
            return out
        else:
            for tape in GradientTape._GRADIENTS_TAPES:
                tape.record(func, '__call__', inputs, kwargs, raw_out)
            return raw_out
        
    @staticmethod
    def _inplace_operation(func, self, other): return func(self, other)
    def __iadd__(self, other): return self._inplace_operation(np.add, self, other)
    def __isub__(self, other): return self._inplace_operation(np.subtract, self, other)
    def __itruediv__(self, other): return self._inplace_operation(np.true_divide, self, other)
    def __imul__(self, other): return self._inplace_operation(np.multiply, self, other)
    def __imatmul__(self, other): return self._inplace_operation(np.matmul, self, other)
    def __ipow__(self, other): return self._inplace_operation(np.pow, self, other)

    @staticmethod
    def _initialize(arr: np.typing.NDArray, shape: Tuple[int], initializer: str):
        match initializer:
            case 'zeros': arr[:] = 0
            case 'ones': arr[:] = 1
            case 'random_normal': arr[:] = np.random.normal(0,1,shape)
            case 'random_uniform': arr[:] = np.random.uniform(0,1,shape)
            case 'xavier_uniform':
                limit = np.sqrt(6/np.sum(shape)); arr[:] = np.random.uniform(-limit,limit,shape)
            case 'xavier_normal':
                std = np.sqrt(2/np.sum(shape)); arr[:] = np.random.normal(0,std,shape)
            case 'he_normal':
                std = np.sqrt(2/shape[0]); arr[:] = np.random.normal(0,std,shape)
            case 'he_uniform':
                limit = np.sqrt(6/shape[0]); arr[:] = np.random.uniform(-limit,limit,shape)
            case 'glorot_normal':
                std = np.sqrt(2/np.sum(shape)); arr[:] = np.random.normal(0,std,shape)
            case 'glorot_uniform':
                limit = np.sqrt(6/np.sum(shape)); arr[:] = np.random.uniform(-limit,limit,shape)
            case 'lecun_normal':
                std = np.sqrt(1/shape[0]); arr[:] = np.random.normal(0,std,shape)
            case 'lecun_uniform':
                limit = np.sqrt(3/shape[0]); arr[:] = np.random.uniform(-limit,limit,shape)

    @property
    def name(self): return self.__name
    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other