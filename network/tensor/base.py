import numpy as np
from typing import Tuple, Union
import warnings
from ..gradient_tape.gradient_tape import GradientTape

class BaseType(np.ndarray):
    __name = None
    def __new__(cls, value: Union[np.typing.NDArray, np.number], shape: Tuple[int], dtype: np.typing.DTypeLike, name: str):
        base_array = np.zeros(shape=shape, dtype=dtype)
        obj = np.asarray(base_array).view(cls)
        obj.__name = name

        if np.isscalar(value):
            obj.fill(value)
        else:
            value_array = np.copy(value).astype(dtype)
            if value_array.shape != shape:
                raise ValueError(f"Value shape {value_array.shape} does not match expected shape {shape}")
            obj[...] = value_array

        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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
        if GradientTape._CURRENT_GRADIENT_TAPE is not None and (out is not None or any(id(i) in GradientTape._CURRENT_GRADIENT_TAPE.watched for i in inputs)):
            GradientTape._CURRENT_GRADIENT_TAPE.record(ufunc, method, inputs, kwargs, result)

        return result

    def __array_function__(self, func, types, args, kwargs):
        raw_args = tuple(x.view(np.ndarray) if isinstance(x, BaseType) else x for x in args)
        raw_out = func(*raw_args, **kwargs)
        if isinstance(raw_out, np.ndarray):
            out = raw_out.view(BaseType)
            if GradientTape._CURRENT_GRADIENT_TAPE is not None:
                GradientTape._CURRENT_GRADIENT_TAPE.record(func, '__call__', args, kwargs, out)
            return out
        else:
            if GradientTape._CURRENT_GRADIENT_TAPE is not None:
                GradientTape._CURRENT_GRADIENT_TAPE.record(func, '__call__', args, kwargs, raw_out)
            return raw_out

    # in-place addition
    def __iadd__(self, other):
        original_value = np.copy(self)

        result = np.add(self, other)

        if GradientTape._CURRENT_GRADIENT_TAPE is not None:
            GradientTape._CURRENT_GRADIENT_TAPE.record(np.add, '__call__', [original_value, other], {}, result)

        return result

    # in-place subtraction
    def __isub__(self, other):
        original_value = np.copy(self)

        result = np.subtract(self, other)

        if GradientTape._CURRENT_GRADIENT_TAPE is not None:
            GradientTape._CURRENT_GRADIENT_TAPE.record(np.subtract, '__call__', [original_value, other], {}, result)

        return result
    
    def __itruediv__(self, other):
        original_value = np.copy(self)

        result = np.true_divide(self, other)

        if GradientTape._CURRENT_GRADIENT_TAPE is not None:
            GradientTape._CURRENT_GRADIENT_TAPE.record(np.true_divide, '__call__', [original_value, other], {}, result)

        return result

    def __imul__(self, other):
        original_value = np.copy(self)

        result = np.multiply(self, other)

        if GradientTape._CURRENT_GRADIENT_TAPE is not None:
            GradientTape._CURRENT_GRADIENT_TAPE.record(np.multiply, '__call__', [original_value, other], {}, result)

        return result

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
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other