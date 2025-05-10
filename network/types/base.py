import numpy as np
from typing import Tuple, Union, Callable, Dict, Optional
from ..gradient_tape.gradient_tape import GradientTape

class BaseType(np.ndarray):
    __name = None
    def __new__(cls, value: Optional[Union[np.typing.NDArray, np.number]], shape: Optional[Tuple[int]] = None, dtype: Optional[np.typing.DTypeLike] = None, name: Optional[str] = None):
        if not shape:
            shape = value.shape if value is not None else ()
        if not dtype:
            dtype = value.dtype if value is not None else np.float32
            
        obj = np.asarray(np.zeros(shape, dtype)).view(cls)
            
        if np.isscalar(value):
            obj.fill(value)
        else:
            value_array = np.asarray(value).astype(dtype)
            if value_array.shape != shape:
                raise ValueError(f"Value shape {value_array.shape} does not match expected shape {shape}")
            obj[...] = value_array
            
        obj.__name = name

        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.__record(ufunc, method, inputs, kwargs)

    def __array_function__(self, func, types, inputs, kwargs):
        return self.__record(func, '__call__', inputs, kwargs)
        
    def __record(self, func: Callable[[np.typing.ArrayLike, np.typing.ArrayLike], np.typing.ArrayLike], method: str, inputs: Tuple[np.typing.ArrayLike], kwargs: Dict[str, np.typing.ArrayLike]) -> np.typing.ArrayLike:
        raw_inputs = [i.view(np.ndarray) if issubclass(type(i), BaseType) else i for i in inputs]
        raw_kwargs = {
            k: v.view(np.ndarray) if issubclass(type(v), BaseType) else v
            for k, v in kwargs.items()
        }
        raw_result = getattr(func, method)(*raw_inputs, **raw_kwargs)
        if issubclass(type(raw_result), np.ndarray):
            result = raw_result.view(BaseType)
        elif issubclass(type(raw_result), np.number):
            result = np.asarray(raw_result).view(BaseType)
        else: result = raw_result

        if tapes := GradientTape._GRADIENTS_TAPES:
            for tape in tapes:
                tape.record(func, method, inputs, kwargs, result)
                
        return result
        
    @staticmethod
    def _inplace_operation(func, self, other): return func(self, other)
    def __iadd__(self, other: np.typing.ArrayLike): return self._inplace_operation(np.add, self, other)
    def __isub__(self, other: np.typing.ArrayLike): return self._inplace_operation(np.subtract, self, other)
    def __itruediv__(self, other: np.typing.ArrayLike): return self._inplace_operation(np.true_divide, self, other)
    def __imul__(self, other: np.typing.ArrayLike): return self._inplace_operation(np.multiply, self, other)
    def __imatmul__(self, other: np.typing.ArrayLike): return self._inplace_operation(np.matmul, self, other)
    def __ipow__(self, other: np.typing.ArrayLike): return self._inplace_operation(np.pow, self, other)
    
    @property
    def real(self): return self.__record(func=np.real, method='__call__', inputs=(self,), kwargs={})
    @property
    def imag(self): return self.__record(func=np.imag, method='__call__', inputs=(self,), kwargs={})

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
    def __eq__(self, other: np.typing.ArrayLike): return self is other