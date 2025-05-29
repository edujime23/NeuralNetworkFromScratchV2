import numpy as np
from typing import Any, Dict, Tuple, Union, Callable, Optional, Self
from .tensor import Tensor
from ..initializers import Initializer, Zeros

class Variable:
    __initializer_map = {
        cls.__name__.lower(): cls
        for cls in Initializer.__subclasses__()
    }
    
    def __init__(
        self,
        value: Optional[np.typing.ArrayLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.typing.DTypeLike] = None,
        trainable: Optional[bool] = True,
        name: Optional[str] = None,
        initializer: Optional[Union[Initializer, Callable, str]] = None,
    ):
        self.__tensor = Tensor(value=value, shape=shape, dtype=dtype, name=name)
        self.__trainable = trainable or True

        if isinstance(initializer, str):
            self.__initializer = self.__initializer_map.get(initializer, Zeros)()
        else:
            self.__initializer = initializer
            
    def copy(self, order = None):
        return Variable(
            value=self.__tensor.copy(order=order), 
            shape=self.__tensor.shape, 
            dtype=self.__tensor.dtype, 
            trainable=self.__trainable, 
            name=self.__tensor.name, 
            initializer=self.__initializer
        ) 
    
    def assign(self, value: np.typing.ArrayLike) -> Self:
        self.__tensor = Tensor(value, shape=self.__tensor.shape, dtype=self.__tensor.dtype)
        return self
    
    def initialize(self):
        self.assign(
            self.initializer(self.__tensor)
        )
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.__do_op__(func=ufunc, method=method, args=inputs, kwargs=kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return self.__do_op__(func=func, method="__call__", args=args, kwargs=kwargs)
    
    def __do_op__(self, func: Callable, method: str, args: Tuple[str, Any], kwargs: Dict[str, Any]) -> Self:
        tensor_inputs = tuple(
            x.__tensor if isinstance(x, type(self)) else x
            for x in args
        )
        return getattr(func, method)(*tensor_inputs, **kwargs)
    
    @property
    def numpy(self) -> np.ndarray:
        return self.__tensor.view(np.ndarray).copy() 
    
    @property
    def trainable(self) -> bool:
        return self.__trainable

    @property
    def name(self) -> Optional[str]:
        return self.__tensor.name
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__tensor.shape

    @property
    def dtype(self) -> np.typing.DTypeLike:
        return self.__tensor.dtype
    
    @property
    def value(self) -> Tensor:
        return self.__tensor
    
    @property
    def initializer(self):
        return self.__initializer 
    
    @property
    def real(self) -> Tensor:
        return self.__tensor.real
    
    @property
    def imag(self) -> Tensor:
        return self.__tensor.imag
    
    def __repr__(self):
        return (f"Variable(name={self.name}, shape={self.__tensor.shape}, "
                f"dtype={self.__tensor.dtype}, trainable={self.trainable}, "
                f"value=\n{self.__tensor.view(np.ndarray)})")
    
    # --- Arithmetic Operators ---
    def __add__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor + other_val

    def __radd__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val + self.__tensor

    def __sub__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor - other_val

    def __rsub__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val - self.__tensor

    def __mul__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor * other_val

    def __rmul__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val * self.__tensor

    def __truediv__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor / other_val

    def __rtruediv__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val / self.__tensor

    def __floordiv__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor // other_val

    def __rfloordiv__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val // self.__tensor

    def __mod__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor % other_val

    def __rmod__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val % self.__tensor

    def __pow__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor ** other_val

    def __rpow__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val ** self.__tensor

    # --- Bitwise Operators ---
    def __and__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor & other_val

    def __rand__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val & self.__tensor

    def __or__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor | other_val

    def __ror__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val | self.__tensor

    def __xor__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor ^ other_val

    def __rxor__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val ^ self.__tensor

    def __lshift__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor << other_val

    def __rlshift__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val << self.__tensor

    def __rshift__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor >> other_val

    def __rrshift__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val >> self.__tensor

    # --- Matmul Operator ---
    def __matmul__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor @ other_val

    def __rmatmul__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val @ self.__tensor

    # --- In-place Operators ---
    def __iadd__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor + other_val)
        return self

    def __isub__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor - other_val)
        return self

    def __imul__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor * other_val)
        return self

    def __itruediv__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor / other_val)
        return self

    def __ifloordiv__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor // other_val)
        return self

    def __imod__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor % other_val)
        return self

    def __ipow__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor ** other_val)
        return self

    def __iand__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor & other_val)
        return self

    def __ior__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor | other_val)
        return self

    def __ixor__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor ^ other_val)
        return self

    def __ilshift__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor << other_val)
        return self

    def __irshift__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        self.assign(self.__tensor >> other_val)
        return self
    
    def __getitem__(self, key):
        return self.__tensor.view(np.ndarray)[key].copy()

    def __setitem__(self, key, value):
        current_numpy_value = self.__tensor.view(np.ndarray).copy()
        current_numpy_value[key] = value
        self.assign(current_numpy_value)