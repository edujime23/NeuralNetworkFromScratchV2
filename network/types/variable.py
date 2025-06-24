from collections.abc import Callable
from typing import Any, Self

import numpy as np

from ..initializers import Initializer, Zeros
from .tensor import Tensor


class Variable:
    __initializer_map = {
        cls.__name__.lower(): cls for cls in Initializer.__subclasses__()
    }

    def __init__(
        self,
        value: np.typing.ArrayLike | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: np.typing.DTypeLike | None = None,
        trainable: bool = True,
        name: str | None = None,
        initializer: Initializer | Callable | str | None = None,
    ):
        self.__tensor = Tensor(
            value=value, shape=shape, dtype=dtype, name=f"{name}/Tensor"
        )
        self.__trainable = trainable or True

        if isinstance(initializer, str):
            self.__initializer = self.__initializer_map.get(initializer, Zeros)()
        else:
            self.__initializer = initializer

    def copy(self, order=None):
        return Variable(
            value=self.__tensor.copy(order=order),
            shape=self.__tensor.shape,
            dtype=self.__tensor.dtype,
            trainable=self.__trainable,
            name=self.__tensor.name,
            initializer=self.__initializer,
        )

    def flatten(self):
        return self.__tensor.flatten()

    def squeeze(self):
        return self.__tensor.squeeze()

    def reshape(self, *shape: int):
        return self.__tensor.reshape(*shape)

    def transpose(self, *axes: int):
        return self.__tensor.transpose(*axes)

    def assign(self, value: np.typing.ArrayLike) -> Self:
        if value.shape != self.shape:
            raise ValueError(
                f"Cannot assign value with shape {value.shape} to Variable with shape {self.shape}. Shapes must match for in-place assignment."
            )
        self.__tensor = Tensor(
            value=value, shape=self.shape, dtype=self.dtype, name=f"{self.name}/Tensor"
        )
        return self

    def assign_add(self, value: np.typing.ArrayLike) -> Self:
        self.assign(
            self.__tensor.data + value,
        )
        return self

    def assign_sub(self, value: np.typing.ArrayLike) -> Self:
        self.assign(
            self.__tensor.data - value,
        )
        return self

    def initialize(self):
        self.assign(self.initializer(self.__tensor))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.__do_op__(func=ufunc, method=method, args=inputs, kwargs=kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return self.__do_op__(func=func, method="__call__", args=args, kwargs=kwargs)

    def __do_op__(
        self, func: Callable, method: str, args: tuple[str, Any], kwargs: dict[str, Any]
    ) -> Self:
        tensor_inputs = tuple(
            x.__tensor if isinstance(x, type(self)) else x for x in args
        )
        return getattr(func, method)(*tensor_inputs, **kwargs)

    @property
    def numpy(self) -> np.ndarray:
        return self.__tensor.numpy

    @property
    def trainable(self) -> bool:
        return self.__trainable

    @property
    def name(self) -> str | None:
        return self.__tensor.name

    @property
    def shape(self) -> tuple[int, ...]:
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
        return self.__tensor.__repr__().replace("Tensor", "Variable")

    # --- Arithmetic Operators ---
    def __add__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return (
            self.__tensor + other_val
        )  # Correct: Delegates to Tensor's __add__ (or ufunc)

    def __radd__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return (
            other_val + self.__tensor
        )  # Correct: Delegates to Tensor's __radd__ (or ufunc)

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
        return self.__tensor**other_val

    def __rpow__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return other_val**self.__tensor

    # --- Bitwise Operators --- (similarly delegate)
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

    # Comparison operators (similarly delegate)
    def __lt__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor < other_val

    def __le__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor <= other_val

    def __eq__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor == other_val

    def __ne__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor != other_val

    def __gt__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor > other_val

    def __ge__(self, other) -> Tensor:
        other_val = other.__tensor if isinstance(other, type(self)) else other
        return self.__tensor >= other_val

    # Unary operators
    def __neg__(self) -> Tensor:
        return -self.__tensor

    def __pos__(self) -> Tensor:
        return +self.__tensor

    def __abs__(self) -> Tensor:
        return abs(self.__tensor)

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
        self.assign(self.__tensor**other_val)
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
        return self.__tensor[key]

    def __setitem__(self, key, value):
        current_numpy_value = self.__tensor.data.copy()
        current_numpy_value[key] = value
        self.assign(current_numpy_value)

    def __len__(self):
        return len(self.__tensor)
