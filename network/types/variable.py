import numpy as np
from typing import Tuple, Union, Callable, Optional
from .base import BaseType

class Variable(BaseType):
    def __new__(
        cls,
        value: np.typing.ArrayLike,
        shape: Tuple[int],
        dtype: np.typing.DTypeLike,
        trainable: bool,
        name: str,
        initializer: Optional[Union[str, Callable]] = None
    ):
        obj = super().__new__(cls, value, shape, dtype, name)
        obj.__trainable = trainable
        obj.__initializer = initializer
        return obj

    def initialize(self):
        if self.__initializer is None:
            pass
        elif callable(self.__initializer):
            self.__initializer(self)
        else:
            self._initialize(self.view(np.ndarray), self.shape, self.__initializer)
        
    @property
    def trainable(self):
        return self.__trainable

    @property
    def initializer(self):
        return self.__initializer