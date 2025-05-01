import numpy as np
from typing import Tuple, Union, Callable
from .base import BaseType

class Variable(BaseType):
    def __new__(cls, shape: Tuple[int], dtype: np.typing.DTypeLike, trainable: bool, name: str, initializer: Union[str, Callable] = 'zeros'):
        obj = super().__new__(cls, shape, dtype, name)
        obj.__trainable = trainable
        obj.__initializer = initializer

        return obj
    
    def initialize(self):
        if callable(self.__initializer):
            self.__initializer(self)
            return
        
        self._initialize(self.real, self.shape, self.initializer)
        if np.iscomplexobj(self):
            self._initialize(self.imag, self.shape, self.initializer)

    @property
    def trainable(self):
        return self.__trainable

    @property
    def initializer(self):
        return self.__initializer