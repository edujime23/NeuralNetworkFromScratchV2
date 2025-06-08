from typing import override
from .base import Initializer
from ..types import Tensor
import numpy as np

class Zeros(Initializer):
    @override
    def __call__(self, tensor: Tensor) -> Tensor:
        return Tensor(np.zeros(tensor.shape, dtype=tensor.dtype), shape=tensor.shape, dtype=tensor.dtype, name=tensor.name)
    
class Ones(Initializer):
    @override
    def __call__(self, tensor: Tensor) -> Tensor:
        return Tensor(np.ones(tensor.shape, dtype=tensor.dtype), shape=tensor.shape, dtype=tensor.dtype, name=tensor.name)