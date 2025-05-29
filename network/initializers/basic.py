from typing import override
from .base import Initializer
from ..types import Tensor
import numpy as np

class Zeros(Initializer):
    @override
    def __call__(self, tensor: Tensor) -> Tensor:
        return np.zeros_like(tensor)
    
class Ones(Initializer):
    @override
    def __call__(self, tensor: Tensor) -> Tensor:
        return np.ones_like(tensor)