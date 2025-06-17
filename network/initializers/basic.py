import numpy as np

from ..types.tensor import Tensor
from .base import Initializer


class Zeros(Initializer):
    def __call__(self, tensor: Tensor) -> Tensor:
        return Tensor(
            np.zeros(tensor.shape, dtype=tensor.dtype),
            shape=tensor.shape,
            dtype=tensor.dtype,
            name=tensor.name,
        )


class Ones(Initializer):
    def __call__(self, tensor: Tensor) -> Tensor:
        return Tensor(
            np.ones(tensor.shape, dtype=tensor.dtype),
            shape=tensor.shape,
            dtype=tensor.dtype,
            name=tensor.name,
        )
