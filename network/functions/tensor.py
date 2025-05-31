from typing import Literal
from ..types import Tensor
import numpy as np

def zeros(shape: tuple[int, ...], 
        dtype: np.typing.DTypeLike = np.float64, 
        name: str | None = None, 
        order: Literal["C", "F"] = "C", 
        like: Tensor | None = None
    ) -> Tensor:
    arr = np.zeros(shape=shape, dtype=dtype, order=order, like=like)
    return Tensor(arr, shape=shape, dtype=dtype, name=name)

def ones(shape: tuple[int, ...], 
        dtype: np.typing.DTypeLike = np.float64, 
        name: str | None = None, 
        order: Literal["C", "F"] = "C", 
        like: Tensor | None = None
    ) -> Tensor:
    arr = np.ones(shape=shape, dtype=dtype, order=order, like=like)
    return Tensor(arr, shape=shape, dtype=dtype, name=name)