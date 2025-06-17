from abc import ABC, abstractmethod

from ..types.tensor import Tensor


class Initializer(ABC):
    @abstractmethod
    def __call__(self, tensor: Tensor) -> Tensor:
        pass
