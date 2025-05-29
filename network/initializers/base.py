from ..types.tensor import Tensor

class Initializer:
    def __call__(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError("Must be implemented by subclass.")