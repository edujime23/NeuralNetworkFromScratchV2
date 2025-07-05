import numpy as np
from network.types.tensor import Tensor
from .base import Layer


class Input(Layer):
    """Input layer that validates and passes through input tensors."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        name: str | None = None,
        dtype: np.typing.DTypeLike = np.float32,
    ):
        super().__init__(name=name, trainable=False, dtype=dtype)
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.input_spec = {
            "shape": input_shape,
            "min_ndim": len(input_shape),
        }
        self._built = True

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Input layer is always built upon initialization."""
        pass

    def call(
        self, inputs: Tensor, training: bool = False, mask: Tensor | None = None
    ) -> Tensor:
        """Pass through input tensor after validation."""
        return inputs

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
            }
        )
        return config
