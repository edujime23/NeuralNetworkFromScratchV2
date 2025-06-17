from collections.abc import Callable

import numpy as np

from ..types.tensor import Tensor
from ..types.variable import Variable
from .base import Layer


class Dense(Layer):
    def __init__(
        self,
        units: int,
        activation: Callable[[Tensor], Tensor] | None = None,
        use_bias: bool = True,
        kernel_initializer: Callable[[tuple[int, int]], Tensor] | None = None,
        bias_initializer: Callable[[tuple[int]], Tensor] | None = None,
        name: str | None = None,
        dtype: np.dtype | None = np.float32,
    ):
        super().__init__(name=name, trainable=True, dtype=(dtype or np.float32))
        self.units: int = units
        self.activation: Callable[[Tensor], Tensor] | None = activation
        self.use_bias: bool = use_bias
        self.kernel_initializer: Callable[[tuple[int, int]], Tensor] = (
            kernel_initializer or self._default_kernel_initializer
        )
        self.bias_initializer: Callable[[tuple[int]], Tensor] = (
            bias_initializer or self._default_bias_initializer
        )
        self.kernel: Variable | None = None
        self.bias: Variable | None = None

    def build(self, input_shape: tuple[int, ...]) -> None:
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        self._built = True

    def call(self, inputs: Tensor, training: bool = False, *args, **kwargs) -> Tensor:
        output: Tensor = np.dot(inputs, self.kernel)  # type: ignore
        if self.use_bias and self.bias is not None:
            # Explicitly reshape bias to (1, units) so broadcasting works cleanly for autodiff
            bias_reshaped = self.bias.reshape((1, -1))  # type: ignore
            output = output + bias_reshaped  # type: ignore
        if self.activation:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape[:-1] + (self.units,)

    def _default_kernel_initializer(self, shape: tuple[int, int]) -> Tensor:
        # Glorot Uniform initializer
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape).astype(self.dtype)

    def _default_bias_initializer(self, shape: tuple[int]) -> Tensor:
        return np.zeros(shape, dtype=self.dtype)
