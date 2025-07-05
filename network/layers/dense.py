from collections.abc import Callable
import numpy as np

from network.types.tensor import Tensor
from network.types.variable import Variable
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
        dtype: np.typing.DTypeLike = np.float32,
        input_dim: int | None = None,
    ):
        super().__init__(name=name, trainable=True, dtype=dtype)

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.input_dim = input_dim

        if input_dim is not None:
            self.input_spec = {"min_ndim": 1, "axes": {-1: input_dim}}
        else:
            self.input_spec = {"min_ndim": 1}

        self.kernel: Variable | None = None
        self.bias: Variable | None = None

    def build(self, input_shape: tuple[int, ...]) -> None:
        input_dim = input_shape[-1]
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"{self.name}: Expected input_dim {self.input_dim}, got {input_dim}"
            )

        self.kernel = self.add_weight(
            "kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer or self._default_kernel_initializer,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(self.units,),
                initializer=self.bias_initializer or self._default_bias_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self, inputs: Tensor, training: bool = False, mask: Tensor | None = None
    ) -> Tensor:
        output = inputs @ self.kernel
        if self.use_bias:
            output = output + self.bias
        return self.activation(output) if self.activation else output

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape[:-1] + (self.units,)

    def _default_kernel_initializer(self, shape: tuple[int, int]) -> Tensor:
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return Tensor(np.random.uniform(-limit, limit, size=shape).astype(self.dtype))

    def _default_bias_initializer(self, shape: tuple[int]) -> Tensor:
        return Tensor(np.zeros(shape, dtype=self.dtype))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation.__name__ if self.activation else None,
                "use_bias": self.use_bias,
                "input_dim": self.input_dim,
            }
        )
        return config
