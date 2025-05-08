from typing import Optional, Callable, Tuple
import numpy as np
from .base import Layer  # Assuming you have a base Layer class as previously defined

class Dense(Layer):
    def __init__(
        self,
        units: int,
        activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[Callable[[Tuple[int, int]], np.ndarray]] = None,
        bias_initializer: Optional[Callable[[Tuple[int]], np.ndarray]] = None,
        name: Optional[str] = None,
        dtype: Optional[np.dtype] = np.float32,
    ):
        super().__init__(name=name, trainable=True, dtype=dtype)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer or self._default_kernel_initializer
        self.bias_initializer = bias_initializer or self._default_bias_initializer
        self.kernel: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

    def build(self, input_shape: Tuple[int, ...]):
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

    def call(self, inputs: np.ndarray, training: bool = False, mask: Optional[np.ndarray] = None) -> np.ndarray:
        output = np.dot(inputs, self.kernel)
        if self.use_bias and self.bias is not None:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-1] + (self.units,)

    def _default_kernel_initializer(self, shape: Tuple[int, int]) -> np.ndarray:
        # Glorot Uniform initializer
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, size=shape).astype(self.dtype)

    def _default_bias_initializer(self, shape: Tuple[int]) -> np.ndarray:
        return np.zeros(shape, dtype=self.dtype)
