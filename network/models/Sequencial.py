from typing import Optional, List, Tuple, Union, Callable
import numpy as np
from .base import Model
from ..layers import Layer
from ..initializers import Initializer
from ..types import Variable
from ..metrics import Metric
from ..optimizers import Optimizer
from ..tape import GradientTape

class Sequential(Model):
    def __init__(self, layers: Optional[List[Layer]] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self._layers: List[Layer] = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer) -> None:
        self._layers.append(layer)

    def build(self, input_shape: Tuple[int, ...], dtype: Optional[np.typing.DTypeLike] = np.float32) -> None:
        self._input_shape = input_shape
        self._dtype = dtype
        shape = input_shape
        for layer in self._layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
            self._variables.extend(layer.trainable_variables)
            self._variables.extend(layer.non_trainable_variables)
        self._built = True

    def call(self, inputs: np.ndarray, training: bool = False, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = inputs
        for layer in self._layers:
            x = layer(x, training=training, mask=mask)
        return x

    def pop(self) -> None:
        if not self._layers:
            raise ValueError("Cannot pop from an empty Sequential model.")
        self._layers.pop()

    def get_weights(self) -> List[np.ndarray]:
        weights = []
        for layer in self._layers:
            weights.extend(layer.get_weights())
        return weights

    def set_weights(self, weights: List[np.ndarray]) -> None:
        for layer in self._layers:
            num_weights = len(layer.get_weights())
            layer.set_weights(weights[:num_weights])
            weights = weights[num_weights:]

    def summary(self) -> None:
        print(f"Model: {self.name}")
        print("Layer (type)                 Output Shape              Param #")
        print("=================================================================")
        total_params = 0
        input_shape = self._input_shape
        for layer in self._layers:
            output_shape = layer.compute_output_shape(input_shape)
            params = layer.count_params()
            total_params += params
            print(f"{layer.name} ({layer.__class__.__name__})    {output_shape}    {params}")
            input_shape = output_shape
        print("=================================================================")
        print(f"Total params: {total_params}")

    @property
    def layers(self) -> List[Layer]:
        return self._layers
