import numpy as np

from ..layers.base import Layer
from ..types.tensor import Tensor
from .base import Model


class Sequential(Model):
    """
    A simple sequential container that chains layers in order.
    Behaves like tf.keras.Sequential.
    """

    def __init__(self, layers: list[Layer] | None = None, name: str | None = None):
        super().__init__(name=name)
        self._layers: list[Layer] = []
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer) -> None:
        """Append a layer to the model."""
        self._layers.append(layer)

    def pop(self) -> None:
        """
        Remove the last layer in the sequence.
        Raises ValueError if no layers remain.
        """
        if not self._layers:
            raise ValueError("Cannot pop from an empty Sequential model.")
        self._layers.pop()

    def build(
        self, input_shape: tuple[int, ...], dtype: np.typing.DTypeLike = np.float32
    ) -> None:
        """
        Build all layers given an input shape (tuple of ints).
        Collects their variables into the model.
        """
        # Delegate to Model._build_layers, but temporarily replace _layers
        self._build_layers(input_shape, dtype)

    def call(
        self, inputs: Tensor, training: bool = False, mask: Tensor | None = None
    ) -> Tensor:
        """
        Forward pass through all layers in order.
        Inherits lazy-build behavior from Model.call.
        """
        return super().call(inputs, training=training, mask=mask)

    def get_weights(self) -> list[np.ndarray]:
        """
        Return a flat list of all layer weights (as NumPy arrays), in layer order.
        """
        weights: list[np.ndarray] = []
        for layer in self._layers:
            weights.extend(var.numpy() for var in layer.get_weights())
        return weights

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """
        Set weights from a flat list of NumPy arrays. Each layer
        consumes as many arrays as it needs (matching layer.get_weights()).
        """
        idx = 0
        for layer in self._layers:
            num = len(layer.get_weights())
            layer.set_weights(weights[idx : idx + num])
            idx += num
        if idx != len(weights):
            raise ValueError(
                f"Provided {len(weights)} weight arrays, "
                f"but Sequential expects {idx} total."
            )

    def summary(self) -> None:
        """
        Print a summary table: layer name (class), output shape, and parameter count.
        """
        if not self._built:
            print("Model not built yet. Run build(...) or a forward pass to build.")
            return

        print(f"Model: {self._name}")
        print(f"{'Layer (type)':<30s}{'Output Shape':<20s}{'# Params':>10s}")
        print("=" * 60)
        total_params = 0
        trainable_params = 0

        shape = self._input_shape or ()
        for layer in self._layers:
            out_shape = layer.output_shape or layer.compute_output_shape(shape)
            params = layer.count_params()
            total_params += params
            if layer.trainable:
                trainable_params += params

            name_type = f"{layer.name} ({layer.__class__.__name__})"
            print(f"{name_type:<30s}{str(out_shape):<20s}{params:>10d}")
            shape = out_shape

        non_trainable = total_params - trainable_params
        print("=" * 60)
        print(f"{'Total params:':<30s}{total_params:>30d}")
        print(f"{'Trainable params:':<30s}{trainable_params:>30d}")
        print(f"{'Non-trainable params:':<30s}{non_trainable:>30d}")

    @property
    def layers(self) -> list[Layer]:
        """List of layers in this Sequential model."""
        return self._layers
