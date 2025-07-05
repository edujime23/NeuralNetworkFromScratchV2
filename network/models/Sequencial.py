import numpy as np

from network.layers.base import Layer
from network.types.tensor import Tensor
from .base import Model


class Sequential(Model):
    """
    A sequential container that chains layers in order.
    Mimics tf.keras.Sequential behavior with lazy building and proper state management.
    """

    def __init__(self, layers: list[Layer] | None = None, name: str | None = None):
        super().__init__(name=name or "sequential")
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer) -> None:
        """
        Add a layer to the end of the model.

        Args:
            layer: Layer instance to add

        Raises:
            ValueError: If layer is None or model is already built
        """
        if layer is None:
            raise ValueError("Cannot add None layer to Sequential model")

        if self._built:
            raise ValueError(
                "Cannot add layers to a Sequential model after it has been built"
            )

        super().add(layer)

    def pop(self) -> Layer:
        """
        Remove and return the last layer in the sequence.

        Returns:
            The removed layer

        Raises:
            ValueError: If no layers remain or model is built
        """
        if not self._layers:
            raise ValueError("Cannot pop from an empty Sequential model")

        if self._built:
            raise ValueError("Cannot pop layers from a built Sequential model")

        return self._layers.pop()

    def build(
        self, input_shape: tuple[int, ...], dtype: np.typing.DTypeLike = np.float32
    ) -> None:
        """
        Explicitly build all layers with given input shape.

        Args:
            input_shape: Shape tuple (excluding batch dimension)
            dtype: Data type for computations
        """
        if self._built:
            return

        if not self._layers:
            raise ValueError("Cannot build Sequential model without layers")

        self._build_layers(input_shape, dtype)

    def call(
        self,
        inputs: Tensor,
        training: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through all layers in sequence.

        Args:
            inputs: Input tensor or array
            training: Whether in training mode
            mask: Optional mask tensor

        Returns:
            Output tensor after passing through all layers
        """
        return super().call(inputs, training=training, mask=mask)

    def get_layer(self, name: str | None = None, index: int | None = None) -> Layer:
        """
        Retrieve a layer by name or index.

        Args:
            name: Layer name to search for
            index: Layer index (0-based)

        Returns:
            The requested layer

        Raises:
            ValueError: If neither name nor index provided, or layer not found
        """
        if name is None and index is None:
            raise ValueError("Must provide either name or index")

        if name is not None and index is not None:
            raise ValueError("Cannot provide both name and index")

        if index is not None:
            if not 0 <= index < len(self._layers):
                raise ValueError(
                    f"Layer index {index} out of range [0, {len(self._layers)})"
                )
            return self._layers[index]

        # Search by name
        for layer in self._layers:
            if layer.name == name:
                return layer

        raise ValueError(f"No layer found with name '{name}'")

    def get_weights(self) -> list[np.ndarray]:
        """
        Get all weights as a flat list of numpy arrays.

        Returns:
            list of weight arrays in layer order
        """
        if not self._built:
            raise ValueError("Model must be built before getting weights")

        weights = []
        for layer in self._layers:
            layer_weights = layer.get_weights() if hasattr(layer, "get_weights") else []
            weights.extend(w.numpy if hasattr(w, "numpy") else w for w in layer_weights)
        return weights

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """
        Set weights from a flat list of numpy arrays.

        Args:
            weights: list of weight arrays

        Raises:
            ValueError: If weight count mismatch or model not built
        """
        if not self._built:
            raise ValueError("Model must be built before setting weights")

        expected_count = sum(
            len(layer.get_weights()) if hasattr(layer, "get_weights") else 0
            for layer in self._layers
        )

        if len(weights) != expected_count:
            raise ValueError(
                f"Expected {expected_count} weight arrays, got {len(weights)}"
            )

        weight_idx = 0
        for layer in self._layers:
            if hasattr(layer, "set_weights") and hasattr(layer, "get_weights"):
                layer_weight_count = len(layer.get_weights())
                if layer_weight_count > 0:
                    layer.set_weights(
                        weights[weight_idx : weight_idx + layer_weight_count]
                    )
                    weight_idx += layer_weight_count

    def count_params(self) -> int:
        """
        Count total number of parameters in the model.

        Returns:
            Total parameter count
        """
        if not self._built:
            return 0
        return sum(layer.count_params() for layer in self._layers)

    def to_json(self) -> str:
        """
        Return JSON representation of model architecture.

        Returns:
            JSON string containing model configuration
        """
        import json

        config = {
            "class_name": self.__class__.__name__,
            "config": {"name": self._name, "layers": []},
        }

        for layer in self._layers:
            layer_config = {
                "class_name": layer.__class__.__name__,
                "config": getattr(layer, "get_config", lambda: {})(),
            }
            config["config"]["layers"].append(layer_config)

        return json.dumps(config, indent=2)

    def reset_states(self) -> None:
        """Reset states of all stateful layers."""
        for layer in self._layers:
            if hasattr(layer, "reset_states"):
                layer.reset_states()

    @property
    def input_shape(self) -> tuple | None:
        """Get input shape (excluding batch dimension)."""
        return self._input_shape

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self._layers)

    def __getitem__(self, index: int) -> Layer:
        """Get layer by index."""
        return self._layers[index]

    def __iter__(self):
        """Iterate over layers."""
        return iter(self._layers)

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<Sequential name={self._name}, layers={len(self._layers)}>"
