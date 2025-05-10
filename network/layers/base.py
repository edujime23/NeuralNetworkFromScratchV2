from typing import Optional, Tuple, Any, List
import numpy as np

class Layer:
    def __init__(self, name: Optional[str] = None, trainable: bool = True, dtype: Optional[np.typing.DTypeLike] = np.float32):
        self.name = name or self.__class__.__name__
        self.trainable = trainable
        self.dtype = dtype
        self._built = False
        self._trainable_weights: List[np.typing.NDArray[Any]] = []
        self._non_trainable_weights: List[np.typing.NDArray[Any]] = []
        self._trainable_variables: List[np.typing.NDArray[Any]] = []
        self._non_trainable_variables: List[np.typing.NDArray[Any]] = []

    def build(self, input_shape: Tuple[int, ...]):
        """
        Creates the variables of the layer. This method is called automatically
        the first time the layer is called.
        """
        self._built = True

    def call(self, inputs: np.typing.NDArray[Any], training: bool = False, mask: Optional[np.typing.NDArray[Any]] = None) -> np.typing.NDArray[Any]:
        """
        Defines the computation from inputs to outputs.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError("The call method must be implemented by subclasses.")

    def __call__(self, inputs: np.typing.NDArray[Any], training: bool = False, mask: Optional[np.typing.NDArray[Any]] = None) -> np.typing.NDArray[Any]:
        if not self._built:
            self.build(inputs.shape)
        return self.call(inputs, training=training, mask=mask)

    def add_weight(self, name: str, shape: Tuple[int, ...], initializer: Optional[callable] = None, trainable: bool = True) -> np.typing.NDArray[Any]:
        """
        Creates a weight variable for the layer.
        """
        if initializer is None:
            initializer = lambda shape: np.random.randn(*shape).astype(self.dtype)
        weight = initializer(shape)
        if trainable:
            self._trainable_weights.append(weight)
            self._trainable_variables.append(weight)
        else:
            self._non_trainable_weights.append(weight)
            self._non_trainable_variables.append(weight)
        return weight

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Computes the output shape of the layer given the input shape.
        Should be overridden by all subclasses.
        """
        return input_shape

    def get_weights(self) -> List[np.typing.NDArray[Any]]:
        """
        Returns the weights of the layer as a list of NumPy arrays.
        """
        return self._trainable_weights + self._non_trainable_weights

    def set_weights(self, weights: List[np.typing.NDArray[Any]]):
        """
        Sets the weights of the layer from a list of NumPy arrays.
        """
        total_weights = len(self._trainable_weights) + len(self._non_trainable_weights)
        if len(weights) != total_weights:
            raise ValueError(f"Expected {total_weights} weights, but got {len(weights)}.")
        for i, weight in enumerate(self._trainable_weights):
            self._trainable_weights[i] = weights[i]
        for i, weight in enumerate(self._non_trainable_weights):
            self._non_trainable_weights[i] = weights[len(self._trainable_weights) + i]

    def count_params(self) -> int:
        """
        Returns the total number of parameters (trainable and non-trainable) in the layer.
        """
        return sum(np.prod(weight.shape) for weight in self.get_weights())

    @property
    def trainable_variables(self) -> List[np.typing.NDArray[Any]]:
        return self._trainable_variables

    @property
    def non_trainable_variables(self) -> List[np.typing.NDArray[Any]]:
        return self._non_trainable_variables

    @property
    def variables(self) -> List[np.typing.NDArray[Any]]:
        return self._trainable_variables + self._non_trainable_variables

    @property
    def built(self) -> bool:
        return self._built