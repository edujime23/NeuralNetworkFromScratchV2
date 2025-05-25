from typing import List, Optional, Union, Tuple, Callable
import numpy as np
from ..optimizers import Optimizer
from ..layers import Layer
from ..types import Variable
from ..metrics import Metric
from ..initializers import Initializer
from ..tape import GradientTape

class Model:
    """
    Base class for neural network models.

    Manages layers, variables, optimizer, loss, and metrics.
    """
    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__
        self._optimizer: Optional[Optimizer] = None
        self._loss_fn: Optional[Callable] = None
        self._metrics: List[Metric] = []
        self._layers: List[Layer] = []
        self._variables: List[Variable] = []
        self._built: bool = False
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._dtype: np.typing.DTypeLike = None

    def compile(
        self,
        optimizer: Union[str, Optimizer],
        loss: Callable,
        metrics: List[Union[str, Metric]] = None
    ) -> None:
        """
        Configure the model for training.
        """
        if isinstance(optimizer, str):
            self._optimizer = Optimizer.from_string(optimizer)
        else:
            self._optimizer = optimizer
        self._loss_fn = loss
        self._metrics = []
        if metrics:
            for m in metrics:
                self._metrics.append(
                    Metric.from_string(m) if isinstance(m, str) else m
                )

    def add(self, layer: Layer) -> None:
        """Add a layer instance to the model."""
        self._layers.append(layer)

    def build(
        self,
        input_shape: Tuple[int, ...],
        dtype: Optional[np.typing.DTypeLike] = np.float32
    ) -> None:
        """
        Build all layers given an input shape and collect variables.
        """
        self._input_shape = input_shape
        self._dtype = dtype
        shape = input_shape
        for layer in self._layers:
            layer.build(shape, dtype)
            shape = layer.compute_output_shape(shape)
            # collect variables
            self._variables.extend(layer.trainable_variables)
            self._variables.extend(layer.non_trainable_variables)
        self._built = True

    def call(
        self,
        inputs: np.typing.NDArray,
        training: bool = False,
        mask: Optional[np.typing.NDArray] = None
    ) -> np.typing.NDArray:
        """Forward pass through the model layers."""
        x = inputs
        for layer in self._layers:
            x = layer(x, training=training, mask=mask)
        return x

    def add_weight(
        self,
        shape: Tuple[int, ...],
        initializer: Union[str, Callable],
        trainable: bool = True,
        name: Optional[str] = None
    ) -> Variable:
        """
        Create and track a new weight variable.
        """
        init_fn = Initializer.from_string(initializer) if isinstance(initializer, str) else initializer
        value = init_fn(shape, dtype=self._dtype)
        var = Variable(value, trainable=trainable, name=name)
        self._variables.append(var)
        return var

    def train_step(
        self,
        x: np.typing.NDArray,
        y: np.typing.NDArray
    ) -> dict:
        """
        Perform a single training step (forward, backward, update).
        Returns a log of loss and metric values.
        """
        if not self._built:
            self.build(x.shape[1:], dtype=x.dtype)
        with GradientTape() as tape:
            y_pred = self.call(x, training=True)
            loss_value = self._loss_fn(y, y_pred)
        grads = tape.gradient(loss_value, list(self.trainable_variables))
        self._optimizer.apply_gradients(grads_and_vars=list(zip(grads, self.trainable_variables)))
        logs = {'loss': loss_value}
        for metric in self._metrics:
            metric.update_state(y, y_pred)
            logs[metric.name] = float(metric.result())
        return logs

    def fit(
        self,
        x: np.typing.NDArray,
        y: np.typing.NDArray,
        epochs: int = 1,
        batch_size: int = 32
    ) -> None:
        """
        Train the model over multiple epochs.
        """
        num_samples = x.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuf, y_shuf = x[indices], y[indices]
            for start in range(0, num_samples, batch_size):
                xb = x_shuf[start:start+batch_size]
                yb = y_shuf[start:start+batch_size]
                logs = self.train_step(xb, yb)
            print(f"Epoch {epoch+1}/{epochs} - ", {k: v for k, v in logs.items()})

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @property
    def trainable_variables(self) -> List[Variable]:
        return [v for v in self._variables if getattr(v, 'trainable', False)]

    @property
    def non_trainable_variables(self) -> List[Variable]:
        return [v for v in self._variables if not v.trainable]

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    @property
    def name(self) -> str:
        return self._name
