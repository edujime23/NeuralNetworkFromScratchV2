from collections.abc import Callable

import numpy as np

from network.gradient_tape.api import GradientTape
from network.layers.base import Layer
from network.metrics.base import Metric
from network.plugins.model.mixin import ModelPluginMixin
from network.plugins.model.hooks import ModelHookPoints
from network.optimizers.base import Optimizer
from network.types.tensor import Tensor
from network.types.variable import Variable


class Model(ModelPluginMixin):
    """
    Base class for neural network models.

    • Lazy-build: first forward pass will build layers automatically.
    • Accepts either `Tensor` or raw NumPy array as input.
    • Tracks layers, variables, optimizer, loss, metrics.
    • Provides summary(), get_weights()/set_weights(), save_weights()/load_weights().
    """

    def __init__(self, name: str | None = None):
        super().__init__()  # Initialize plugin system
        self._name: str = name or self.__class__.__name__
        self._optimizer: Optimizer | None = None
        self._loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None
        self._metrics: list[Metric] = []
        self._layers: list[Layer] = []
        self._variables: list[Variable] = []
        self._built: bool = False
        self._input_shape: tuple[int, ...] | None = None
        self._dtype: np.typing.DTypeLike | None = None
        self.output_shape: tuple[int, ...] | None = None

    def compile(
        self,
        optimizer: str | Optimizer,
        loss: Callable[[Tensor, Tensor], Tensor],
        metrics: list[str | Metric] | None = None,
    ) -> None:
        self.call_hooks(
            ModelHookPoints.PRE_COMPILE, optimizer=optimizer, loss=loss, metrics=metrics
        )

        if isinstance(optimizer, str):
            self._optimizer = Optimizer.from_string(optimizer)
        else:
            self._optimizer = optimizer

        self._loss_fn = loss
        self._metrics = []
        if metrics is not None:
            self._metrics.extend(
                Metric.from_string(m) if isinstance(m, str) else m for m in metrics
            )

        self.call_hooks(ModelHookPoints.POST_COMPILE)

    def add(self, layer: Layer) -> None:
        """Add a layer instance to the model."""
        self._layers.append(layer)

    def _build_layers(
        self, input_shape: tuple[int, ...], dtype: np.typing.DTypeLike
    ) -> None:
        self.call_hooks(ModelHookPoints.PRE_BUILD, input_shape=input_shape, dtype=dtype)

        self._input_shape = input_shape
        self._dtype = dtype
        shape: tuple[int, ...] = input_shape
        self._variables.clear()

        for layer in self._layers:
            if layer.input_spec is not None:
                spec = layer.input_spec
                if "ndim" in spec and len(shape) != spec["ndim"]:
                    raise ValueError(
                        f"Layer '{layer.name}' expects ndim={spec['ndim']}, but got {len(shape)}."
                    )
                if "shape" in spec:
                    expected = spec["shape"]
                    for idx, (exp_dim, actual_dim) in enumerate(zip(expected, shape)):
                        if exp_dim is not None and exp_dim != actual_dim:
                            raise ValueError(
                                f"Layer '{layer.name}' expects input_shape[{idx}]={exp_dim}, "
                                f"but got {actual_dim}."
                            )

            layer.build(shape)
            layer.input_shape = shape
            shape = layer.compute_output_shape(shape)
            layer.output_shape = shape

            self._variables.extend(layer.trainable_variables)
            self._variables.extend(layer.non_trainable_variables)

        self.output_shape = shape
        self._built = True

        self.call_hooks(
            ModelHookPoints.POST_BUILD, input_shape=input_shape, dtype=dtype
        )

    def call(
        self,
        inputs: Tensor,
        training: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        if isinstance(inputs, np.ndarray):
            inputs = Tensor(inputs, dtype=inputs.dtype)

        x: Tensor = inputs.astype(self._dtype or inputs.dtype)

        if not self._built:
            self._build_layers(inputs.shape[1:], inputs.dtype)

        self.call_hooks(ModelHookPoints.PRE_FORWARD, x=x, training=training, mask=mask)

        for layer in self._layers:
            x = layer(x, training=training, mask=mask)

        self.call_hooks(ModelHookPoints.POST_FORWARD, x=x)
        return x

    def __call__(
        self,
        inputs: Tensor,
        training: bool = False,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.call(inputs, training=training, mask=mask)

    def train_step(self, x, y):
        if isinstance(x, np.ndarray):
            x = Tensor(x, dtype=x.dtype)
        if isinstance(y, np.ndarray):
            y = Tensor(y, dtype=y.dtype)

        if not self._built:
            self._build_layers(x.shape[1:], x.dtype)

        self.call_hooks(ModelHookPoints.PRE_TRAIN_STEP, x=x, y=y)

        with GradientTape() as tape:
            tape.watch(*self.trainable_variables)
            y_pred = self.call(x, training=True)
            loss_value = self._loss_fn(y, y_pred)

            reg_losses = []
            for layer in self._layers:
                reg_losses.extend(layer.losses)
            if reg_losses:
                total_reg = sum(reg_losses, Tensor(0.0))
                loss_value += total_reg

        grads = tape.gradient(loss_value, self.trainable_variables)
        for i in range(len(grads)):
            if grads[i] is None:
                continue
            if np.iscomplexobj(self.trainable_variables[i]):
                grads[i] = grads[i].ah
            elif np.isrealobj(self.trainable_variables[i]):
                grads[i] = grads[i].total

        self._optimizer.apply_gradients(list(zip(grads, self.trainable_variables)))

        logs = {"loss": float(loss_value.numpy.item())}
        for metric in self._metrics:
            metric.update_state(y, y_pred)
            logs[metric.name] = float(metric.result().numpy.item())

        self.call_hooks(ModelHookPoints.POST_TRAIN_STEP, logs=logs)
        return logs

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 1,
        batch_size: int = 32,
        validation_data: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        """
        Train the model over multiple epochs with optional validation.
        Accepts either Tensor or NumPy array for x, y, and validation_data.
        """
        # Wrap entire datasets into Tensors if needed
        if isinstance(x, np.ndarray):
            x = Tensor(x, dtype=x.dtype)
        if isinstance(y, np.ndarray):
            y = Tensor(y, dtype=y.dtype)

        if validation_data is not None:
            x_val, y_val = validation_data
            if isinstance(x_val, np.ndarray):
                x_val = Tensor(x_val, dtype=x_val.dtype)
            if isinstance(y_val, np.ndarray):
                y_val = Tensor(y_val, dtype=y_val.dtype)

        num_samples = x.shape[0]

        for epoch in range(epochs):
            self.call_hooks(ModelHookPoints.PRE_EPOCH, epoch=epoch)
            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            x_arr = x.numpy[indices]
            y_arr = y.numpy[indices]
            x_shuf = Tensor(x_arr, dtype=x.dtype)
            y_shuf = Tensor(y_arr, dtype=y.dtype)

            # Reset metrics
            for metric in self._metrics:
                metric.reset_states()

            epoch_loss = 0.0
            steps = 0
            for batch_start in range(0, num_samples, batch_size):
                xb = x_shuf[batch_start : batch_start + batch_size]
                yb = y_shuf[batch_start : batch_start + batch_size]

                self.call_hooks(
                    ModelHookPoints.PRE_BATCH,
                    epoch=epoch,
                    batch_idx=steps,
                    inputs=xb,
                    targets=yb,
                )
                batch_logs = self.train_step(xb, yb)

                self.call_hooks(
                    ModelHookPoints.POST_BATCH,
                    epoch=epoch,
                    batch_idx=steps,
                    logs=batch_logs,
                )

                epoch_loss += batch_logs["loss"]
                steps += 1

            avg_loss = epoch_loss / steps
            epoch_metrics = {
                m.name: float(m.result().numpy.item()) for m in self._metrics
            }
            logs_epoch = {"loss": avg_loss, **epoch_metrics}

            # Validation at epoch end
            if validation_data is not None:
                y_val_pred = self.call(x_val, training=False)
                val_loss_tensor = self._loss_fn(y_val, y_val_pred)
                val_loss = float(val_loss_tensor.numpy.item())

                for m in self._metrics:
                    m.reset_states()
                    m.update_state(y_val, y_val_pred)
                val_metrics = {
                    m.name: float(m.result().numpy.item()) for m in self._metrics
                }

                logs_epoch |= {
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }

            self.call_hooks(ModelHookPoints.POST_EPOCH, epoch=epoch, logs=logs_epoch)

            if (epoch + 1) % np.ceil(epoch / 100) == 0:
                print(f"Epoch {epoch+1}/{epochs} - ", logs_epoch)

    def summary(self) -> None:
        """
        Print a tabular summary of layers, output shapes, and parameter counts.
        """
        if not self._built:
            print(
                "Model not built yet. Call model.build(...) or run one batch through fit()."
            )
            return

        print(f"Model: {self._name}")
        print(f"{'Layer (type)':<30s}{'Output Shape':<20s}{'# Params':>10s}")
        print("=" * 70)
        total_params = 0
        trainable_params = 0

        for layer in self._layers:
            layer_name = f"{layer.name} ({layer.__class__.__name__})"
            out_shape = layer.output_shape or ()
            params = layer.count_params()
            total_params += params
            if layer.trainable:
                trainable_params += params
            print(f"{layer_name:<30s}{str(out_shape):<20s}{params:>10d}")

        non_trainable = total_params - trainable_params
        print("=" * 70)
        print(f"{'Plugin (name)':<30s}{'State':<20s}{'Priority':>10s}")
        for plugin_name, plugin in self._plugins.items():
            print(f"{plugin_name:<30s}{plugin.state.name:<20s}{plugin.priority:>10d}")

        print("=" * 70)
        print(f"{'Hook (name):':<30s}{'# of Calls':>20s}")
        print("=" * 70)
        for hook_name, count in self._hook_call_counts.items():
            print(hook_name, count)
            print(f"{hook_name:<30s}{count:>20d}")
        print("=" * 70)
        print(f"{'Total params:':<30s}{total_params:>30d}")
        print(f"{'Trainable params:':<30s}{trainable_params:>30d}")
        print(f"{'Non-trainable params:':<30s}{non_trainable:>30d}")

    def gradient(self, x: Tensor, y: Tensor) -> list[Tensor]:
        """
        Compute gradients of loss w.r.t. trainable variables for inputs x, y.
        Accepts either Tensor or NumPy array.
        """
        if isinstance(x, np.ndarray):
            x = Tensor(x, dtype=x.dtype)
        if isinstance(y, np.ndarray):
            y = Tensor(y, dtype=y.dtype)

        if not self._built:
            self._build_layers(x.shape[1:], x.dtype)

        with GradientTape() as tape:
            tape.watch(*self.trainable_variables)
            y_pred = self.call(x, training=True)
            loss = self._compute_loss(y, y_pred)

        return tape.gradient(loss, self.trainable_variables)

    def _compute_loss(self, y: Tensor, y_pred: Tensor) -> Tensor:
        loss_value = self._loss_fn(y, y_pred)
        reg_losses = []
        for layer in self._layers:
            reg_losses.extend(layer.losses)
        if reg_losses:
            total_reg = sum(reg_losses, Tensor(0.0))
            loss_value = loss_value + total_reg
        return loss_value

    def get_weights(self) -> list[np.ndarray]:
        return [v.value.numpy for v in self._variables]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        if len(weights) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} weights, but got {len(weights)}."
            )
        for var, arr in zip(self._variables, weights):
            var.assign(arr)

    def save_weights(self, filepath: str) -> None:
        arrays = {f"var_{i}": v.value.numpy for i, v in enumerate(self._variables)}
        np.savez(filepath, **arrays)

    def load_weights(self, filepath: str) -> None:
        data = np.load(filepath)
        expected = len(self._variables)
        if len(data.files) != expected:
            raise ValueError(
                f"Expected {expected} arrays in checkpoint, found {len(data.files)}."
            )
        for i, key in enumerate(sorted(data.files, key=lambda s: int(s.split("_")[1]))):
            self._variables[i].assign(data[key])

    @property
    def variables(self) -> list[Variable]:
        return self._variables

    @property
    def trainable_variables(self) -> list[Variable]:
        return [v for v in self._variables if getattr(v, "trainable", False)]

    @property
    def non_trainable_variables(self) -> list[Variable]:
        return [v for v in self._variables if not v.trainable]

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    @property
    def name(self) -> str:
        return self._name
