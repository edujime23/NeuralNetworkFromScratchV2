from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import numpy as np

from network.types.tensor import Tensor
from network.types.variable import Variable


class Layer(ABC):
    _layer_instance_count: dict[str, int] = {}

    def __init__(
        self,
        name: str | None = None,
        trainable: bool = True,
        dtype: np.typing.DTypeLike = np.float32,
    ):
        base = (name or self.__class__.__name__).lower()
        idx = Layer._layer_instance_count.get(base, 0)
        Layer._layer_instance_count[base] = idx + 1

        self.name: str = base if idx == 0 else f"{base}_{idx}"
        self.trainable: bool = trainable
        self.dtype: np.dtype = np.dtype(dtype)

        self._built: bool = False
        self._trainable_weights: list[Variable] = []
        self._non_trainable_weights: list[Variable] = []

        self._losses: list[float] = []
        self._updates: list[Callable[[], None]] = []

        self.input_shape: tuple[int, ...] | None = None
        self.output_shape: tuple[int, ...] | None = None
        self.input_spec: dict[str, Any] | None = None

    def __call__(
        self, inputs: Tensor, training: bool = False, mask: Tensor | None = None
    ) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError(f"Input to layer '{self.name}' must be a Tensor.")

        inputs = inputs.astype(self.dtype)
        self._validate_input_spec(inputs)

        if not self._built:
            self.build(inputs.shape)
            self.input_shape = inputs.shape
            self.output_shape = self.compute_output_shape(inputs.shape)

        return self.call(inputs, training=training, mask=mask)

    def _validate_input_spec(self, inputs: Tensor) -> None:
        spec = self.input_spec
        if spec is None:
            return

        # extract per-sample info
        total_ndim = inputs.ndim
        sample_shape = inputs.shape[1:]
        sample_ndim = total_ndim - 1

        if "min_ndim" in spec and sample_ndim < spec["min_ndim"]:
            raise ValueError(
                f"{self.name} expects sample ndim >= {spec['min_ndim']}, "
                f"got {sample_ndim}"
            )

        if "shape" in spec:
            if len(sample_shape) != len(spec["shape"]):
                raise ValueError(
                    f"{self.name} expects sample rank {len(spec['shape'])}, "
                    f"got {len(sample_shape)}"
                )
            for idx, dim in enumerate(spec["shape"]):
                if dim is not None and sample_shape[idx] != dim:
                    raise ValueError(
                        f"{self.name} expects sample_shape[{idx}] = {dim}, "
                        f"got {sample_shape[idx]}"
                    )

        if "axes" in spec:
            for axis, dim in spec["axes"].items():
                if sample_shape[axis] != dim:
                    raise ValueError(
                        f"{self.name} expects sample_shape[{axis}] = {dim}, "
                        f"got {sample_shape[axis]}"
                    )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """To be optionally overridden: weight creation and setup."""
        self._built = True

    @abstractmethod
    def call(
        self, inputs: Tensor, training: bool = False, mask: Tensor | None = None
    ) -> Tensor:
        """Defines computation. Must be implemented by subclass."""
        pass

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def add_weight(
        self,
        name: str,
        shape: tuple[int, ...],
        initializer: Callable[[tuple[int, ...]], Tensor] | None = None,
        trainable: bool = True,
    ) -> Variable:
        initializer = initializer or self._get_default_initializer(shape)
        full_name = f"{self.name}/{name}"
        tensor = initializer(shape)

        var = Variable(
            value=tensor,
            trainable=self.trainable and trainable,
            name=full_name,
            dtype=self.dtype,
        )

        (
            self._trainable_weights if var.trainable else self._non_trainable_weights
        ).append(var)
        return var

    def _get_default_initializer(
        self, shape: tuple[int, ...]
    ) -> Callable[[tuple[int, ...]], Tensor]:
        def glorot_uniform(shp: tuple[int, ...]) -> Tensor:
            fan_in = shp[-2] if len(shp) >= 2 else shp[0] if shp else 1
            fan_out = shp[-1] if len(shp) >= 2 else shp[0] if shp else 1
            limit = np.sqrt(6 / (fan_in + fan_out))
            return Tensor(np.random.uniform(-limit, limit, size=shp), dtype=self.dtype)

        return glorot_uniform

    def get_weights(self) -> list[Variable]:
        return self._trainable_weights + self._non_trainable_weights

    def set_weights(self, weights: list[Tensor]) -> None:
        variables = self.get_weights()
        if len(weights) != len(variables):
            raise ValueError(
                f"{self.name}: expected {len(variables)} weights, got {len(weights)}"
            )
        for var, weight in zip(variables, weights):
            var.assign(weight.astype(self.dtype))

    def count_params(self) -> int:
        return sum(int(np.prod(var.value.shape)) for var in self.get_weights())

    @property
    def variables(self) -> list[Variable]:
        return self._trainable_weights + self._non_trainable_weights

    @property
    def trainable_variables(self) -> list[Variable]:
        return self._trainable_weights

    @property
    def non_trainable_variables(self) -> list[Variable]:
        return self._non_trainable_weights

    @property
    def built(self) -> bool:
        return self._built

    def add_loss(self, loss_value: float) -> None:
        self._losses.append(loss_value)

    @property
    def losses(self) -> list[float]:
        return self._losses

    def add_update(self, update_fn: Callable[[], None]) -> None:
        self._updates.append(update_fn)

    @property
    def updates(self) -> list[Callable[[], None]]:
        return self._updates

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "trainable": self.trainable,
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_config(cls, config: dict) -> "Layer":
        return cls(
            name=config.get("name"),
            trainable=config.get("trainable", True),
            dtype=np.dtype(config.get("dtype", "float32")),
        )
