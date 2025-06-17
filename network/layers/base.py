from collections.abc import Callable

import numpy as np

from ..types.tensor import Tensor
from ..types.variable import Variable


class Layer:
    # Global counter to ensure unique layer names
    _layer_instance_count: dict[str, int] = {}

    def __init__(
        self,
        name: str | None = None,
        trainable: bool = True,
        dtype: np.typing.DTypeLike = np.float32,
    ):
        # Assign a unique name based on class name or user-provided name
        base_name = (name or self.__class__.__name__).lower()
        count = Layer._layer_instance_count.get(base_name, 0)
        Layer._layer_instance_count[base_name] = count + 1
        self.name: str = base_name if count == 0 else f"{base_name}_{count}"

        # Layer attributes
        self.trainable: bool = trainable
        self.dtype: np.dtype = np.dtype(dtype)
        self._built: bool = False

        # Weight/variable storage (Variable objects)
        self._trainable_weights: list[Variable] = []
        self._non_trainable_weights: list[Variable] = []
        self._trainable_variables: list[Variable] = []
        self._non_trainable_variables: list[Variable] = []
        # Weight names storage, parallel to weight lists
        self._trainable_weight_names: list[str] = []
        self._non_trainable_weight_names: list[str] = []

        # Losses and updates hooks
        self._losses: list[float] = []
        self._updates: list[Callable[[], None]] = []

        # Shape bookkeeping and input validation
        self.input_spec: dict[str, any] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.output_shape: tuple[int, ...] | None = None

    def build(self, input_shape: tuple[int, ...]) -> None:
        """
        Creates the variables of the layer. Subclasses should override this
        to call self.add_weight(...) as needed. After creating weights,
        set self._built = True.
        """
        self._built = True

    def call(self, *args, **kwargs) -> Tensor:
        """
        Defines the computation from inputs to outputs. Must be overridden
        by subclasses.
        """
        raise NotImplementedError("The call method must be implemented by subclasses.")

    def __call__(
        self, inputs: Tensor, training: bool = False, mask: Tensor | None = None
    ) -> Tensor:
        # 1) Cast inputs to layer dtype
        inputs = inputs.astype(self.dtype)

        # 2) Check input_spec (e.g. expected ndim, min/max rank, shape constraints)
        if self.input_spec is not None:
            if "ndim" in self.input_spec:
                expected_ndim = self.input_spec["ndim"]
                if inputs.ndim != expected_ndim:
                    raise ValueError(
                        f"Layer '{self.name}' expects input ndim={expected_ndim}, "
                        f"but got ndim={inputs.ndim}."
                    )
            if "shape" in self.input_spec:
                expected_shape = self.input_spec["shape"]
                # expected_shape can contain None for unknown dims
                for idx, (dim_expected, dim_actual) in enumerate(
                    zip(expected_shape, inputs.shape)
                ):
                    if dim_expected is not None and dim_expected != dim_actual:
                        raise ValueError(
                            f"Layer '{self.name}' expects input_shape[{idx}]={dim_expected}, "
                            f"but got {dim_actual}."
                        )

        # 3) Lazy build on first call
        if not self._built:
            self.build(inputs.shape)
            self.input_shape = inputs.shape
            self.output_shape = self.compute_output_shape(inputs.shape)

        return self.call(inputs, training=training, mask=mask)

    def add_weight(
        self,
        name: str,
        shape: tuple[int, ...],
        initializer: Callable[[tuple[int, ...]], Tensor] | None = None,
        trainable: bool = True,
    ) -> Variable:
        """
        Creates a weight Variable for the layer. If the layer itself is set
        trainable=False, the new Variable will always be non-trainable. The
        `name` argument is prepended with the layer's name to ensure uniqueness.
        """
        # Determine final "trainable" status for this weight
        final_trainable = self.trainable and trainable

        # Default initializer: random normal
        if initializer is None:

            def _default_init(shp: tuple[int, ...]) -> Tensor:
                return np.random.randn(*shp).astype(self.dtype)

            initializer = _default_init

        # Full weight name: "<layer_name>/<weight_name>"
        full_name = f"{self.name}/{name}"

        init_array = initializer(shape).astype(self.dtype)
        var = Variable(value=init_array, trainable=final_trainable, name=full_name)

        if final_trainable:
            self._trainable_weights.append(var)
            self._trainable_variables.append(var)
            self._trainable_weight_names.append(full_name)
        else:
            self._non_trainable_weights.append(var)
            self._non_trainable_variables.append(var)
            self._non_trainable_weight_names.append(full_name)

        return var

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Computes the output shape of the layer given the input shape.
        Subclasses should override when the output shape differs from input.
        """
        return input_shape

    def get_weights(self) -> list[Variable]:
        """
        Returns a flat list of all Variables (trainable + non-trainable).
        """
        return self._trainable_weights + self._non_trainable_weights

    def get_weight_names(self) -> list[str]:
        """
        Returns a flat list of all weight names (trainable + non-trainable), in the
        same order as get_weights().
        """
        return self._trainable_weight_names + self._non_trainable_weight_names

    def set_weights(self, weights: list[Tensor]) -> None:
        """
        Sets the Variables of the layer from a list of NumPy arrays or Tensors.
        The order must match get_weights(). Variables are updated via .assign().
        """
        total = len(self._trainable_weights) + len(self._non_trainable_weights)
        if len(weights) != total:
            raise ValueError(
                f"Layer '{self.name}' expected {total} weight arrays, but got {len(weights)}."
            )

        # Assign to trainable first, then non-trainable
        idx = 0
        for i in range(len(self._trainable_weights)):
            self._trainable_weights[i].assign(weights[idx].astype(self.dtype))
            idx += 1
        for i in range(len(self._non_trainable_weights)):
            self._non_trainable_weights[i].assign(weights[idx].astype(self.dtype))
            idx += 1

    def count_params(self) -> int:
        """
        Returns the total number of parameters (trainable + non-trainable).
        """
        return sum(int(np.prod(var.value.shape)) for var in self.get_weights())

    @property
    def trainable_variables(self) -> list[Variable]:
        return self._trainable_variables

    @property
    def non_trainable_variables(self) -> list[Variable]:
        return self._non_trainable_variables

    @property
    def variables(self) -> list[Variable]:
        return self._trainable_variables + self._non_trainable_variables

    @property
    def built(self) -> bool:
        return self._built

    # ----- Loss and Update Hooks -----
    def add_loss(self, loss_value: float) -> None:
        """
        Adds a scalar loss term (e.g., a regularization penalty) to this layer.
        """
        self._losses.append(loss_value)

    @property
    def losses(self) -> list[float]:
        """
        Returns any losses associated with this layer (e.g., regularizers).
        """
        return self._losses

    def add_update(self, update_fn: Callable[[], None]) -> None:
        """
        Registers a callable to be run later (e.g., BatchNorm moving-average update).
        """
        self._updates.append(update_fn)

    @property
    def updates(self) -> list[Callable[[], None]]:
        """
        Returns the list of update functions that should be executed after the forward pass.
        """
        return self._updates

    # ----- Serialization -----
    def get_config(self) -> dict:
        """
        Returns the configuration of the layer (hyperparameters needed to recreate it).
        Subclasses should extend this to include their own arguments.
        """
        return {
            "name": self.name,
            "trainable": self.trainable,
            "dtype": self.dtype.name,
        }

    @classmethod
    def from_config(cls, config: dict) -> "Layer":
        """
        Instantiates a layer from its config dictionary. Subclasses should override
        if they have extra args in get_config().
        """
        dtype = np.dtype(config["dtype"])
        return cls(
            name=config.get("name"),
            trainable=config.get("trainable", True),
            dtype=dtype,
        )
