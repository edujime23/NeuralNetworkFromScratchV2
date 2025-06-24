from __future__ import annotations

import numpy as np

from network.types.tensor import Tensor


class Metric:
    """
    Base class for metrics. Keeps internal state across batches.
    Subclasses must implement:
      - update_state(y_true: Tensor, y_pred: Tensor) -> None
      - result() -> Tensor
      - reset_states() -> None
    """

    _registry: dict[str, type[Metric]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register subclasses by lowercase class name (and any aliases they specify)
        name = getattr(cls, "NAME", cls.__name__).lower()
        cls._registry[name] = cls

    @classmethod
    def from_string(cls, name: str) -> Metric:
        """
        Instantiate a metric by its string name (case-insensitive).
        e.g. "mse" or "MeanSquaredError" → MeanSquaredError()
        """
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    def update_state(self, y_true: Tensor, y_pred: Tensor) -> None:
        """
        Accumulate state from a single batch. Must be overridden.
        """
        raise NotImplementedError("Must implement update_state in subclass.")

    def result(self) -> Tensor:
        """
        Compute and return the metric value (scalar Tensor) from accumulated state.
        Must be overridden.
        """
        raise NotImplementedError("Must implement result in subclass.")

    def reset_states(self) -> None:
        """
        Reset internal variables to initial state. Must be overridden.
        """
        raise NotImplementedError("Must implement reset_states in subclass.")

    @property
    def name(self) -> str:
        """
        Return a user-friendly name for this metric instance.
        By default, uses the lowercase class name.
        """
        return getattr(self, "NAME", self.__class__.__name__).lower()


class MeanSquaredError(Metric):
    """
    Computes the (running) mean squared error:
      state: sum_of_squared_errors, total_samples
      result: sum_of_squared_errors / total_samples
    """

    NAME = "mse"

    def __init__(self):
        self.reset_states()

    def update_state(self, y_true: Tensor, y_pred: Tensor) -> None:
        """
        y_true, y_pred: Tensor of shape (batch_size, ...).
        Accumulate sum of squared errors and sample count.
        """
        # Convert to NumPy for accumulation
        y_true_arr = y_true.numpy()
        y_pred_arr = y_pred.numpy()

        # Compute batch squared error
        err = y_true_arr - y_pred_arr
        sq_err = np.square(err)

        # Sum over all elements in the batch
        batch_sum = float(np.sum(sq_err))
        batch_count = float(np.prod(y_true_arr.shape))

        self._sum_squared_error += batch_sum
        self._total_count += batch_count

    def result(self) -> Tensor:
        """
        Return the overall MSE as a scalar Tensor.
        """
        if self._total_count == 0:
            # If no samples seen yet, return 0.0
            return Tensor(0.0, dtype=np.float32)
        mse_value = self._sum_squared_error / self._total_count
        return Tensor(mse_value, dtype=np.float32)

    def reset_states(self) -> None:
        """
        Reset sum and count to zero.
        """
        self._sum_squared_error: float = 0.0
        self._total_count: float = 0.0


class MeanAbsoluteError(Metric):
    """
    Computes the (running) mean absolute error.
    """

    NAME = "mae"

    def __init__(self):
        self.reset_states()

    def update_state(self, y_true: Tensor, y_pred: Tensor) -> None:
        y_true_arr = y_true.numpy()
        y_pred_arr = y_pred.numpy()

        err = np.abs(y_true_arr - y_pred_arr)
        batch_sum = float(np.sum(err))
        batch_count = float(np.prod(y_true_arr.shape))

        self._sum_abs_error += batch_sum
        self._total_count += batch_count

    def result(self) -> Tensor:
        if self._total_count == 0:
            return Tensor(0.0, dtype=np.float32)
        mae_value = self._sum_abs_error / self._total_count
        return Tensor(mae_value, dtype=np.float32)

    def reset_states(self) -> None:
        self._sum_abs_error: float = 0.0
        self._total_count: float = 0.0
