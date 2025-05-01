import numpy as np
from typing import Tuple, Union, Callable
from .base import BaseType

class Variable(BaseType):
    def __new__(
        cls,
        value: np.typing.ArrayLike,
        shape: Tuple[int],
        dtype: np.typing.DTypeLike,
        trainable: bool,
        name: str,
        initializer: Union[str, Callable] = 'zeros'
    ):
        obj = super().__new__(cls, value, shape, dtype, name)
        obj.__trainable = trainable
        obj.__initializer = initializer
        return obj

    def initialize(self):
        if callable(self.__initializer):
            self.__initializer(self)
        else:
            self._initialize(self.view(np.ndarray), self.shape, self.__initializer)
            
    def assign(self, value: np.typing.ArrayLike):
        """
        Assigns a new value to the variable.

        Args:
            value: The new value to assign. Must be compatible with the variable's shape and dtype.
        """
        new_value = np.asarray(value, dtype=self.dtype)
        if new_value.shape != self.shape:
            # Attempt to reshape if possible, otherwise raise error
            try:
                new_value = new_value.reshape(self.shape)
            except ValueError:
                    raise ValueError(
                    f"Cannot assign value with shape {new_value.shape} to variable '{self.name}' "
                    f"with shape {self.shape}. Shapes are incompatible."
                )
        # Assign the new value to the underlying numpy array view
        self[:] = new_value
        # In a real framework, this assignment might also need to be recorded
        # by the GradientTape if you want to track operations that depend on
        # the updated variable value for higher-order gradients or control flow.
        # For a simple clone, direct assignment like this is common for optimizers.

    def assign_add(self, delta: np.typing.ArrayLike):
        """
        Adds a value to the variable's current value.

        Args:
            delta: The value to add. Must be compatible with the variable's shape and dtype.
        """
        delta_value = np.asarray(delta, dtype=self.dtype)
        # NumPy handles broadcasting for the addition operation
        try:
            self[:] = self + delta_value
        except ValueError as e:
                raise ValueError(
                f"Cannot add value with shape {delta_value.shape} to variable '{self.name}' "
                f"with shape {self.shape} due to broadcasting issues: {e}"
            )
        # Similar considerations for GradientTape recording as in assign()

    def assign_sub(self, delta: np.typing.ArrayLike):
        """
        Subtracts a value from the variable's current value.

        Args:
            delta: The value to subtract. Must be compatible with the variable's shape and dtype.
        """
        delta_value = np.asarray(delta, dtype=self.dtype)
            # NumPy handles broadcasting for the subtraction operation
        try:
            self[:] = self - delta_value
        except ValueError as e:
                raise ValueError(
                f"Cannot subtract value with shape {delta_value.shape} from variable '{self.name}' "
                f"with shape {self.shape} due to broadcasting issues: {e}"
            )

    def assign_mul(self, delta: np.typing.ArrayLike):
        """
        Multiplies the variable's current value by a value.

        Args:
            delta: The value to multiply
        """
        delta_value = np.asarray(delta, dtype=self.dtype)
        # NumPy handles broadcasting for the multiplication operation
        try:
            self[:] = self * delta_value
        except ValueError as e:
                raise ValueError(
                f"Cannot multiply variable '{self.name}' with value with shape {delta_value.shape} due to broadcasting issues: {e}"
            )

    def assign_div(self, delta: np.typing.ArrayLike):
        """
        Dives the variable's current value by a value.

        Args:
            delta: The value to divide
        """
        delta_value = np.asarray(delta, dtype=self.dtype)
        # NumPy handles broadcasting for the division operation
        try:
            self[:] = self / delta_value
        except ValueError as e:
            raise ValueError(
                f"Cannot multiply variable '{self.name}' with value with shape {delta_value.shape} due to broadcasting issues: {e}"
            )
        
        
    @property
    def trainable(self):
        return self.__trainable

    @property
    def initializer(self):
        return self.__initializer