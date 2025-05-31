from collections.abc import Callable
from typing import Any, Self

import numpy as np

from ..tape import GradientTape
from ..types import Tensor, Variable


class Optimizer:
    """
    A base implementation of a TensorFlow-like optimizer.

    Handles iteration count, gradient computation, slot creation, and apply_gradients logic.
    Subclasses should implement update_step() to define variable updates.
    """

    def __init__(self) -> None:
        self._iterations: int = 0
        # Slots: {slot_name: {var_id: slot_variable}}
        self._slots: dict[str, dict[int, Variable]] = {}
        self._built: bool = False

    @staticmethod
    def __get_all_subclasses() -> dict[str, Self]:
        return {subclass.__name__: subclass for subclass in Optimizer.__subclasses__()}

    @classmethod
    def from_string(cls, optimizer_name: str) -> Self:
        return cls.__get_all_subclasses()[optimizer_name]

    @property
    def iterations(self) -> int:
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        self._iterations = value

    def compute_gradients(
        self,
        loss: Callable[[], Any],
        var_list: list[Variable],
        grad_loss: Tensor = None,
        tape: GradientTape = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Compute gradients of loss w.r.t. var_list using GradientTape.
        If tape is not provided, a new one is created.
        Returns a list of (gradient, variable) tuples.
        """
        own_tape = False
        if tape is None:
            tape = GradientTape()
            own_tape = True

        if own_tape:
            tape.__enter__()
            try:
                loss_value = loss()
            finally:
                tape.__exit__(None, None, None)
        else:
            loss_value = loss()

        # Compute gradients
        grads = tape.gradient(loss_value, var_list, grad_loss)
        return list(zip(grads, var_list))

    def apply_gradients(self, grads_and_vars: list[tuple[Tensor, Variable]]) -> None:
        """
        Apply gradients to variables. Build slots if needed, then call update_step
        for each gradient-variable pair, incrementing iteration count.
        """
        if not self._built:
            self.build([v for _, v in grads_and_vars])
            self._built = True

        for grad, var in grads_and_vars:
            if grad is None or not var.trainable:
                continue
            # We still take the conjugate here (for complex support).
            self.update_step(np.conj(grad), var)

        self._iterations += 1

    def build(self, var_list: list[Variable]) -> None:
        """
        Initialize any optimizer-specific slots for each variable.
        Called once before the first application of gradients.
        """
        # Subclasses will typically call add_slot(var, slot_name) here.
        pass

    def add_slot(self, var: Variable, slot_name: str) -> None:
        """
        Create a slot Variable (initialized to zeros) for each variable under slot_name.
        The new slot lives independently of 'var', but has the same shape and dtype.
        """
        if slot_name not in self._slots:
            self._slots[slot_name] = {}
        var_id = id(var)
        if var_id in self._slots[slot_name]:
            return  # slot already exists

        # Build a NumPy array of zeros matching var.value's shape and dtype
        zero_arr = np.zeros_like(var.value)

        # Create a new Variable to hold these zeros (non-trainable)
        slot_var = Variable(
            value=zero_arr,
            trainable=False,
            name=f"{var.name}/{slot_name}",
            initializer="zeros",
        )
        slot_var.initialize()

        self._slots[slot_name][var_id] = slot_var

    def get_slot(self, var: Variable, slot_name: str) -> Variable:
        """
        Get the slot Variable (e.g. 'm' or 'v') for a given primary variable.
        """
        try:
            return self._slots[slot_name][id(var)]
        except KeyError:
            raise ValueError(f"Slot '{slot_name}' not found for variable {var}.") from None

    def update_step(self, gradient: Tensor, variable: Variable) -> None:
        """
        Apply one step of the update to a single variable.
        Subclasses must override this to implement specific optimizer rules.
        """
        raise NotImplementedError("Subclasses must implement update_step().")

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the optimizer for serialization.
        """
        return {"iterations": self._iterations}

    @classmethod
    def get_slot_names(cls) -> list[str]:
        """
        Return a list of slot names defined by this optimizer class.
        Subclasses should override if they create slots.
        """
        return []
