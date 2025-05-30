import numpy as np
from typing import Self, Tuple, Callable, Dict, List, Any

from ..types import Variable, Tensor
from ..tape import GradientTape

class Optimizer:
    """
    A base implementation of a TensorFlow-like optimizer.

    Handles iteration count, gradient computation, slot creation, and apply_gradients logic.
    Subclasses should implement update_step() to define variable updates.
    """
    def __init__(self) -> None:
        self._iterations: int = 0
        # Slots: {slot_name: {var_id: slot_value}}
        self._slots: Dict[str, Dict[int, Variable]] = {}
        self._built: bool = False
        
    
    @staticmethod
    def __get_all_subclasses() -> Dict[str, Self]:
        return {
            subclass.__name__: subclass
            for subclass in Optimizer.__subclasses__()
        }
    
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
        var_list: List[Variable],
        grad_loss: Tensor = None,
        tape: GradientTape = None
    ) -> List[Tuple[Tensor, Tensor]]:
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

    def apply_gradients(
        self,
        grads_and_vars: List[Tuple[Tensor, Variable]]
    ) -> None:
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
            self.update_step(np.conj(grad), var)

        self._iterations += 1

    def build(self, var_list: List[Variable]) -> None:
        """
        Initialize any optimizer-specific slots for each variable.
        Called once before the first application of gradients.
        """
        # Example: subclasses may call add_slot(var, name)
        pass

    def add_slot(self, var: Variable, slot_name: str) -> None:
        """
        Create a slot tensor for a given variable under slot_name.
        """
        if slot_name not in self._slots:
            self._slots[slot_name] = {}
        var_id = id(var)
        if var_id in self._slots[slot_name]:
            return  # slot already exists
        # initialize slot with zeros of same shape as var
        self._slots[slot_name][var_id] = var

    def get_slot(self, var: Variable, slot_name: str) -> Variable:
        """
        Get the value of a slot for a given variable.
        """
        try:
            return self._slots[slot_name][id(var)]
        except KeyError:
            raise ValueError(f"Slot '{slot_name}' not found for variable {var}.")

    def update_step(self, gradient: Tensor, variable: Variable) -> None:
        """
        Apply one step of the update to a single variable.
        Subclasses must override this method to implement specific optimizer rules.
        """
        raise NotImplementedError("Subclasses must implement update_step().")

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the optimizer for serialization.
        """
        return {"iterations": self._iterations}

    @classmethod
    def get_slot_names(cls) -> List[str]:
        """
        Return a list of slot names defined by this optimizer class.
        Subclasses should override if they create slots.
        """
        return []
