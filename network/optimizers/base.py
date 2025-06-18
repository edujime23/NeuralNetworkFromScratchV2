import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Self

import numpy as np

from ..tape.gradient.gradient_tape import GradientTape
from ..types.tensor import Tensor
from ..types.variable import Variable
from .add_on import (
    AddonHookPoint,
    OptimizerAddon,
    OptimizerAddonMixin,
)

# Type aliases
GradVarPair = tuple[Tensor, Variable]
GradVarList = list[GradVarPair]
SlotDict = dict[str, dict[int, Variable]]


class Optimizer(ABC, OptimizerAddonMixin):
    """
    Enhanced TensorFlow-like optimizer with comprehensive add-on support.

    Features:
    - Advanced addon lifecycle management
    - Context-aware hook system
    - Error handling and recovery
    - Addon grouping and profiles
    - Performance monitoring
    """

    _registry: dict[str, type[Self]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        cls._registry[name] = cls

    @classmethod
    def from_string(cls, optimizer_name: str) -> Self:
        """
        Instantiate an optimizer by its class name (case-insensitive).
        e.g. "adam" → AdamOptimizer()
        """
        key = optimizer_name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown optimizer '{optimizer_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    def __init__(self, lr: float = 0.005) -> None:
        super().__init__()
        self._iterations: int = 0
        self._lr: float = lr
        # Slots: {slot_name: {var_id: Variable}}
        self._slots: dict[str, dict[int, Variable]] = {}
        self._built: bool = False

        # Enhanced add-on system
        self._addons: dict[str, OptimizerAddon] = {}
        self._hook_cache: dict[AddonHookPoint, list[OptimizerAddon]] = {}
        self._hooks_dirty = True
        self._logger = logging.getLogger(f"optimizer.{self.__class__.__name__}")

        # Performance tracking
        self._hook_call_counts: dict[str, int] = {}
        self._hook_timings: dict[str, float] = {}

        # Variable tracking for context
        self._current_variables: list[Variable] = []
        self._current_loss_value: Tensor | None = None

    @property
    def iterations(self) -> int:
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        self._iterations = value

    @property
    def lr(self) -> float:
        return self._lr

    def minimize(
        self,
        loss: Callable[[], Any],
        var_list: list[Variable],
        grad_loss: Tensor | None = None,
        tape: GradientTape | None = None,
    ) -> None:
        """
        Minimize loss function with full addon support.
        Combines compute_gradients and apply_gradients.
        """
        # Store variables for context
        self._current_variables = var_list

        # Pre-step hook
        self._call_hooks(AddonHookPoint.PRE_STEP)

        # Compute gradients
        grads_and_vars = self.compute_gradients(loss, var_list, grad_loss, tape)

        # Apply gradients
        self.apply_gradients(grads_and_vars)

        # Check convergence
        convergence_result = self._call_hooks(AddonHookPoint.ON_CONVERGENCE_CHECK)
        if convergence_result is True:
            self._logger.info("Convergence detected by addon")

        # Post-step hook
        self._call_hooks(AddonHookPoint.POST_STEP)

    def compute_gradients(
        self,
        loss: Callable[[], Any],
        var_list: list[Variable],
        grad_loss: Tensor | None = None,
        tape: GradientTape | None = None,
    ) -> GradVarList:
        """
        Compute gradients with enhanced addon support.
        """
        # Store variables for context
        self._current_variables = var_list

        # Pre-compute gradients hook
        loss, var_list, grad_loss, tape = self._call_hooks(
            AddonHookPoint.PRE_COMPUTE_GRADIENTS, loss, var_list, grad_loss, tape
        )

        own_tape = False
        if tape is None:
            tape = GradientTape()
            own_tape = True

        if own_tape:
            tape.__enter__()
            try:
                loss_value = loss()
                self._current_loss_value = loss_value
            finally:
                tape.__exit__(None, None, None)
        else:
            loss_value = loss()
            self._current_loss_value = loss_value

        grads = tape.gradient(loss_value, var_list, grad_loss)
        grads_and_vars = list(zip(grads, var_list))

        # Post-compute gradients hook
        grads_and_vars = self._call_hooks(
            AddonHookPoint.POST_COMPUTE_GRADIENTS,
            grads_and_vars,
            gradients=[g for g, _ in grads_and_vars],
        )

        return grads_and_vars

    def apply_gradients(self, grads_and_vars: GradVarList) -> None:
        """
        Apply gradients with comprehensive addon support.
        """
        # Pre-apply gradients hook
        grads_and_vars = self._call_hooks(
            AddonHookPoint.PRE_APPLY_GRADIENTS, grads_and_vars
        )

        # Build optimizer if needed
        if not self._built:
            self._lazy_build(grads_and_vars)
        # Apply updates to each variable
        for grad, var in grads_and_vars:
            if grad is None or not var.trainable:
                continue

            # Complex conjugate for complex support
            processed_grad = np.conj(grad) if np.iscomplexobj(grad) else grad

            # Get optimizer slots
            slots = {
                slot_name: self.get_slot(var, slot_name) for slot_name in self._slots
            }

            # Pre-update step hook
            processed_grad, var = self._call_hooks(
                AddonHookPoint.PRE_UPDATE_STEP, processed_grad, var
            )

            # Compute update
            update = self.update_step(processed_grad, slots)
            # Pre-apply update hook (new)
            update, var = self._call_hooks(AddonHookPoint.PRE_APPLY_UPDATE, update, var)

            # Post-update step hook
            self._call_hooks(
                AddonHookPoint.POST_UPDATE_STEP, processed_grad, var, update
            )

            # Ensure dtype consistency
            original_dtype = var.value.dtype
            if update.dtype != original_dtype:
                if np.iscomplexobj(update) and not np.iscomplexobj(var.value):
                    update = update.real.astype(original_dtype)
                else:
                    update = update.astype(original_dtype)

            # Apply update
            var.assign_sub(update)

        self._iterations += 1

        # Post-apply gradients hook
        self._call_hooks(AddonHookPoint.POST_APPLY_GRADIENTS, grads_and_vars)

    # TODO Rename this here and in `apply_gradients`
    def _lazy_build(self, grads_and_vars):
        var_list = [v for _, v in grads_and_vars]
        dtypes = [v.dtype for v in var_list]

        # Pre-build hook
        var_list, dtypes, self._slots = self._call_hooks(
            AddonHookPoint.PRE_BUILD, var_list, dtypes, self._slots
        )

        self.build(var_list, dtypes=dtypes)
        self._built = True

        # Post-build hook
        self._call_hooks(AddonHookPoint.POST_BUILD, var_list, self._slots)

        # Notify addons about new variables
        for var in var_list:
            self._call_hooks(AddonHookPoint.ON_VARIABLE_CREATED, var)

    @abstractmethod
    def build(
        self, var_list: list[Variable], dtypes: list[np.dtype] | None = None
    ) -> None:
        """
        Initialize optimizer-specific slots for each variable.
        """
        pass

    def add_slot(
        self, var: Variable, slot_name: str, dtype: np.dtype | None = None
    ) -> None:
        """
        Create slot Variable with enhanced error handling.
        """
        if slot_name not in self._slots:
            self._slots[slot_name] = {}

        var_id = id(var)
        if var_id in self._slots[slot_name]:
            return  # Slot already exists

        try:
            zero_arr = np.zeros_like(var.value, dtype=dtype)
            slot_var = Variable(
                value=zero_arr,
                dtype=dtype or var.dtype,
                trainable=False,
                name=f"{var.name}/{slot_name}",
                initializer="zeros",
            )
            slot_var.initialize()
            self._slots[slot_name][var_id] = slot_var

        except Exception as e:
            self._logger.error(f"Failed to create slot {slot_name} for {var.name}: {e}")
            raise

    def get_slot(self, var: Variable, slot_name: str) -> Variable:
        """
        Get slot Variable with enhanced error handling.
        """
        try:
            return self._slots[slot_name][id(var)]
        except KeyError as e:
            available_slots = list(self._slots.keys())
            raise ValueError(
                f"Slot '{slot_name}' not found for variable {var.name}. "
                f"Available slots: {available_slots}"
            ) from e

    @abstractmethod
    def update_step(
        self, gradient: Tensor, variable: Variable, slots: dict[str, Variable]
    ) -> Tensor:
        """
        Enhanced update step with slots dictionary.
        Returns the update tensor to be subtracted from the variable.
        """
        pass

    @staticmethod
    @abstractmethod
    def _update_step_math(*args, **kwargs):
        """Static method for update mathematics."""
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Enhanced configuration including addon information.
        """
        config = {
            "iterations": self._iterations,
            "built": self._built,
            "slot_names": list(self._slots.keys()),
        }

        if self._addons:
            config["addons"] = {
                name: addon.get_config() for name, addon in self._addons.items()
            }
            config["addon_groups"] = self._addon_groups.copy()

        # Performance statistics
        config["performance"] = {
            "hook_call_counts": self._hook_call_counts.copy(),
            "active_addons": len([a for a in self._addons.values() if a.enabled]),
        }

        return config

    @classmethod
    @abstractmethod
    def get_slot_names(cls) -> list[str]:
        """
        Return slot names defined by this optimizer class.
        """
        pass

    # Utility methods
    def summary(self) -> str:
        """Return a summary of the optimizer state."""
        addon_stats = self.get_addon_statistics()

        summary_lines = [
            f"Optimizer: {self.__class__.__name__}",
            f"Iterations: {self._iterations}",
            f"Built: {self._built}",
            f"Slots: {list(self._slots.keys())}",
            f"Addons: {addon_stats['total_addons']} total, {addon_stats['active_addons']} active",
        ]

        if addon_stats["error_addons"] > 0:
            summary_lines.append(f"Error addons: {addon_stats['error_addons']}")

        return "\n".join(summary_lines)

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current optimizer configuration."""
        issues = self.validate_addon_setup()

        # Add optimizer-specific validation
        optimizer_issues = {
            "unbuilt_with_slots": not self._built and bool(self._slots),
            "missing_slot_implementations": [],
        }

        # Check if required slots are properly implemented
        required_slots = self.get_slot_names()
        for slot_name in required_slots:
            if slot_name not in self._slots and self._built:
                optimizer_issues["missing_slot_implementations"].append(slot_name)

        return {"addon_issues": issues, "optimizer_issues": optimizer_issues}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(iterations={self._iterations}, addons={len(self._addons)})"
