import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Self

import numpy as np

from ..plugins.base.plugin import PluginContext
from ..plugins.optimizer.base import OptimizerPluginMixin
from ..plugins.optimizer.hooks import OptimizerHookPoints
from ..tape.gradient.api import GradientTape
from ..types.tensor import Tensor
from ..types.variable import Variable

# Type aliases
GradVarPair = tuple[Tensor, Variable]
GradVarList = list[GradVarPair]
SlotDict = dict[str, dict[int, Variable]]


class Optimizer(ABC, OptimizerPluginMixin):
    """
    Enhanced TensorFlow-like optimizer with comprehensive plugin support.

    Features:
    - Advanced plugin lifecycle management
    - Context-aware hook system
    - Error handling and recovery
    - Plugin grouping and profiles
    - Performance monitoring
    """

    _registry: dict[str, type[Self]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        cls._registry[name] = cls

    @classmethod
    def from_string(cls, optimizer_name: str) -> Self:
        """Create optimizer instance by class name (case-insensitive)."""
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
        self._slots: SlotDict = {}
        self._built: bool = False

        # Context tracking for plugins
        self._current_variables: list[Variable] = []
        self._current_loss_value: Tensor | None = None
        self._last_gradients: list[Tensor] | None = None

        self._logger = logging.getLogger(f"optimizer.{self.__class__.__name__}")

    @property
    def iterations(self) -> int:
        return self._iterations

    @iterations.setter
    def iterations(self, value: int) -> None:
        self._iterations = value

    @property
    def lr(self) -> float:
        return self._lr

    def create_context(self, **kwargs) -> PluginContext:
        """Create optimizer-specific plugin context."""
        metadata = {
            "step": self._iterations,
            "lr": self._lr,
            "built": self._built,
            "variables": self._current_variables,
            "loss_value": self._current_loss_value,
            "gradients": self._last_gradients,
            "slots": self._slots,
            **kwargs,  # Allow additional context data
        }
        return PluginContext(host=self, metadata=metadata)

    def minimize(
        self,
        loss: Callable[[], Any],
        var_list: list[Variable],
        grad_loss: Tensor | None = None,
        tape: GradientTape | None = None,
    ) -> None:
        """Minimize loss function with full plugin support."""
        self._current_variables = var_list

        # Pre-step hook
        self._call_hooks(OptimizerHookPoints.PRE_STEP.value)

        # Compute gradients
        grads_and_vars = self.compute_gradients(loss, var_list, grad_loss, tape)

        # Apply gradients
        self.apply_gradients(grads_and_vars)

        # Check convergence
        convergence_result = self._call_hooks(
            OptimizerHookPoints.ON_CONVERGENCE_CHECK.value
        )
        if convergence_result is True:
            self._logger.info("Convergence detected by plugin")

        # Post-step hook
        self._call_hooks(OptimizerHookPoints.POST_STEP.value)

    def compute_gradients(
        self,
        loss: Callable[[], Any],
        var_list: list[Variable],
        grad_loss: Tensor | None = None,
        tape: GradientTape | None = None,
    ) -> GradVarList:
        """Compute gradients with plugin support."""
        self._current_variables = var_list

        # Pre-compute hook - plugins can modify inputs through context
        context = self.create_context(
            loss=loss, var_list=var_list, grad_loss=grad_loss, tape=tape
        )
        result = self._call_hooks(
            OptimizerHookPoints.PRE_COMPUTE_GRADIENTS.value, context
        )

        # Extract modified values from context if plugins changed them
        if isinstance(result, PluginContext):
            loss = result.metadata.get("loss", loss)
            var_list = result.metadata.get("var_list", var_list)
            grad_loss = result.metadata.get("grad_loss", grad_loss)
            tape = result.metadata.get("tape", tape)

        own_tape = tape is None
        if own_tape:
            tape = GradientTape()

        if own_tape:
            with tape:
                loss_value = loss()
                self._current_loss_value = loss_value
        else:
            loss_value = loss()
            self._current_loss_value = loss_value

        grads = tape.gradient(loss_value, var_list, grad_loss)
        self._last_gradients = grads
        grads_and_vars = list(zip(grads, var_list))

        # Post-compute hook
        context = self.create_context(grads_and_vars=grads_and_vars, gradients=grads)
        result = self._call_hooks(
            OptimizerHookPoints.POST_COMPUTE_GRADIENTS.value, context
        )

        # Extract modified grads_and_vars if plugins changed them
        if isinstance(result, PluginContext):
            grads_and_vars = result.metadata.get("grads_and_vars", grads_and_vars)

        return grads_and_vars

    def apply_gradients(self, grads_and_vars: GradVarList) -> None:
        """Apply gradients with comprehensive plugin support."""
        # Pre-apply hook
        context = self.create_context(grads_and_vars=grads_and_vars)
        result = self._call_hooks(
            OptimizerHookPoints.PRE_APPLY_GRADIENTS.value, context
        )

        if isinstance(result, PluginContext):
            grads_and_vars = result.metadata.get("grads_and_vars", grads_and_vars)

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
            context = self.create_context(
                gradient=processed_grad, variable=var, slots=slots
            )
            result = self._call_hooks(
                OptimizerHookPoints.PRE_UPDATE_STEP.value, context
            )

            if isinstance(result, PluginContext):
                processed_grad = result.metadata.get("gradient", processed_grad)
                var = result.metadata.get("variable", var)

            # Compute update
            update = self.update_step(processed_grad, var, slots)

            # Pre-apply update hook
            context = self.create_context(update=update, variable=var)
            result = self._call_hooks(
                OptimizerHookPoints.PRE_APPLY_UPDATE.value, context
            )

            if isinstance(result, PluginContext):
                update = result.metadata.get("update", update)
                var = result.metadata.get("variable", var)

            # Post-update step hook
            context = self.create_context(
                gradient=processed_grad, variable=var, update=update
            )
            self._call_hooks(OptimizerHookPoints.POST_UPDATE_STEP.value, context)

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
        context = self.create_context(grads_and_vars=grads_and_vars)
        self._call_hooks(OptimizerHookPoints.POST_APPLY_GRADIENTS.value, context)

    def _lazy_build(self, grads_and_vars: GradVarList) -> None:
        """Build optimizer with plugin support."""
        var_list = [v for _, v in grads_and_vars]
        dtypes = [v.dtype for v in var_list]

        # Pre-build hook
        context = self.create_context(
            var_list=var_list, dtypes=dtypes, slots=self._slots
        )
        result = self._call_hooks(OptimizerHookPoints.PRE_BUILD.value, context)

        if isinstance(result, PluginContext):
            var_list = result.metadata.get("var_list", var_list)
            dtypes = result.metadata.get("dtypes", dtypes)
            self._slots = result.metadata.get("slots", self._slots)

        self.build(var_list, dtypes=dtypes)
        self._built = True

        # Post-build hook
        context = self.create_context(var_list=var_list, slots=self._slots)
        self._call_hooks(OptimizerHookPoints.POST_BUILD.value, context)

        # Notify plugins about new variables
        for var in var_list:
            context = self.create_context(variable=var)
            self._call_hooks(OptimizerHookPoints.ON_VARIABLE_CREATED.value, context)

    @abstractmethod
    def build(
        self, var_list: list[Variable], dtypes: list[np.dtype] | None = None
    ) -> None:
        """Initialize optimizer-specific slots for each variable."""
        pass

    def add_slot(
        self, var: Variable, slot_name: str, dtype: np.dtype | None = None
    ) -> None:
        """Create slot Variable with enhanced error handling."""
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
        """Get slot Variable with enhanced error handling."""
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
        """Enhanced configuration including plugin information."""
        config = {
            "iterations": self._iterations,
            "lr": self._lr,
            "built": self._built,
            "slot_names": list(self._slots.keys()),
        }

        # Include plugin information
        plugin_stats = self.get_plugin_statistics()
        config |= {
            "plugins": {
                name: plugin.get_config() for name, plugin in self._plugins.items()
            },
            "plugin_groups": self._plugin_groups.copy(),
            "performance": {
                "hook_call_counts": self._hook_call_counts.copy(),
                "active_plugins": plugin_stats["active_plugins"],
            },
        }

        return config

    @classmethod
    @abstractmethod
    def get_slot_names(cls) -> list[str]:
        """Return slot names defined by this optimizer class."""
        pass

    def summary(self) -> str:
        """Return a summary of the optimizer state."""
        plugin_stats = self.get_plugin_statistics()

        summary_lines = [
            f"Optimizer: {self.__class__.__name__}",
            f"Learning Rate: {self._lr}",
            f"Iterations: {self._iterations}",
            f"Built: {self._built}",
            f"Slots: {list(self._slots.keys())}",
            f"Plugins: {plugin_stats['total_plugins']} total, {plugin_stats['active_plugins']} active",
        ]

        if plugin_stats["error_plugins"] > 0:
            summary_lines.append(f"Error plugins: {plugin_stats['error_plugins']}")

        return "\n".join(summary_lines)

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current optimizer configuration."""
        issues = self.validate_plugin_setup()

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

        return {"plugin_issues": issues, "optimizer_issues": optimizer_issues}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self._lr}, iterations={self._iterations}, plugins={len(self._plugins)})"
