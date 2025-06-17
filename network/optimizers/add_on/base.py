from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

import numpy as np

from ...tape.gradient.gradient_tape import GradientTape
from ...types.tensor import Tensor
from ...types.variable import Variable

# Type aliases for better readability
GradVarPair = tuple[Tensor, Variable]
GradVarList = list[GradVarPair]
SlotDict = dict[str, dict[int, Variable]]

T = TypeVar("T")


class AddonHookPoint(Enum):
    """Define when addons can hook into the optimization process."""

    PRE_COMPUTE_GRADIENTS = "pre_compute_gradients"
    POST_COMPUTE_GRADIENTS = "post_compute_gradients"
    PRE_APPLY_GRADIENTS = "pre_apply_gradients"
    PRE_BUILD = "pre_build"
    POST_BUILD = "post_build"
    PRE_UPDATE_STEP = "pre_update_step"
    PRE_APPLY_UPDATE = "pre_apply_update"  # New hook point
    POST_UPDATE_STEP = "post_update_step"
    POST_APPLY_GRADIENTS = "post_apply_gradients"
    # New hook points for enhanced functionality
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    ON_VARIABLE_CREATED = "on_variable_created"
    ON_CONVERGENCE_CHECK = "on_convergence_check"


@dataclass
class AddonContext:
    """Context object passed to addon hooks with useful information."""

    optimizer: "Optimizer"
    step: int
    lr: float
    variables: list[Variable]
    gradients: list[Tensor] | None = None
    loss_value: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_slot(self, var: Variable, slot_name: str) -> Variable:
        """Convenience method to get optimizer slot."""
        return self.optimizer.get_slot(var, slot_name)

    def add_slot(self, var: Variable, slot_name: str, dtype: np.dtype = None) -> None:
        """Convenience method to add optimizer slot."""
        self.optimizer.add_slot(var, slot_name, dtype)


class AddonPriority:
    """Predefined priority levels for common addon types."""

    CRITICAL = 1000  # System-level addons (logging, debugging)
    HIGH = 500  # Important modifications (gradient clipping)
    NORMAL = 100  # Regular addons (momentum, regularization)
    LOW = 50  # Cosmetic/optional addons (visualization)
    BACKGROUND = 0  # Background tasks (metrics collection)


class AddonState(Enum):
    """State of an addon."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"


@dataclass
class AddonConfig:
    """Configuration for addons with validation."""

    name: str
    priority: int = AddonPriority.NORMAL
    enabled: bool = True
    auto_enable: bool = True
    dependencies: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    requires_slots: list[str] = field(default_factory=list)
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)


class AddonError(Exception):
    """Base exception for addon-related errors."""

    pass


class AddonDependencyError(AddonError):
    """Raised when addon dependencies are not met."""

    pass


class AddonConflictError(AddonError):
    """Raised when addon conflicts with another addon."""

    pass


class OptimizerAddon:
    """Enhanced base class for optimizer add-ons with robust error handling."""

    def __init__(
        self,
        name: str = None,
        priority: int = AddonPriority.NORMAL,
        enabled: bool = True,
        auto_enable: bool = True,
        dependencies: list[str] = None,
        conflicts: list[str] = None,
        requires_slots: list[str] = None,
        **kwargs,
    ):
        self.config = AddonConfig(
            name=name or self.__class__.__name__.lower(),
            priority=priority,
            enabled=enabled,
            auto_enable=auto_enable,
            dependencies=dependencies or [],
            conflicts=conflicts or [],
            requires_slots=requires_slots or [],
            metadata=kwargs,
        )

        self._optimizer: "Optimizer" | None = None
        self._state = AddonState.INACTIVE
        self._error_count = 0
        self._logger = logging.getLogger(f"addon.{self.config.name}")
        self._hooks_cache: set[AddonHookPoint] = set()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def priority(self) -> int:
        return self.config.priority

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self._state == AddonState.ACTIVE

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.config.enabled = value
        if value and self._state == AddonState.SUSPENDED:
            self._state = AddonState.ACTIVE
        elif not value and self._state == AddonState.ACTIVE:
            self._state = AddonState.SUSPENDED

    @property
    def state(self) -> AddonState:
        return self._state

    def attach(self, optimizer: "Optimizer") -> None:
        """Called when addon is attached to optimizer."""
        self._optimizer = optimizer
        self._validate_dependencies()
        self._validate_conflicts()
        self._setup_required_slots()
        self._state = AddonState.ACTIVE if self.config.enabled else AddonState.SUSPENDED
        self._hooks_cache = set(self.get_hook_points())
        self._logger.debug(f"Addon {self.name} attached to optimizer")

    def detach(self) -> None:
        """Called when addon is detached from optimizer."""
        self._cleanup()
        self._optimizer = None
        self._state = AddonState.INACTIVE
        self._hooks_cache.clear()
        self._logger.debug(f"Addon {self.name} detached from optimizer")

    def _validate_dependencies(self) -> None:
        """Validate that all dependencies are met."""
        if not self.config.dependencies:
            return

        available_addons = set(self._optimizer.list_addons())
        if missing_deps := set(self.config.dependencies) - available_addons:
            raise AddonDependencyError(
                f"Addon {self.name} requires dependencies: {missing_deps}"
            )

    def _validate_conflicts(self) -> None:
        """Check for conflicts with existing addons."""
        if not self.config.conflicts:
            return

        available_addons = set(self._optimizer.list_addons())
        if conflicts := set(self.config.conflicts) & available_addons:
            raise AddonConflictError(f"Addon {self.name} conflicts with: {conflicts}")

    def _setup_required_slots(self) -> None:
        """Setup any required slots for this addon."""
        pass

    def _cleanup(self) -> None:
        """Cleanup resources when addon is detached."""
        pass

    def get_hook_points(self) -> list[AddonHookPoint]:
        """Return list of hook points this addon uses."""
        hooks = []
        for hook_point in AddonHookPoint:
            method_name = f"on_{hook_point.value}"
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                if callable(method) and method != getattr(
                    OptimizerAddon, method_name, None
                ):
                    hooks.append(hook_point)
        return hooks

    def _safe_call(self, method_name: str, *args, **kwargs):
        """Safely call a hook method with error handling."""
        if not self.enabled:
            return args[0] if len(args) == 1 else args

        try:
            method = getattr(self, method_name, None)
            if method and callable(method):
                result = method(*args, **kwargs)
                self._error_count = 0  # Reset error count on success
                return result
            return args[0] if len(args) == 1 else args

        except Exception as e:
            self._error_count += 1
            self._logger.error(f"Error in {method_name}: {e}")

            if self._error_count >= self.config.max_retries:
                self._state = AddonState.ERROR
                self._logger.error(f"Addon {self.name} disabled due to repeated errors")

            return args[0] if len(args) == 1 else args

    # Context manager support
    @contextmanager
    def temporary_disable(self):
        """Temporarily disable addon."""
        was_enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = was_enabled

    @contextmanager
    def temporary_priority(self, priority: int):
        """Temporarily change addon priority."""
        old_priority = self.config.priority
        self.config.priority = priority
        try:
            yield
        finally:
            self.config.priority = old_priority

    # Enhanced hook methods with context support
    def on_pre_compute_gradients(
        self,
        context: AddonContext,
        loss: Callable,
        var_list: list[Variable],
        grad_loss: Tensor = None,
        tape: GradientTape = None,
    ) -> tuple:
        """Called before computing gradients."""
        return loss, var_list, grad_loss, tape

    def on_post_compute_gradients(
        self, context: AddonContext, grads_and_vars: GradVarList
    ) -> GradVarList:
        """Called after computing gradients."""
        return grads_and_vars

    def on_pre_apply_gradients(
        self, context: AddonContext, grads_and_vars: GradVarList
    ) -> GradVarList:
        """Called before applying gradients."""
        return grads_and_vars

    def on_pre_update_step(
        self, context: AddonContext, gradient: Tensor, variable: Variable
    ) -> tuple[Tensor, Variable]:
        """Called before each variable update."""
        return gradient, variable

    def on_pre_apply_update(
        self, context: AddonContext, update: Tensor, variable: Variable
    ) -> Tensor:
        """Called before applying update to variable."""
        return update, variable

    def on_post_update_step(
        self,
        context: AddonContext,
        gradient: Tensor,
        variable: Variable,
        update: Tensor,
    ) -> None:
        """Called after each variable update."""
        pass

    def on_post_apply_gradients(
        self, context: AddonContext, grads_and_vars: GradVarList
    ) -> None:
        """Called after all gradients applied."""
        pass

    def on_pre_build(
        self,
        context: AddonContext,
        var_list: list[Variable],
        dtypes: list[np.dtype],
        slots: SlotDict,
    ) -> tuple:
        """Called before building optimizer slots."""
        return var_list, dtypes, slots

    def on_post_build(
        self, context: AddonContext, var_list: list[Variable], slots: SlotDict
    ) -> None:
        """Called after building optimizer slots."""
        pass

    def on_pre_step(self, context: AddonContext) -> None:
        """Called at the beginning of optimization step."""
        pass

    def on_post_step(self, context: AddonContext) -> None:
        """Called at the end of optimization step."""
        pass

    def on_variable_created(self, context: AddonContext, variable: Variable) -> None:
        """Called when a new variable is created."""
        pass

    def on_convergence_check(self, context: AddonContext) -> bool | None:
        """Called to check convergence. Return True to stop optimization."""
        return None

    # Utility methods
    def get_optimizer_slot(self, var: Variable, slot_name: str) -> Variable:
        """Get optimizer slot variable."""
        if not self._optimizer:
            raise RuntimeError("Addon not attached to optimizer")
        return self._optimizer.get_slot(var, slot_name)

    def add_optimizer_slot(
        self, var: Variable, slot_name: str, dtype: np.dtype = None
    ) -> None:
        """Add new optimizer slot."""
        if not self._optimizer:
            raise RuntimeError("Addon not attached to optimizer")
        self._optimizer.add_slot(var, slot_name, dtype)

    def get_optimizer_iterations(self) -> int:
        """Get current optimizer iterations."""
        if not self._optimizer:
            raise RuntimeError("Addon not attached to optimizer")
        return self._optimizer.iterations

    def get_config(self) -> dict[str, Any]:
        """Return addon configuration."""
        return {
            "name": self.config.name,
            "priority": self.config.priority,
            "enabled": self.config.enabled,
            "state": self._state.value,
            "dependencies": self.config.dependencies,
            "conflicts": self.config.conflicts,
            "requires_slots": self.config.requires_slots,
            "error_count": self._error_count,
            "metadata": self.config.metadata,
        }

    def reset_errors(self) -> None:
        """Reset error count and reactivate addon if it was in error state."""
        self._error_count = 0
        if self._state == AddonState.ERROR:
            self._state = (
                AddonState.ACTIVE if self.config.enabled else AddonState.SUSPENDED
            )


# Utility classes for common addon patterns
class SimpleAddon(OptimizerAddon):
    """Simple addon for single-hook implementations."""

    def __init__(self, hook_point: AddonHookPoint, hook_func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.hook_point = hook_point
        self.hook_func = hook_func

        # Dynamically set the hook method
        method_name = f"on_{hook_point.value}"
        setattr(self, method_name, self._call_hook)

    def _call_hook(self, context: AddonContext, *args, **kwargs):
        return self.hook_func(context, *args, **kwargs)

    def get_hook_points(self) -> list[AddonHookPoint]:
        return [self.hook_point]


class ConditionalAddon(OptimizerAddon):
    """Addon that only activates under certain conditions."""

    def __init__(self, condition: Callable[[AddonContext], bool], **kwargs):
        super().__init__(**kwargs)
        self.condition = condition

    def _safe_call(self, method_name: str, *args, **kwargs):
        if not self.enabled:
            return args[0] if len(args) == 1 else args

        # Check condition before calling hook
        if (
            hasattr(args[0], "__class__")
            and args[0].__class__.__name__ == "AddonContext"
        ):
            context = args[0]
            if not self.condition(context):
                return args[0] if len(args) == 1 else args

        return super()._safe_call(method_name, *args, **kwargs)


# Decorator for creating simple addons
def addon_hook(
    hook_point: AddonHookPoint, priority: int = AddonPriority.NORMAL, **kwargs
):
    """Decorator to create simple addons from functions."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Create addon class dynamically
        addon_name = kwargs.get("name", func.__name__)
        addon = SimpleAddon(
            hook_point=hook_point,
            hook_func=func,
            name=addon_name,
            priority=priority,
            **kwargs,
        )

        wrapper.addon = addon
        return wrapper

    return decorator


# Enhanced Optimizer Addon Management Methods
class OptimizerAddonMixin:
    """Mixin class for enhanced addon management in optimizers."""

    def __init__(self):
        super().__init__()
        self._addon_groups: dict[str, list[str]] = {}
        self._addon_profiles: dict[str, dict[str, Any]] = {}

    def add_addon_group(self, group_name: str, addon_names: list[str]) -> None:
        """Group addons for batch operations."""
        self._addon_groups[group_name] = addon_names

    def enable_addon_group(self, group_name: str) -> None:
        """Enable all addons in a group."""
        if group_name not in self._addon_groups:
            raise ValueError(f"Addon group '{group_name}' not found")

        for addon_name in self._addon_groups[group_name]:
            if addon_name in self._addons:
                self.enable_addon(addon_name)

    def disable_addon_group(self, group_name: str) -> None:
        """Disable all addons in a group."""
        if group_name not in self._addon_groups:
            raise ValueError(f"Addon group '{group_name}' not found")

        for addon_name in self._addon_groups[group_name]:
            if addon_name in self._addons:
                self.disable_addon(addon_name)

    def create_addon_profile(self, profile_name: str) -> None:
        """Create a profile of current addon configuration."""
        self._addon_profiles[profile_name] = {
            "addons": {
                name: addon.get_config() for name, addon in self._addons.items()
            },
            "groups": self._addon_groups.copy(),
        }

    def load_addon_profile(self, profile_name: str) -> None:
        """Load an addon profile."""
        if profile_name not in self._addon_profiles:
            raise ValueError(f"Addon profile '{profile_name}' not found")

        profile = self._addon_profiles[profile_name]

        # Restore addon states
        for addon_name, config in profile["addons"].items():
            if addon_name in self._addons:
                addon = self._addons[addon_name]
                addon.config.enabled = config["enabled"]
                addon.config.priority = config["priority"]

        # Restore groups
        self._addon_groups = profile["groups"].copy()

        # Rebuild hook cache
        self._hooks_dirty = True

    def get_addon_statistics(self) -> dict[str, Any]:
        """Get statistics about addon usage."""
        stats = {
            "total_addons": len(self._addons),
            "active_addons": sum(
                bool(addon.enabled) for addon in self._addons.values()
            ),
            "suspended_addons": sum(
                addon.state == AddonState.SUSPENDED for addon in self._addons.values()
            ),
            "error_addons": sum(
                addon.state == AddonState.ERROR for addon in self._addons.values()
            ),
            "hook_points_used": {},
            "priority_distribution": {},
        }

        # Count hook points usage
        for addon in self._addons.values():
            for hook_point in addon.get_hook_points():
                hook_name = hook_point.value
                stats["hook_points_used"][hook_name] = (
                    stats["hook_points_used"].get(hook_name, 0) + 1
                )

        # Priority distribution
        for addon in self._addons.values():
            priority_range = self._get_priority_range(addon.priority)
            stats["priority_distribution"][priority_range] = (
                stats["priority_distribution"].get(priority_range, 0) + 1
            )

        return stats

    def _get_priority_range(self, priority: int) -> str:
        """Get priority range name for statistics."""
        if priority >= AddonPriority.CRITICAL:
            return "critical"
        elif priority >= AddonPriority.HIGH:
            return "high"
        elif priority >= AddonPriority.NORMAL:
            return "normal"
        elif priority >= AddonPriority.LOW:
            return "low"
        else:
            return "background"

    def validate_addon_setup(self) -> dict[str, list[str]]:
        """Validate current addon setup and return issues."""
        issues = {
            "conflicts": [],
            "missing_dependencies": [],
            "circular_dependencies": [],
            "error_addons": [],
        }

        # Check for conflicts
        for addon_name, addon in self._addons.items():
            for conflict in addon.config.conflicts:
                if conflict in self._addons and self._addons[conflict].enabled:
                    issues["conflicts"].append(
                        f"{addon_name} conflicts with {conflict}"
                    )

        # Check dependencies
        for addon_name, addon in self._addons.items():
            for dep in addon.config.dependencies:
                if dep not in self._addons:
                    issues["missing_dependencies"].append(
                        f"{addon_name} requires {dep}"
                    )
                elif not self._addons[dep].enabled:
                    issues["missing_dependencies"].append(
                        f"{addon_name} requires {dep} to be enabled"
                    )

        # Check for error state addons
        for addon_name, addon in self._addons.items():
            if addon.state == AddonState.ERROR:
                issues["error_addons"].append(addon_name)

        return issues
