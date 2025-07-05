from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable
from weakref import WeakSet

T = TypeVar("T")
HostType = TypeVar("HostType")
ContextType = TypeVar("ContextType")


# Base Hook Point Protocol
@runtime_checkable
class PluginHookPoint(Protocol):
    """Protocol for hook points that can be used with the plugin system."""

    value: str


# Base Plugin Context
@dataclass
class PluginContext:
    """Base context object passed to plugin hooks."""

    host: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_host_attribute(self, attr_name: str, default=None):
        """Safely get attribute from host."""
        return getattr(self.host, attr_name, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """Get metadata value."""
        return self.metadata.get(key, default)


# Plugin Priority System
class PluginPriority:
    """Universal priority levels for plugins."""

    CRITICAL = 1000  # System-level plugins (logging, debugging)
    HIGH = 500  # Important modifications (validation, security)
    NORMAL = 100  # Regular plugins (business logic)
    LOW = 50  # Optional enhancements (UI, convenience)
    BACKGROUND = 0  # Background tasks (metrics, cleanup)


# Plugin State Management
class PluginState(Enum):
    """State of an plugin."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"


# Plugin Configuration
@dataclass
class PluginConfig:
    """Universal plugin configuration."""

    name: str
    priority: int = PluginPriority.NORMAL
    enabled: bool = True
    auto_enable: bool = True
    dependencies: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    requires_resources: list[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: float = 30.0  # Timeout for plugin operations
    metadata: dict[str, Any] = field(default_factory=dict)


# Plugin Exceptions
class PluginError(Exception):
    """Base exception for plugin-related errors."""

    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies are not met."""

    pass


class PluginConflictError(PluginError):
    """Raised when plugin conflicts with another plugin."""

    pass


class PluginTimeoutError(PluginError):
    """Raised when plugin operation times out."""

    pass


class PluginValidationError(PluginError):
    """Raised when plugin validation fails."""

    pass


# Base Plugin Class
class Plugin(Generic[HostType, ContextType], ABC):
    """Universal base class for all plugins."""

    def __init__(
        self,
        name: str = None,
        priority: int = PluginPriority.NORMAL,
        enabled: bool = True,
        auto_enable: bool = True,
        dependencies: list[str] = None,
        conflicts: list[str] = None,
        requires_resources: list[str] = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        self.config = PluginConfig(
            name=name or self.__class__.__name__.lower().replace("plugin", ""),
            priority=priority,
            enabled=enabled,
            auto_enable=auto_enable,
            dependencies=dependencies or [],
            conflicts=conflicts or [],
            requires_resources=requires_resources or [],
            timeout=timeout,
            metadata=kwargs,
        )

        self._host: HostType | None = None
        self._state = PluginState.INACTIVE
        self._error_count = 0
        self._logger = logging.getLogger(f"plugin.{self.config.name}")
        self._hooks_cache: set[PluginHookPoint] = set()
        self._listeners: WeakSet = WeakSet()

    # Properties
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def priority(self) -> int:
        return self.config.priority

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self._state == PluginState.ACTIVE

    @enabled.setter
    def enabled(self, value: bool) -> None:
        old_enabled = self.enabled
        self.config.enabled = value

        if value and self._state == PluginState.SUSPENDED:
            self._state = PluginState.ACTIVE
        elif not value and self._state == PluginState.ACTIVE:
            self._state = PluginState.SUSPENDED

        if old_enabled != self.enabled:
            self._notify_state_change()

    @property
    def state(self) -> PluginState:
        return self._state

    @property
    def host(self) -> HostType:
        return self._host

    # Lifecycle Methods
    def attach(self, host: HostType) -> None:
        """Called when plugin is attached to host."""
        if self._host.__class__ == host.__class__:
            raise PluginError(f"Plugin {self.name} is already attached")

        self._host = host
        self._validate_host(host)
        self._validate_dependencies()
        self._validate_conflicts()
        self._setup_resources()

        self._state = (
            PluginState.ACTIVE if self.config.enabled else PluginState.SUSPENDED
        )
        self._hooks_cache = set(self.get_hook_points())

        self._on_attach(host)
        self._logger.debug(f"Plugin {self.name} attached to {type(host).__name__}")

    def detach(self) -> None:
        """Called when plugin is detached from host."""
        if self._host is None:
            return

        host = self._host
        self._on_detach(host)
        self._cleanup_resources()

        self._host = None
        self._state = PluginState.INACTIVE
        self._hooks_cache.clear()

        self._logger.debug(f"Plugin {self.name} detached from {type(host).__name__}")

    # Abstract Methods
    @abstractmethod
    def get_hook_points(self) -> list[PluginHookPoint]:
        """Return list of hook points this plugin uses."""
        pass

    @abstractmethod
    def _validate_host(self, host: HostType) -> None:
        """Validate that the host is compatible with this plugin."""
        pass

    # Hook Methods (to be overridden by subclasses)
    def _on_attach(self, host: HostType) -> None:
        """Called during attachment process. Override for custom logic."""
        pass

    def _on_detach(self, host: HostType) -> None:
        """Called during detachment process. Override for custom logic."""
        pass

    def _setup_resources(self) -> None:
        """Setup required resources. Override for custom logic."""
        pass

    def _cleanup_resources(self) -> None:
        """Cleanup resources. Override for custom logic."""
        pass

    # Validation Methods
    def _validate_dependencies(self) -> None:
        """Validate that all dependencies are met."""
        if not self.config.dependencies or not hasattr(self._host, "list_plugins"):
            return

        available_plugins = set(self._host.list_plugins())
        if missing_deps := set(self.config.dependencies) - available_plugins:
            raise PluginDependencyError(
                f"Plugin {self.name} requires dependencies: {missing_deps}"
            )

    def _validate_conflicts(self) -> None:
        """Check for conflicts with existing plugins."""
        if not self.config.conflicts or not hasattr(self._host, "list_plugins"):
            return

        available_plugins = set(self._host.list_plugins())
        if conflicts := set(self.config.conflicts) & available_plugins:
            raise PluginConflictError(f"Plugin {self.name} conflicts with: {conflicts}")

    def _call_hook(self, hook_method: str, context: ContextType):
        """Safely call a hook method with error handling and timeout."""
        if not self.enabled:
            return

        try:
            method = getattr(self, hook_method, None)
            if not method or not callable(method):
                return

            result = method(context=context)
            self._error_count = 0
            return result

        except Exception as e:
            self._handle_error(hook_method, e)
            return

    def _handle_error(self, method_name: str, error: Exception) -> None:
        """Handle plugin errors with retry logic."""
        self._error_count += 1
        self._logger.error(f"Error in {self.name}.{method_name}: {error}")

        if self._error_count >= self.config.max_retries:
            self._state = PluginState.ERROR
            self._logger.error(f"Plugin {self.name} disabled due to repeated errors")
            self._notify_state_change()

    # State Management
    def reset_errors(self) -> None:
        """Reset error count and reactivate if possible."""
        self._error_count = 0
        if self._state == PluginState.ERROR:
            self._state = (
                PluginState.ACTIVE if self.config.enabled else PluginState.SUSPENDED
            )
            self._notify_state_change()

    def _notify_state_change(self) -> None:
        """Notify listeners of state changes."""
        for listener in self._listeners:
            try:
                listener.on_plugin_state_change(self)
            except Exception as e:
                self._logger.warning(f"Error notifying listener: {e}")

    # Context Managers
    @contextmanager
    def temporary_disable(self):
        """Temporarily disable plugin."""
        was_enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = was_enabled

    @contextmanager
    def temporary_priority(self, priority: int):
        """Temporarily change plugin priority."""
        old_priority = self.config.priority
        self.config.priority = priority
        try:
            yield
        finally:
            self.config.priority = old_priority

    # Utility Methods
    def get_config(self) -> dict[str, Any]:
        """Return plugin configuration as dictionary."""
        return {
            "name": self.config.name,
            "priority": self.config.priority,
            "enabled": self.config.enabled,
            "state": self._state.value,
            "dependencies": self.config.dependencies,
            "conflicts": self.config.conflicts,
            "requires_resources": self.config.requires_resources,
            "error_count": self._error_count,
            "timeout": self.config.timeout,
            "metadata": self.config.metadata,
        }

    def add_listener(self, listener) -> None:
        """Add state change listener."""
        self._listeners.add(listener)

    def remove_listener(self, listener) -> None:
        """Remove state change listener."""
        self._listeners.discard(listener)


# Base Plugin Host Mixin
class PluginHostMixin(Generic[ContextType]):
    """Universal mixin class for hosts that support plugins."""

    def __init__(self):
        super().__init__()
        self._plugins: dict[str, Plugin] = {}
        self._plugin_groups: dict[str, list[str]] = {}
        self._plugin_profiles: dict[str, dict[str, Any]] = {}
        self._hook_cache: dict[PluginHookPoint, list[Plugin]] = {}
        self._hooks_dirty = True
        self._hook_call_counts: dict[str, int] = {}
        self._logger = logging.getLogger(f"{self.__class__.__name__}.plugins")

    # Core Plugin Management
    def add_plugin(self, plugin: Plugin) -> None:
        """Add an plugin to the host."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already exists")

        try:
            self._plugins[plugin.name] = plugin
            plugin.attach(self)
            self._hooks_dirty = True
            self._logger.info(
                f"Added plugin: {plugin.name} (priority: {plugin.priority})"
            )

        except (PluginDependencyError, PluginConflictError) as e:
            if plugin.name in self._plugins:
                del self._plugins[plugin.name]
            raise e

    def add_plugins(self, plugins: list[Plugin]) -> None:
        """Add multiple plugins to the host."""
        for plugin in plugins:
            self.add_plugin(plugin)

    def remove_plugin(self, plugin_name: str) -> None:
        """Remove plugin by name."""
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")

        plugin = self._plugins.pop(plugin_name)
        plugin.detach()
        self._hooks_dirty = True
        self._logger.info(f"Removed plugin: {plugin_name}")

        # Clean up groups
        for group_plugins in self._plugin_groups.values():
            if plugin_name in group_plugins:
                group_plugins.remove(plugin_name)

    def remove_plugins(self, plugin_names: list[str]) -> None:
        """Remove multiple plugins by name."""
        for plugin_name in plugin_names:
            self.remove_plugin(plugin_name)

    def get_plugin(self, plugin_name: str) -> Plugin:
        """Get plugin by name."""
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        return self._plugins[plugin_name]

    def list_plugins(self) -> list[str]:
        """Return list of all plugin names."""
        return list(self._plugins.keys())

    def has_plugin(self, plugin_name: str) -> bool:
        """Check if plugin exists."""
        return plugin_name in self._plugins

    # Plugin State Management
    def enable_plugin(self, plugin_name: str) -> None:
        """Enable specific plugin."""
        self.get_plugin(plugin_name).enabled = True
        self._hooks_dirty = True

    def disable_plugin(self, plugin_name: str) -> None:
        """Disable specific plugin."""
        self.get_plugin(plugin_name).enabled = False
        self._hooks_dirty = True

    def reset_plugin_errors(self, plugin_name: str) -> None:
        """Reset errors for specific plugin."""
        self.get_plugin(plugin_name).reset_errors()
        self._hooks_dirty = True

    # Group Management
    def create_plugin_group(self, group_name: str, plugin_names: list[str]) -> None:
        """Create a group of plugins for batch operations."""
        if invalid_plugins := [
            name for name in plugin_names if name not in self._plugins
        ]:
            raise ValueError(f"Invalid plugin names: {invalid_plugins}")

        self._plugin_groups[group_name] = plugin_names.copy()

    def enable_plugin_group(self, group_name: str) -> None:
        """Enable all plugins in a group."""
        if group_name not in self._plugin_groups:
            raise ValueError(f"Group '{group_name}' not found")

        for plugin_name in self._plugin_groups[group_name]:
            if plugin_name in self._plugins:
                self.enable_plugin(plugin_name)

    def disable_plugin_group(self, group_name: str) -> None:
        """Disable all plugins in a group."""
        if group_name not in self._plugin_groups:
            raise ValueError(f"Group '{group_name}' not found")

        for plugin_name in self._plugin_groups[group_name]:
            if plugin_name in self._plugins:
                self.disable_plugin(plugin_name)

    # Profile Management
    def save_plugin_profile(self, profile_name: str) -> None:
        """Save current plugin configuration as a profile."""
        self._plugin_profiles[profile_name] = {
            "plugins": {
                name: plugin.get_config() for name, plugin in self._plugins.items()
            },
            "groups": self._plugin_groups.copy(),
            "timestamp": self._get_current_timestamp(),
        }

    def load_plugin_profile(self, profile_name: str) -> None:
        """Load an plugin profile."""
        if profile_name not in self._plugin_profiles:
            raise ValueError(f"Profile '{profile_name}' not found")

        profile = self._plugin_profiles[profile_name]

        # Restore plugin configurations
        for plugin_name, config in profile["plugins"].items():
            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                plugin.config.enabled = config["enabled"]
                plugin.config.priority = config["priority"]

        # Restore groups
        self._plugin_groups = profile["groups"].copy()
        self._hooks_dirty = True

    def list_plugin_profiles(self) -> list[str]:
        """List available plugin profiles."""
        return list(self._plugin_profiles.keys())

    # Hook System
    def _call_hooks(self, hook_point: PluginHookPoint, context: ContextType) -> Any:
        """Call all registered hooks for a given hook point."""
        if self._hooks_dirty:
            self._rebuild_hook_cache()

        # Track hook calls
        hook_name = hook_point.value
        self._hook_call_counts[hook_name] = self._hook_call_counts.get(hook_name, 0) + 1

        # Get active plugins for this hook point
        plugins = self._hook_cache.get(hook_point, [])

        # Call hooks in priority order
        result = None
        for plugin in plugins:
            if plugin.state != PluginState.ACTIVE:
                continue

            hook_method = f"on_{hook_point.value}"
            result = plugin._call_hook(hook_method=hook_method, context=context)

        return result

    def _rebuild_hook_cache(self) -> None:
        """Rebuild hook cache for optimal performance."""
        self._hook_cache.clear()

        # Group plugins by hook points
        for plugin in self._plugins.values():
            if plugin.state != PluginState.ACTIVE:
                continue

            for hook_point in plugin.get_hook_points():
                if hook_point not in self._hook_cache:
                    self._hook_cache[hook_point] = []
                self._hook_cache[hook_point].append(plugin)

        # Sort by priority (higher first)
        for _, plugins in self._hook_cache.items():
            plugins.sort(key=lambda a: a.priority, reverse=True)

        self._hooks_dirty = False

    @abstractmethod
    def create_context(self, **kwargs) -> ContextType:
        """Create context object for hooks. Must be implemented by subclasses."""
        pass

    # Statistics and Monitoring
    def get_plugin_statistics(self) -> dict[str, Any]:
        """Get comprehensive plugin statistics."""
        return {
            "total_plugins": len(self._plugins),
            "active_plugins": sum(bool(a.enabled) for a in self._plugins.values()),
            "suspended_plugins": sum(
                a.state == PluginState.SUSPENDED for a in self._plugins.values()
            ),
            "error_plugins": sum(
                a.state == PluginState.ERROR for a in self._plugins.values()
            ),
            "hook_call_counts": self._hook_call_counts.copy(),
            "priority_distribution": self._get_priority_distribution(),
            "groups": {
                name: len(plugins) for name, plugins in self._plugin_groups.items()
            },
            "profiles": len(self._plugin_profiles),
        }

    def _get_priority_distribution(self) -> dict[str, int]:
        """Get distribution of plugin priorities."""
        distribution = {
            "critical": 0,
            "high": 0,
            "normal": 0,
            "low": 0,
            "background": 0,
        }

        for plugin in self._plugins.values():
            if plugin.priority >= PluginPriority.CRITICAL:
                distribution["critical"] += 1
            elif plugin.priority >= PluginPriority.HIGH:
                distribution["high"] += 1
            elif plugin.priority >= PluginPriority.NORMAL:
                distribution["normal"] += 1
            elif plugin.priority >= PluginPriority.LOW:
                distribution["low"] += 1
            else:
                distribution["background"] += 1

        return distribution

    def validate_plugin_setup(self) -> dict[str, list[str]]:
        """Validate current plugin setup."""
        issues = {
            "conflicts": [],
            "missing_dependencies": [],
            "error_plugins": [],
        }

        for plugin_name, plugin in self._plugins.items():
            # Check conflicts
            for conflict in plugin.config.conflicts:
                if conflict in self._plugins and self._plugins[conflict].enabled:
                    issues["conflicts"].append(
                        f"{plugin_name} conflicts with {conflict}"
                    )

            # Check dependencies
            for dep in plugin.config.dependencies:
                if dep not in self._plugins:
                    issues["missing_dependencies"].append(
                        f"{plugin_name} requires {dep}"
                    )
                elif not self._plugins[dep].enabled:
                    issues["missing_dependencies"].append(
                        f"{plugin_name} requires {dep} to be enabled"
                    )

            # Check error state
            if plugin.state == PluginState.ERROR:
                issues["error_plugins"].append(plugin_name)

        return issues

    def _get_current_timestamp(self) -> float:
        """Get current timestamp. Can be overridden for testing."""
        import time

        return time.time()

    # Event Handlers
    def on_plugin_state_change(self, plugin: Plugin) -> None:
        """Called when an plugin changes state."""
        self._hooks_dirty = True
        self._logger.debug(
            f"Plugin {plugin.name} changed state to {plugin.state.value}"
        )


# Utility Classes
class SimplePlugin(Plugin):
    """Simple plugin for single-hook implementations."""

    def __init__(self, hook_point: PluginHookPoint, hook_func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.hook_point = hook_point
        self.hook_func = hook_func

        # Set hook method dynamically
        method_name = f"on_{hook_point.value}"
        setattr(self, method_name, self._call_hook)

    def _call_hook(self, context, *args, **kwargs):
        return self.hook_func(context, *args, **kwargs)

    def get_hook_points(self) -> list[PluginHookPoint]:
        return [self.hook_point]

    def _validate_host(self, host) -> None:
        # Simple plugins work with any host
        pass


class ConditionalPlugin(Plugin):
    """Plugin that only executes under certain conditions."""

    def __init__(self, condition: Callable[[Any], bool], **kwargs):
        super().__init__(**kwargs)
        self.condition = condition

    def _call_hook(self, hook_method: str, context, *args, **kwargs):
        if not self.enabled or not self.condition(context):
            return args[0] if len(args) == 1 else args

        return super()._call_hook(hook_method, context, *args, **kwargs)


# Decorators
def plugin_hook(
    hook_point: PluginHookPoint, priority: int = PluginPriority.NORMAL, **kwargs
):
    """Decorator to create simple plugins from functions."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        plugin_name = kwargs.get("name", func.__name__)
        wrapper.plugin = SimplePlugin(
            hook_point=hook_point,
            hook_func=func,
            name=plugin_name,
            priority=priority,
            **kwargs,
        )
        return wrapper

    return decorator
