from __future__ import annotations

import logging
from typing import Any

import numpy as np

from network.plugins.base.plugin import Plugin, PluginContext, PluginPriority
from network.plugins.optimizer.hooks import OptimizerHookPoints


class LookaheadPlugin(Plugin):
    """
    Lookahead Plugin for optimizers.

    Maintains slow weights (θ̄) and fast weights (θ). Every K steps:
    - θ̄ ← θ̄ + α(θ - θ̄)  (interpolate slow weights)
    - θ ← θ̄              (sync fast weights to slow)

    This smooths gradient oscillations, accelerates convergence,
    and improves stability without modifying the base optimizer logic.
    """

    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.5,
        name: str = "lookahead",
        priority: int = PluginPriority.HIGH,
        **kwargs,
    ):
        """
        Initialize Lookahead plugin.

        Args:
            k: Update frequency for slow weights (every k steps)
            alpha: Interpolation factor for slow weight updates
            name: Plugin name
            priority: Plugin priority
            **kwargs: Additional plugin configuration
        """
        super().__init__(name=name, priority=priority, **kwargs)

        # Lookahead hyperparameters
        self.k = max(1, k)
        self.alpha = max(0.0, min(1.0, alpha))

        # Internal state
        self._slow_weights: dict[str, np.ndarray] = {}
        self._step_count = 0
        self._is_initialized = False

        self._logger = logging.getLogger(f"plugin.{self.name}")

        # Validate parameters
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    def get_hook_points(self) -> list:
        """Return hook points this plugin uses."""
        return [
            OptimizerHookPoints.PRE_STEP.value,
            OptimizerHookPoints.POST_STEP.value,
        ]

    def _validate_host(self, host) -> None:
        """Validate that host is compatible with this plugin."""
        # Check if host has required methods/attributes
        required_attrs = ["_current_variables", "_slots"]
        for attr in required_attrs:
            if not hasattr(host, attr):
                raise ValueError(
                    f"Host must have '{attr}' attribute for Lookahead plugin"
                )

    def _on_attach(self, host) -> None:
        """Called when plugin is attached to host."""
        self._logger.info(f"Lookahead plugin attached (k={self.k}, alpha={self.alpha})")

    def _on_detach(self, host) -> None:
        """Called when plugin is detached from host."""
        self._cleanup_resources()
        self._logger.info("Lookahead plugin detached")

    def _setup_resources(self) -> None:
        """Setup required resources."""
        self._reset()

    def _cleanup_resources(self) -> None:
        """Cleanup resources."""
        self._slow_weights.clear()
        self._is_initialized = False

    def on_pre_step(self, context: PluginContext, *args, **kwargs):
        """
        Called before optimizer step. Initialize slow weights if needed.
        """
        if not self._is_initialized:
            self._initialize_slow_weights(context)

        return args[0] if len(args) == 1 else args

    def on_post_step(self, context: PluginContext, *args, **kwargs):
        """
        Called after optimizer step. Update slow weights every k steps.
        """
        self._step_count += 1

        # Check if it's time to update slow weights
        if self._step_count % self.k == 0:
            self._update_slow_weights(context)
            self._sync_fast_to_slow(context)

            self._logger.debug(
                f"Lookahead update at step {self._step_count} "
                f"(every {self.k} steps, alpha={self.alpha})"
            )

        return args[0] if len(args) == 1 else args

    def _initialize_slow_weights(self, context: PluginContext) -> None:
        """Initialize slow weights with current variable values."""
        variables = context.get_metadata("variables", [])

        if not variables:
            self._logger.warning("No variables found for Lookahead initialization")
            return

        self._slow_weights.clear()

        for i, var in enumerate(variables):
            if hasattr(var, "numpy"):
                # TensorFlow/Keras variable
                var_value = var.numpy()
            elif isinstance(var, np.ndarray):
                # NumPy array
                var_value = var
            else:
                # Try to convert to numpy
                try:
                    var_value = np.array(var)
                except Exception as e:
                    self._logger.warning(
                        f"Could not convert variable {i} to numpy: {e}"
                    )
                    continue

            # Store slow weight as copy
            var_key = f"var_{i}"
            self._slow_weights[var_key] = var_value.copy()

        self._is_initialized = True
        self._logger.debug(f"Initialized {len(self._slow_weights)} slow weights")

    def _update_slow_weights(self, context: PluginContext) -> None:
        """Update slow weights: θ̄ ← θ̄ + α(θ - θ̄)"""
        variables = context.get_metadata("variables", [])

        if not variables or not self._slow_weights:
            self._logger.warning("Cannot update slow weights: missing variables")
            return

        updated_count = 0

        for i, var in enumerate(variables):
            var_key = f"var_{i}"

            if var_key not in self._slow_weights:
                continue

            try:
                # Get current fast weight
                if hasattr(var, "numpy"):
                    fast_weight = var.numpy()
                elif isinstance(var, np.ndarray):
                    fast_weight = var
                else:
                    fast_weight = np.array(var)

                # Get slow weight
                slow_weight = self._slow_weights[var_key]

                # Ensure shapes match
                if fast_weight.shape != slow_weight.shape:
                    self._logger.warning(
                        f"Shape mismatch for variable {i}: "
                        f"fast={fast_weight.shape}, slow={slow_weight.shape}"
                    )
                    continue

                # Update slow weight: θ̄ ← θ̄ + α(θ - θ̄)
                weight_diff = fast_weight - slow_weight
                self._slow_weights[var_key] += self.alpha * weight_diff

                updated_count += 1

            except Exception as e:
                self._logger.warning(
                    f"Error updating slow weight for variable {i}: {e}"
                )
                continue

        if updated_count > 0:
            self._logger.debug(f"Updated {updated_count} slow weights")

    def _sync_fast_to_slow(self, context: PluginContext) -> None:
        """Sync fast weights to slow weights: θ ← θ̄"""
        variables = context.get_metadata("variables", [])

        if not variables or not self._slow_weights:
            self._logger.warning("Cannot sync weights: missing variables")
            return

        synced_count = 0

        for i, var in enumerate(variables):
            var_key = f"var_{i}"

            if var_key not in self._slow_weights:
                continue

            try:
                slow_weight = self._slow_weights[var_key]

                # Update variable with slow weight
                if hasattr(var, "assign"):
                    # TensorFlow/Keras variable
                    var.assign(slow_weight)
                elif isinstance(var, np.ndarray):
                    # NumPy array (in-place update)
                    var[:] = slow_weight
                else:
                    # Cannot update in-place, log warning
                    self._logger.warning(
                        f"Cannot sync variable {i}: unsupported type {type(var)}"
                    )
                    continue

                synced_count += 1

            except Exception as e:
                self._logger.warning(f"Error syncing variable {i}: {e}")
                continue

        if synced_count > 0:
            self._logger.debug(f"Synced {synced_count} fast weights to slow weights")

    def get_state(self) -> dict[str, Any]:
        """Get current plugin state for debugging/monitoring."""
        return {
            "k": self.k,
            "alpha": self.alpha,
            "step_count": self._step_count,
            "is_initialized": self._is_initialized,
            "num_slow_weights": len(self._slow_weights),
            "next_update_step": (
                (self._step_count // self.k + 1) * self.k
                if self._is_initialized
                else None
            ),
        }

    def reset_state(self) -> None:
        """Reset plugin state (useful for training restarts)."""
        self._reset()
        self._logger.info("Lookahead plugin state reset")

    def _reset(self):
        self._slow_weights.clear()
        self._step_count = 0
        self._is_initialized = False

    def __repr__(self) -> str:
        return (
            f"LookaheadPlugin(k={self.k}, alpha={self.alpha}, "
            f"step_count={self._step_count}, initialized={self._is_initialized})"
        )
