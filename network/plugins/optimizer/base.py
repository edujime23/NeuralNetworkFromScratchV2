from __future__ import annotations

import logging
from typing import Any

from network.plugins.base.plugin import PluginContext, PluginHookPoint, PluginHostMixin


class OptimizerPluginMixin(PluginHostMixin[PluginContext]):
    """
    Mixin enabling any optimizer to register and manage plugins via the
    PluginHostMixin base, using the plugin API.
    Metadata for optimizer-specific state is passed in PluginContext.metadata.
    """

    def __init__(self):
        # Initialize plugin machinery
        super().__init__()
        self._logger = logging.getLogger(f"{type(self).__name__}.plugins")

    def call_hooks(
        self,
        hook_point: PluginHookPoint,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Delegate hook invocation to PluginHostMixin, injecting optimizer state
        into the context metadata.
        """
        # Prepare metadata with optimizer-specific fields
        metadata: dict[str, Any] = {
            # Basic state
            "step": getattr(self, "iterations", 0),
            "lr": getattr(self, "lr", None),
            "built": getattr(self, "_built", False),
            "variables": getattr(self, "_current_variables", []),
            "loss_value": getattr(self, "_current_loss_value", None),
            "gradients": getattr(self, "_last_gradients", None),
            "slots": getattr(self, "_slots", {}),
        }
        # Create base context with metadata
        context = PluginContext(host=self, metadata=metadata)
        # Delegate to base implementation
        return super().call_hooks(hook_point, context, *args)
