from __future__ import annotations

import logging
from typing import Any

from network.plugins.base.plugin import PluginContext, PluginHookPoint, PluginHostMixin

class IntegralTapeHostMixin(PluginHostMixin[PluginContext]):
    """
    Mixin enabling the complex integrator to register and manage plugins
    for custom behaviors (e.g. new methods, logging, metrics).
    """
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(f"{type(self).__name__}.plugins")

    def call_hooks(
        self,
        hook_point: PluginHookPoint,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Prepare metadata with integrator-specific state
        metadata: dict[str, Any] = {
            "method": getattr(self, 'current_method', None),
            "evaluations": getattr(self, 'function_evals', None),
            **kwargs,
        }
        context = PluginContext(host=self, metadata=metadata)
        return super()._call_hooks(hook_point=hook_point.value, context=context)
