import logging
from typing import Any
from network.plugins.base.plugin import PluginContext, PluginHostMixin, PluginHookPoint


class ModelPluginMixin(PluginHostMixin[PluginContext]):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(f"{type(self).__name__}.plugins")

    def call_hooks(
        self,
        hook_point: PluginHookPoint,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        context = PluginContext(
            host=self,
            metadata={
                "built": getattr(self, "_built", False),
                "layers": getattr(self, "_layers", []),
                "variables": getattr(self, "_variables", []),
                "optimizer": getattr(self, "_optimizer", None),
                "metrics": getattr(self, "_metrics", []),
                "loss_fn": getattr(self, "_loss_fn", None),
                "input_shape": getattr(self, "_input_shape", None),
                "dtype": getattr(self, "_dtype", None),
                **kwargs,
            },
        )
        return super()._call_hooks(hook_point=hook_point.value, context=context)
