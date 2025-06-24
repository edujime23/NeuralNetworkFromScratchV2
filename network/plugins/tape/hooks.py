from dataclasses import dataclass
from enum import Enum

from network.plugins.base.plugin import PluginHookPoint


@dataclass(frozen=True)
class TapeHookPoint(PluginHookPoint):
    """Concrete implementation of PluginHookPoint for optimizers."""

    value: str


class TapeHookPoint(Enum):
    """Define los puntos de anclaje para los plugins de GradientTape."""

    ON_TAPE_CREATE = TapeHookPoint("on_tape_create")
    BEFORE_BACKPROP = TapeHookPoint("before_backprop")
    AFTER_BACKPROP = TapeHookPoint("after_backprop")
    BEFORE_NODE_COMPUTATION = TapeHookPoint("before_node_computation")
    AFTER_NODE_COMPUTATION = TapeHookPoint("after_node_computation")
    ON_GRADIENT_ACCUMULATION = TapeHookPoint("on_gradient_accumulation")
