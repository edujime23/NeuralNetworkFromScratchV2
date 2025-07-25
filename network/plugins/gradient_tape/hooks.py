from dataclasses import dataclass
from enum import Enum

from network.plugins.base.plugin import PluginHookPoint


@dataclass(frozen=True)
class GradientTapeHookPoint(PluginHookPoint):
    """Concrete implementation of PluginHookPoint for optimizers."""

    value: str


class GradientTapeHookPoints(Enum):
    """Define hook points for GradientTape."""

    ON_TAPE_CREATE = GradientTapeHookPoint("on_tape_create")
    BEFORE_BACKPROP = GradientTapeHookPoint("before_backprop")
    AFTER_BACKPROP = GradientTapeHookPoint("after_backprop")
    BEFORE_NODE_COMPUTATION = GradientTapeHookPoint("before_node_computation")
    AFTER_NODE_COMPUTATION = GradientTapeHookPoint("after_node_computation")
    ON_GRADIENT_ACCUMULATION = GradientTapeHookPoint("on_gradient_accumulation")
