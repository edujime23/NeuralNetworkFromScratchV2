from dataclasses import dataclass
from enum import Enum

from network.plugins.base.plugin import PluginHookPoint

@dataclass(frozen=True)
class IntegralTapeHookPoint(PluginHookPoint):
    """Concrete hook point for integrator events."""
    value: str

class IntegralTapeHookPoints(Enum):
    REGISTER_METHODS   = IntegralTapeHookPoint("register_methods")
    PRE_INTEGRATE      = IntegralTapeHookPoint("pre_integrate")
    POST_INTEGRATE     = IntegralTapeHookPoint("post_integrate")
    DETECT_SINGULAR    = IntegralTapeHookPoint("detect_singular")
    AFTER_DETECTION    = IntegralTapeHookPoint("after_detection")
