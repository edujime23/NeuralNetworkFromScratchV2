from dataclasses import dataclass
from enum import Enum

from ..base.plugin import PluginHookPoint


@dataclass(frozen=True)
class OptimizerHookPoint(PluginHookPoint):
    """Concrete implementation of PluginHookPoint for optimizers."""

    value: str


class OptimizerHookPoints(Enum):
    PRE_COMPUTE_GRADIENTS = OptimizerHookPoint("pre_compute_gradients")
    POST_COMPUTE_GRADIENTS = OptimizerHookPoint("post_compute_gradients")
    PRE_APPLY_GRADIENTS = OptimizerHookPoint("pre_apply_gradients")
    PRE_BUILD = OptimizerHookPoint("pre_build")
    POST_BUILD = OptimizerHookPoint("post_build")
    PRE_UPDATE_STEP = OptimizerHookPoint("pre_update_step")
    PRE_APPLY_UPDATE = OptimizerHookPoint("pre_apply_update")
    POST_UPDATE_STEP = OptimizerHookPoint("post_update_step")
    POST_APPLY_GRADIENTS = OptimizerHookPoint("post_apply_gradients")
    PRE_STEP = OptimizerHookPoint("pre_step")
    POST_STEP = OptimizerHookPoint("post_step")
    ON_VARIABLE_CREATED = OptimizerHookPoint("on_variable_created")
    ON_CONVERGENCE_CHECK = OptimizerHookPoint("on_convergence_check")
