from dataclasses import dataclass
from enum import Enum
from network.plugins.base.plugin import PluginHookPoint


@dataclass(frozen=True)
class ModelHookPoint(PluginHookPoint):
    value: str


class ModelHookPoints(Enum):
    PRE_COMPILE = ModelHookPoint("pre_compile")
    POST_COMPILE = ModelHookPoint("post_compile")
    PRE_BUILD = ModelHookPoint("pre_build")
    POST_BUILD = ModelHookPoint("post_build")
    PRE_FORWARD = ModelHookPoint("pre_forward")
    POST_FORWARD = ModelHookPoint("post_forward")
    PRE_TRAIN_STEP = ModelHookPoint("pre_train_step")
    POST_TRAIN_STEP = ModelHookPoint("post_train_step")
    PRE_EPOCH = ModelHookPoint("pre_epoch")
    POST_EPOCH = ModelHookPoint("post_epoch")
    PRE_BATCH = ModelHookPoint("pre_batch")
    POST_BATCH = ModelHookPoint("post_batch")
