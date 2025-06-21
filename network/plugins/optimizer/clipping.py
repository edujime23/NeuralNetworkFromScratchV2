from typing import Any

import numpy as np

from ..base.plugin import Plugin, PluginContext, PluginHookPoint
from .hooks import OptimizerHookPoints


class AdaptiveGradientClippingPlugin(Plugin):
    """
    Clips gradients by an adaptive threshold based on running average of gradient norms.
    Hook: pre_apply_gradients
    """

    def __init__(
        self,
        clip_factor: float = 0.01,
        decay: float = 0.99,
        name: str = "adaptive_gradient_clipping",
        priority: int = 500,
        **kwargs,
    ):
        super().__init__(
            name=name,
            priority=priority,
            **kwargs,
        )
        # Factor to multiply current learning rate for threshold
        self.clip_factor = clip_factor
        self.decay = decay
        # Initialize running norm in metadata
        self.running_norm = None

    def get_hook_points(self) -> list[PluginHookPoint]:
        # Clip before applying gradients
        return [OptimizerHookPoints.PRE_APPLY_GRADIENTS.value]

    def _validate_host(self, host: Any) -> None:
        # No special host requirements
        pass

    def on_pre_apply_gradients(
        self,
        context: PluginContext,
        *args,
        **kwargs,
    ) -> PluginContext:
        """
        Compute threshold = clip_factor * lr, update running norm,
        clip each gradient tensor by threshold, return modified context.
        """
        # Extract grads_and_vars from context
        grads_and_vars = context.metadata.get("grads_and_vars", [])

        # Skip if no gradients or variables
        if not grads_and_vars:
            return context

        # Compute total norm of gradients
        total_norm_sq = 0.0
        for g, _ in grads_and_vars:
            if g is not None:
                total_norm_sq += float((g**2).sum())

        total_norm = total_norm_sq**0.5

        # Initialize or update running average
        if self.running_norm is None:
            self.running_norm = total_norm
        else:
            self.running_norm = (
                self.decay * self.running_norm + (1 - self.decay) * total_norm
            )

        # Determine threshold based on running norm
        lr = context.metadata.get("lr", 1.0)
        threshold = self.clip_factor * lr * self.running_norm

        # Clip gradients
        clipped = []
        for grad, var in grads_and_vars:
            if grad is None:
                clipped.append((grad, var))
                continue

            grad_norm = float((grad**2).sum() ** 0.5)
            if grad_norm > threshold and grad_norm > 0:
                clipped_grad = grad * (threshold / grad_norm)
            else:
                clipped_grad = grad
            clipped.append((clipped_grad, var))

        # Update context with clipped gradients
        context.metadata["grads_and_vars"] = clipped
        return context


class AdaptivePercentileClippingPlugin(Plugin):
    """
    Clips gradients based on percentile statistics.
    Maintains clipping threshold as percentile of gradient magnitudes.
    """

    def __init__(
        self,
        percentile: float = 95.0,
        decay: float = 0.99,
        name: str = "adaptive_percentile_clipping",
        priority: int = 500,
        **kwargs,
    ):
        super().__init__(name=name, priority=priority, **kwargs)
        self.percentile = percentile
        self.decay = decay
        self.running_threshold = None

    def get_hook_points(self) -> list[PluginHookPoint]:
        return [OptimizerHookPoints.PRE_APPLY_GRADIENTS.value]

    def _validate_host(self, host: Any) -> None:
        pass

    def on_pre_apply_gradients(self, context: PluginContext) -> PluginContext:
        grads_and_vars = context.metadata.get("grads_and_vars", [])
        if not grads_and_vars:
            return context

        # Collect all gradient magnitudes
        all_mags = []
        for g, _ in grads_and_vars:
            if g is not None:
                all_mags.extend(np.abs(g).flatten())

        if not all_mags:
            return context

        # Compute current percentile threshold
        current_threshold = np.percentile(all_mags, self.percentile)

        # Update running threshold
        if self.running_threshold is None:
            self.running_threshold = current_threshold
        else:
            self.running_threshold = (
                self.decay * self.running_threshold
                + (1 - self.decay) * current_threshold
            )

        # Apply clipping
        clipped = []
        for grad, var in grads_and_vars:
            if grad is None:
                clipped.append((grad, var))
                continue

            clipped_grad = np.clip(
                grad, -self.running_threshold, self.running_threshold
            )
            clipped.append((clipped_grad, var))

        context.metadata["grads_and_vars"] = clipped
        return context


class StochasticGradientClippingPlugin(Plugin):
    """
    Probabilistic gradient clipping based on gradient magnitude.
    Higher magnitude gradients have higher probability of being clipped.
    """

    def __init__(
        self,
        base_threshold: float = 1.0,
        max_clip_prob: float = 0.9,
        temperature: float = 1.0,
        name: str = "stochastic_gradient_clipping",
        priority: int = 500,
        **kwargs,
    ):
        super().__init__(name=name, priority=priority, **kwargs)
        self.base_threshold = base_threshold
        self.max_clip_prob = max_clip_prob
        self.temperature = temperature

    def get_hook_points(self) -> list[PluginHookPoint]:
        return [OptimizerHookPoints.PRE_APPLY_GRADIENTS.value]

    def _validate_host(self, host: Any) -> None:
        pass

    def on_pre_apply_gradients(self, context: PluginContext) -> PluginContext:
        grads_and_vars = context.metadata.get("grads_and_vars", [])
        if not grads_and_vars:
            return context

        clipped = []
        for grad, var in grads_and_vars:
            if grad is None:
                clipped.append((grad, var))
                continue

            grad_norm = float(np.sqrt(np.sum(grad**2)))

            # Compute clipping probability based on magnitude
            clip_prob = self.max_clip_prob * (
                1 - np.exp(-grad_norm / (self.base_threshold * self.temperature))
            )

            # Stochastic clipping decision
            if np.random.random() < clip_prob:
                clipped_grad = grad * (self.base_threshold / max(grad_norm, 1e-8))
            else:
                clipped_grad = grad

            clipped.append((clipped_grad, var))

        context.metadata["grads_and_vars"] = clipped
        return context
