import logging
import time
from collections import deque
from typing import Any

import numpy as np

from ...types.tensor import Tensor
from ...types.variable import Variable
from .base import (
    AddonContext,
    AddonPriority,
    OptimizerAddon,
)

# ==================== GRADIENT CLIPPING ADD-ONS ====================


class GradientClippingAddon(OptimizerAddon):
    """Gradient clipping by norm or value."""

    def __init__(
        self,
        clip_norm: float | None = None,
        clip_value: float | None = None,
        global_clipnorm: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=kwargs.get("name", "gradient_clipping"),
            priority=AddonPriority.HIGH,
            **kwargs,
        )
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        self.global_clipnorm = global_clipnorm

        if not clip_norm and not clip_value:
            raise ValueError("Must specify either clip_norm or clip_value")

    def on_post_compute_gradients(self, context: AddonContext, grads_and_vars):
        """Apply gradient clipping after gradients are computed."""
        if not grads_and_vars:
            return grads_and_vars

        gradients = [g for g, _ in grads_and_vars if g is not None]
        variables = [v for g, v in grads_and_vars if g is not None]

        if not gradients:
            return grads_and_vars

        # Clip by norm
        if self.clip_norm:
            if self.global_clipnorm:
                # Global norm clipping
                global_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
                if global_norm > self.clip_norm:
                    clip_coeff = self.clip_norm / global_norm
                    gradients = [g * clip_coeff for g in gradients]
            else:
                # Per-gradient norm clipping
                gradients = [
                    g * min(1.0, self.clip_norm / (np.linalg.norm(g) + 1e-8))
                    for g in gradients
                ]

        # Clip by value
        if self.clip_value:
            gradients = [
                np.clip(g, -self.clip_value, self.clip_value) for g in gradients
            ]

        return list(zip(gradients, variables))


class AdaptiveGradientClippingAddon(OptimizerAddon):
    """Adaptive gradient clipping based on gradient history."""

    def __init__(self, percentile: float = 90.0, history_size: int = 1000, **kwargs):
        super().__init__(
            name=kwargs.get("name", "adaptive_gradient_clipping"),
            priority=AddonPriority.HIGH,
            **kwargs,
        )
        self.percentile = percentile
        self.history_size = history_size
        self.gradient_norms = deque(maxlen=history_size)

    def on_post_compute_gradients(self, context: AddonContext, grads_and_vars):
        """Apply adaptive gradient clipping."""
        if not grads_and_vars:
            return grads_and_vars

        gradients = [g for g, _ in grads_and_vars if g is not None]
        variables = [v for g, v in grads_and_vars if g is not None]

        if not gradients:
            return grads_and_vars

        # Calculate current global norm
        global_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
        self.gradient_norms.append(global_norm)

        # Calculate adaptive threshold
        if len(self.gradient_norms) >= 10:  # Need some history
            threshold = np.percentile(self.gradient_norms, self.percentile)

            if global_norm > threshold:
                clip_coeff = threshold / global_norm
                gradients = [g * clip_coeff for g in gradients]

        return list(zip(gradients, variables))


# ==================== REGULARIZATION ADD-ONS ====================


class L1L2RegularizationAddon(OptimizerAddon):
    """L1 and L2 regularization."""

    def __init__(self, l1: float = 0.0, l2: float = 0.0, **kwargs):
        super().__init__(
            name=kwargs.get("name", "l1_l2_regularization"),
            priority=AddonPriority.NORMAL,
            **kwargs,
        )
        self.l1 = l1
        self.l2 = l2

    def on_pre_apply_update(
        self, context: AddonContext, update: Tensor, variable: Variable
    ):
        """Add regularization to the update."""
        if not variable.trainable:
            return update, variable

        reg_update = update

        # L1 regularization
        if self.l1 > 0:
            reg_update = reg_update + self.l1 * np.sign(variable.value)

        # L2 regularization
        if self.l2 > 0:
            reg_update = reg_update + self.l2 * variable.value

        return reg_update, variable


class DropoutAddon(OptimizerAddon):
    """Dropout regularization during training."""

    def __init__(self, rate: float = 0.1, **kwargs):
        super().__init__(
            name=kwargs.get("name", "dropout"), priority=AddonPriority.NORMAL, **kwargs
        )
        self.rate = rate
        self.training = True

    def on_pre_apply_update(
        self, context: AddonContext, update: Tensor, variable: Variable
    ):
        """Apply dropout to updates."""
        if not self.training or self.rate <= 0:
            return update

        mask = np.random.random(update.shape) > self.rate
        return (
            update * mask / (1.0 - self.rate),
            variable,
        )  # Scale to maintain expected value

    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training = training


# ==================== LEARNING RATE SCHEDULING ADD-ONS ====================


class LearningRateSchedulerAddon(OptimizerAddon):
    """Base class for learning rate scheduling."""

    def __init__(self, initial_lr: float, **kwargs):
        super().__init__(
            name=kwargs.get("name", "lr_scheduler"),
            priority=AddonPriority.NORMAL,
            **kwargs,
        )
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def get_lr(self, step: int) -> float:
        """Override this method to implement scheduling logic."""
        return self.initial_lr

    def on_pre_step(self, context: AddonContext):
        """Update learning rate before step."""
        self.current_lr = self.get_lr(context.step)

        # Update optimizer's learning rate if it has one
        if hasattr(context.optimizer, "learning_rate"):
            context.optimizer.learning_rate = self.current_lr


class ExponentialDecaySchedulerAddon(LearningRateSchedulerAddon):
    """Exponential decay learning rate scheduler."""

    def __init__(
        self,
        initial_lr: float,
        decay_rate: float = 0.96,
        decay_steps: int = 1000,
        staircase: bool = False,
        **kwargs,
    ):
        super().__init__(initial_lr, **kwargs)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        self.config.name = kwargs.get("name", "exponential_decay")

    def get_lr(self, step: int) -> float:
        """Exponential decay formula."""
        if self.staircase:
            return self.initial_lr * (self.decay_rate ** (step // self.decay_steps))
        else:
            return self.initial_lr * (self.decay_rate ** (step / self.decay_steps))


class CosineDecaySchedulerAddon(LearningRateSchedulerAddon):
    """Cosine decay learning rate scheduler."""

    def __init__(
        self, initial_lr: float, decay_steps: int, alpha: float = 0.0, **kwargs
    ):
        super().__init__(initial_lr, **kwargs)
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.config.name = kwargs.get("name", "cosine_decay")

    def get_lr(self, step: int) -> float:
        """Cosine decay formula."""
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_lr * decayed


class WarmupSchedulerAddon(LearningRateSchedulerAddon):
    """Learning rate warmup scheduler."""

    def __init__(self, initial_lr: float, warmup_steps: int, **kwargs):
        super().__init__(initial_lr, **kwargs)
        self.warmup_steps = warmup_steps
        self.config.name = kwargs.get("name", "warmup")

    def get_lr(self, step: int) -> float:
        """Linear warmup formula."""
        if step < self.warmup_steps:
            return self.initial_lr * (step / self.warmup_steps)
        return self.initial_lr


# ==================== MOMENTUM ENHANCEMENT ADD-ONS ====================


class NesterovMomentumAddon(OptimizerAddon):
    """Nesterov momentum with look-ahead to any optimizer"""

    def __init__(self, momentum: float = 0.9, **kwargs):
        super().__init__(
            name=kwargs.get("name", "nesterov_momentum"),
            priority=AddonPriority.NORMAL,
            requires_slots=["momentum"],
            **kwargs,
        )
        self.momentum = momentum

    def on_post_build(self, context: AddonContext, var_list, slots):
        """Initialize momentum slots."""
        for var in var_list:
            if var.trainable:
                context.optimizer.add_slot(var, "momentum")

    def on_pre_update_step(
        self, context: AddonContext, gradient: Tensor, variable: Variable
    ) -> tuple[Tensor, Variable]:
        if not variable.trainable:
            return gradient, variable

        # Get momentum slot
        momentum_slot = context.get_slot(variable, "momentum")
        v_prev = momentum_slot.value

        nesterov_gradient = self.momentum * v_prev + gradient

        momentum_slot.assign(nesterov_gradient)

        return nesterov_gradient, variable


class LookaheadAddon(OptimizerAddon):
    """
    Lookahead optimizer addon that implements the k-step forward, 1-step back algorithm.

    The algorithm maintains slow weights and periodically updates them by interpolating
    with the fast weights after k optimization steps.
    """

    def __init__(self, k: int = 5, alpha: float = 0.5, **kwargs):
        """
        Initialize Lookahead addon.

        Args:
            k: Number of fast weight updates before slow weight update
            alpha: Interpolation factor for slow weight update (0 < alpha <= 1)
        """
        super().__init__(
            name=kwargs.get("name", "lookahead"),
            priority=AddonPriority.NORMAL,
            requires_slots=["slow_weights"],
            **kwargs,
        )

        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.k = k  # Update frequency
        self.alpha = alpha  # Interpolation factor
        self._step_count = 0
        self._initialized = False

    def on_post_build(self, context: AddonContext, var_list, slots):
        """Initialize slow weights slots."""
        for var in var_list:
            if var.trainable:
                context.optimizer.add_slot(var, "slow_weights", dtype=var.dtype)

    def on_post_apply_gradients(self, context: AddonContext, grads_and_vars) -> None:
        """
        Called after all gradients are applied. This is where we perform the
        lookahead update logic.
        """
        if not self._initialized:
            self._initialize_slow_weights(context, grads_and_vars)
            self._initialized = True
            return

        self._step_count += 1

        # Perform lookahead update every k steps
        if self._step_count % self.k == 0:
            self._perform_lookahead_update(context, grads_and_vars)

    def _initialize_slow_weights(self, context: AddonContext, grads_and_vars):
        """Initialize slow weights with current fast weights."""
        for grad, var in grads_and_vars:
            if var.trainable and grad is not None:
                slow_weights = context.get_slot(var, "slow_weights")
                slow_weights.assign(var.value)

    def _perform_lookahead_update(self, context: AddonContext, grads_and_vars):
        """
        Perform the lookahead update:
        φ_t = φ_{t-k} + α(θ_t - φ_{t-k})
        θ_t = φ_t

        Where:
        - φ_t: slow weights at step t
        - θ_t: fast weights at step t
        - α: interpolation factor
        """
        for grad, var in grads_and_vars:
            if var.trainable and grad is not None:
                slow_weights = context.get_slot(var, "slow_weights")

                # φ_t = φ_{t-k} + α(θ_t - φ_{t-k})
                # Equivalent to: φ_t = (1-α)φ_{t-k} + αθ_t
                new_slow_weights = (
                    1.0 - self.alpha
                ) * slow_weights.value + self.alpha * var.value

                # Update slow weights
                slow_weights.assign(new_slow_weights)

                # Set fast weights to new slow weights: θ_t = φ_t
                var.assign(new_slow_weights)

    def on_pre_step(self, context: AddonContext) -> None:
        """Log lookahead status at the beginning of each step."""
        if context.step % (self.k * 10) == 0:  # Log every 10 lookahead cycles
            next_update_step = ((context.step // self.k) + 1) * self.k
            self._logger.debug(
                f"Lookahead: step {context.step}, "
                f"next update at step {next_update_step}, "
                f"total updates: {context.step // self.k}"
            )

    def reset_step_count(self):
        """Reset the internal step counter. Useful for resuming training."""
        self._step_count = 0
        self._initialized = False

    def get_config(self) -> dict[str, Any]:
        """Return addon configuration including hyperparameters."""
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "alpha": self.alpha,
                "step_count": self._step_count,
                "initialized": self._initialized,
            }
        )
        return config

    @property
    def next_update_step(self) -> int:
        """Return the step number when next lookahead update will occur."""
        return ((self._step_count // self.k) + 1) * self.k

    @property
    def updates_performed(self) -> int:
        """Return the number of lookahead updates performed so far."""
        return self._step_count // self.k


# ==================== LOGGING AND MONITORING ADD-ONS ====================


class LoggingAddon(OptimizerAddon):
    """Comprehensive logging addon."""

    def __init__(
        self,
        log_frequency: int = 100,
        log_gradients: bool = False,
        log_variables: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=kwargs.get("name", "logging"),
            priority=AddonPriority.CRITICAL,
            **kwargs,
        )
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients
        self.log_variables = log_variables
        self.logger = logging.getLogger("optimizer.logging")

        # Metrics tracking
        self.metrics = {
            "loss_history": deque(maxlen=1000),
            "gradient_norms": deque(maxlen=1000),
            "variable_norms": deque(maxlen=1000),
            "step_times": deque(maxlen=1000),
        }
        self.step_start_time = None

    def on_pre_step(self, context: AddonContext):
        """Record step start time."""
        self.step_start_time = time.time()

    def on_post_step(self, context: AddonContext):
        """Log metrics after each step."""
        if context.step % self.log_frequency != 0:
            return

        # Calculate step time
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            self.metrics["step_times"].append(step_time)

        # Log basic info
        self.logger.info(f"Step {context.step}: Loss={context.loss_value}")

        # Log gradient statistics
        if self.log_gradients and context.gradients:
            if grad_norms := [
                np.linalg.norm(g) for g in context.gradients if g is not None
            ]:
                self._save_metrics(
                    grad_norms, "gradient_norms", "  Avg gradient norm: "
                )
        # Log variable statistics
        if self.log_variables and context.variables:
            if var_norms := [
                np.linalg.norm(v.value) for v in context.variables if v.trainable
            ]:
                self._save_metrics(var_norms, "variable_norms", "  Avg variable norm: ")

    def _save_metrics(self, grad_norms: list[float], slot: str, log_prefix: str):
        avg_grad_norm = np.mean(grad_norms)
        self.metrics[slot].append(avg_grad_norm)
        self.logger.info(f"{log_prefix}{avg_grad_norm:.6f}")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        return {
            "loss_history": list(self.metrics["loss_history"]),
            "avg_gradient_norm": (
                np.mean(self.metrics["gradient_norms"])
                if self.metrics["gradient_norms"]
                else 0
            ),
            "avg_variable_norm": (
                np.mean(self.metrics["variable_norms"])
                if self.metrics["variable_norms"]
                else 0
            ),
            "avg_step_time": (
                np.mean(self.metrics["step_times"]) if self.metrics["step_times"] else 0
            ),
        }


class EarlyStoppingAddon(OptimizerAddon):
    """Early stopping based on loss or metrics."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        monitor: str = "loss",
        mode: str = "min",
        **kwargs,
    ):
        super().__init__(
            name=kwargs.get("name", "early_stopping"),
            priority=AddonPriority.HIGH,
            **kwargs,
        )
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.best_value = np.inf if mode == "min" else -np.inf
        self.wait_count = 0
        self.stopped = False

    def on_convergence_check(self, context: AddonContext) -> bool | None:
        """Check if early stopping criteria are met."""
        if self.stopped:
            return True

        # Get current value to monitor
        current_value = None
        if self.monitor == "loss" and context.loss_value is not None:
            current_value = float(context.loss_value)

        if current_value is None:
            return None

        # Check for improvement
        improved = False
        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.stopped = True
            self._logger.info(
                f"Early stopping triggered after {self.patience} steps without improvement"
            )
            return True

        return None


# ==================== ADVANCED OPTIMIZATION ADD-ONS ====================


class AdaptiveNoiseAddon(OptimizerAddon):
    """Add adaptive noise to gradients for better generalization."""

    def __init__(self, initial_stddev: float = 0.1, decay_rate: float = 0.99, **kwargs):
        super().__init__(
            name=kwargs.get("name", "adaptive_noise"),
            priority=AddonPriority.NORMAL,
            **kwargs,
        )
        self.initial_stddev = initial_stddev
        self.decay_rate = decay_rate
        self.current_stddev = initial_stddev

    def on_pre_step(self, context: AddonContext):
        """Update noise level."""
        self.current_stddev = self.initial_stddev * (self.decay_rate**context.step)

    def on_post_compute_gradients(self, context: AddonContext, grads_and_vars):
        """Add adaptive noise to gradients."""
        if self.current_stddev <= 0:
            return grads_and_vars

        noisy_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                noise = np.random.normal(0, self.current_stddev, grad.shape)
                noisy_grad = grad + noise
                noisy_grads_and_vars.append((noisy_grad, var))
            else:
                noisy_grads_and_vars.append((grad, var))

        return noisy_grads_and_vars


class GradientAccumulationAddon(OptimizerAddon):
    """Gradient accumulation for large batch training."""

    def __init__(self, accumulate_steps: int = 4, **kwargs):
        super().__init__(
            name=kwargs.get("name", "gradient_accumulation"),
            priority=AddonPriority.HIGH,
            requires_slots=["accumulated_gradients"],
            **kwargs,
        )
        self.accumulate_steps = accumulate_steps
        self.current_step = 0

    def on_post_build(self, context: AddonContext, var_list, slots):
        """Initialize accumulation slots."""
        for var in var_list:
            if var.trainable:
                context.optimizer.add_slot(var, "accumulated_gradients")

    def on_post_compute_gradients(self, context: AddonContext, grads_and_vars):
        """Accumulate gradients."""
        self.current_step += 1

        accumulated_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None and var.trainable:
                acc_slot = context.get_slot(var, "accumulated_gradients")

                # Accumulate gradients
                if self.current_step == 1:
                    acc_slot.assign(grad)
                else:
                    acc_slot.assign_add(grad)

                # Return accumulated gradients when ready
                if self.current_step >= self.accumulate_steps:
                    accumulated_grads_and_vars.append(
                        (acc_slot.value / self.accumulate_steps, var)
                    )
                else:
                    accumulated_grads_and_vars.append((None, var))  # Skip this update
            else:
                accumulated_grads_and_vars.append((grad, var))

        # Reset accumulation counter
        if self.current_step >= self.accumulate_steps:
            self.current_step = 0
            # Clear accumulated gradients
            for grad, var in grads_and_vars:
                if grad is not None and var.trainable:
                    acc_slot = context.get_slot(var, "accumulated_gradients")
                    acc_slot.assign(np.zeros_like(acc_slot.value))

        return accumulated_grads_and_vars
