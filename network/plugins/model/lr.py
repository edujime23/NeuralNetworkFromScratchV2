from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from network.plugins.base.plugin import Plugin, PluginContext, PluginPriority
from network.plugins.base.plugin import PluginHookPoint
from network.plugins.model.hooks import ModelHookPoints


class TrendState(Enum):
    STRONG_IMPROVEMENT = auto()
    STEADY_IMPROVEMENT = auto()
    MILD_IMPROVEMENT = auto()
    STAGNATION = auto()
    MILD_DEGRADATION = auto()
    STRONG_DEGRADATION = auto()
    PLATEAU = auto()
    EXPLOSION = auto()


@dataclass
class AdaptationMetrics:
    trend_momentum: float
    confidence_score: float
    stability_index: float
    volatility_ratio: float
    improvement_streak: int
    degradation_streak: int
    convergence_rate: float


class LearningRateState(NamedTuple):
    current_lr: float
    target_lr: float
    adjustment_factor: float
    reason: str
    confidence: float


class AdaptiveLRPlugin(Plugin):
    """
    Stable adaptive learning rate plugin with explosion prevention.

    Key stability improvements:
    - Conservative growth factors with gradual increases
    - Robust explosion detection with multiple safety checks
    - Gradual adaptation to prevent oscillations
    - Better plateau detection with noise filtering
    - Safe bounds management
    """

    def __init__(
        self,
        # Core parameters - balanced for stability
        history_size: int = 100,
        short_term_window: int = 10,
        medium_term_window: int = 25,
        long_term_window: int = 50,
        # Conservative growth factors
        mild_growth_factor: float = 1.05,
        moderate_growth_factor: float = 1.15,
        strong_growth_factor: float = 1.3,
        # Robust decay factors
        mild_decay_factor: float = 0.95,
        moderate_decay_factor: float = 0.8,
        strong_decay_factor: float = 0.6,
        emergency_decay_factor: float = 0.3,
        # Stability parameters
        momentum_alpha: float = 0.1,
        confidence_threshold: float = 0.7,
        trend_sensitivity: float = 0.01,
        volatility_penalty: float = 0.3,
        # Plateau detection
        plateau_patience: int = 15,
        plateau_variance_threshold: float = 0.005,
        plateau_escape_multiplier: float = 1.5,
        stagnation_threshold: float = 0.001,
        # Explosion prevention - more conservative
        explosion_z_score_threshold: float = 2.0,
        explosion_ratio_threshold: float = 1.5,
        consecutive_explosion_limit: int = 2,
        recovery_steps: int = 5,
        # Safe bounds
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        max_lr_change_ratio: float = 2.0,
        # Warmup period
        warmup_steps: int = 10,
        name: str = "stable_adaptive_lr",
        priority: int = PluginPriority.NORMAL,
        **kwargs,
    ):
        super().__init__(name=name, priority=priority, **kwargs)

        # Validate parameters
        self.history_size = max(20, history_size)
        self.short_term_window = min(short_term_window, self.history_size // 10)
        self.medium_term_window = min(medium_term_window, self.history_size // 4)
        self.long_term_window = min(long_term_window, self.history_size // 2)

        # Growth factors - conservative
        self.mild_growth_factor = mild_growth_factor
        self.moderate_growth_factor = moderate_growth_factor
        self.strong_growth_factor = strong_growth_factor

        # Decay factors
        self.mild_decay_factor = mild_decay_factor
        self.moderate_decay_factor = moderate_decay_factor
        self.strong_decay_factor = strong_decay_factor
        self.emergency_decay_factor = emergency_decay_factor

        # Stability parameters
        self.momentum_alpha = momentum_alpha
        self.confidence_threshold = confidence_threshold
        self.trend_sensitivity = trend_sensitivity
        self.volatility_penalty = volatility_penalty

        # Plateau parameters
        self.plateau_patience = plateau_patience
        self.plateau_variance_threshold = plateau_variance_threshold
        self.plateau_escape_multiplier = plateau_escape_multiplier
        self.stagnation_threshold = stagnation_threshold

        # Explosion prevention
        self.explosion_z_score_threshold = explosion_z_score_threshold
        self.explosion_ratio_threshold = explosion_ratio_threshold
        self.consecutive_explosion_limit = consecutive_explosion_limit
        self.recovery_steps = recovery_steps

        # Bounds
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.max_lr_change_ratio = max_lr_change_ratio

        # Warmup
        self.warmup_steps = warmup_steps

        # State tracking
        self._loss_history: deque[float] = deque(maxlen=self.history_size)
        self._lr_history: deque[float] = deque(maxlen=self.history_size)
        self._step_counter = 0

        # Metrics
        self._trend_momentum = 0.0
        self._confidence_score = 0.5
        self._stability_index = 0.0
        self._volatility_ratio = 1.0

        # Streak tracking
        self._improvement_streak = 0
        self._degradation_streak = 0
        self._plateau_counter = 0
        self._explosion_counter = 0

        # EMAs
        self._short_term_ema = None
        self._medium_term_ema = None
        self._long_term_ema = None
        self._loss_variance_ema = None

        # Safety mechanisms
        self._recovery_mode = False
        self._recovery_countdown = 0
        self._consecutive_explosions = 0
        self._last_stable_lr = None
        self._best_loss = float("inf")
        self._steps_since_best = 0

        # Smoothing for gradual changes
        self._lr_smoothing_factor = 0.7

        self._logger = logging.getLogger(f"plugin.{self.name}")

    def get_hook_points(self) -> list[PluginHookPoint]:
        return [ModelHookPoints.POST_TRAIN_STEP.value]

    def _validate_host(self, host: Any) -> None:
        for base_class in host.__class__.__bases__:
            if base_class.__name__ in ["Model", "Optimizer"]:
                return
        raise ValueError("Host model must be an Optimizer or a Model attribute.")

    def _update_exponential_averages(self, current_loss: float) -> None:
        """Update EMA windows with numerical stability."""
        if not math.isfinite(current_loss):
            self._logger.warning(f"Non-finite loss detected: {current_loss}")
            return

        if self._short_term_ema is None:
            self._short_term_ema = current_loss
            self._medium_term_ema = current_loss
            self._long_term_ema = current_loss
            self._loss_variance_ema = 0.0
        else:
            self._extracted_from__update_exponential_averages_7(current_loss)

    # TODO Rename this here and in `_update_exponential_averages`
    def _extracted_from__update_exponential_averages_7(self, current_loss):
        alpha_short = 2.0 / (self.short_term_window + 1)
        self._short_term_ema = (
            alpha_short * current_loss + (1 - alpha_short) * self._short_term_ema
        )
        alpha_medium = 2.0 / (self.medium_term_window + 1)
        self._medium_term_ema = (
            alpha_medium * current_loss + (1 - alpha_medium) * self._medium_term_ema
        )
        alpha_long = 2.0 / (self.long_term_window + 1)

        self._long_term_ema = (
            alpha_long * current_loss + (1 - alpha_long) * self._long_term_ema
        )

        # Update variance EMA with clipping
        squared_deviation = (current_loss - self._short_term_ema) ** 2
        self._loss_variance_ema = (
            alpha_short * squared_deviation
            + (1 - alpha_short) * self._loss_variance_ema
        )

    def _calculate_trend_momentum(self) -> float:
        """Calculate trend momentum with noise filtering."""
        if self._short_term_ema is None or self._medium_term_ema is None:
            return 0.0

        # Multi-scale momentum
        short_medium_trend = (self._medium_term_ema - self._short_term_ema) / (
            self._short_term_ema + 1e-12
        )
        medium_long_trend = (self._long_term_ema - self._medium_term_ema) / (
            self._medium_term_ema + 1e-12
        )

        # Weighted combination favoring recent trends but with stability
        combined_momentum = 0.6 * short_medium_trend + 0.4 * medium_long_trend

        # Smooth momentum updates
        self._trend_momentum = (
            1 - self.momentum_alpha
        ) * self._trend_momentum + self.momentum_alpha * combined_momentum

        return self._trend_momentum

    def _calculate_confidence_score(self) -> float:
        """Calculate confidence with stability bias."""
        if len(self._loss_history) < 5:
            return 0.5

        recent_losses = list(self._loss_history)[-10:]

        # Trend consistency
        improvements = sum(
            recent_losses[i] < recent_losses[i - 1]
            for i in range(1, len(recent_losses))
        )
        consistency_score = improvements / (len(recent_losses) - 1)

        # Stability score (lower volatility = higher confidence)
        if len(recent_losses) > 1:
            volatility = np.std(recent_losses) / (np.mean(recent_losses) + 1e-12)
            stability_score = 1.0 / (1.0 + volatility)
        else:
            stability_score = 0.5

        # Recent improvement score
        if len(recent_losses) >= 3:
            recent_improvement = (recent_losses[-3] - recent_losses[-1]) / (
                recent_losses[-3] + 1e-12
            )
            improvement_score = min(1.0, max(0.0, recent_improvement * 10))
        else:
            improvement_score = 0.5

        # Combined confidence
        raw_confidence = (
            0.4 * consistency_score + 0.4 * stability_score + 0.2 * improvement_score
        )

        # Smooth confidence updates
        self._confidence_score = 0.9 * self._confidence_score + 0.1 * raw_confidence

        return self._confidence_score

    def _detect_explosion(self, current_loss: float) -> bool:
        """Robust explosion detection with multiple criteria."""
        if len(self._loss_history) < 3:
            return False

        # Check for non-finite values
        if not math.isfinite(current_loss):
            return True

        # Z-score based detection
        if len(self._loss_history) > 10:
            historical_losses = list(self._loss_history)[:-2]
            mean_loss = np.mean(historical_losses)
            std_loss = np.std(historical_losses)

            if std_loss > 1e-12:
                z_score = (current_loss - mean_loss) / std_loss
                if z_score > self.explosion_z_score_threshold:
                    return True

        # Ratio-based detection
        if len(self._loss_history) >= 2:
            prev_loss = self._loss_history[-2]
            if prev_loss > 0:
                ratio = current_loss / prev_loss
                if ratio > self.explosion_ratio_threshold:
                    return True

        # Gradient-based detection
        if len(self._loss_history) >= 3:
            recent_losses = list(self._loss_history)[-3:]
            gradients = [
                recent_losses[i] - recent_losses[i - 1]
                for i in range(1, len(recent_losses))
            ]
            if all(g > 0 for g in gradients) and gradients[-1] > gradients[-2] * 2:
                return True

        return False

    def _detect_plateau(self) -> bool:
        """Detect plateau with noise filtering."""
        if len(self._loss_history) < self.plateau_patience:
            return False

        recent_losses = list(self._loss_history)[-self.plateau_patience :]

        # Variance-based plateau detection
        variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)
        cv = np.sqrt(variance) / (mean_loss + 1e-12)

        # Trend-based plateau detection
        trend_strength = abs(self._trend_momentum)

        is_plateau = (
            cv < self.plateau_variance_threshold
            and trend_strength < self.stagnation_threshold
        )

        if is_plateau:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0

        return self._plateau_counter >= 3  # Require consecutive plateau detections

    def _detect_trend_state(self, current_loss: float) -> TrendState:
        """Detect trend state with stability focus."""
        if len(self._loss_history) < 5:
            return TrendState.STAGNATION

        # Check explosion first
        if self._detect_explosion(current_loss):
            return TrendState.EXPLOSION

        # Check plateau
        if self._detect_plateau():
            return TrendState.PLATEAU

        # Analyze recent trend
        recent_losses = list(self._loss_history)[-min(10, len(self._loss_history)) :]
        if len(recent_losses) < 2:
            return TrendState.STAGNATION

        # Calculate relative changes
        relative_changes = [
            (recent_losses[i - 1] - recent_losses[i]) / (recent_losses[i - 1] + 1e-12)
            for i in range(1, len(recent_losses))
        ]

        avg_change = np.mean(relative_changes)
        change_consistency = max(0.0, 1.0 - np.std(relative_changes) * 2)

        # Conservative trend classification
        if avg_change > self.trend_sensitivity * 3 and change_consistency > 0.7:
            return TrendState.STRONG_IMPROVEMENT
        elif avg_change > self.trend_sensitivity * 1.5 and change_consistency > 0.5:
            return TrendState.STEADY_IMPROVEMENT
        elif avg_change > self.trend_sensitivity * 0.5:
            return TrendState.MILD_IMPROVEMENT
        elif avg_change < -self.trend_sensitivity * 2:
            return TrendState.STRONG_DEGRADATION
        elif avg_change < -self.trend_sensitivity * 0.5:
            return TrendState.MILD_DEGRADATION
        else:
            return TrendState.STAGNATION

    def _calculate_adaptation_multiplier(
        self, trend_state: TrendState, confidence: float
    ) -> float:
        """Calculate conservative adaptation multiplier."""
        base_multiplier = 1.0

        # Conservative multiplier mapping
        multiplier_map = {
            TrendState.STRONG_IMPROVEMENT: self.strong_growth_factor,
            TrendState.STEADY_IMPROVEMENT: self.moderate_growth_factor,
            TrendState.MILD_IMPROVEMENT: self.mild_growth_factor,
            TrendState.STAGNATION: 1.0,
            TrendState.MILD_DEGRADATION: self.mild_decay_factor,
            TrendState.STRONG_DEGRADATION: self.moderate_decay_factor,
            TrendState.PLATEAU: self.plateau_escape_multiplier,
            TrendState.EXPLOSION: self.emergency_decay_factor,
        }

        base_multiplier = multiplier_map.get(trend_state, 1.0)

        # Apply confidence scaling (more conservative)
        if trend_state in [
            TrendState.STRONG_IMPROVEMENT,
            TrendState.STEADY_IMPROVEMENT,
        ]:
            confidence_scaling = 0.5 + 0.5 * confidence  # Range [0.5, 1.0]
            base_multiplier = 1.0 + (base_multiplier - 1.0) * confidence_scaling

        # Apply streak bonuses (conservative)
        if trend_state in [
            TrendState.STRONG_IMPROVEMENT,
            TrendState.STEADY_IMPROVEMENT,
            TrendState.MILD_IMPROVEMENT,
        ]:
            streak_bonus = min(1.2, 1.0 + 0.02 * self._improvement_streak)
            base_multiplier *= streak_bonus

        # Apply volatility penalty
        if self._volatility_ratio > 1.5:
            penalty = max(
                0.8, 1.0 - (self._volatility_ratio - 1.0) * self.volatility_penalty
            )
            base_multiplier *= penalty

        return base_multiplier

    def _apply_learning_rate_update(
        self, optimizer: Any, old_lr: float
    ) -> LearningRateState:
        """Apply stable learning rate update."""
        current_loss = self._loss_history[-1]

        # Skip during warmup
        if self._step_counter < self.warmup_steps:
            return LearningRateState(
                current_lr=old_lr,
                target_lr=old_lr,
                adjustment_factor=1.0,
                reason="warmup phase",
                confidence=0.5,
            )

        # Update metrics
        self._update_exponential_averages(current_loss)
        self._calculate_trend_momentum()
        confidence = self._calculate_confidence_score()
        trend_state = self._detect_trend_state(current_loss)

        # Update streaks
        if trend_state in [
            TrendState.STRONG_IMPROVEMENT,
            TrendState.STEADY_IMPROVEMENT,
            TrendState.MILD_IMPROVEMENT,
        ]:
            self._improvement_streak += 1
            self._degradation_streak = 0
        elif trend_state in [
            TrendState.MILD_DEGRADATION,
            TrendState.STRONG_DEGRADATION,
        ]:
            self._degradation_streak += 1
            self._improvement_streak = 0

        # Update volatility ratio
        if self._loss_variance_ema is not None and self._short_term_ema is not None:
            self._volatility_ratio = np.sqrt(self._loss_variance_ema) / (
                self._short_term_ema + 1e-12
            )

        # Track best loss
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._steps_since_best = 0
            self._last_stable_lr = old_lr
        else:
            self._steps_since_best += 1

        # Handle recovery mode
        if self._recovery_mode:
            if self._recovery_countdown > 0:
                multiplier = self.strong_decay_factor
                reason = f"recovery mode ({self._recovery_countdown} steps remaining)"
                self._recovery_countdown -= 1
            else:
                self._recovery_mode = False
                multiplier = 1.0
                reason = "exiting recovery mode"
        elif trend_state == TrendState.EXPLOSION:
            self._consecutive_explosions += 1
            if self._consecutive_explosions >= self.consecutive_explosion_limit:
                self._recovery_countdown = self.recovery_steps
                multiplier = self.emergency_decay_factor
                reason = (
                    f"explosion detected! ({self._consecutive_explosions} consecutive)"
                )
                self._recovery_mode = True
                self._logger.warning(f"Loss explosion detected: {current_loss:.6f}")
            else:
                multiplier = self.strong_decay_factor
                reason = f"explosion warning ({self._consecutive_explosions}/{self.consecutive_explosion_limit})"
        else:
            self._consecutive_explosions = 0
            multiplier = self._calculate_adaptation_multiplier(trend_state, confidence)
            reason = f"{trend_state.name.lower()} (conf: {confidence:.2f})"

        # Calculate target learning rate
        target_lr = old_lr * multiplier

        # Apply conservative bounds
        target_lr = max(self.min_lr, min(self.max_lr, target_lr))

        # Limit maximum change per step
        max_change = old_lr * self.max_lr_change_ratio
        min_change = old_lr / self.max_lr_change_ratio
        target_lr = max(min_change, min(max_change, target_lr))

        # Apply smoothing for gradual changes
        if abs(multiplier - 1.0) < 0.5:  # Small changes
            target_lr = old_lr * self._lr_smoothing_factor + target_lr * (
                1 - self._lr_smoothing_factor
            )

        return LearningRateState(
            current_lr=old_lr,
            target_lr=target_lr,
            adjustment_factor=target_lr / old_lr,
            reason=reason,
            confidence=confidence,
        )

    def on_post_train_step(self, context: PluginContext, *args, **kwargs):
        """Main hook function with error handling."""
        try:
            loss = context.metadata["logs"]["loss"]
            self._step_counter += 1

            # Validate loss
            if not math.isfinite(loss) or loss < 0:
                self._logger.warning(f"Invalid loss value: {loss}")
                return

            # Attach to optimizer
            model = context.host
            optimizer = getattr(model, "_optimizer", None)

            # Record loss
            self._loss_history.append(loss)

            # Get current learning rate
            old_lr = getattr(optimizer, "lr", None)
            if old_lr is None or not math.isfinite(old_lr):
                self._logger.warning(f"Invalid learning rate: {old_lr}")
                return

            self._lr_history.append(old_lr)

            # Apply updates
            if len(self._loss_history) >= 2:
                lr_state = self._apply_learning_rate_update(optimizer, old_lr)

                # Update optimizer
                if abs(lr_state.target_lr - lr_state.current_lr) > 1e-12:
                    optimizer._lr = lr_state.target_lr

                    # Log changes
                    if abs(lr_state.adjustment_factor - 1.0) > 0.01:
                        self._logger.debug(
                            f"Step {self._step_counter} | "
                            f"Loss: {loss:.6f} | "
                            f"LR: {lr_state.current_lr:.2e} -> {lr_state.target_lr:.2e} "
                            f"({lr_state.adjustment_factor:.3f}x) | "
                            f"{lr_state.reason}"
                        )

        except Exception as e:
            self._logger.error(f"Error in adaptive LR plugin: {e}")
            # Don't raise to avoid breaking training
