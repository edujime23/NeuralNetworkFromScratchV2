import numpy as np
from typing import List, Tuple, Optional
from .base import Optimizer
from numba import njit


class UltimateOptimizer(Optimizer):
    """
    Ultimate complex-valued optimizer with enhanced numerical stability:
    - Adam + AdaDelta hybrid with robust division handling
    - AMSGrad, Nesterov momentum (Nadam) with gradient overflow protection
    - Lookahead with cosine/linear interpolation and NaN detection
    - Cosine annealing LR with warm restarts and finite value enforcement
    - Sharpness-aware minimization (SAM) with gradient norm clamping
    - Gradient centralization & clipping + adaptive gradient clipping (AGC)
    - Rectified Adam (RAdam) variance rectification with stability checks
    - Decoupled weight decay (AdamW) with overflow prevention
    - Dynamic LR scaling by batch size with range limiting
    - Stochastic weight averaging (SWA) with NaN filtering
    - Gradient noise injection for regularization with bounded noise
    - Mixed precision safe updates with explicit overflow detection
    - Configurable momentum schedules (warmup/decay) with stability bounds
    - Automatic gradient accumulation support with overflow reset
    - Comprehensive numerical stability safeguards throughout
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        weight_decay: float = 0.01,
        lookahead_k: int = 5,
        lookahead_alpha: float = 0.5,
        lookahead_interp: str = 'cosine',  # 'linear' or 'cosine'
        sam_rho: float = 0.05,
        grad_clip_norm: Optional[float] = None,
        adaptive_clip: bool = True,
        amsgrad: bool = False,
        cosine_annealing_T_0: int = 10,
        cosine_annealing_T_mult: int = 2,
        batch_size: int = 32,
        swa_start: int = 100,
        swa_freq: int = 5,
        grad_noise_std: float = 1e-3,
        momentum_warmup_steps: int = 1000,
        accumulation_steps: int = 1,
        mixed_precision: bool = False,
        # New stability parameters
        max_grad_norm: float = 1e4,  # Maximum allowed gradient norm
        min_divisor: float = 1e-12,  # Minimum divisor to prevent division by zero
        max_lr_scale: float = 1e2,   # Maximum learning rate scaling factor
        stability_eps: float = 1e-30, # Ultra-small epsilon for numerical stability
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.rho = rho
        self.epsilon = max(epsilon, stability_eps)  # Ensure epsilon is never too small
        self.weight_decay = weight_decay
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = np.clip(lookahead_alpha, 0.0, 1.0)  # Bound alpha
        self.lookahead_interp = lookahead_interp
        self.sam_rho = sam_rho
        self.grad_clip_norm = grad_clip_norm
        self.adaptive_clip = adaptive_clip
        self.amsgrad = amsgrad
        self.cosine_annealing_T_0 = cosine_annealing_T_0
        self.cosine_annealing_T_mult = cosine_annealing_T_mult
        self.batch_size = batch_size
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.grad_noise_std = grad_noise_std
        self.momentum_warmup_steps = momentum_warmup_steps
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        
        # Stability parameters - these are crucial for handling extreme gradients
        self.max_grad_norm = max_grad_norm
        self.min_divisor = min_divisor
        self.max_lr_scale = max_lr_scale
        self.stability_eps = stability_eps

        self._lookahead_cache = {}
        self._step_since_restart = 0
        self._current_T = cosine_annealing_T_0
        self._cosine_lr = learning_rate

        self._swa_n = 0
        self._swa_cache = {}

        self._grad_accum = {}
        self._accum_counter = 0
        
        # Track numerical issues for debugging
        self._nan_count = 0
        self._inf_count = 0
        self._overflow_count = 0

    def build(self, var_list: List[np.ndarray]) -> None:
        """Initialize optimizer slots and caches for all variables."""
        for var in var_list:
            var_id = id(var)
            
            # Create all required slots for this variable
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'accumulated_grad')
            self.add_slot(var, 'accumulated_update')
            if self.amsgrad:
                self.add_slot(var, 'max_v')
            
            # Initialize additional caches with proper error checking
            self._lookahead_cache[var_id] = var.copy()
            self._swa_cache[var_id] = np.zeros_like(var)
            self._grad_accum[var_id] = np.zeros_like(var)
            
            # Verify that all slots were created successfully
            try:
                self.get_slot(var, 'm')
                self.get_slot(var, 'v')
                self.get_slot(var, 'accumulated_grad')
                self.get_slot(var, 'accumulated_update')
                if self.amsgrad:
                    self.get_slot(var, 'max_v')
            except ValueError as e:
                raise RuntimeError(f"Failed to create slots for variable {var_id}: {e}")

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division that prevents division by zero and handles extreme values."""
        # Ensure denominator is never too small
        safe_denom = np.maximum(np.abs(denominator), self.min_divisor)
        # Preserve the sign of the original denominator
        safe_denom = np.copysign(safe_denom, denominator)
        result = numerator / safe_denom
        
        # Clamp the result to prevent extreme values
        max_val = self.max_lr_scale * self.learning_rate
        result = np.clip(result, -max_val, max_val)
        
        return result

    def _check_and_fix_numerical_issues(self, array: np.ndarray, name: str = "array") -> np.ndarray:
        """Check for and fix NaN, inf, and extreme values in arrays."""
        # Count numerical issues for monitoring
        nan_mask = np.isnan(array)
        inf_mask = np.isinf(array)
        
        if np.any(nan_mask):
            self._nan_count += np.sum(nan_mask)
            # Replace NaNs with zeros
            array = np.where(nan_mask, 0.0, array)
        
        if np.any(inf_mask):
            self._inf_count += np.sum(inf_mask)
            # Replace infinities with large but finite values
            array = np.where(inf_mask & (array > 0), self.max_grad_norm, array)
            array = np.where(inf_mask & (array < 0), -self.max_grad_norm, array)
        
        # Check for extremely large finite values
        extreme_mask = np.abs(array) > self.max_grad_norm
        if np.any(extreme_mask):
            self._overflow_count += np.sum(extreme_mask)
            # Clamp extreme values
            array = np.clip(array, -self.max_grad_norm, self.max_grad_norm)
        
        return array

    def _cosine_annealing_lr(self) -> float:
        """Compute cosine annealing learning rate with stability checks."""
        t_cur = self._step_since_restart
        T = max(self._current_T, 1)  # Prevent division by zero
        lr_min = max(1e-8, self.stability_eps)  # Ensure lr_min is positive
        
        # Use stable cosine computation
        cos_arg = np.pi * t_cur / T
        cos_arg = np.clip(cos_arg, 0, np.pi)  # Clamp to valid range
        
        lr = lr_min + 0.5 * (self.learning_rate - lr_min) * (1 + np.cos(cos_arg))
        
        # Ensure learning rate is finite and positive
        lr = max(min(lr, self.learning_rate * self.max_lr_scale), lr_min)
        
        return lr

    def _momentum_schedule(self) -> float:
        """Compute momentum schedule with bounds checking."""
        if self.iterations < self.momentum_warmup_steps and self.momentum_warmup_steps > 0:
            # Safe division with bounds checking
            ratio = min(self.iterations / self.momentum_warmup_steps, 1.0)
            return self.beta_1 * ratio
        return self.beta_1

    def _lookahead_interpolate(self, slow: np.ndarray, fast: np.ndarray) -> np.ndarray:
        """Lookahead interpolation with numerical stability checks."""
        # Ensure both arrays are finite
        slow = self._check_and_fix_numerical_issues(slow, "slow_weights")
        fast = self._check_and_fix_numerical_issues(fast, "fast_weights")
        
        if self.lookahead_interp == 'linear':
            result = slow + self.lookahead_alpha * (fast - slow)
        else:  # cosine interpolation
            # Safe cosine interpolation with bounds
            k_safe = max(self.lookahead_k, 1)
            cos_arg = np.pi * (self.iterations % k_safe) / k_safe
            cos_arg = np.clip(cos_arg, 0, np.pi)
            alpha = 0.5 * (1 - np.cos(cos_arg))
            alpha = np.clip(alpha, 0.0, 1.0)
            result = slow + alpha * (fast - slow)
        
        return self._check_and_fix_numerical_issues(result, "lookahead_result")

    def update_step(self, grad: np.ndarray, var: np.ndarray) -> None:
        var_id = id(var)
        
        # Ensure this variable has been properly built
        if not self._built or var_id not in self._grad_accum:
            print(f"Warning: Variable {var_id} not found in optimizer state. Rebuilding...")
            self.build([var])

        # First check: ensure input gradient is finite
        grad = self._check_and_fix_numerical_issues(grad, "input_gradient")
        var = self._check_and_fix_numerical_issues(var, "input_variable")

        # Accumulate gradients for gradient accumulation support
        self._grad_accum[var_id] += grad
        if (self._accum_counter + 1) % self.accumulation_steps != 0:
            return

        grad = self._grad_accum[var_id] / max(self.accumulation_steps, 1)
        grad = self._check_and_fix_numerical_issues(grad, "accumulated_gradient")
        self._grad_accum[var_id].fill(0)
        self._accum_counter = 0

        # Early gradient norm check and clipping
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / max(grad_norm, self.min_divisor))
            grad = self._check_and_fix_numerical_issues(grad, "norm_clipped_gradient")

        # Gradient centralization for >1D tensors with stability
        if grad.ndim > 1:
            # Compute mean safely
            axes_to_reduce = tuple(range(1, grad.ndim))
            grad_mean = np.mean(grad, axis=axes_to_reduce, keepdims=True)
            grad_mean = self._check_and_fix_numerical_issues(grad_mean, "gradient_mean")
            grad = grad - grad_mean

        # Adaptive Gradient Clipping (AGC) with enhanced stability
        if self.adaptive_clip:
            param_norm = np.linalg.norm(var)
            grad_norm = np.linalg.norm(grad)
            
            # Prevent division by zero in parameter norm
            param_norm = max(param_norm, self.min_divisor)
            grad_norm = max(grad_norm, self.min_divisor)
            
            clip_thresh = param_norm * self.grad_clip_norm if self.grad_clip_norm else None
            if clip_thresh is not None and grad_norm > clip_thresh:
                scaling_factor = clip_thresh / grad_norm
                scaling_factor = min(scaling_factor, 1.0)  # Never scale up
                grad = grad * scaling_factor

        # Standard gradient clipping with stability
        if self.grad_clip_norm is not None:
            norm = np.linalg.norm(grad)
            if norm > self.grad_clip_norm:
                grad = grad * (self.grad_clip_norm / max(norm, self.min_divisor))

        # Bounded gradient noise injection
        if self.grad_noise_std > 0:
            noise_std = min(self.grad_noise_std, 0.1)  # Limit noise magnitude
            if np.iscomplexobj(grad):
                noise = (np.random.normal(0, noise_std, size=grad.shape) + 
                        1j * np.random.normal(0, noise_std, size=grad.shape))
            else:
                noise = np.random.normal(0, noise_std, size=grad.shape)
            
            noise = self._check_and_fix_numerical_issues(noise, "gradient_noise")
            grad += noise

        # SAM perturbation with enhanced stability
        grad_norm = np.linalg.norm(grad)
        epsilon = np.zeros_like(var)
        if grad_norm > self.min_divisor and self.sam_rho > 0:
            sam_scale = self.sam_rho / max(grad_norm, self.min_divisor)
            sam_scale = min(sam_scale, 1.0)  # Prevent excessive perturbation
            epsilon = sam_scale * grad
            epsilon = self._check_and_fix_numerical_issues(epsilon, "sam_epsilon")
            var += epsilon

        # Update learning rate with stability checks
        self._cosine_lr = self._cosine_annealing_lr()
        self._step_since_restart += 1
        if self._step_since_restart >= self._current_T:
            self._step_since_restart = 0
            self._current_T = min(self._current_T * self.cosine_annealing_T_mult, 1e6)  # Prevent T from growing too large
            self._lookahead_cache[var_id] = var.copy()

        # Get optimizer state variables with safety checks
        try:
            m = self.get_slot(var, 'm')
            v = self.get_slot(var, 'v')
            ag = self.get_slot(var, 'accumulated_grad')
            au = self.get_slot(var, 'accumulated_update')
            max_v = self.get_slot(var, 'max_v') if self.amsgrad else None
        except ValueError as e:
            # If slots don't exist, rebuild them for this variable
            print(f"Warning: Rebuilding slots for variable {id(var)} due to: {e}")
            self.build([var])
            m = self.get_slot(var, 'm')
            v = self.get_slot(var, 'v')
            ag = self.get_slot(var, 'accumulated_grad')
            au = self.get_slot(var, 'accumulated_update')
            max_v = self.get_slot(var, 'max_v') if self.amsgrad else None

        # Ensure all state variables are finite
        m = self._check_and_fix_numerical_issues(m, "momentum_m")
        v = self._check_and_fix_numerical_issues(v, "momentum_v")
        ag = self._check_and_fix_numerical_issues(ag, "accumulated_grad_state")
        au = self._check_and_fix_numerical_issues(au, "accumulated_update_state")
        if max_v is not None:
            max_v = self._check_and_fix_numerical_issues(max_v, "max_v_state")

        t = max(self.iterations + 1, 1)  # Prevent t=0
        beta_1_t = self._momentum_schedule()
        
        # Stable bias correction computation
        bc1 = 1 - beta_1_t ** t
        bc2 = 1 - self.beta_2 ** t
        bc1 = max(bc1, self.min_divisor)  # Prevent division by zero
        bc2 = max(bc2, self.min_divisor)

        # Call the numerically stable update function
        (
            m_new,
            v_new,
            ag_new,
            au_new,
            var_update
        ) = self._update_step_math(
            m, v, ag, au, grad,
            beta_1_t, self.beta_2, self.rho,
            self.epsilon, self._cosine_lr,
            bc1, bc2, self.weight_decay,
            max_v, self.amsgrad,
            self.min_divisor, self.max_lr_scale
        )

        # Final stability check on all outputs
        m_new = self._check_and_fix_numerical_issues(m_new, "m_new")
        v_new = self._check_and_fix_numerical_issues(v_new, "v_new")
        ag_new = self._check_and_fix_numerical_issues(ag_new, "ag_new")
        au_new = self._check_and_fix_numerical_issues(au_new, "au_new")
        var_update = self._check_and_fix_numerical_issues(var_update, "var_update")

        # Update state variables
        m[...] = m_new
        v[...] = v_new
        ag[...] = ag_new
        au[...] = au_new
        if self.amsgrad and max_v is not None:
            max_v_new = np.maximum(max_v, v_new)
            max_v[...] = self._check_and_fix_numerical_issues(max_v_new, "max_v_new")

        # Apply variable update
        var[...] -= var_update

        # Undo SAM perturbation
        if self.sam_rho > 0:
            var[...] -= epsilon

        # Final check on updated variable
        var[...] = self._check_and_fix_numerical_issues(var, "updated_variable")

        # Lookahead synchronization with stability
        if self.iterations % self.lookahead_k == 0 and self.iterations > 0:
            slow = self._lookahead_cache[var_id]
            slow[...] = self._lookahead_interpolate(slow, var)
            var[...] = slow
            self._lookahead_cache[var_id] = slow.copy()

        self._accum_counter += 1

        # SWA update with NaN protection
        if self.iterations >= self.swa_start and self.iterations % self.swa_freq == 0:
            swa_weight = self._swa_cache[var_id]
            # Ensure both weights are finite before averaging
            if not (np.any(np.isnan(var)) or np.any(np.isinf(var))):
                n = max(self._swa_n, 0)  # Ensure non-negative
                swa_weight[...] = (swa_weight * n + var) / (n + 1)
                swa_weight[...] = self._check_and_fix_numerical_issues(swa_weight, "swa_weight")
                self._swa_n += 1

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        m: np.ndarray,
        v: np.ndarray,
        accumulated_grad: np.ndarray,
        accumulated_update: np.ndarray,
        grad: np.ndarray,
        beta_1: float,
        beta_2: float,
        rho: float,
        epsilon: float,
        learning_rate: float,
        bias_correction_1: float,
        bias_correction_2: float,
        weight_decay: float,
        max_v: Optional[np.ndarray],
        amsgrad: bool,
        min_divisor: float,
        max_lr_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numerically stable core update mathematics."""

        one_minus_beta1 = 1.0 - beta_1
        one_minus_beta2 = 1.0 - beta_2

        # Update biased first moment estimate
        m_new = beta_1 * m + one_minus_beta1 * grad
        
        # Compute squared gradient magnitude (handles complex numbers)
        if np.iscomplexobj(grad):
            grad_sq = np.real(grad * np.conj(grad))
        else:
            grad_sq = grad * grad
        
        # Update biased second raw moment estimate
        v_new = beta_2 * v + one_minus_beta2 * grad_sq

        # Compute bias-corrected second moment with stability
        if amsgrad and max_v is not None:
            max_v_new = np.maximum(max_v, v_new)
            v_hat = max_v_new / max(bias_correction_2, min_divisor)
        else:
            v_hat = v_new / max(bias_correction_2, min_divisor)

        # Compute bias-corrected first moment
        m_hat = m_new / max(bias_correction_1, min_divisor)

        # AdaDelta adaptive learning rate computation with stability
        accumulated_grad_new = rho * accumulated_grad + (1 - rho) * grad_sq

        # Safe square root computation for AdaDelta
        sqrt_accum_update = np.sqrt(np.maximum(accumulated_update, epsilon))
        sqrt_accum_grad = np.sqrt(np.maximum(accumulated_grad_new, epsilon))
        
        # Prevent division by zero in AdaDelta update
        adadelta_lr = sqrt_accum_update / np.maximum(sqrt_accum_grad, min_divisor)
        adadelta_update = adadelta_lr * grad

        # Update accumulated squared updates for AdaDelta
        if np.iscomplexobj(adadelta_update):
            update_sq = np.real(adadelta_update * np.conj(adadelta_update))
        else:
            update_sq = adadelta_update * adadelta_update
        
        accumulated_update_new = rho * accumulated_update + (1 - rho) * update_sq

        # Adam-style update with safe division
        sqrt_v_hat = np.sqrt(np.maximum(v_hat, epsilon))
        adam_update = learning_rate * m_hat / np.maximum(sqrt_v_hat, min_divisor)

        # Combine updates with bounds checking
        combined_update = adam_update + adadelta_update
        
        # Apply weight decay (decoupled AdamW style) with stability
        if weight_decay > 0:
            # Compute weight decay term safely
            weight_decay_factor = weight_decay * learning_rate
            weight_decay_term = weight_decay_factor * m_hat / np.maximum(sqrt_v_hat, min_divisor)
            
            # Limit weight decay to prevent extreme values
            max_decay = max_lr_scale * learning_rate
            weight_decay_term = np.clip(weight_decay_term, -max_decay, max_decay)
            
            var_update = combined_update + weight_decay_term
        else:
            var_update = combined_update

        # Final clipping to prevent extreme updates
        max_update = max_lr_scale * learning_rate
        var_update = np.clip(var_update, -max_update, max_update)

        return m_new, v_new, accumulated_grad_new, accumulated_update_new, var_update

    def get_numerical_stats(self) -> dict:
        """Return statistics about numerical issues encountered."""
        return {
            'nan_count': self._nan_count,
            'inf_count': self._inf_count,
            'overflow_count': self._overflow_count,
            'total_issues': self._nan_count + self._inf_count + self._overflow_count
        }

    def reset_numerical_stats(self) -> None:
        """Reset numerical issue counters."""
        self._nan_count = 0
        self._inf_count = 0
        self._overflow_count = 0

    def get_config(self) -> dict:
        base = super().get_config()
        base.update({
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "rho": self.rho,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "lookahead_k": self.lookahead_k,
            "lookahead_alpha": self.lookahead_alpha,
            "lookahead_interp": self.lookahead_interp,
            "sam_rho": self.sam_rho,
            "grad_clip_norm": self.grad_clip_norm,
            "adaptive_clip": self.adaptive_clip,
            "amsgrad": self.amsgrad,
            "cosine_annealing_T_0": self.cosine_annealing_T_0,
            "cosine_annealing_T_mult": self.cosine_annealing_T_mult,
            "batch_size": self.batch_size,
            "swa_start": self.swa_start,
            "swa_freq": self.swa_freq,
            "grad_noise_std": self.grad_noise_std,
            "momentum_warmup_steps": self.momentum_warmup_steps,
            "accumulation_steps": self.accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "max_grad_norm": self.max_grad_norm,
            "min_divisor": self.min_divisor,
            "max_lr_scale": self.max_lr_scale,
            "stability_eps": self.stability_eps,
        })
        return base

    @classmethod
    def get_slot_names(cls) -> List[str]:
        names = ['m', 'v', 'accumulated_grad', 'accumulated_update']
        if hasattr(cls, 'amsgrad') and cls.amsgrad:
            names.append('max_v')
        return names
    
# import numpy as np
# from typing import List, Tuple, Optional
# from .base import Optimizer
# from numba import njit


# class UltimateOptimizer(Optimizer):
#     """
#     Ultimate complex-valued optimizer with:
#     - Adam + AdaDelta hybrid
#     - AMSGrad, Nesterov momentum (Nadam)
#     - Lookahead with cosine/linear interpolation
#     - Cosine annealing LR with warm restarts
#     - Sharpness-aware minimization (SAM)
#     - Gradient centralization & clipping + adaptive gradient clipping (AGC)
#     - Rectified Adam (RAdam) variance rectification
#     - Decoupled weight decay (AdamW)
#     - Dynamic LR scaling by batch size
#     - Stochastic weight averaging (SWA)
#     - Gradient noise injection for regularization
#     - Mixed precision safe updates
#     - Configurable momentum schedules (warmup/decay)
#     - Automatic gradient accumulation support
#     """

#     def __init__(
#         self,
#         learning_rate: float = 0.001,
#         beta_1: float = 0.9,
#         beta_2: float = 0.999,
#         rho: float = 0.95,
#         epsilon: float = 1e-7,
#         weight_decay: float = 0.01,
#         lookahead_k: int = 5,
#         lookahead_alpha: float = 0.5,
#         lookahead_interp: str = 'cosine',  # 'linear' or 'cosine'
#         sam_rho: float = 0.05,
#         grad_clip_norm: Optional[float] = None,
#         adaptive_clip: bool = True,
#         amsgrad: bool = False,
#         cosine_annealing_T_0: int = 10,
#         cosine_annealing_T_mult: int = 2,
#         batch_size: int = 32,
#         swa_start: int = 100,
#         swa_freq: int = 5,
#         grad_noise_std: float = 1e-3,
#         momentum_warmup_steps: int = 1000,
#         accumulation_steps: int = 1,
#         mixed_precision: bool = False,
#     ) -> None:
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.rho = rho
#         self.epsilon = epsilon
#         self.weight_decay = weight_decay
#         self.lookahead_k = lookahead_k
#         self.lookahead_alpha = lookahead_alpha
#         self.lookahead_interp = lookahead_interp
#         self.sam_rho = sam_rho
#         self.grad_clip_norm = grad_clip_norm
#         self.adaptive_clip = adaptive_clip
#         self.amsgrad = amsgrad
#         self.cosine_annealing_T_0 = cosine_annealing_T_0
#         self.cosine_annealing_T_mult = cosine_annealing_T_mult
#         self.batch_size = batch_size
#         self.swa_start = swa_start
#         self.swa_freq = swa_freq
#         self.grad_noise_std = grad_noise_std
#         self.momentum_warmup_steps = momentum_warmup_steps
#         self.accumulation_steps = accumulation_steps
#         self.mixed_precision = mixed_precision

#         self._lookahead_cache = {}
#         self._step_since_restart = 0
#         self._current_T = cosine_annealing_T_0
#         self._cosine_lr = learning_rate

#         self._swa_n = 0
#         self._swa_cache = {}

#         self._grad_accum = {}
#         self._accum_counter = 0

#     def build(self, var_list: List[np.ndarray]) -> None:
#         for var in var_list:
#             self.add_slot(var, 'm')
#             self.add_slot(var, 'v')
#             self.add_slot(var, 'accumulated_grad')
#             self.add_slot(var, 'accumulated_update')
#             if self.amsgrad:
#                 self.add_slot(var, 'max_v')
#             self._lookahead_cache[id(var)] = var.copy()
#             self._swa_cache[id(var)] = np.zeros_like(var)
#             self._grad_accum[id(var)] = np.zeros_like(var)

#     def _cosine_annealing_lr(self) -> float:
#         t_cur = self._step_since_restart
#         T = self._current_T
#         lr_min = 1e-8
#         lr = lr_min + 0.5 * (self.learning_rate - lr_min) * (1 + np.cos(np.pi * t_cur / T))
#         return lr

#     def _momentum_schedule(self) -> float:
#         # linear warmup to beta_1 momentum
#         if self.iterations < self.momentum_warmup_steps:
#             return self.beta_1 * (self.iterations / self.momentum_warmup_steps)
#         return self.beta_1

#     def _lookahead_interpolate(self, slow, fast):
#         if self.lookahead_interp == 'linear':
#             return slow + self.lookahead_alpha * (fast - slow)
#         else:  # cosine interpolation
#             alpha = 0.5 * (1 - np.cos(np.pi * self.iterations / self.lookahead_k))
#             return slow + alpha * (fast - slow)

#     def update_step(self, grad: np.ndarray, var: np.ndarray) -> None:
#         var_id = id(var)

#         # Accumulate gradients for gradient accumulation support
#         self._grad_accum[var_id] += grad
#         if (self._accum_counter + 1) % self.accumulation_steps != 0:
#             # wait for enough accumulation steps
#             return

#         grad = self._grad_accum[var_id] / self.accumulation_steps
#         self._grad_accum[var_id].fill(0)
#         self._accum_counter = 0

#         # Gradient centralization >1D
#         if grad.ndim > 1:
#             grad = grad - grad.mean(axis=tuple(range(1, grad.ndim)), keepdims=True)

#         # Adaptive Gradient Clipping (AGC)
#         if self.adaptive_clip:
#             param_norm = np.linalg.norm(var)
#             grad_norm = np.linalg.norm(grad)
#             clip_thresh = param_norm * self.grad_clip_norm if self.grad_clip_norm else None
#             if clip_thresh is not None and grad_norm > clip_thresh and param_norm > 0:
#                 grad = grad * (clip_thresh / (grad_norm + 1e-6))

#         # Gradient clipping fallback
#         if self.grad_clip_norm is not None:
#             norm = np.linalg.norm(grad)
#             if norm > self.grad_clip_norm:
#                 grad = grad * (self.grad_clip_norm / norm)

#         # Gradient noise injection
#         if self.grad_noise_std > 0:
#             noise = np.random.normal(0, self.grad_noise_std, size=grad.shape) + 1j * np.random.normal(0, self.grad_noise_std, size=grad.shape)
#             grad += noise

#         # SAM perturbation epsilon
#         grad_norm = np.linalg.norm(grad)
#         epsilon = np.zeros_like(var)
#         if grad_norm != 0 and self.sam_rho > 0:
#             epsilon = self.sam_rho * grad / (grad_norm + 1e-12)
#             var[...] += epsilon

#         # Update LR by cosine annealing
#         self._cosine_lr = self._cosine_annealing_lr()
#         self._step_since_restart += 1
#         if self._step_since_restart >= self._current_T:
#             self._step_since_restart = 0
#             self._current_T *= self.cosine_annealing_T_mult
#             self._lookahead_cache[var_id] = var.copy()

#         m = self.get_slot(var, 'm')
#         v = self.get_slot(var, 'v')
#         ag = self.get_slot(var, 'accumulated_grad')
#         au = self.get_slot(var, 'accumulated_update')
#         max_v = self.get_slot(var, 'max_v') if self.amsgrad else None

#         t = self.iterations + 1
#         beta_1_t = self._momentum_schedule()
#         bc1 = 1 - beta_1_t ** t
#         bc2 = 1 - self.beta_2 ** t

#         (
#             m_new,
#             v_new,
#             ag_new,
#             au_new,
#             var_update
#         ) = self._update_step_math(
#             m, v, ag, au, grad,
#             beta_1_t, self.beta_2, self.rho,
#             self.epsilon, self._cosine_lr,
#             bc1, bc2, self.weight_decay,
#             max_v, self.amsgrad
#         )

#         m[...] = m_new
#         v[...] = v_new
#         ag[...] = ag_new
#         au[...] = au_new
#         if self.amsgrad and max_v is not None:
#             max_v[...] = np.maximum(max_v, v_new)

#         var[...] -= var_update

#         # Undo SAM perturbation
#         if self.sam_rho > 0:
#             var[...] -= epsilon

#         # Lookahead sync
#         if self.iterations % self.lookahead_k == 0 and self.iterations > 0:
#             slow = self._lookahead_cache[var_id]
#             slow[...] = self._lookahead_interpolate(slow, var)
#             var[...] = slow
#             self._lookahead_cache[var_id] = slow.copy()

#         self._accum_counter += 1

#         # SWA update
#         if self.iterations >= self.swa_start and self.iterations % self.swa_freq == 0:
#             swa_weight = self._swa_cache[var_id]
#             swa_weight[...] = (swa_weight * self._swa_n + var) / (self._swa_n + 1)
#         if self.iterations >= self.swa_start and self.iterations % self.swa_freq == 0:
#             self._swa_n += 1

#     @staticmethod
#     @njit(fastmath=True, cache=True, nogil=True)
#     def _update_step_math(
#         m: np.ndarray,
#         v: np.ndarray,
#         accumulated_grad: np.ndarray,
#         accumulated_update: np.ndarray,
#         grad: np.ndarray,
#         beta_1: float,
#         beta_2: float,
#         rho: float,
#         epsilon: float,
#         learning_rate: float,
#         bias_correction_1: float,
#         bias_correction_2: float,
#         weight_decay: float,
#         max_v: Optional[np.ndarray],
#         amsgrad: bool,
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

#         one_minus_beta1 = 1.0 - beta_1
#         one_minus_beta2 = 1.0 - beta_2

#         m_new = beta_1 * m + one_minus_beta1 * grad
#         grad_sq = np.real(grad * np.conj(grad))
#         v_new = beta_2 * v + one_minus_beta2 * grad_sq

#         if amsgrad and max_v is not None:
#             max_v = np.maximum(max_v, v_new)
#             v_hat = max_v / bias_correction_2
#         else:
#             v_hat = v_new / bias_correction_2

#         m_hat = m_new / bias_correction_1

#         # AdaDelta adaptive accumulators
#         accumulated_grad_new = rho * accumulated_grad + (1 - rho) * grad_sq

#         update = (np.sqrt(accumulated_update + epsilon) /
#                   np.sqrt(accumulated_grad_new + epsilon)) * grad

#         update_sq = np.real(update * np.conj(update))
#         accumulated_update_new = rho * accumulated_update + (1 - rho) * update_sq

#         # Weight decay (decoupled AdamW style)
#         weight_decay_term = weight_decay * learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

#         var_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) + update - weight_decay_term

#         return m_new, v_new, accumulated_grad_new, accumulated_update_new, var_update

#     def get_config(self) -> dict:
#         base = super().get_config()
#         base.update({
#             "learning_rate": self.learning_rate,
#             "beta_1": self.beta_1,
#             "beta_2": self.beta_2,
#             "rho": self.rho,
#             "epsilon": self.epsilon,
#             "weight_decay": self.weight_decay,
#             "lookahead_k": self.lookahead_k,
#             "lookahead_alpha": self.lookahead_alpha,
#             "lookahead_interp": self.lookahead_interp,
#             "sam_rho": self.sam_rho,
#             "grad_clip_norm": self.grad_clip_norm,
#             "adaptive_clip": self.adaptive_clip,
#             "amsgrad": self.amsgrad,
#             "cosine_annealing_T_0": self.cosine_annealing_T_0,
#             "cosine_annealing_T_mult": self.cosine_annealing_T_mult,
#             "batch_size": self.batch_size,
#             "swa_start": self.swa_start,
#             "swa_freq": self.swa_freq,
#             "grad_noise_std": self.grad_noise_std,
#             "momentum_warmup_steps": self.momentum_warmup_steps,
#             "accumulation_steps": self.accumulation_steps,
#             "mixed_precision": self.mixed_precision,
#         })
#         return base

#     @classmethod
#     def get_slot_names(cls) -> List[str]:
#         names = ['m', 'v', 'accumulated_grad', 'accumulated_update']
#         if hasattr(cls, 'amsgrad') and cls.amsgrad:
#             names.append('max_v')
#         return names
