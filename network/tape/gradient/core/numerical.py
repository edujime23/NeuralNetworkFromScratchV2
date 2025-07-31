from collections.abc import Callable
from typing import Any
import warnings
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
import time
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor

@dataclass
class DerivativeConfig:
    """Configuration class for derivative computation parameters"""
    verbose: bool = False
    return_real_if_input_real: bool = True
    adaptive_step: bool = True
    high_precision: bool = False
    max_order: int = 6
    richardson_extrapolation: bool = True
    condition_threshold: float = 1e12
    min_step_size: float = 1e-16
    max_step_size: float = 1e-2
    step_optimization_tolerance: float = 1e-3
    use_optimal_step_search: bool = True
    parallel_evaluation: bool = False

class WirtingerDifferentiator:
    """
    Wirtinger derivatives numerical approximator that computes Jacobian matrices.
    Accepts integer inputs by casting them to float64.
    """
    def __init__(self, config: DerivativeConfig | None = None):
        self.config = config or DerivativeConfig()
        self.cache: dict[tuple[int, int], Tensor] = {}
        self.evaluation_count = 0
        self.step_size_history: list[tuple[float, float]] = []

        # Determine working precision
        if self.config.high_precision:
            try:
                self.work_dtype = Tensor.complex256 if hasattr(Tensor, 'complex256') else Tensor.complex128
                self.real_dtype = Tensor.float128 if hasattr(Tensor, 'float128') else Tensor.float64
                self.machine_eps = 1.084e-19  # approx epsilon for float128
            except:
                self.work_dtype = Tensor.complex128
                self.real_dtype = Tensor.float64
                self.machine_eps = 2.220446049250313e-16  # float64 epsilon
        else:
            self.work_dtype = Tensor.complex128
            self.real_dtype = Tensor.float64
            self.machine_eps = 2.220446049250313e-16

        if self.config.verbose:
            print(f"Initialized with precision: {self.work_dtype}")
            print(f"Machine epsilon: {self.machine_eps:.2e}")

    def _robust_hash(self, tensor: Tensor) -> int:
        """Create robust hash for caching function evaluations"""
        tolerance = max(self.machine_eps * 1000, 1e-12)

        if tensor.is_complex():
            real_rounded = tensor.real.round_to_multiple(tolerance)
            imag_rounded = tensor.imag.round_to_multiple(tolerance)
            rounded = real_rounded + 1j * imag_rounded
        else:
            rounded = tensor.round_to_multiple(tolerance)

        return hash(rounded.to_bytes())

    def _validate_inputs(
        self,
        func: Callable,
        inputs: tuple[Tensor, ...],
        kwargs: dict[str, Any]
    ):
        """
        Input validation with automatic integer to float64 promotion.
        """
        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(inputs, tuple):
            raise TypeError("inputs must be a tuple of Tensor")

        if not inputs:
            raise ValueError("inputs tuple cannot be empty")

        sanitized: list[Tensor] = []
        for i, x in enumerate(inputs):
            if not isinstance(x, Tensor):
                raise TypeError(f"Input {i} must be Tensor, got {type(x)}")

            # Cast integer tensors to float64
            if x.is_integer():
                x = x.astype(self.real_dtype)

            # Check for supported dtypes
            if not (x.is_floating() or x.is_complex()):
                raise TypeError(f"Input {i} has unsupported dtype {x.dtype}")

            if x.size == 0:
                raise ValueError(f"Input {i} is empty array")

            if not x.isfinite().all():
                count_bad = (~x.isfinite()).sum()
                raise ValueError(f"Input {i} contains non-finite values: {count_bad} elements")

            sanitized.append(x)

        inputs = tuple(sanitized)

        # Test function evaluation
        try:
            test_output = func(*inputs, **kwargs)
            if not isinstance(test_output, Tensor):
                test_output = Tensor(test_output)
            if test_output.size == 0:
                raise ValueError("Function returns empty array")
            if not test_output.isfinite().all():
                raise ValueError("Function returns non-finite values")
        except Exception as e:
            raise RuntimeError(f"Function evaluation test failed: {e}")

        self.original_inputs = [x.astype(self.work_dtype) for x in inputs]
        self.current_inputs = [x.copy() for x in self.original_inputs]

    def _get_finite_difference_stencil(self, order: int) -> tuple[Tensor, Tensor]:
        """Get optimized finite difference coefficients"""
        stencils = {
            2: {
                'coefficients': Tensor([-1.0, 1.0]) / 1.0,
                'points': Tensor([-1, 1])
            },
            4: {
                'coefficients': Tensor([1.0, -8.0, 8.0, -1.0]) / 12.0,
                'points': Tensor([-2, -1, 1, 2])
            },
            6: {
                'coefficients': Tensor([-1.0, 9.0, -45.0, 45.0, -9.0, 1.0]) / 60.0,
                'points': Tensor([-3, -2, -1, 1, 2, 3])
            },
            8: {
                'coefficients': Tensor([3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0]) / 840.0,
                'points': Tensor([-4, -3, -2, -1, 1, 2, 3, 4])
            }
        }

        if order not in stencils:
            raise ValueError(f"Unsupported finite difference order: {order}")

        stencil = stencils[order]
        return stencil['coefficients'], stencil['points']

    def _find_optimal_step_size(
        self,
        func: Callable,
        x: Tensor,
        idx: int,
        element_idx: int,
        direction: str,
        kwargs: dict[str, Any]
    ) -> float:
        """Find optimal step size with error handling"""
        x_flat = x.flatten()
        original_val = x_flat[element_idx]

        # Initial step size estimate
        if direction == 'complex_step':
            base_step = (self.machine_eps ** 0.5) * max(abs(original_val), 1.0)
        else:
            scale = max(abs(original_val.real if direction == 'real' else original_val.imag), 1.0)
            base_step = (self.machine_eps ** (1.0 / (self.config.max_order + 2))) * scale

        if not self.config.use_optimal_step_search:
            return max(min(base_step, self.config.max_step_size), self.config.min_step_size)

        def error_estimate(log_step: float) -> float:
            """Estimate total error for given step size"""
            step = Tensor.exp(Tensor(log_step)).item()
            if step < self.config.min_step_size or step > self.config.max_step_size:
                return 1e10

            try:
                if direction == 'complex_step':
                    x_pert = x_flat.copy().astype(self.work_dtype)
                    x_pert[element_idx] = original_val + 1j * step
                    f_pert = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)

                    deriv1 = f_pert.flatten()[0].imag / step
                    x_pert[element_idx] = original_val + 1j * (step / 2)
                    f_pert_half = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                    deriv2 = f_pert_half.flatten()[0].imag / (step / 2)

                    error = abs(deriv1 - deriv2)
                else:
                    coeffs, points = self._get_finite_difference_stencil(self.config.max_order)
                    f_vals1: list[float] = []
                    f_vals2: list[float] = []

                    for point in points:
                        x_pert = x_flat.copy().astype(self.work_dtype)
                        if direction == 'real':
                            x_pert[element_idx] = (original_val.real + point * step) + 1j * original_val.imag
                        else:
                            x_pert[element_idx] = original_val.real + 1j * (original_val.imag + point * step)

                        f_val1 = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                        f_vals1.append(f_val1.flatten()[0])

                        # Half-step
                        if direction == 'real':
                            x_pert[element_idx] = (original_val.real + point * (step / 2)) + 1j * original_val.imag
                        else:
                            x_pert[element_idx] = original_val.real + 1j * (original_val.imag + point * (step / 2))

                        f_val2 = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                        f_vals2.append(f_val2.flatten()[0])

                    deriv1 = sum(c * f for c, f in zip(coeffs, f_vals1)) / step
                    deriv2 = sum(c * f for c, f in zip(coeffs, f_vals2)) / (step / 2)
                    error = abs(deriv1 - deriv2) / (2**self.config.max_order - 1)

                return float(error) + self.machine_eps

            except Exception:
                return 1e10

        try:
            from math import log, exp
            result = minimize_scalar(
                error_estimate,
                bounds=(log(self.config.min_step_size), log(self.config.max_step_size)),
                method='bounded',
                options={'xatol': self.config.step_optimization_tolerance}
            )
            optimal_step = exp(result.x)
        except Exception:
            optimal_step = base_step

        if self.config.verbose:
            print(f"Optimal step size for element {element_idx}, direction {direction}: {optimal_step:.2e}")

        return optimal_step

    def _call_function_safe(
        self,
        func: Callable,
        input_idx: int,
        perturbed_input: Tensor,
        kwargs: dict[str, Any]
    ) -> Tensor:
        """Safely call function with caching"""
        cache_key = (input_idx, self._robust_hash(perturbed_input))

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Manage cache size
        if len(self.cache) > 50000:
            keys_to_remove = list(self.cache.keys())[: len(self.cache) // 10]
            for key in keys_to_remove:
                del self.cache[key]

        try:
            original_inputs = [x.copy() for x in self.current_inputs]
            self.current_inputs[input_idx] = perturbed_input

            result = func(*self.current_inputs, **kwargs)
            if not isinstance(result, Tensor):
                result = Tensor(result).astype(self.work_dtype)

            # Validate result shape
            if result.shape != self.f0_shape:
                raise ValueError(f"Function output shape changed: {result.shape} vs {self.f0_shape}")

            if not result.isfinite().all():
                warnings.warn("Function output contains non-finite values")
                result = result.where(result.isfinite(), 0.0)

            self.cache[cache_key] = result
            self.evaluation_count += 1
            self.current_inputs = original_inputs

            return result

        except Exception as e:
            if hasattr(self, 'original_inputs'):
                self.current_inputs = [x.copy() for x in self.original_inputs]
            raise RuntimeError(f"Function evaluation failed: {e}")

    def _compute_complex_step_derivative(
        self,
        func: Callable,
        x: Tensor,
        input_idx: int,
        kwargs: dict[str, Any]
    ) -> Gradient:
        """Compute derivatives using complex-step differentiation"""
        x_flat = x.flatten()
        n_inputs = x_flat.size
        n_outputs = self.f0.size

        grad_z = Tensor.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        grad_conj_z = Tensor.zeros((n_outputs, n_inputs), dtype=self.work_dtype)

        for i in range(n_inputs):
            original_val = x_flat[i]

            if self.config.adaptive_step:
                step_size = self._find_optimal_step_size(func, x, input_idx, i, 'complex_step', kwargs)
            else:
                step_size = (self.machine_eps ** 0.5) * max(abs(original_val.real), abs(original_val.imag), 1.0)

            self.step_size_history.append((step_size, 0.0))

            x_pert = x_flat.copy().astype(self.work_dtype)
            x_pert[i] = original_val + 1j * step_size

            f_pert = self._call_function_safe(func, input_idx, x_pert.reshape(x.shape), kwargs)
            f_pert_flat = f_pert.flatten()

            for j in range(n_outputs):
                grad_z[j, i] = (f_pert_flat[j] - self.f0.flatten()[j]) / (1j * step_size)
                grad_conj_z[j, i] = 0.0  # For analytic functions

        output_shape = self.f0.shape + x.shape
        return Gradient(
            h=grad_z.reshape(output_shape),
            ah=grad_conj_z.reshape(output_shape)
        )

    def _compute_finite_difference_derivative(
        self,
        func: Callable,
        x: Tensor,
        input_idx: int,
        kwargs: dict[str, Any]
    ) -> Gradient:
        """Compute derivatives using finite differences"""
        x_flat = x.flatten()
        n_inputs = x_flat.size
        n_outputs = self.f0.size

        grad_z = Tensor.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        grad_conj_z = Tensor.zeros((n_outputs, n_inputs), dtype=self.work_dtype)

        coeffs, points = self._get_finite_difference_stencil(self.config.max_order)

        for i in range(n_inputs):
            original_val = x_flat[i]

            if self.config.adaptive_step:
                eps_real = self._find_optimal_step_size(func, x, input_idx, i, 'real', kwargs)
                eps_imag = self._find_optimal_step_size(func, x, input_idx, i, 'imag', kwargs)
            else:
                scale_real = max(abs(original_val.real), 1.0)
                scale_imag = max(abs(original_val.imag), 1.0)
                order_factor = 1.0 / (self.config.max_order + 1)
                eps_real = (self.machine_eps ** order_factor) * scale_real
                eps_imag = (self.machine_eps ** order_factor) * scale_imag

            self.step_size_history.append((eps_real, eps_imag))

            def create_perturbation(delta: float, direction: str) -> Tensor:
                y = x_flat.copy().astype(self.work_dtype)
                if direction == 'real':
                    y[i] = (original_val.real + delta) + 1j * original_val.imag
                else:
                    y[i] = original_val.real + 1j * (original_val.imag + delta)
                return y.reshape(x.shape)

            # Compute d/dx (real part)
            if self.config.richardson_extrapolation:
                f_vals_h = [
                    self._call_function_safe(func, input_idx, create_perturbation(pt * eps_real, 'real'), kwargs)
                    for pt in points
                ]
                dfx_h = [
                    sum(c * f.flatten()[j] for c, f in zip(coeffs, f_vals_h)) / eps_real
                    for j in range(n_outputs)
                ]

                f_vals_h2 = [
                    self._call_function_safe(
                        func, input_idx,
                        create_perturbation(pt * (eps_real / 2), 'real'),
                        kwargs
                    )
                    for pt in points
                ]
                dfx_h2 = [
                    sum(c * f.flatten()[j] for c, f in zip(coeffs, f_vals_h2)) / (eps_real / 2)
                    for j in range(n_outputs)
                ]

                dfx = [
                    (2**self.config.max_order * h2 - h) / (2**self.config.max_order - 1)
                    for h, h2 in zip(dfx_h, dfx_h2)
                ]
            else:
                f_vals_real = [
                    self._call_function_safe(func, input_idx, create_perturbation(pt * eps_real, 'real'), kwargs)
                    for pt in points
                ]
                dfx = [
                    sum(c * f.flatten()[j] for c, f in zip(coeffs, f_vals_real)) / eps_real
                    for j in range(n_outputs)
                ]

            # Compute d/dy (imaginary part)
            if self.config.richardson_extrapolation:
                f_vals_h = [
                    self._call_function_safe(func, input_idx, create_perturbation(pt * eps_imag, 'imag'), kwargs)
                    for pt in points
                ]
                dfy_h = [
                    sum(c * f.flatten()[j] for c, f in zip(coeffs, f_vals_h)) / eps_imag
                    for j in range(n_outputs)
                ]

                f_vals_h2 = [
                    self._call_function_safe(
                        func, input_idx,
                        create_perturbation(pt * (eps_imag / 2), 'imag'),
                        kwargs
                    )
                    for pt in points
                ]
                dfy_h2 = [
                    sum(c * f.flatten()[j] for c, f in zip(coeffs, f_vals_h2)) / (eps_imag / 2)
                    for j in range(n_outputs)
                ]

                dfy = [
                    (2**self.config.max_order * h2 - h) / (2**self.config.max_order - 1)
                    for h, h2 in zip(dfy_h, dfy_h2)
                ]
            else:
                f_vals_imag = [
                    self._call_function_safe(func, input_idx, create_perturbation(pt * eps_imag, 'imag'), kwargs)
                    for pt in points
                ]
                dfy = [
                    sum(c * f.flatten()[j] for c, f in zip(coeffs, f_vals_imag)) / eps_imag
                    for j in range(n_outputs)
                ]

            for j in range(n_outputs):
                # Wirtinger derivatives: ∂f/∂z = 0.5 * (∂f/∂x - i∂f/∂y)
                grad_z[j, i] = 0.5 * (dfx[j] - 1j * dfy[j])
                grad_conj_z[j, i] = 0.5 * (dfx[j] + 1j * dfy[j])

                if (abs(dfx[j]) > self.config.condition_threshold or
                    abs(dfy[j]) > self.config.condition_threshold):
                    warnings.warn(
                        f"Large derivative detected at output {j}, input {i}: "
                        f"|∂f/∂x|={abs(dfx[j]):.2e}, |∂f/∂y|={abs(dfy[j]):.2e}"
                    )

        output_shape = self.f0.shape + x.shape
        return Gradient(
            h=grad_z.reshape(output_shape),
            ah=grad_conj_z.reshape(output_shape)
        )

    def compute_derivatives(
        self,
        func: Callable[..., Tensor],
        inputs: tuple[Tensor, ...],
        kwargs: dict[str, Any] | None = None
    ) -> list[Gradient]:
        """
        Compute Wirtinger derivatives with maximum accuracy.
        Returns list of Gradient objects, one for each input.
        """
        if kwargs is None:
            kwargs = {}

        start_time = time.time()

        self._validate_inputs(func, inputs, kwargs)

        f0 = func(*self.current_inputs, **kwargs)
        if not isinstance(f0, Tensor):
            f0 = Tensor(f0)
        self.f0 = f0.astype(self.work_dtype)
        self.f0_shape = self.f0.shape
        self.f0_norm = self.f0.norm()

        if self.config.verbose:
            print(f"Function output shape: {self.f0_shape}")
            print(f"Function output norm: {self.f0_norm:.2e}")

        self.evaluation_count = 1
        self.step_size_history.clear()
        self.cache.clear()

        grads: list[Gradient] = []

        for idx, x in enumerate(self.current_inputs):
            if self.config.verbose:
                print(f"\nProcessing input {idx}: shape={x.shape}, dtype={x.dtype}, complex={x.is_complex()}")

            is_complex = x.is_complex()
            if not is_complex:
                grad = self._compute_complex_step_derivative(func, x, idx, kwargs)
            else:
                grad = self._compute_finite_difference_derivative(func, x, idx, kwargs)

            # Return real values if input was real and config requests it
            if self.config.return_real_if_input_real and not is_complex:
                grad = Gradient(
                    h=grad.h.real.astype(self.real_dtype),
                    ah=grad.ah.real.astype(self.real_dtype)
                )
            elif not self.config.high_precision:
                target_dtype = self.work_dtype if is_complex else x.dtype
                grad = Gradient(
                    h=grad.h.astype(target_dtype),
                    ah=grad.ah.astype(target_dtype)
                )

            grads.append(grad)

            if self.config.verbose:
                print(f"Input {idx} completed:")
                print(f"  Jacobian shape: {grad.h.shape}")
                print(f"  max |∂f/∂z| = {grad.h.abs().max():.2e}")
                print(f"  max |∂f/∂z̄| = {grad.ah.abs().max():.2e}")

        computation_time = time.time() - start_time
        if self.config.verbose:
            print(f"\nComputation completed in {computation_time:.3f} seconds")
            print(f"Total function evaluations: {self.evaluation_count}")
            print(f"Cache entries: {len(self.cache)}")

        return grads


def numerical_derivative(
    func: Callable[..., Tensor],
    inputs: tuple[Tensor, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
    return_real_if_input_real: bool = True,
    adaptive_step: bool = True,
    high_precision: bool = False,
    max_order: int = 6,
    richardson_extrapolation: bool = True,
    condition_threshold: float = 1e12,
    use_optimal_step_search: bool = True,
) -> list[Gradient]:
    """Convenience wrapper for WirtingerDifferentiator class."""
    config = DerivativeConfig(
        verbose=verbose,
        return_real_if_input_real=return_real_if_input_real,
        adaptive_step=adaptive_step,
        high_precision=high_precision,
        max_order=max_order,
        richardson_extrapolation=richardson_extrapolation,
        condition_threshold=condition_threshold,
        use_optimal_step_search=use_optimal_step_search,
    )

    differentiator = WirtingerDifferentiator(config)
    return differentiator.compute_derivatives(func, inputs, kwargs)