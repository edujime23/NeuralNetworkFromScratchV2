import numpy as np
from typing import Callable, Tuple, Dict, Any, List, Optional
import warnings
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
import time

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
    Fixed Wirtinger‐derivatives numerical approximator that correctly computes
    Jacobian matrices instead of summed derivatives.

    This version now also accepts integer‐typed inputs by silently casting them
    to float64 before proceeding.
    """
    def __init__(self, config: Optional[DerivativeConfig] = None):
        self.config = config or DerivativeConfig()
        self.cache: Dict[Tuple[int, int], np.ndarray] = {}
        self.evaluation_count = 0
        self.step_size_history: List[Tuple[float, float]] = []

        # Determine working precision based on configuration
        if self.config.high_precision and hasattr(np, 'complex256'):
            self.work_dtype = np.complex256
            self.real_dtype = np.float128
            self.machine_eps = np.finfo(np.float128).eps
        else:
            self.work_dtype = np.complex128
            self.real_dtype = np.float64
            self.machine_eps = np.finfo(np.float64).eps

        if self.config.verbose:
            print(f"Initialized with precision: {self.work_dtype}")
            print(f"Machine epsilon: {self.machine_eps:.2e}")

    def _robust_hash(self, arr: np.ndarray) -> int:
        """Create a robust hash for caching function evaluations"""
        tolerance = max(self.machine_eps * 1000, 1e-12)

        if np.iscomplexobj(arr):
            real_rounded = np.round(arr.real / tolerance) * tolerance
            imag_rounded = np.round(arr.imag / tolerance) * tolerance
            rounded = real_rounded + 1j * imag_rounded
        else:
            rounded = np.round(arr / tolerance) * tolerance

        return hash(rounded.tobytes())

    def _validate_inputs(
        self,
        func: Callable,
        inputs: Tuple[np.ndarray, ...],
        kwargs: Dict[str, Any]
    ):
        """
        Comprehensive input validation with detailed error messages,
        plus automatic promotion of integer arrays to float64.
        """
        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(inputs, tuple):
            raise TypeError("inputs must be a tuple of np.ndarray")

        if not inputs:
            raise ValueError("inputs tuple cannot be empty")

        sanitized: List[np.ndarray] = []
        for i, x in enumerate(inputs):
            if not isinstance(x, np.ndarray):
                raise TypeError(f"Input {i} must be np.ndarray, got {type(x)}")

            # If dtype is integer, cast to float64 immediately:
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float64)

            # Now only float or complex‐float are acceptable:
            if not (np.issubdtype(x.dtype, np.floating) or
                    np.issubdtype(x.dtype, np.complexfloating)):
                raise TypeError(f"Input {i} has unsupported dtype {x.dtype}")

            if x.size == 0:
                raise ValueError(f"Input {i} is empty array")

            if not np.all(np.isfinite(x)):
                count_bad = np.sum(~np.isfinite(x))
                raise ValueError(f"Input {i} contains non‐finite values: {count_bad} elements")

            sanitized.append(x)

        # Replace inputs tuple with sanitized (possibly cast) arrays:
        inputs = tuple(sanitized)

        # Test‐evaluate func(*inputs, **kwargs) exactly as before:
        try:
            test_output = func(*inputs, **kwargs)
            test_output = np.asarray(test_output)
            if test_output.size == 0:
                raise ValueError("Function returns empty array")
            if not np.all(np.isfinite(test_output)):
                raise ValueError("Function returns non‐finite values")
        except Exception as e:
            raise RuntimeError(f"Function evaluation test failed: {e}")

        # Finally, store the sanitized inputs to be used downstream:
        self.original_inputs = [x.astype(self.work_dtype) for x in inputs]
        self.current_inputs = [x.copy() for x in self.original_inputs]

    def _get_finite_difference_stencil(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get optimized finite difference coefficients for maximum accuracy"""
        stencils = {
            2: {
                'coefficients': np.array([-1.0, 1.0]) / 1.0,
                'points': np.array([-1, 1])
            },
            4: {
                'coefficients': np.array([1.0, -8.0, 8.0, -1.0]) / 12.0,
                'points': np.array([-2, -1, 1, 2])
            },
            6: {
                'coefficients': np.array([-1.0, 9.0, -45.0, 45.0, -9.0, 1.0]) / 60.0,
                'points': np.array([-3, -2, -1, 1, 2, 3])
            },
            8: {
                'coefficients': np.array([3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0]) / 840.0,
                'points': np.array([-4, -3, -2, -1, 1, 2, 3, 4])
            }
        }

        if order not in stencils:
            raise ValueError(f"Unsupported finite difference order: {order}")

        stencil = stencils[order]
        return stencil['coefficients'], stencil['points']

    def _find_optimal_step_size(
        self,
        func: Callable,
        x: np.ndarray,
        idx: int,
        element_idx: int,
        direction: str,
        kwargs: Dict[str, Any]
    ) -> float:
        """
        Find optimal step size with improved error handling.
        Fixed to avoid issues when error estimates are very small.
        """
        x_flat = x.ravel()
        original_val = x_flat[element_idx]

        # Initial step size estimate based on function and input characteristics
        if direction == 'complex_step':
            base_step = np.sqrt(self.machine_eps) * max(abs(original_val), 1.0)
        else:
            # If we're in "real"/"imag" mode, we pick scale from the real or imag part
            scale = max(abs(original_val.real if direction == 'real' else original_val.imag), 1.0)
            base_step = (self.machine_eps ** (1.0 / (self.config.max_order + 2))) * scale

        if not self.config.use_optimal_step_search:
            return np.clip(base_step, self.config.min_step_size, self.config.max_step_size)

        def error_estimate(log_step: float) -> float:
            """Estimate total error for a given step size"""
            step = np.exp(log_step)
            if step < self.config.min_step_size or step > self.config.max_step_size:
                return 1e10

            try:
                if direction == 'complex_step':
                    x_pert = x_flat.copy().astype(self.work_dtype)
                    x_pert[element_idx] = original_val + 1j * step
                    f_pert = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)

                    deriv1 = f_pert.ravel()[0].imag / step
                    x_pert[element_idx] = original_val + 1j * (step / 2)
                    f_pert_half = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                    deriv2 = f_pert_half.ravel()[0].imag / (step / 2)

                    error = abs(deriv1 - deriv2)
                else:
                    coeffs, points = self._get_finite_difference_stencil(self.config.max_order)
                    f_vals1: List[float] = []
                    f_vals2: List[float] = []

                    for point in points:
                        x_pert = x_flat.copy().astype(self.work_dtype)
                        if direction == 'real':
                            x_pert[element_idx] = (original_val.real + point * step) + 1j * original_val.imag
                        else:
                            x_pert[element_idx] = original_val.real + 1j * (original_val.imag + point * step)

                        f_val1 = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                        f_vals1.append(f_val1.ravel()[0])

                        # half‐step
                        if direction == 'real':
                            x_pert[element_idx] = (original_val.real + point * (step / 2)) + 1j * original_val.imag
                        else:
                            x_pert[element_idx] = original_val.real + 1j * (original_val.imag + point * (step / 2))

                        f_val2 = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                        f_vals2.append(f_val2.ravel()[0])

                    deriv1 = sum(c * f for c, f in zip(coeffs, f_vals1)) / step
                    deriv2 = sum(c * f for c, f in zip(coeffs, f_vals2)) / (step / 2)
                    error = abs(deriv1 - deriv2) / (2**self.config.max_order - 1)

                return float(error) + self.machine_eps

            except Exception:
                return 1e10

        try:
            result = minimize_scalar(
                error_estimate,
                bounds=(np.log(self.config.min_step_size), np.log(self.config.max_step_size)),
                method='bounded',
                options={'xatol': self.config.step_optimization_tolerance}
            )
            optimal_step = np.exp(result.x)
        except Exception:
            optimal_step = base_step

        if self.config.verbose:
            print(f"Optimal step size for element {element_idx}, direction {direction}: {optimal_step:.2e}")

        return optimal_step

    def _call_function_safe(
        self,
        func: Callable,
        input_idx: int,
        perturbed_input: np.ndarray,
        kwargs: Dict[str, Any]
    ) -> np.ndarray:
        """Safely call function with comprehensive error handling and caching"""
        cache_key = (input_idx, self._robust_hash(perturbed_input))

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Manage cache size
        if len(self.cache) > 50000:
            keys_to_remove = list(self.cache.keys())[: len(self.cache) // 10]
            for key in keys_to_remove:
                del self.cache[key]

        try:
            # Store original inputs (so we can restore after calling)
            original_inputs = [x.copy() for x in self.current_inputs]

            # Replace the perturbed input
            self.current_inputs[input_idx] = perturbed_input

            # Call function
            result = func(*self.current_inputs, **kwargs)
            result = np.asarray(result, dtype=self.work_dtype)

            # Validate result shape
            if result.shape != self.f0_shape:
                raise ValueError(f"Function output shape changed: {result.shape} vs {self.f0_shape}")

            if not np.all(np.isfinite(result)):
                warnings.warn("Function output contains non‐finite values")
                result = np.where(np.isfinite(result), result, 0.0)

            # Cache and restore
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
        x: np.ndarray,
        input_idx: int,
        kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives using complex‐step differentiation.
        Now produces full Jacobian matrix for each input element.
        """
        x_flat = x.ravel()
        n_inputs = x_flat.size
        n_outputs = self.f0.size

        grad_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        grad_conj_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)

        for i in range(n_inputs):
            original_val = x_flat[i]

            if self.config.adaptive_step:
                step_size = self._find_optimal_step_size(func, x, input_idx, i, 'complex_step', kwargs)
            else:
                step_size = np.sqrt(self.machine_eps) * max(abs(np.real(original_val)), abs(np.imag(original_val)), 1.0)

            self.step_size_history.append((step_size, 0.0))

            x_pert = x_flat.copy().astype(self.work_dtype)
            x_pert[i] = original_val + 1j * step_size

            f_pert = self._call_function_safe(func, input_idx, x_pert.reshape(x.shape), kwargs)
            f_pert_flat = f_pert.ravel()

            for j in range(n_outputs):
                # (f(x+ i⋅h) – f(x)) / (i⋅h)
                grad_z[j, i] = (f_pert_flat[j] - self.f0.ravel()[j]) / (1j * step_size)
                # ∂f/∂z̄ = 0 for an analytic (real‐to‐complex) path
                grad_conj_z[j, i] = 0.0

        output_shape = self.f0.shape + x.shape
        return grad_z.reshape(output_shape), grad_conj_z.reshape(output_shape)

    def _compute_finite_difference_derivative(
        self,
        func: Callable,
        x: np.ndarray,
        input_idx: int,
        kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives using high‐order finite differences on complex‐typed inputs.
        """
        x_flat = x.ravel()
        n_inputs = x_flat.size
        n_outputs = self.f0.size

        grad_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        grad_conj_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)

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

            def create_perturbation(delta: float, direction: str) -> np.ndarray:
                y = x_flat.copy().astype(self.work_dtype)
                if direction == 'real':
                    y[i] = (original_val.real + delta) + 1j * original_val.imag
                else:
                    y[i] = original_val.real + 1j * (original_val.imag + delta)
                return y.reshape(x.shape)

            # ∂/∂x (real part) via high‐order FD + optional Richardson:
            if self.config.richardson_extrapolation:
                # first with step = eps_real
                f_vals_h = [
                    self._call_function_safe(func, input_idx, create_perturbation(pt * eps_real, 'real'), kwargs)
                    for pt in points
                ]
                dfx_h = [
                    sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h)) / eps_real
                    for j in range(n_outputs)
                ]

                # second with half‐step = eps_real/2
                f_vals_h2 = [
                    self._call_function_safe(
                        func, input_idx,
                        create_perturbation(pt * (eps_real / 2), 'real'),
                        kwargs
                    )
                    for pt in points
                ]
                dfx_h2 = [
                    sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h2)) / (eps_real / 2)
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
                    sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_real)) / eps_real
                    for j in range(n_outputs)
                ]

            # ∂/∂y (imaginary part) via high‐order FD + optional Richardson:
            if self.config.richardson_extrapolation:
                f_vals_h = [
                    self._call_function_safe(func, input_idx, create_perturbation(pt * eps_imag, 'imag'), kwargs)
                    for pt in points
                ]
                dfy_h = [
                    sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h)) / eps_imag
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
                    sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h2)) / (eps_imag / 2)
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
                    sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_imag)) / eps_imag
                    for j in range(n_outputs)
                ]

            for j in range(n_outputs):
                # Wirtinger formulas:
                #   ∂f/∂z  =  0.5 * (            ∂f/∂x  – i⋅∂f/∂y )
                #   ∂f/∂z̄ =  0.5 * (            ∂f/∂x  + i⋅∂f/∂y )
                grad_z[j, i] = 0.5 * (dfx[j] - 1j * dfy[j])
                grad_conj_z[j, i] = 0.5 * (dfx[j] + 1j * dfy[j])

                if (abs(dfx[j]) > self.config.condition_threshold or
                    abs(dfy[j]) > self.config.condition_threshold):
                    warnings.warn(
                        f"Large derivative detected at output {j}, input {i}: "
                        f"|∂f/∂x|={abs(dfx[j]):.2e}, |∂f/∂y|={abs(dfy[j]):.2e}"
                    )

        output_shape = self.f0.shape + x.shape
        return grad_z.reshape(output_shape), grad_conj_z.reshape(output_shape)

    def compute_derivatives(
        self,
        func: Callable[..., np.ndarray],
        inputs: Tuple[np.ndarray, ...],
        kwargs: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Main method to compute Wirtinger derivatives with maximum accuracy.

        Returns a list of (∂f/∂z, ∂f/∂z̄) tuples, one for each input array.
        Each tuple contains arrays with shape (output_shape + input_shape).
        """
        if kwargs is None:
            kwargs = {}

        start_time = time.time()

        # --- Input validation (and integer→float promotion) happens here ---
        self._validate_inputs(func, inputs, kwargs)

        # At this point, self.current_inputs / self.original_inputs have been set
        # to float‐promoted arrays (dtype=float64 for real inputs, or complex128
        # if originally complex).  Now evaluate f0 and proceed exactly as before:

        f0 = func(*self.current_inputs, **kwargs)
        self.f0 = np.asarray(f0, dtype=self.work_dtype)
        self.f0_shape = self.f0.shape
        self.f0_norm = np.linalg.norm(self.f0)

        if self.config.verbose:
            print(f"Function output shape: {self.f0_shape}")
            print(f"Function output norm: {self.f0_norm:.2e}")

        self.evaluation_count = 1
        self.step_size_history.clear()
        self.cache.clear()

        grads: List[Tuple[np.ndarray, np.ndarray]] = []

        for idx, x in enumerate(self.current_inputs):
            if self.config.verbose:
                print(f"\nProcessing input {idx}: shape={x.shape}, dtype={x.dtype}, complex={np.iscomplexobj(x)}")

            is_complex = np.iscomplexobj(x)
            if not is_complex:
                grad_z, grad_conj_z = self._compute_complex_step_derivative(func, x, idx, kwargs)
            else:
                grad_z, grad_conj_z = self._compute_finite_difference_derivative(func, x, idx, kwargs)

            # If the original caller said “return real if input was real,” strip the imaginary parts
            if self.config.return_real_if_input_real and not is_complex:
                grad_z = grad_z.real.astype(np.float64)
                grad_conj_z = grad_conj_z.real.astype(np.float64)
            elif not self.config.high_precision:
                target_dtype = np.complex128 if is_complex else x.dtype
                grad_z = grad_z.astype(target_dtype)
                grad_conj_z = grad_conj_z.astype(target_dtype)

            grads.append((grad_z, grad_conj_z))

            if self.config.verbose:
                print(f"Input {idx} completed:")
                print(f"  Jacobian shape: {grad_z.shape}")
                print(f"  max |∂f/∂z| = {np.max(np.abs(grad_z)):.2e}")
                print(f"  max |∂f/∂z̄| = {np.max(np.abs(grad_conj_z)):.2e}")

        computation_time = time.time() - start_time
        if self.config.verbose:
            print(f"\nComputation completed in {computation_time:.3f} seconds")
            print(f"Total function evaluations: {self.evaluation_count}")
            print(f"Cache entries: {len(self.cache)}")

        return grads


# Convenience wrapper
def numerical_derivative(
    func: Callable[..., np.ndarray],
    inputs: Tuple[np.ndarray, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    verbose: bool = False,
    return_real_if_input_real: bool = True,
    adaptive_step: bool = True,
    high_precision: bool = False,
    max_order: int = 6,
    richardson_extrapolation: bool = True,
    condition_threshold: float = 1e12,
    use_optimal_step_search: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience wrapper for the WirtingerDifferentiator class.
    Returns full Jacobian matrices as requested.
    """
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


# Example usage—now supporting integer inputs seamlessly:
if __name__ == "__main__":
    def test_func(x, y):
        # x can be real (float or int) → becomes float64 internally
        # y can be complex (complex‐typed) → stays complex128
        return np.array([x[0]**2, x[1]**2, y[0] + y[1]])

    # Real input given as integers:
    x_int = np.array([1, 2], dtype=np.int64)
    # Complex input stays as complex:
    y_cplx = np.array([1 + 1j, 2 + 2j], dtype=np.complex128)

    print("Testing with integer input (auto‐promoted) and complex input:")
    grads = numerical_derivative(test_func, (x_int, y_cplx), verbose=True)

    print("\nGradient w.r.t. x (integer‐input → promoted to float):")
    print(f"  ∂f/∂x shape: {grads[0][0].shape}")  # should be (3, 2)
    print(f"  ∂f/∂x values:\n{grads[0][0]}")

    print("\nGradient w.r.t. y (complex input):")
    print(f"  ∂f/∂z shape: {grads[1][0].shape}")    # should be (3, 2)
    print(f"  ∂f/∂z values:\n{grads[1][0]}")
    print(f"  ∂f/∂z̄ values:\n{grads[1][1]}")
