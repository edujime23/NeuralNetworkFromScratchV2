import math
import warnings
import cmath
from dataclasses import dataclass
from collections.abc import Callable

from network.types.tensor import Tensor
from network.utils.cache import cache
from network.plugins.integral_tape.mixin import IntegralTapeHostMixin
from network.plugins.integral_tape.hooks import IntegralTapeHookPoints
from scipy import integrate
import numpy as np
from numba import njit, float64, complex128

__all__ = ["IntegrationResult", "OptimizedComplexPathIntegral", "complex_path_integral"]

@dataclass
class IntegrationResult:
    """Result of complex path integration."""
    value: complex
    error_estimate: float
    method_used: str
    convergence_achieved: bool
    singularities_detected: list[complex]
    subdivisions_used: int
    function_evaluations: int
    warnings: list[str]
    alternative_results: dict[str, complex]

@njit(complex128(complex128), cache=True)
def _safe_complex_evaluation(z: complex) -> complex:
    """Safely evaluate a complex number, scaling if magnitude is too large."""
    real_part = z.real
    imag_part = z.imag
    magnitude = math.sqrt(real_part**2 + imag_part**2)
    if magnitude > 700:
        scale_factor = 700 / magnitude
        real_part *= scale_factor
        imag_part *= scale_factor
    return complex(real_part, imag_part)

@njit(float64(complex128), cache=True)
def _complex_magnitude(z: complex) -> float:
    """Compute the magnitude of a complex number."""
    return math.sqrt(z.real**2 + z.imag**2)

@njit(complex128(float64), cache=True)
def _unit_circle_fast(t: float) -> complex:
    """Parameterize the unit circle."""
    return complex(math.cos(t), math.sin(t))

@njit(complex128[:](float64[:], float64[:]), cache=True)
def _safe_complex_evaluation_array(z_real: np.ndarray, z_imag: np.ndarray) -> np.ndarray:
    """Vectorized safe complex evaluation."""
    result = np.empty(len(z_real), dtype=complex128)
    for i in range(len(z_real)):
        z = complex(z_real[i], z_imag[i])
        result[i] = _safe_complex_evaluation(z)
    return result

@njit(float64[:](complex128[:]), cache=True)
def _complex_magnitude_array(z: np.ndarray) -> np.ndarray:
    """Vectorized complex magnitude computation."""
    result = np.empty(len(z), dtype=float64)
    for i in range(len(z)):
        result[i] = _complex_magnitude(z[i])
    return result

class OptimizedComplexPathIntegral(IntegralTapeHostMixin):
    """Class for optimized complex path integration."""

    def __init__(self, max_subdivisions: int = 100, abs_tol: float = 1e-12, min_step: float = 1e-10):
        super().__init__()
        self.max_subdivisions = max_subdivisions
        self.abs_tol = abs_tol
        self.min_step = min_step
        self.function_evals = 0

    @cache(max_limit=1000)
    def safe_evaluate(self, f: Callable, z: complex) -> tuple[complex, bool, str]:
        """Safely evaluate function f at complex point z."""
        self.function_evals += 1
        try:
            z = _safe_complex_evaluation(z)
            result = f(z)
            if np.isnan(result) or np.isinf(result):
                return 0j, True, "NaN/Inf result"
            magnitude = abs(result)
            if magnitude > 1e50:
                phase = cmath.phase(result)
                result = 1e50 * cmath.exp(1j * phase)
                return result, True, f"Extreme magnitude {magnitude:.2e} scaled"
            if magnitude < 1e-100 and magnitude > 0:
                return result, True, f"Very small magnitude {magnitude:.2e}"
            return result, False, "OK"
        except ZeroDivisionError:
            return complex(np.inf, 0), True, "Division by zero"
        except OverflowError:
            return 1e50 + 0j, True, "Overflow error"
        except Exception as e:
            return 0j, True, f"Evaluation error: {str(e)[:50]}"

    @cache(max_limit=500)
    def detect_singularities(self, f: Callable, gamma: Callable, a: float, b: float, n_test: int = 200) -> list[tuple[float, complex]]:
        """Detect potential singularities along the path."""
        self.call_hooks(IntegralTapeHookPoints.DETECT_SINGULAR)

        singularities = []
        t_vals = Tensor(np.linspace(a, b, n_test))
        dt = (b - a) / n_test

        for i in range(n_test):
            t = t_vals[i].item()
            try:
                z = gamma(t)
                val, has_issues, desc = self.safe_evaluate(f, z)

                # Check derivative for rapid changes
                if i < n_test - 1:
                    z_next = gamma(t_vals[i + 1].item())
                    val_next, _, _ = self.safe_evaluate(f, z_next)
                    derivative_approx = abs(val_next - val) / dt
                    if derivative_approx > 1e6 or has_issues:
                        singularities.append((t, z))
                if has_issues and ("Division by zero" in desc or "Inf" in desc):
                    singularities.append((t, z))
            except Exception:
                singularities.append((t, gamma(t)))

        # Filter duplicates using tensors for efficient computation
        filtered_singularities = []
        tolerance = dt * 1.5
        for t, z in singularities:
            is_duplicate = any(
                abs(t - t_existing) < tolerance and abs(z - z_existing) < 1e-8
                for t_existing, z_existing in filtered_singularities
            )
            if not is_duplicate:
                filtered_singularities.append((t, z))

        self.call_hooks(IntegralTapeHookPoints.AFTER_DETECTION)
        return filtered_singularities[:5]

    @cache(max_limit=1000)
    def path_derivative(self, gamma: Callable, t: float, dt: float = 1e-10) -> complex:
        """Compute the derivative of the path gamma at t."""
        try:
            if hasattr(gamma, 'derivative'):
                return gamma.derivative(t)
            return (gamma(t + dt) - gamma(t - dt)) / (2 * dt)
        except:
            return (gamma(t + dt) - gamma(t)) / dt

    def adaptive_subdivision(self, integrand: Callable, a: float, b: float,
                           singularities: list[tuple[float, complex]], tol: float) -> tuple[complex, float, list[str]]:
        """Perform adaptive quadrature with subdivision near singularities."""
        warnings_list = []
        intervals = [(a, b)]

        for t, _ in singularities:
            intervals.extend([(t - self.min_step, t), (t, t + self.min_step)])

        # Use tensor operations for interval processing
        interval_tensor = Tensor(intervals)
        intervals = sorted([(max(a, min(t1, t2)), min(b, max(t1, t2)))
                          for t1, t2 in interval_tensor.data if t1 < b and t2 > a])

        total_result = 0j
        total_error = 0.0

        for t1, t2 in intervals:
            if t2 - t1 < self.min_step:
                continue
            result, error, method_warnings = self.scipy_quad_integration(integrand, t1, t2, tol / len(intervals))
            total_result += result
            total_error += error
            warnings_list.extend(method_warnings)

        return total_result, total_error, warnings_list

    @cache(max_limit=200)
    def scipy_quad_integration(self, integrand: Callable, a: float, b: float,
                             tol: float = 1e-12) -> tuple[complex, float, list[str]]:
        """Perform integration using SciPy's adaptive quadrature."""
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                real_result, real_error = integrate.quad(
                    lambda t: integrand(t).real, a, b, epsabs=tol/2, epsrel=tol/2, limit=100
                )
                imag_result, imag_error = integrate.quad(
                    lambda t: integrand(t).imag, a, b, epsabs=tol/2, epsrel=tol/2, limit=100
                )
                result = complex(real_result, imag_result)
                error = math.sqrt(real_error**2 + imag_error**2)
                warning_msgs = [str(warn.message) for warn in w]
                return result, error, warning_msgs
        except Exception as e:
            return 0j, float('inf'), [f"Scipy quad integration failed: {str(e)}"]

    @cache(max_limit=100)
    def gauss_legendre_integration(self, integrand: Callable, a: float, b: float,
                                 n_points: int = 64) -> tuple[complex, float, list[str]]:
        """Perform Gauss-Legendre quadrature using tensors."""
        try:
            # Get Gauss-Legendre points and weights using numpy (for scipy compatibility)
            xi_np, wi_np = np.polynomial.legendre.leggauss(n_points)

            # Convert to tensors for computation
            xi = Tensor(xi_np)
            wi = Tensor(wi_np)

            # Transform to integration interval using tensor operations
            t_points = (b - a) * xi * 0.5 + (b + a) * 0.5
            weights = (b - a) * wi * 0.5

            # Compute integral using tensor operations where possible
            integral = 0j
            for i in range(n_points):
                t_val = t_points[i].item()
                weight = weights[i].item()
                integral += weight * integrand(t_val)

            error_estimate = abs(integral) / (n_points**2)
            return integral, error_estimate, []
        except Exception as e:
            return 0j, float('inf'), [f"Gauss-Legendre failed: {str(e)}"]

    def integrate(self, f: Callable[[complex], complex], gamma: Callable[[float], complex],
                 a: float, b: float, tol: float = 1e-12) -> IntegrationResult:
        """Integrate f(z) along the complex path gamma(t) from a to b."""
        self.call_hooks(IntegralTapeHookPoints.PRE_INTEGRATE)

        self.function_evals = 0
        warnings_list = []
        alternative_results = {}

        def integrand(t: float) -> complex:
            z = gamma(t)
            dzdt = self.path_derivative(gamma, t)
            fz, has_issues, desc = self.safe_evaluate(f, z)
            if has_issues:
                warnings_list.append(f"Issue at t={t:.6f}: {desc}")
            return fz * dzdt

        singularities = self.detect_singularities(f, gamma, a, b)
        sing_locations = [s[1] for s in singularities]
        path_closed = abs(gamma(a) - gamma(b)) < 1e-10

        # Choose integration method based on path type
        if path_closed:
            result, error, method_warnings = self.gauss_legendre_integration(integrand, a, b, n_points=64)
            method_used = 'gauss_legendre'
        else:
            result, error, method_warnings = self.adaptive_subdivision(integrand, a, b, singularities, tol)
            method_used = 'adaptive_scipy_quad'

        warnings_list.extend(method_warnings)

        if error == float('inf'):
            integration_result = IntegrationResult(
                value=0j,
                error_estimate=float('inf'),
                method_used='none',
                convergence_achieved=False,
                singularities_detected=sing_locations,
                subdivisions_used=len(singularities) * 2,
                function_evaluations=self.function_evals,
                warnings=warnings_list + [f"{method_used} failed"],
                alternative_results=alternative_results
            )
        else:
            integration_result = IntegrationResult(
                value=result,
                error_estimate=error,
                method_used=method_used,
                convergence_achieved=error <= tol,
                singularities_detected=sing_locations,
                subdivisions_used=len(singularities) * 2,
                function_evaluations=self.function_evals,
                warnings=warnings_list,
                alternative_results=alternative_results
            )

        self.call_hooks(IntegralTapeHookPoints.POST_INTEGRATE)
        return integration_result

def complex_path_integral(f: Callable[[complex], complex], gamma: Callable[[float], complex],
                         a: float, b: float, tol: float = 1e-12) -> IntegrationResult:
    """Top-level function for complex path integration."""
    integrator = OptimizedComplexPathIntegral()
    integrator.call_hooks(IntegralTapeHookPoints.REGISTER_METHODS)
    return integrator.integrate(f, gamma, a, b, tol)