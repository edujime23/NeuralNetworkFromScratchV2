import numpy as np
from typing import Callable, Tuple, List, Dict
from scipy import integrate
import cmath
import warnings
from numba import jit, complex128, float64
import math
from dataclasses import dataclass
import uuid

@dataclass
class IntegrationResult:
    """Result of complex path integration."""
    value: complex
    error_estimate: float
    method_used: str
    convergence_achieved: bool
    singularities_detected: List[complex]
    subdivisions_used: int
    function_evaluations: int
    warnings: List[str]
    alternative_results: Dict[str, complex]

@jit(complex128(complex128), nopython=True, cache=True)
def safe_complex_evaluation(z: complex) -> complex:
    """Safely evaluate a complex number, scaling if magnitude is too large."""
    real_part = z.real
    imag_part = z.imag
    magnitude = math.sqrt(real_part**2 + imag_part**2)
    if magnitude > 700:
        scale_factor = 700 / magnitude
        real_part *= scale_factor
        imag_part *= scale_factor
    return complex(real_part, imag_part)

@jit(float64(complex128), nopython=True, cache=True)
def complex_magnitude(z: complex) -> float:
    """Compute the magnitude of a complex number."""
    return math.sqrt(z.real**2 + z.imag**2)

@jit(complex128(float64), nopython=True, cache=True)
def unit_circle_fast(t: float) -> complex:
    """Parameterize the unit circle."""
    return complex(math.cos(t), math.sin(t))

class OptimizedComplexPathIntegral:
    """Class for optimized complex path integration."""
    def __init__(self, max_subdivisions: int = 100, abs_tol: float = 1e-12, min_step: float = 1e-10):
        self.max_subdivisions = max_subdivisions
        self.abs_tol = abs_tol
        self.min_step = min_step
        self.function_evals = 0

    def safe_evaluate(self, f: Callable, z: complex) -> Tuple[complex, bool, str]:
        """Safely evaluate function f at complex point z."""
        self.function_evals += 1
        try:
            z = safe_complex_evaluation(z)
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

    def detect_singularities(self, f: Callable, gamma: Callable, a: float, b: float, n_test: int = 200) -> List[Tuple[float, complex]]:
        """Detect potential singularities along the path."""
        singularities = []
        t_vals = np.linspace(a, b, n_test)
        dt = (b - a) / n_test
        for i, t in enumerate(t_vals):
            try:
                z = gamma(t)
                val, has_issues, desc = self.safe_evaluate(f, z)
                # Check derivative for rapid changes
                if i < n_test - 1:
                    z_next = gamma(t_vals[i + 1])
                    val_next, _, _ = self.safe_evaluate(f, z_next)
                    derivative_approx = abs(val_next - val) / dt
                    if derivative_approx > 1e6 or has_issues:
                        singularities.append((t, z))
                if has_issues and ("Division by zero" in desc or "Inf" in desc):
                    singularities.append((t, z))
            except Exception:
                singularities.append((t, gamma(t)))
        # Filter duplicates
        filtered_singularities = []
        tolerance = dt * 1.5
        for t, z in singularities:
            is_duplicate = False
            for t_existing, z_existing in filtered_singularities:
                if abs(t - t_existing) < tolerance and abs(z - z_existing) < 1e-8:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_singularities.append((t, z))
        return filtered_singularities[:5]

    def path_derivative(self, gamma: Callable, t: float, dt: float = 1e-10) -> complex:
        """Compute the derivative of the path gamma at t."""
        try:
            if hasattr(gamma, 'derivative'):
                return gamma.derivative(t)
            return (gamma(t + dt) - gamma(t - dt)) / (2 * dt)
        except:
            return (gamma(t + dt) - gamma(t)) / dt

    def adaptive_subdivision(self, integrand: Callable, a: float, b: float, singularities: List[Tuple[float, complex]], tol: float) -> Tuple[complex, float, List[str]]:
        """Perform adaptive quadrature with subdivision near singularities."""
        warnings_list = []
        intervals = [(a, b)]
        for t, _ in singularities:
            intervals.extend([(t - self.min_step, t), (t, t + self.min_step)])
        intervals = sorted([(max(a, min(t1, t2)), min(b, max(t1, t2))) for t1, t2 in intervals if t1 < b and t2 > a])

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

    def scipy_quad_integration(self, integrand: Callable, a: float, b: float, tol: float = 1e-12) -> Tuple[complex, float, List[str]]:
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

    def gauss_legendre_integration(self, integrand: Callable, a: float, b: float, n_points: int = 64) -> Tuple[complex, float, List[str]]:
        """Perform Gauss-Legendre quadrature."""
        try:
            xi, wi = np.polynomial.legendre.leggauss(n_points)
            t_points = 0.5 * (b - a) * xi + 0.5 * (b + a)
            weights = 0.5 * (b - a) * wi
            integral = sum(w * integrand(t) for w, t in zip(weights, t_points))
            error_estimate = abs(integral) / n_points**2
            return integral, error_estimate, []
        except Exception as e:
            return 0j, float('inf'), [f"Gauss-Legendre failed: {str(e)}"]

    def integrate(self, f: Callable[[complex], complex], gamma: Callable[[float], complex], a: float, b: float, tol: float = 1e-12) -> IntegrationResult:
        """Integrate f(z) along the complex path gamma(t) from a to b."""
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
            return IntegrationResult(
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

        return IntegrationResult(
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

def complex_path_integral(f: Callable[[complex], complex], gamma: Callable[[float], complex], a: float, b: float, tol: float = 1e-12) -> IntegrationResult:
    """Top-level function for complex path integration."""
    integrator = OptimizedComplexPathIntegral()
    return integrator.integrate(f, gamma, a, b, tol)

if __name__ == "__main__":
    print("=== OPTIMIZED COMPLEX PATH INTEGRAL TESTS ===\n")

    def reciprocal(z):
        """Test function: 1/z."""
        return 1/z if abs(z) > 1e-15 else 1e15+0j

    def unit_circle(t):
        """Unit circle parameterization."""
        return np.exp(1j * t)

    print("Test 1: ∮ 1/z dz around unit circle")
    result1 = complex_path_integral(reciprocal, unit_circle, 0, 2*np.pi, 1e-12)
    expected = 2j * np.pi
    print(f"Result: {result1.value:.12f}")
    print(f"Expected: {expected:.12f}")
    print(f"Error: {abs(result1.value - expected):.2e}")
    print(f"Method: {result1.method_used}")
    print(f"Convergence: {result1.convergence_achieved}")
    print(f"Function evaluations: {result1.function_evaluations}")

    def oscillatory(z):
        """Test function: exp(10iz)."""
        return np.exp(10j * z)

    print(f"\nTest 2: Oscillatory function exp(10iz)")
    result2 = complex_path_integral(oscillatory, unit_circle, 0, 2*np.pi, 1e-10)
    print(f"Result: {result2.value:.12f}")
    print(f"Method: {result2.method_used}")
    print(f"Function evaluations: {result2.function_evaluations}")

    def simple_pole(z):
        """Test function: 1/(z - 0.5)."""
        return 1/(z - 0.5) if abs(z - 0.5) > 1e-15 else 1e15+0j

    print(f"\nTest 3: Simple pole at z=0.5")
    result3 = complex_path_integral(simple_pole, unit_circle, 0, 2*np.pi, 1e-12)
    print(f"Result: {result3.value:.12f}")
    print(f"Expected (2πi): {2j * np.pi:.12f}")
    print(f"Method: {result3.method_used}")
    print(f"Singularities detected: {len(result3.singularities_detected)}")

    def polynomial(z):
        """Test function: z^3 + 2z^2 - z + 1."""
        return z**3 + 2*z**2 - z + 1

    def line_segment(t):
        """Line segment parameterization."""
        return t * (1 + 1j)

    print(f"\nTest 4: Polynomial along line segment")
    result4 = complex_path_integral(polynomial, line_segment, 0, 1, 1e-12)
    print(f"Result: {result4.value:.12f}")
    print(f"Method: {result4.method_used}")
    print(f"Error estimate: {result4.error_estimate:.2e}")

    def challenging(z):
        """Test function: exp(z)/(z^2 + 1)."""
        try:
            return np.exp(z) / (z**2 + 1)
        except:
            return 0j

    print(f"\nTest 5: exp(z)/(z^2 + 1)")
    result5 = complex_path_integral(challenging, unit_circle, 0, 2*np.pi, 1e-10)
    print(f"Result: {result5.value:.12f}")
    print(f"Method: {result5.method_used}")
    if result5.warnings:
        print(f"Warnings: {len(result5.warnings)}")

    def linear_function(z):
        """Test function: 2z."""
        return 2 * z  # ∫₀² 2x dx = 4

    def real_line_0_to_2(t):
        """Real line parameterization from 0 to 2."""
        return t

    print(f"\nTest 6: ∫₀² 2x dx = 4")
    result6 = complex_path_integral(linear_function, real_line_0_to_2, 0.0, 2.0, 1e-12)
    print(f"Result: {result6.value.real:.12f}")
    print(f"Expected: {4.0:.12f}")
    print(f"Error: {abs(result6.value.real - 4.0):.2e}")
    print(f"Method: {result6.method_used}")
    print(f"Convergence: {result6.convergence_achieved}")

    def log_over_x(z):
        """Test function: ln(z)/z."""
        return cmath.log(z) / z if abs(z) > 1e-10 else 0.0

    def real_line_1_to_e(t):
        """Real line parameterization from 1 to e."""
        return 1 + (math.e - 1) * t

    print(f"\nTest 7: ∫₁ᵉ ln(x)/x dx = 0.5")
    result7 = complex_path_integral(log_over_x, real_line_1_to_e, 0.0, 1.0, 1e-12)
    print(f"Result: {result7.value.real:.12f}")
    print(f"Expected: {0.5:.12f}")
    print(f"Error: {abs(result7.value.real - 0.5):.2e}")
    print(f"Method: {result7.method_used}")
    print(f"Convergence: {result7.convergence_achieved}")

    print(f"\n=== ALL TESTS COMPLETED ===")