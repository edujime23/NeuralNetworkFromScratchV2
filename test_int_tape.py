from network.tape.integral.core.tape import complex_path_integral
import numpy as np

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
    print(f"Error: {np.abs(result1.value - expected):.2e}")
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
        import cmath
        return cmath.log(z) / z if abs(z) > 1e-10 else 0.0

    def real_line_1_to_e(t):
        """Real line parameterization from 1 to e."""
        return 1 + (np.e - 1) * t

    print(f"\nTest 7: ∫₁ᵉ ln(x)/x dx = 0.5")
    result7 = complex_path_integral(log_over_x, real_line_1_to_e, 0.0, 1.0, 1e-12)
    print(f"Result: {result7.value.real:.12f}")
    print(f"Expected: {0.5:.12f}")
    print(f"Error: {abs(result7.value.real - 0.5):.2e}")
    print(f"Method: {result7.method_used}")
    print(f"Convergence: {result7.convergence_achieved}")

    print(f"\n=== ALL TESTS COMPLETED ===")