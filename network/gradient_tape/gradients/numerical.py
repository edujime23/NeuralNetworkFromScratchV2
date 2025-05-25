import numpy as np
from typing import Callable, Tuple, Dict, Any, List, Union, Optional
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
    Fixed Wirtinger derivatives numerical approximator that correctly computes
    Jacobian matrices instead of summed derivatives.
    
    Key fixes:
    1. Computes full Jacobian: ∂f_j/∂x_i for each output j and input i
    2. Correctly applies Wirtinger formulas to vector-valued functions
    3. Improved step size optimization with proper error handling
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

    def _validate_inputs(self, func: Callable, inputs: Tuple[np.ndarray, ...], kwargs: Dict[str, Any]):
        """Comprehensive input validation with detailed error messages"""
        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(inputs, tuple):
            raise TypeError("inputs must be a tuple of np.ndarray")

        if not inputs:
            raise ValueError("inputs tuple cannot be empty")

        for i, x in enumerate(inputs):
            if not isinstance(x, np.ndarray):
                raise TypeError(f"Input {i} must be np.ndarray, got {type(x)}")
            if x.size == 0:
                raise ValueError(f"Input {i} is empty array")
            if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.complexfloating):
                raise TypeError(f"Input {i} has unsupported dtype {x.dtype}")
            if not np.all(np.isfinite(x)):
                raise ValueError(f"Input {i} contains non-finite values: {np.sum(~np.isfinite(x))} elements")

        # Test function evaluation
        try:
            test_output = func(*inputs, **kwargs)
            test_output = np.asarray(test_output)
            if test_output.size == 0:
                raise ValueError("Function returns empty array")
            if not np.all(np.isfinite(test_output)):
                raise ValueError("Function returns non-finite values")
        except Exception as e:
            raise RuntimeError(f"Function evaluation test failed: {e}")

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

    def _find_optimal_step_size(self, func: Callable, x: np.ndarray, idx: int, 
                              element_idx: int, direction: str, kwargs: Dict[str, Any]) -> float:
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
                # Compute derivative with this step size
                if direction == 'complex_step':
                    x_pert = x_flat.copy().astype(self.work_dtype)
                    x_pert[element_idx] = original_val + 1j * step
                    f_pert = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                    
                    # Take derivative of first output element for error estimation
                    deriv1 = f_pert.ravel()[0].imag / step
                    
                    # Compute with half step for error estimation
                    x_pert[element_idx] = original_val + 1j * (step / 2)
                    f_pert_half = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                    deriv2 = f_pert_half.ravel()[0].imag / (step / 2)
                    
                    # Estimate error from difference
                    error = abs(deriv1 - deriv2)
                    
                else:
                    # For finite differences, use Richardson extrapolation error estimate
                    coeffs, points = self._get_finite_difference_stencil(self.config.max_order)
                    
                    f_vals1 = []
                    f_vals2 = []
                    
                    for point in points:
                        x_pert = x_flat.copy().astype(self.work_dtype)
                        if direction == 'real':
                            x_pert[element_idx] = original_val.real + point * step + 1j * original_val.imag
                        else:  # imaginary
                            x_pert[element_idx] = original_val.real + 1j * (original_val.imag + point * step)
                        
                        f_val1 = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                        f_vals1.append(f_val1.ravel()[0])  # Use first output element
                        
                        # Half step for error estimate
                        if direction == 'real':
                            x_pert[element_idx] = original_val.real + point * (step/2) + 1j * original_val.imag
                        else:
                            x_pert[element_idx] = original_val.real + 1j * (original_val.imag + point * (step/2))
                        
                        f_val2 = self._call_function_safe(func, idx, x_pert.reshape(x.shape), kwargs)
                        f_vals2.append(f_val2.ravel()[0])
                    
                    deriv1 = sum(c * f for c, f in zip(coeffs, f_vals1)) / step
                    deriv2 = sum(c * f for c, f in zip(coeffs, f_vals2)) / (step/2)
                    
                    # Richardson extrapolation error estimate
                    error = abs(deriv1 - deriv2) / (2**self.config.max_order - 1)
                
                # Add small baseline to prevent optimization issues with very small errors
                return float(error) + self.machine_eps
                
            except Exception:
                return 1e10

        # Optimize step size
        log_base = np.log(base_step)
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

    def _call_function_safe(self, func: Callable, input_idx: int, 
                          perturbed_input: np.ndarray, kwargs: Dict[str, Any]) -> np.ndarray:
        """Safely call function with comprehensive error handling and caching"""
        cache_key = (input_idx, self._robust_hash(perturbed_input))
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Manage cache size
        if len(self.cache) > 50000:
            keys_to_remove = list(self.cache.keys())[:len(self.cache) // 10]
            for key in keys_to_remove:
                del self.cache[key]

        try:
            # Store original inputs
            original_inputs = [x.copy() for x in self.current_inputs]
            
            # Replace the perturbed input
            self.current_inputs[input_idx] = perturbed_input
            
            # Call function
            result = func(*self.current_inputs, **kwargs)
            result = np.asarray(result, dtype=self.work_dtype)
            
            # Validate result
            if result.shape != self.f0_shape:
                raise ValueError(f"Function output shape changed: {result.shape} vs {self.f0_shape}")
            
            if not np.all(np.isfinite(result)):
                warnings.warn("Function output contains non-finite values")
                result = np.where(np.isfinite(result), result, 0.0)
            
            # Cache the result
            self.cache[cache_key] = result
            self.evaluation_count += 1
            
            # Restore original inputs
            self.current_inputs = original_inputs
            
            return result
            
        except Exception as e:
            if hasattr(self, 'current_inputs'):
                self.current_inputs = [x.copy() for x in self.original_inputs]
            raise RuntimeError(f"Function evaluation failed: {e}")

    def _compute_complex_step_derivative(self, func: Callable, x: np.ndarray,
                                         input_idx: int, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives using complex-step differentiation.
        FIXED: Now computes full Jacobian matrix instead of summed derivatives.
        FIXED: Correctly handles real inputs to functions that may produce complex outputs.
        """
        x_flat = x.ravel()
        n_inputs = x_flat.size
        n_outputs = self.f0.size # f0 is the output of func(*self.current_inputs)

        grad_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        grad_conj_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)

        # Ensure self.f0 is available and has the correct shape, corresponding to func(*original_inputs)
        # self.f0 should have been computed once per call to compute_derivatives,
        # for the unperturbed inputs.

        for i in range(n_inputs):
            original_x_flat_i = x_flat[i] # Store original value to ensure clean perturbation
            if self.config.adaptive_step:
                step_size = self._find_optimal_step_size(func, x, input_idx, i, 'complex_step', kwargs)
            else:
                # Ensure original_x_flat_i is scalar for abs() if x_flat[i] could be an array (it shouldn't here)
                step_size = np.sqrt(self.machine_eps) * max(abs(np.real(original_x_flat_i)), abs(np.imag(original_x_flat_i)), 1.0)


            self.step_size_history.append((step_size, 0.0))

            # Complex step perturbation
            x_pert_flat = x_flat.copy().astype(self.work_dtype)
            x_pert_flat[i] = original_x_flat_i + 1j * step_size # Perturb from original scalar value

            # Evaluate function with perturbed x_i, other inputs (from self.current_inputs) remain same
            # _call_function_safe temporarily modifies self.current_inputs[input_idx]
            f_pert = self._call_function_safe(func, input_idx, x_pert_flat.reshape(x.shape), kwargs)
            f_pert_flat = f_pert.ravel()

            # Extract derivatives for each output element
            for j in range(n_outputs):
                # Correct complex step formula for f: R^N -> C^M
                # (f(x_i + ih) - f(x_i)) / ih
                # self.f0.ravel()[j] is f_j(x_original)
                grad_z[j, i] = (f_pert_flat[j] - self.f0.ravel()[j]) / (1j * step_size)
                
                # For real inputs, ∂f/∂z̄ should be 0 if f is an analytic continuation.
                # More robustly, if f(x) is seen as f(z) where z=x (real), then df/dx = ∂f/∂z.
                # And ∂f/∂z̄ is typically taken as 0.
                grad_conj_z[j, i] = 0.0

        output_shape = self.f0.shape + x.shape
        grad_z_reshaped = grad_z.reshape(output_shape)
        grad_conj_z_reshaped = grad_conj_z.reshape(output_shape)

        return grad_z_reshaped, grad_conj_z_reshaped

    def _compute_finite_difference_derivative(self, func: Callable, x: np.ndarray,
                                            input_idx: int, kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute derivatives using high-order finite differences.
        FIXED: Now computes full Jacobian matrix with correct Wirtinger formulas.
        """
        x_flat = x.ravel()
        n_inputs = x_flat.size
        n_outputs = self.f0.size
        
        # Initialize full Jacobian matrices
        grad_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        grad_conj_z = np.zeros((n_outputs, n_inputs), dtype=self.work_dtype)
        
        # Get finite difference coefficients
        coeffs, points = self._get_finite_difference_stencil(self.config.max_order)
        
        for i in range(n_inputs):
            original_val = x_flat[i]
            
            # Find optimal step sizes for real and imaginary directions
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
                """Create perturbed input array"""
                x_pert = x_flat.copy().astype(self.work_dtype)
                if direction == 'real':
                    x_pert[i] = (original_val.real + delta) + 1j * original_val.imag
                else:  # imaginary
                    x_pert[i] = original_val.real + 1j * (original_val.imag + delta)
                return x_pert.reshape(x.shape)

            # Compute partial derivatives with respect to real part
            if self.config.richardson_extrapolation:
                f_vals_h = [self._call_function_safe(func, input_idx, 
                           create_perturbation(point * eps_real, 'real'), kwargs) 
                           for point in points]
                dfx_h = [sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h)) / eps_real 
                         for j in range(n_outputs)]
                
                f_vals_h2 = [self._call_function_safe(func, input_idx,
                            create_perturbation(point * eps_real / 2, 'real'), kwargs)
                            for point in points]
                dfx_h2 = [sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h2)) / (eps_real / 2)
                          for j in range(n_outputs)]
                
                # Richardson extrapolation
                dfx = [(2**self.config.max_order * h2 - h) / (2**self.config.max_order - 1) 
                       for h, h2 in zip(dfx_h, dfx_h2)]
            else:
                f_vals_real = [self._call_function_safe(func, input_idx,
                              create_perturbation(point * eps_real, 'real'), kwargs)
                              for point in points]
                dfx = [sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_real)) / eps_real
                       for j in range(n_outputs)]

            # Compute partial derivatives with respect to imaginary part
            if self.config.richardson_extrapolation:
                f_vals_h = [self._call_function_safe(func, input_idx,
                           create_perturbation(point * eps_imag, 'imag'), kwargs)
                           for point in points]
                dfy_h = [sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h)) / eps_imag
                         for j in range(n_outputs)]
                
                f_vals_h2 = [self._call_function_safe(func, input_idx,
                            create_perturbation(point * eps_imag / 2, 'imag'), kwargs)
                            for point in points]
                dfy_h2 = [sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_h2)) / (eps_imag / 2)
                          for j in range(n_outputs)]
                
                dfy = [(2**self.config.max_order * h2 - h) / (2**self.config.max_order - 1) 
                       for h, h2 in zip(dfy_h, dfy_h2)]
            else:
                f_vals_imag = [self._call_function_safe(func, input_idx,
                              create_perturbation(point * eps_imag, 'imag'), kwargs)
                              for point in points]
                dfy = [sum(c * f.ravel()[j] for c, f in zip(coeffs, f_vals_imag)) / eps_imag
                       for j in range(n_outputs)]

            # Apply Wirtinger formulas correctly for each output element
            for j in range(n_outputs):
                # ∂f/∂z = 0.5 * (∂f/∂x - i * ∂f/∂y)
                # ∂f/∂z̄ = 0.5 * (∂f/∂x + i * ∂f/∂y)
                grad_z[j, i] = 0.5 * (dfx[j] - 1j * dfy[j])
                grad_conj_z[j, i] = 0.5 * (dfx[j] + 1j * dfy[j])
            
                # Check for numerical issues
                if (abs(dfx[j]) > self.config.condition_threshold or 
                    abs(dfy[j]) > self.config.condition_threshold):
                    warnings.warn(f"Large derivative detected at output {j}, input {i}: "
                                f"|∂f/∂x|={abs(dfx[j]):.2e}, |∂f/∂y|={abs(dfy[j]):.2e}")

        # Reshape to match input shape while preserving output dimension structure
        output_shape = self.f0.shape + x.shape
        grad_z_reshaped = grad_z.reshape(output_shape)
        grad_conj_z_reshaped = grad_conj_z.reshape(output_shape)
        
        return grad_z_reshaped, grad_conj_z_reshaped

    def compute_derivatives(self, func: Callable[..., np.ndarray], 
                          inputs: Tuple[np.ndarray, ...],
                          kwargs: Optional[Dict[str, Any]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Main method to compute Wirtinger derivatives with maximum accuracy.
        
        Returns a list of (∂f/∂z, ∂f/∂z̄) tuples, one for each input array.
        Each tuple contains arrays with shape (output_shape + input_shape).
        """
        if kwargs is None:
            kwargs = {}

        start_time = time.time()
        
        # Comprehensive input validation
        self._validate_inputs(func, inputs, kwargs)
        
        # Store inputs for safe function evaluation
        self.original_inputs = [x.astype(self.work_dtype) for x in inputs]
        self.current_inputs = [x.copy() for x in self.original_inputs]
        
        # Initial function evaluation
        f0 = func(*self.current_inputs, **kwargs)
        self.f0 = np.asarray(f0, dtype=self.work_dtype)
        self.f0_shape = self.f0.shape
        self.f0_norm = np.linalg.norm(self.f0)
        
        if self.config.verbose:
            print(f"Function output shape: {self.f0_shape}")
            print(f"Function output norm: {self.f0_norm:.2e}")
        
        # Reset counters and history
        self.evaluation_count = 1
        self.step_size_history.clear()
        self.cache.clear()
        
        grads = []
        
        # Compute derivatives for each input array
        for idx, x in enumerate(inputs):
            if self.config.verbose:
                print(f"\nProcessing input {idx}: shape={x.shape}, dtype={x.dtype}, "
                      f"complex={np.iscomplexobj(x)}")
            
            is_complex = np.iscomplexobj(x)
            
            if not is_complex:
                # Use complex-step differentiation for real inputs
                grad_z, grad_conj_z = self._compute_complex_step_derivative(func, x, idx, kwargs)
            else:
                # Use finite differences for complex inputs
                grad_z, grad_conj_z = self._compute_finite_difference_derivative(func, x, idx, kwargs)
            
            # Convert precision if needed
            if self.config.return_real_if_input_real and not is_complex:
                grad_z = grad_z.real.astype(x.dtype)
                grad_conj_z = grad_conj_z.real.astype(x.dtype)
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


# Convenience function that maintains the original API
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


# Example usage showing the fix
if __name__ == "__main__":
    # Test with a function that returns a vector
    def test_func(x, y):
        """Function that takes two inputs and returns a vector"""
        return np.array([x[0]**2, x[1]**2, y[0] + y[1]])
    
    # Test inputs
    x = np.array([1.0, 2.0])  # Real input
    y = np.array([1+1j, 2+2j])  # Complex input
    
    print("Testing vector-valued function with mixed real/complex inputs:")
    grads = numerical_derivative(test_func, (x, y), verbose=True)
    
    print(f"\nGradient w.r.t. x (real input):")
    print(f"  ∂f/∂x shape: {grads[0][0].shape}")  # Should be (3, 2)
    print(f"  ∂f/∂x values:\n{grads[0][0]}")
    
    print(f"\nGradient w.r.t. y (complex input):")
    print(f"  ∂f/∂z shape: {grads[1][0].shape}")  # Should be (3, 2)
    print(f"  ∂f/∂z values:\n{grads[1][0]}")
    print(f"  ∂f/∂z̄ values:\n{grads[1][1]}")