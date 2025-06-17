import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from ....functions import zeros
from ....types import Tensor, Variable


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
    Numerical Wirtinger‐derivatives approximator for Tensor/Variable.
    This version never mutates any Tensor in place, and never calls
    tensor.numpy() to build perturbations.  All masks are built from
    Python lists (wrapped into Tensor), so that Tensor remains “write-only.”
    """

    def __init__(self, config: DerivativeConfig | None = None):
        self.config = config or DerivativeConfig()
        self.cache: dict[tuple[int, int], Tensor] = {}
        self.evaluation_count = 0
        self.step_size_history: list[tuple[float, float]] = []

        # Choose working and real dtypes
        if self.config.high_precision:
            self.work_dtype = np.complex128
            self.real_dtype = np.float64
            self.machine_eps = np.finfo(np.float64).eps
        else:
            self.work_dtype = np.complex64
            self.real_dtype = np.float32
            self.machine_eps = np.finfo(np.float32).eps

        if self.config.verbose:
            print(f"Initialized with precision: {self.work_dtype}")
            print(f"Machine epsilon: {self.machine_eps:.2e}")

    def _extract_tensor(self, input_obj: Tensor | Variable) -> Tensor:
        """Extract the raw Tensor from a Variable or return it directly."""
        return input_obj.value if isinstance(input_obj, Variable) else input_obj

    def _robust_hash(self, tensor: Tensor) -> int:
        """
        Hash a Tensor by rounding its data to a tolerance, then hashing the bytes.
        We only read from tensor.data (the underlying numpy array), never mutate it.
        """
        tol = max(self.machine_eps * 1000, 1e-12)
        arr = tensor.data  # read-only numpy array

        if np.iscomplexobj(arr):
            # Round real and imag separately
            real_rounded = np.round(arr.real / tol) * tol
            imag_rounded = np.round(arr.imag / tol) * tol
            combined = real_rounded + 1j * imag_rounded
            return hash(combined.tobytes())
        else:
            rounded = np.round(arr / tol) * tol
            return hash(rounded.tobytes())

    def _validate_inputs(
        self,
        func: Callable,
        inputs: tuple[Tensor | Variable, ...],
        kwargs: dict[str, Any],
    ):
        """
        Validate that func is callable, each input is a Tensor or Variable,
        ensure no empty tensors, promote integers to floats, check finiteness.
        Store two lists:
          - self.original_inputs: all inputs as Tensors (dtype=work_dtype)
          - self.current_inputs: a copy (which we will never mutate in place)
        """
        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(inputs, tuple):
            raise TypeError("inputs must be a tuple of Tensor or Variable")

        if not inputs:
            raise ValueError("inputs tuple cannot be empty")

        sanitized: list[Tensor] = []
        self.input_is_variable: list[bool] = []

        for idx, x in enumerate(inputs):
            if not isinstance(x, (Tensor, Variable)):
                x = Tensor(x)
            is_var = isinstance(x, Variable)
            self.input_is_variable.append(is_var)
            tensor = self._extract_tensor(x)

            # Check empty
            if tensor.size == 0:
                raise ValueError(f"Input {idx} is empty tensor")

            # Promote integer‐typed Tensors to float
            if tensor.dtype.kind in ("i", "u"):
                tensor = tensor.astype("float64")

            # Check finiteness via tensor.data
            arr = tensor.data
            if not np.isfinite(arr).all():
                count_bad = int((~np.isfinite(arr)).sum())
                raise ValueError(f"Input {idx} contains non-finite values: {count_bad}")

            # Finally cast to work_dtype
            tensor = tensor.astype(self.work_dtype)
            sanitized.append(tensor)

        # Test a single call: func(*sanitized)
        try:
            out = func(*sanitized, **kwargs)
            if isinstance(out, Variable):
                out = out.value
            if not isinstance(out, Tensor):
                raise ValueError("Function must return a Tensor or Variable")
            if out.size == 0:
                raise ValueError("Function returns empty tensor")
            if not np.isfinite(out.data).all():
                raise ValueError("Function returns non-finite values")
        except Exception as e:
            raise RuntimeError(f"Function evaluation test failed: {e}") from e

        # Store them (we never mutate these lists in place)
        self.original_inputs = sanitized[:]
        self.current_inputs = sanitized[:]

    def _get_finite_difference_stencil(self, order: int) -> tuple[Tensor, Tensor]:
        """Return (coefficients, points) as Tensors, based on the requested order."""
        stencils = {
            2: {"coefficients": [-1.0, 1.0], "points": [-1, 1]},
            4: {
                "coefficients": [1.0, -8.0, 8.0, -1.0],
                "divisor": 12.0,
                "points": [-2, -1, 1, 2],
            },
            6: {
                "coefficients": [-1.0, 9.0, -45.0, 45.0, -9.0, 1.0],
                "divisor": 60.0,
                "points": [-3, -2, -1, 1, 2, 3],
            },
            8: {
                "coefficients": [3.0, -32.0, 168.0, -672.0, 672.0, -168.0, 32.0, -3.0],
                "divisor": 840.0,
                "points": [-4, -3, -2, -1, 1, 2, 3, 4],
            },
        }
        if order not in stencils:
            raise ValueError(f"Unsupported finite difference order: {order}")

        data = stencils[order]
        coeffs_list = data["coefficients"]
        if "divisor" in data:
            coeffs_array = (
                np.array(coeffs_list, dtype=self.work_dtype) / data["divisor"]
            )
        else:
            coeffs_array = np.array(coeffs_list, dtype=self.work_dtype)
        points_array = np.array(data["points"], dtype=self.real_dtype)

        # Wrap into Tensors
        coeffs = Tensor(coeffs_array, dtype=coeffs_array.dtype)
        points = Tensor(points_array, dtype=points_array.dtype)
        return coeffs, points

    def _find_optimal_step_size(
        self,
        func: Callable,
        x: Tensor,
        idx: int,
        element_idx: int,
        direction: str,
        kwargs: dict[str, Any],
    ) -> float:
        """
        Search for the step size (scalar float) that minimizes the local error estimate.
        All Tensor perturbations are created anew (never in place).
        """

        # Flatten x to grab the single element
        x_flat = x.flatten()
        orig_val = x_flat[element_idx].item()  # Python scalar

        # Base‐step estimate
        if direction == "complex_step":
            base = np.sqrt(self.machine_eps) * max(abs(orig_val), 1.0)
        else:
            if isinstance(orig_val, complex):
                scale = max(abs(orig_val.real), abs(orig_val.imag), 1.0)
            else:
                scale = max(abs(orig_val), 1.0)
            base = (self.machine_eps ** (1.0 / (self.config.max_order + 2))) * scale

        if not self.config.use_optimal_step_search:
            return float(
                np.clip(base, self.config.min_step_size, self.config.max_step_size)
            )

        def error_estimate(log_h: float) -> float:
            h = float(np.exp(log_h))
            if h < self.config.min_step_size or h > self.config.max_step_size:
                return 1e10

            try:
                if direction == "complex_step":
                    # Build a mask of zeros, except +i*h at index element_idx
                    n = x_flat.size
                    mask_imag_list = [0.0] * n
                    mask_imag_list[element_idx] = h
                    mask_imag = Tensor(np.array(mask_imag_list, dtype=self.real_dtype))

                    # base complex‐typed Tensor:
                    base_real = x_flat.real()
                    base_imag = x_flat.imag()
                    base_complex = Tensor(base_real, dtype=self.work_dtype) + (
                        Tensor(base_imag, dtype=self.work_dtype)
                        * Tensor(1j, dtype=self.work_dtype)
                    )

                    perturbed = base_complex + (
                        mask_imag * Tensor(1j, dtype=self.work_dtype)
                    )
                    x_pert = perturbed.reshape(x.shape)
                    f1 = self._call_function_safe(func, idx, x_pert, kwargs)
                    deriv1 = float(f1.flatten()[0].imag().item()) / h

                    # half‐step
                    mask_imag_half_list = [val / 2.0 for val in mask_imag_list]
                    mask_imag_half = Tensor(
                        np.array(mask_imag_half_list, dtype=self.real_dtype)
                    )
                    perturbed2 = base_complex + (
                        mask_imag_half * Tensor(1j, dtype=self.work_dtype)
                    )
                    x_pert2 = perturbed2.reshape(x.shape)
                    f2 = self._call_function_safe(func, idx, x_pert2, kwargs)
                    deriv2 = float(f2.flatten()[0].imag().item()) / (h / 2.0)

                    err = abs(deriv1 - deriv2)
                else:
                    # high-order finite differences: “real” or “imag” direction
                    coeffs, points = self._get_finite_difference_stencil(
                        self.config.max_order
                    )
                    n_out = self.f0.size

                    # Evaluate at full‐step
                    fvals_full = []
                    for p in points.data.tolist():
                        delta = float(p) * h
                        if direction == "real":
                            # build mask that adds delta to real part at element_idx
                            mask_list = [0.0] * x_flat.size
                            mask_list[element_idx] = delta
                            mask = Tensor(np.array(mask_list, dtype=self.real_dtype))
                            base = (
                                x_flat
                                if x_flat.dtype.kind == "c"
                                else x_flat.astype(self.work_dtype)
                            )
                            pert = base + mask
                        else:  # “imag” direction
                            mask_list = [0.0] * x_flat.size
                            mask_list[element_idx] = delta
                            mask_im = Tensor(np.array(mask_list, dtype=self.real_dtype))
                            base = (
                                x_flat
                                if x_flat.dtype.kind == "c"
                                else x_flat.astype(self.work_dtype)
                            )
                            pert = base + (mask_im * Tensor(1j, dtype=self.work_dtype))

                        x_p = pert.reshape(x.shape)
                        fvals_full.append(
                            self._call_function_safe(func, idx, x_p, kwargs)
                        )

                    # Evaluate at half‐step
                    fvals_half = []
                    for p in points.data.tolist():
                        delta = float(p) * (h / 2.0)
                        if direction == "real":
                            mask_list2 = [0.0] * x_flat.size
                            mask_list2[element_idx] = delta
                            mask2 = Tensor(np.array(mask_list2, dtype=self.real_dtype))
                            base2 = (
                                x_flat
                                if x_flat.dtype.kind == "c"
                                else x_flat.astype(self.work_dtype)
                            )
                            pert2 = base2 + mask2
                        else:
                            mask_list2 = [0.0] * x_flat.size
                            mask_list2[element_idx] = delta
                            mask_im2 = Tensor(
                                np.array(mask_list2, dtype=self.real_dtype)
                            )
                            base2 = (
                                x_flat
                                if x_flat.dtype.kind == "c"
                                else x_flat.astype(self.work_dtype)
                            )
                            pert2 = base2 + (
                                mask_im2 * Tensor(1j, dtype=self.work_dtype)
                            )

                        x_p2 = pert2.reshape(x.shape)
                        fvals_half.append(
                            self._call_function_safe(func, idx, x_p2, kwargs)
                        )

                    # d_full and d_half as lists of n_out floats
                    d_full = []
                    d_half = []
                    coeffs_list = coeffs.data.tolist()
                    for j in range(n_out):
                        num_full = sum(
                            coeffs_list[k] * float(fvals_full[k].flatten()[j].item())
                            for k in range(len(coeffs_list))
                        )
                        d_full.append(num_full / h)

                        num_half = sum(
                            coeffs_list[k] * float(fvals_half[k].flatten()[j].item())
                            for k in range(len(coeffs_list))
                        )
                        d_half.append(num_half / (h / 2.0))

                    # Richardson error estimate
                    err = max(abs(dh - df) for dh, df in zip(d_half, d_full)) / (
                        2**self.config.max_order - 1
                    )

                return float(err + self.machine_eps)
            except Exception:
                return 1e10

        try:
            res = minimize_scalar(
                error_estimate,
                bounds=(
                    np.log(self.config.min_step_size),
                    np.log(self.config.max_step_size),
                ),
                method="bounded",
                options={"xatol": self.config.step_optimization_tolerance},
            )
            optimal = float(np.exp(res.x))
        except Exception:
            optimal = base

        if self.config.verbose:
            print(
                f"Optimal step size for element {element_idx}, direction {direction}: {optimal:.2e}"
            )
        return float(
            np.clip(optimal, self.config.min_step_size, self.config.max_step_size)
        )

    def _call_function_safe(
        self,
        func: Callable,
        input_idx: int,
        perturbed_input: Tensor,
        kwargs: dict[str, Any],
    ) -> Tensor:
        """
        Call func(*…) safely, without ever mutating self.current_inputs in place.
        Caches based on (input_idx, hash(perturbed_input)).
        """
        key = (input_idx, self._robust_hash(perturbed_input))
        if key in self.cache:
            return self.cache[key]

        # Evict if too large
        if len(self.cache) > 50000:
            to_remove = list(self.cache.keys())[: len(self.cache) // 10]
            for k in to_remove:
                del self.cache[k]

        try:
            # Build a fresh list of inputs, replacing only index=input_idx
            inputs_for_call = [
                (perturbed_input if i == input_idx else orig)
                for i, orig in enumerate(self.current_inputs)
            ]
            result = func(*inputs_for_call, **kwargs)
            if isinstance(result, Variable):
                result = result.value
            if not isinstance(result, Tensor):
                raise ValueError("Function must return a Tensor or Variable")

            result = result.astype(self.work_dtype)

            if result.shape != self.f0_shape:
                raise ValueError(
                    f"Function output shape changed: {result.shape} vs {self.f0_shape}"
                )

            # Replace any non-finite entries with zero (using a temporary numpy array)
            data_arr = result.data
            if not np.isfinite(data_arr).all():
                warnings.warn(
                    "Function output contains non-finite values", stacklevel=2
                )
                copy_arr = data_arr.copy()
                copy_arr[~np.isfinite(copy_arr)] = 0.0
                result = Tensor(copy_arr, dtype=copy_arr.dtype)

            self.cache[key] = result
            self.evaluation_count += 1
            return result

        except Exception as e:
            raise RuntimeError(f"Function evaluation failed: {e}") from e

    def _compute_complex_step_derivative(
        self, func: Callable, x: Tensor, input_idx: int, kwargs: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        """
        Compute ∂f/∂z and ∂f/∂z̄ by complex‐step on a real‐typed input x.
        We build each x_pert by adding a freshly constructed “mask” Tensor
        of shape x_flat, never mutating x_flat itself.
        """
        x_flat = x.flatten()
        n_in = x_flat.size
        n_out = self.f0.size

        grad_z = zeros((n_out, n_in), dtype=self.work_dtype)
        grad_conj_z = zeros((n_out, n_in), dtype=self.work_dtype)

        for i in range(n_in):
            orig = x_flat[i].item()
            # Pick step size
            if self.config.adaptive_step:
                h = self._find_optimal_step_size(
                    func, x, input_idx, i, "complex_step", kwargs
                )
            else:
                scale = (
                    max(abs(orig.real), abs(orig.imag))
                    if isinstance(orig, complex)
                    else abs(orig)
                )
                scale = max(scale, 1.0)
                h = np.sqrt(self.machine_eps) * scale

            self.step_size_history.append((h, 0.0))

            # Build mask_imag_list: zeros except index i is h
            mask_list = [0.0] * n_in
            mask_list[i] = h
            mask_imag = Tensor(np.array(mask_list, dtype=self.real_dtype))

            # Base complex‐typed Tensor from x_flat
            base_real = x_flat.real()
            base_imag = x_flat.imag()
            base_complex = Tensor(base_real, dtype=self.work_dtype) + (
                Tensor(base_imag, dtype=self.work_dtype)
                * Tensor(1j, dtype=self.work_dtype)
            )

            pert_flat = base_complex + (mask_imag * Tensor(1j, dtype=self.work_dtype))
            x_pert = pert_flat.reshape(x.shape)

            f_pert = self._call_function_safe(func, input_idx, x_pert, kwargs)
            f_pert_flat = f_pert.flatten()
            f0_flat = self.f0.flatten()

            for j in range(n_out):
                diff = f_pert_flat[j] - f0_flat[j]
                grad_z = grad_z  # (we'll overwrite via Tensor indexing)
                # But since grad_z is a Tensor, we cannot do grad_z[j,i] = ...
                # Instead, extract as numpy, modify, and wrap again at the very end.
                # However, the provided Tensor class forbids in-place assignment.
                # So we’ll create a small 2D numpy array to hold one column, then reinsert it:
                col_arr = grad_z.data[:, :]  # read the entire (n_out, n_in) as numpy
                col_arr = col_arr.copy()  # mutable copy
                col_arr[j, i] = complex(diff.item() / (1j * h))
                grad_z = Tensor(col_arr.reshape((n_out, n_in)), dtype=col_arr.dtype)

                # ∂f/∂z̄ = 0 for analytic f
                col_arr2 = grad_conj_z.data.copy()
                col_arr2[j, i] = 0.0
                grad_conj_z = Tensor(
                    col_arr2.reshape((n_out, n_in)), dtype=col_arr2.dtype
                )

        out_shape = self.f0.shape + x.shape
        grad_z = grad_z.reshape(out_shape)
        grad_conj_z = grad_conj_z.reshape(out_shape)
        return grad_z, grad_conj_z

    def _compute_finite_difference_derivative(
        self, func: Callable, x: Tensor, input_idx: int, kwargs: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        """
        Compute ∂f/∂z and ∂f/∂z̄ by high-order finite differences on a complex-typed input x.
        No in-place ops: whenever we need to perturb x_flat at index i, we build a fresh mask list.
        """
        x_flat = x.flatten()
        n_in = x_flat.size
        n_out = self.f0.size

        grad_z = zeros((n_out, n_in), dtype=self.work_dtype)
        grad_conj_z = zeros((n_out, n_in), dtype=self.work_dtype)

        coeffs, points = self._get_finite_difference_stencil(self.config.max_order)
        coeffs_list = coeffs.data.tolist()
        points_list = points.data.tolist()

        for i in range(n_in):
            orig = x_flat[i].item()

            # Step sizes in real/imag
            if self.config.adaptive_step:
                eps_r = self._find_optimal_step_size(
                    func, x, input_idx, i, "real", kwargs
                )
                eps_i = self._find_optimal_step_size(
                    func, x, input_idx, i, "imag", kwargs
                )
            else:
                if isinstance(orig, complex):
                    scale_r = max(abs(orig.real), 1.0)
                    scale_i = max(abs(orig.imag), 1.0)
                else:
                    scale_r = max(abs(orig), 1.0)
                    scale_i = 1.0
                factor = 1.0 / (self.config.max_order + 1)
                eps_r = (self.machine_eps**factor) * scale_r
                eps_i = (self.machine_eps**factor) * scale_i

            self.step_size_history.append((eps_r, eps_i))

            def make_perturbed(delta: float, dir_flag: str, i: int) -> Tensor:
                """
                Build a new Tensor of shape x.shape, where only x_flat[i] is perturbed:
                - If dir_flag=="real", add +delta to real part
                - If dir_flag=="imag", add +delta to imag part
                """
                n = x_flat.size
                if dir_flag == "real":
                    mask_list = [0.0] * n
                    mask_list[i] = delta
                    mask = Tensor(np.array(mask_list, dtype=self.real_dtype))
                    base = (
                        x_flat
                        if x_flat.dtype.kind == "c"
                        else x_flat.astype(self.work_dtype)
                    )
                    pert_flat = base + mask
                else:  # "imag"
                    mask_list = [0.0] * n
                    mask_list[i] = delta
                    mask_im = Tensor(np.array(mask_list, dtype=self.real_dtype))
                    base = (
                        x_flat
                        if x_flat.dtype.kind == "c"
                        else x_flat.astype(self.work_dtype)
                    )
                    pert_flat = base + (mask_im * Tensor(1j, dtype=self.work_dtype))

                return pert_flat.reshape(x.shape)

            # === Real-part derivative (possibly Richardson) ===
            if self.config.richardson_extrapolation:
                f_full = [
                    self._call_function_safe(
                        func, input_idx, make_perturbed(pt * eps_r, "real", i), kwargs
                    )
                    for pt in points_list
                ]
                f_half = [
                    self._call_function_safe(
                        func,
                        input_idx,
                        make_perturbed(pt * (eps_r / 2.0), "real", i),
                        kwargs,
                    )
                    for pt in points_list
                ]

                d_full = []
                d_half = []
                for j in range(n_out):
                    num_f = sum(
                        coeffs_list[k] * f_full[k].flatten()[j].item()
                        for k in range(len(coeffs_list))
                    )
                    d_full.append(num_f / eps_r)

                    num_h = sum(
                        coeffs_list[k] * f_half[k].flatten()[j].item()
                        for k in range(len(coeffs_list))
                    )
                    d_half.append(num_h / (eps_r / 2.0))

                d_real = [
                    ((2**self.config.max_order) * dh - df)
                    / (2**self.config.max_order - 1)
                    for df, dh in zip(d_full, d_half)
                ]
            else:
                f_vals = [
                    self._call_function_safe(
                        func, input_idx, make_perturbed(pt * eps_r, "real", i), kwargs
                    )
                    for pt in points_list
                ]
                d_real = []
                for j in range(n_out):
                    num = sum(
                        coeffs_list[k] * f_vals[k].flatten()[j].item()
                        for k in range(len(coeffs_list))
                    )
                    d_real.append(num / eps_r)

            # === Imag-part derivative (possibly Richardson) ===
            if self.config.richardson_extrapolation:
                f_full_i = [
                    self._call_function_safe(
                        func, input_idx, make_perturbed(pt * eps_i, "imag", i), kwargs
                    )
                    for pt in points_list
                ]
                f_half_i = [
                    self._call_function_safe(
                        func,
                        input_idx,
                        make_perturbed(pt * (eps_i / 2.0), "imag", i),
                        kwargs,
                    )
                    for pt in points_list
                ]

                d_full_i = []
                d_half_i = []
                for j in range(n_out):
                    num_fi = sum(
                        coeffs_list[k] * f_full_i[k].flatten()[j].item()
                        for k in range(len(coeffs_list))
                    )
                    d_full_i.append(num_fi / eps_i)

                    num_hi = sum(
                        coeffs_list[k] * f_half_i[k].flatten()[j].item()
                        for k in range(len(coeffs_list))
                    )
                    d_half_i.append(num_hi / (eps_i / 2.0))

                d_imag = [
                    ((2**self.config.max_order) * dhi - dfi)
                    / (2**self.config.max_order - 1)
                    for dfi, dhi in zip(d_full_i, d_half_i)
                ]
            else:
                f_vals_i = [
                    self._call_function_safe(
                        func, input_idx, make_perturbed(pt * eps_i, "imag", i), kwargs
                    )
                    for pt in points_list
                ]
                d_imag = []
                for j in range(n_out):
                    num_i = sum(
                        coeffs_list[k] * f_vals_i[k].flatten()[j].item()
                        for k in range(len(coeffs_list))
                    )
                    d_imag.append(num_i / eps_i)

            # ==== Assemble Wirtinger partials into grad_z and grad_conj_z ====
            # Since grad_z is a read-only Tensor, extract its data as numpy, modify the column, wrap back
            gz_arr = grad_z.data.copy()
            gc_arr = grad_conj_z.data.copy()
            for j in range(n_out):
                re_val = d_real[j]
                im_val = d_imag[j]
                gz_arr[j, i] = 0.5 * (re_val - 1j * im_val)
                gc_arr[j, i] = 0.5 * (re_val + 1j * im_val)

                if (
                    abs(re_val) > self.config.condition_threshold
                    or abs(im_val) > self.config.condition_threshold
                ):
                    warnings.warn(
                        f"Large derivative at output {j}, input {i}: "
                        f"|∂f/∂x|={abs(re_val):.2e}, |∂f/∂y|={abs(im_val):.2e}",
                        stacklevel=2,
                    )

            grad_z = Tensor(gz_arr, dtype=gz_arr.dtype)
            grad_conj_z = Tensor(gc_arr, dtype=gc_arr.dtype)

        out_shape = self.f0.shape + x.shape
        return grad_z.reshape(out_shape), grad_conj_z.reshape(out_shape)

    def compute_derivatives(
        self,
        func: Callable[..., Tensor | Variable],
        inputs: tuple[Tensor | Variable, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Compute Wirtinger derivatives for each input; returns a list of
        (∂f/∂z, ∂f/∂z̄) pairs.  No in-place ops on any Tensor anywhere.
        """
        if kwargs is None:
            kwargs = {}

        start = time.time()

        # 1) Validate inputs (stores self.current_inputs, self.original_inputs)
        self._validate_inputs(func, inputs, kwargs)

        # 2) Evaluate f0 once
        f0 = func(*self.current_inputs, **kwargs)
        if isinstance(f0, Variable):
            f0 = f0.value
        self.f0 = f0.astype(self.work_dtype)
        self.f0_shape = self.f0.shape
        self.f0_norm = float(self.f0.norm().item())

        if self.config.verbose:
            print(f"Function output shape: {self.f0_shape}")
            print(f"Function output norm: {self.f0_norm:.2e}")

        self.evaluation_count = 1
        self.step_size_history.clear()
        self.cache.clear()

        grads: list[tuple[Tensor, Tensor]] = []

        for idx, x in enumerate(self.current_inputs):
            is_complex = x.dtype.kind == "c"
            if self.config.verbose:
                print(
                    f"\nProcessing input {idx}: shape={x.shape}, dtype={x.dtype}, complex={is_complex}"
                )

            if not is_complex:
                gz, gc = self._compute_complex_step_derivative(func, x, idx, kwargs)
            else:
                gz, gc = self._compute_finite_difference_derivative(
                    func, x, idx, kwargs
                )

            # Convert output dtypes if requested
            if self.config.return_real_if_input_real and not is_complex:
                gz = gz.real().astype(self.real_dtype)
                gc = gc.real().astype(self.real_dtype)
            elif not self.config.high_precision:
                target_dtype = np.complex128 if is_complex else x.dtype
                gz = gz.astype(target_dtype)
                gc = gc.astype(target_dtype)

            grads.append((gz, gc))

            if self.config.verbose:
                print(f"Input {idx} done:")
                print(f"  Jacobian shape: {gz.shape}")
                max_z = float(gz.abs().max().item())
                max_c = float(gc.abs().max().item())
                print(f"  max |∂f/∂z| = {max_z:.2e}")
                print(f"  max |∂f/∂z̄| = {max_c:.2e}")

        elapsed = time.time() - start
        if self.config.verbose:
            print(f"\nComputation completed in {elapsed:.3f} seconds")
            print(f"Total function evaluations: {self.evaluation_count}")
            print(f"Cache entries: {len(self.cache)}")

        return grads


def numerical_derivative(
    func: Callable[..., Tensor | Variable],
    inputs: tuple[Tensor | Variable, ...],
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
) -> list[tuple[Tensor, Tensor]]:
    """
    Convenience wrapper around WirtingerDifferentiator that never mutates
    any Tensor in place.
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
    diff = WirtingerDifferentiator(config)
    return diff.compute_derivatives(func, inputs, kwargs)


# Example usage
if __name__ == "__main__":

    def test_func(x, y):
        # x is real‐typed → promoted. y is complex‐typed → stays complex
        return Tensor([x[0] ** 2, x[1] ** 2, y[0] + y[1]])

    x_int = Tensor([1, 2], dtype="int64")
    y_cplx = Tensor([1 + 1j, 2 + 2j], dtype="complex128")

    print("Testing with integer (auto‒promoted) and complex inputs:")
    grads = numerical_derivative(test_func, (x_int, y_cplx), verbose=True)

    print("\nGradient w.r.t. x:")
    print(f"  ∂f/∂x shape: {grads[0][0].shape}")  # (3,2)
    print(f"  ∂f/∂x values:\n{grads[0][0].numpy()}")

    print("\nGradient w.r.t. y:")
    print(f"  ∂f/∂z shape: {grads[1][0].shape}")  # (3,2)
    print(f"  ∂f/∂z values:\n{grads[1][0].numpy()}")
    print(f"  ∂f/∂z̄ values:\n{grads[1][1].numpy()}")
