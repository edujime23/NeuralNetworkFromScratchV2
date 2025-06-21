# autodiff/tape/autodiff_api.py

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Literal, Union

import numpy as np

from ...types import Tensor, Variable

# Assuming _backpropagate and _get_final_gradient_for_source are in backprop_logic.py
from .backprop_logic import _backpropagate, _get_final_gradient_for_source
from .tape_core import GradientTapeCore  # For type hinting and methods access
from .types import Gradient


# The main GradientTape class which combines core and API
class GradientTape(GradientTapeCore):
    """
    Records operations on Tensors/Variables to enable automatic differentiation.
    This class combines the core tape functionalities with the public API methods.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _grad_scalar_or_tensor(
        self,
        target_arr: Tensor,
        source_arr: Tensor,
        seed: Tensor,
        unconnected_gradients: Literal["zero", "none"],
    ) -> tuple[Tensor, Tensor] | None:
        """
        Internal helper for Jacobian-vector products (JVP) and Jacobians.
        This uses the tape's internal state and backpropagation logic.
        """
        self.grads.clear()

        dtype = self._get_gradient_dtype(target_arr.dtype, self.forced_dtype)
        zero_ah = Tensor(np.zeros_like(seed.data, dtype=dtype))
        self.grads[id(target_arr)] = self.grads[id(target_arr)] = Gradient(
            holomorphic=seed, antiholomorphic=zero_ah
        )

        self._initialize_input_gradient(source_arr)

        _backpropagate(self, target_arr)  # Call the external backprop function

        return _get_final_gradient_for_source(self, source_arr, unconnected_gradients)

    def gradient(
        self,
        target: Tensor | Variable,
        sources: Tensor | Variable | Iterable[Tensor | Variable],
        output_gradients: Tensor | Variable | None = None,
        unconnected_gradients: Literal["zero", "none"] = "zero",
    ) -> Union[
        tuple[Tensor, Tensor],
        list[Union[tuple[Tensor, Tensor], None]],
        None,
    ]:
        """
        Computes the gradients of a `target` Tensor with respect to one or more `sources`.
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "GradientTape has already been used and is not persistent. "
                "Set `persistent=True` during initialization for multiple gradient() calls."
            )
        self._used = True

        target_arr = target.value if isinstance(target, Variable) else target
        if not isinstance(target_arr, Tensor):
            target_arr = Tensor(target_arr)

        is_single_source_input = not isinstance(sources, Iterable)
        sources_list = [sources] if is_single_source_input else list(sources)
        raw_sources: list[Tensor] = []
        for s in sources_list:
            if isinstance(s, Variable):
                raw_sources.append(s.value)
            elif isinstance(s, Tensor):
                raw_sources.append(s)
            else:
                raw_sources.append(Tensor(s))

        filtered_sources = [s for s in raw_sources if id(s) in self.watched]

        if not filtered_sources:
            if unconnected_gradients == "zero":
                result_list_unconnected: list[tuple[Tensor, Tensor]] = []
                for s in raw_sources:
                    dtype_s = self._get_gradient_dtype(s.dtype, self.forced_dtype)
                    zero1 = Tensor(np.zeros(s.shape, dtype=dtype_s))
                    zero2 = Tensor(np.zeros(s.shape, dtype=dtype_s))
                    result_list_unconnected.append((zero1, zero2))
                return (
                    result_list_unconnected[0]
                    if is_single_source_input
                    else result_list_unconnected
                )
            return None if is_single_source_input else [None] * len(raw_sources)

        gh_init: Tensor
        gah_init: Tensor
        if output_gradients is not None:
            og_arr = (
                output_gradients.value
                if isinstance(output_gradients, Variable)
                else output_gradients
            )
            if not isinstance(og_arr, Tensor):
                og_arr = Tensor(og_arr)

            if og_arr.shape != target_arr.shape:
                raise ValueError(
                    f"Output gradient shape {og_arr.shape} must match target shape {target_arr.shape}."
                )
            gh_init = og_arr
            gah_init = Tensor(np.zeros_like(og_arr.data))
        else:
            dtype_seed = self._get_gradient_dtype(target_arr.dtype, self.forced_dtype)
            if (
                self._ones_seed is None
                or self._ones_seed.shape != target_arr.shape
                or self._ones_seed.dtype != dtype_seed
            ):
                self._ones_seed = Tensor(np.ones(target_arr.shape, dtype=dtype_seed))
            if (
                self._zeros_seed is None
                or self._zeros_seed.shape != target_arr.shape
                or self._zeros_seed.dtype != dtype_seed
            ):
                self._zeros_seed = Tensor(np.zeros(target_arr.shape, dtype=dtype_seed))
            gh_init = self._ones_seed
            gah_init = self._zeros_seed

        self.grads.clear()
        for src in filtered_sources:
            dtype_src = self._get_gradient_dtype(src.dtype, self.forced_dtype)
            zero_h = Tensor(np.zeros(src.shape, dtype=dtype_src))
            zero_ah = Tensor(np.zeros(src.shape, dtype=dtype_src))
            self.grads[id(src)] = Gradient(holomorphic=zero_h, antiholomorphic=zero_ah)

        self.grads[id(target_arr)] = Gradient(
            holomorphic=gh_init, antiholomorphic=gah_init
        )

        _backpropagate(self, target_arr)

        final_list: list[tuple[Tensor, Tensor] | None] = []
        for s in raw_sources:
            grad_pair = _get_final_gradient_for_source(self, s, unconnected_gradients)
            if grad_pair is not None:
                final_list.append(
                    (np.real_if_close(grad_pair[0]), np.real_if_close(grad_pair[1]))
                )
            else:
                final_list.append(None)

        return final_list[0] if is_single_source_input else final_list

    @staticmethod
    def _jac_row_worker(args: tuple[Any, ...]) -> np.ndarray:
        r"""
        Worker function for parallel Jacobian row computation.
        """
        (
            target_np,
            source_np,
            idx,
            shape_target,
            shape_source,
            og_np,
            unconnected_gradients,
            forced_dtype,
        ) = args

        flat_t_seed_np = np.zeros(np.prod(shape_target), dtype=forced_dtype)
        flat_t_seed_np[idx] = 1.0
        seed_np = flat_t_seed_np.reshape(shape_target)
        seed = Tensor(seed_np)

        if og_np is not None:
            seed = seed * Tensor(og_np)

        target_tensor = Tensor(target_np)
        source_tensor = Tensor(source_np)

        # Create a temporary GradientTape for each worker.
        with GradientTape(
            persistent=False, watch_accessed_variables=False, dtype=forced_dtype
        ) as tape:
            tape.watch(source_tensor)
            grad_pair = tape._grad_scalar_or_tensor(
                target_tensor, source_tensor, seed, unconnected_gradients
            )

        if grad_pair is None:
            return np.zeros(np.prod(shape_source), dtype=forced_dtype)

        combined = grad_pair[0] + grad_pair[1].conj()
        combined_np = combined.data
        if np.isrealobj(combined_np):
            return combined_np.reshape(-1)
        if np.allclose(combined_np.imag, 0):
            return combined_np.real.reshape(-1)
        return combined_np.reshape(-1)

    def jacobian(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        unconnected_gradients: Literal["zero", "none"] = "none",
        output_gradients: Tensor | Variable | None = None,
    ) -> Tensor:
        r"""
        Computes the full Jacobian matrix $\frac{\partial \text{target}}{\partial \text{source}}$ using parallel processes.
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "GradientTape has already been used and is not persistent. "
                "Set `persistent=True` during initialization for reuse."
            )
        self._used = True

        target_np = self._extract_raw_numpy_data(target)
        source_np = self._extract_raw_numpy_data(source)

        flat_target_size = target_np.size
        shape_target = target_np.shape
        shape_source = source_np.shape
        forced_dtype = self._get_gradient_dtype(source_np.dtype, self.forced_dtype)

        og_np: np.ndarray | None = None
        if output_gradients is not None:
            og_arr = (
                output_gradients.value
                if isinstance(output_gradients, Variable)
                else output_gradients
            )
            if not isinstance(og_arr, Tensor):
                og_arr = Tensor(og_arr)
            if og_arr.shape != shape_target:
                raise ValueError(
                    f"Output gradients shape {og_arr.shape} must match target shape {shape_target}."
                )
            og_np = og_arr.data

        args_list: list[tuple[Any, ...]] = [
            (
                target_np,
                source_np,
                idx,
                shape_target,
                shape_source,
                og_np,
                unconnected_gradients,
                forced_dtype,
            )
            for idx in range(flat_target_size)
        ]

        with mp.Pool() as pool:
            rows = pool.map(GradientTape._jac_row_worker, args_list)

        jac_np = np.stack(rows, axis=0)
        full_shape = shape_target + shape_source
        jac_np = jac_np.reshape(full_shape)
        return Tensor(jac_np)

    def jvp(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        vector: Tensor | Variable,
    ) -> Tensor:
        r"""
        Computes the Jacobian-vector product (JVP) or forward-mode automatic differentiation.
        """
        source_arr = source.value if isinstance(source, Variable) else source
        if not isinstance(source_arr, Tensor):
            source_arr = Tensor(source_arr)

        vec_arr = vector.value if isinstance(vector, Variable) else vector
        if not isinstance(vec_arr, Tensor):
            vec_arr = Tensor(vec_arr)

        if vec_arr.shape != source_arr.shape:
            raise ValueError(
                f"Vector shape {vec_arr.shape} must match source shape {source_arr.shape} for JVP."
            )

        f = self._prepare_target_function(target)

        with GradientTape(
            persistent=False,
            watch_accessed_variables=self.watch_on_read,
            dtype=self.forced_dtype,
        ) as inner_tape:
            inner_tape.watch(source_arr)
            y = f(source_arr)

        gy_result = inner_tape.gradient(y, source_arr)
        combined_grad_y = self._get_combined_gradient_from_tape_output(
            gy_result, source_arr
        )

        if isinstance(y, Tensor) and (y.ndim == 0 or y.size == 1):
            return (combined_grad_y * vec_arr).sum()

        warnings.warn(
            "JVP for vector-valued targets is not implemented as a true Jacobian-vector product "
            "(J*V). This call returns (∂f/∂x)^T @ vector instead (a VJP-like result). "
            "To compute an explicit JVP (J*V), consider using `jacobian()` and then "
            "performing the matrix multiplication manually.",
            stacklevel=2,
        )
        vjp_result = inner_tape.gradient(
            y, source_arr, output_gradients=vec_arr, unconnected_gradients="zero"
        )
        return self._get_combined_gradient_from_tape_output(vjp_result, source_arr)

    def vjp(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        vector: Tensor | Variable,
    ) -> Tensor:
        r"""
        Computes the Vector-Jacobian product (VJP) or reverse-mode automatic differentiation.
        """
        source_arr = source.value if isinstance(source, Variable) else source
        if not isinstance(source_arr, Tensor):
            source_arr = Tensor(source_arr)

        vec_arr = vector.value if isinstance(vector, Variable) else vector
        if not isinstance(vec_arr, Tensor):
            vec_arr = Tensor(vec_arr)

        f = self._prepare_target_function(target)

        with GradientTape(
            persistent=False,
            watch_accessed_variables=self.watch_on_read,
            dtype=self.forced_dtype,
        ) as tape:
            tape.watch(source_arr)
            y = f(source_arr)

            vjp_pair = tape.gradient(
                y, source_arr, output_gradients=vec_arr, unconnected_gradients="zero"
            )
        return self._get_combined_gradient_from_tape_output(vjp_pair, source_arr)

    def derivative(
        self, f: Callable[[Tensor], Tensor], x: Tensor | Variable, order: int
    ) -> Tensor:
        """
        Computes higher-order derivatives recursively.
        """
        x_arr = x.value if isinstance(x, Variable) else x
        if not isinstance(x_arr, Tensor):
            x_arr = Tensor(x_arr)

        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        if order == 0:
            return f(x_arr)

        inner_derivative_result = self.derivative(f, x_arr, order - 1)

        with GradientTape(
            persistent=False,
            watch_accessed_variables=self.watch_on_read,
            dtype=self.forced_dtype,
        ) as tape:
            tape.watch(x_arr)
            grad_pair = tape.gradient(inner_derivative_result, x_arr)

        return self._get_combined_gradient_from_tape_output(grad_pair, x_arr)

    def hessian(self, f: Callable[[Tensor], Tensor], x: Tensor | Variable) -> Tensor:
        r"""
        Computes the Hessian matrix $\frac{\partial^2 f}{\partial x_i \partial x_j}$ using parallel processes.
        """
        x_arr = x.value if isinstance(x, Variable) else x
        if not isinstance(x_arr, Tensor):
            x_arr = Tensor(x_arr)

        n = x_arr.size
        hess_dtype = self._get_gradient_dtype(x_arr.dtype, self.forced_dtype)

        with GradientTape(
            persistent=False,
            watch_accessed_variables=self.watch_on_read,
            dtype=self.forced_dtype,
        ) as g1:
            g1.watch(x_arr)
            y = f(x_arr)
            grad_f_pair = g1.gradient(y, x_arr)

        if grad_f_pair is None or grad_f_pair[0] is None or grad_f_pair[1] is None:
            raise ValueError(
                "Could not compute the first-order gradient for Hessian. "
                "Ensure that the function `f` is differentiable with respect to `x`."
            )

        first_grad_tensor = self._get_combined_gradient_from_tape_output(
            grad_f_pair, x_arr
        )
        first_grad_np = first_grad_tensor.data

        def _hess_worker(args: tuple[Any, ...]) -> np.ndarray:
            r"""
            Internal worker for parallel Hessian column computation.
            """
            (i, x_np, first_grad_np_worker, watch_on_read, forced_dtype_np) = args

            seed_i_np = np.zeros(
                np.prod(first_grad_np_worker.shape), dtype=forced_dtype_np
            )
            seed_i_np[i] = 1.0
            seed_i = Tensor(seed_i_np.reshape(first_grad_np_worker.shape))

            x_tensor = Tensor(x_np)

            with GradientTape(
                persistent=False,
                watch_accessed_variables=watch_on_read,
                dtype=forced_dtype_np,
            ) as t2:
                t2.watch(x_tensor)
                grad_inner_pair = t2.gradient(
                    f(x_tensor),
                    x_tensor,
                    output_gradients=seed_i,
                    unconnected_gradients="zero",
                )

            if (
                grad_inner_pair is None
                or grad_inner_pair[0] is None
                or grad_inner_pair[1] is None
            ):
                return np.zeros(n, dtype=forced_dtype_np)

            combined_inner = grad_inner_pair[0] + grad_inner_pair[1].conj()
            return combined_inner.data.reshape(-1)

        x_np = x_arr.data

        args_list = [
            (i, x_np, first_grad_np, self.watch_on_read, hess_dtype) for i in range(n)
        ]

        with mp.Pool() as pool:
            columns = pool.map(_hess_worker, args_list)

        H_np = np.zeros((n, n), dtype=hess_dtype)
        for i in range(n):
            H_np[:, i] = columns[i]
        return Tensor(H_np)
