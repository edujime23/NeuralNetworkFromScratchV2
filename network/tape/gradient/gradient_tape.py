# network/tape/gradient/gradient_tape.py

from __future__ import annotations
from typing import Any, Callable, Iterable
from collections.abc import Sequence
from dataclasses import dataclass, field
import numpy as np
import multiprocessing as mp
import warnings

# Assuming these are defined elsewhere in your project
from .funcs import GRADIENTS, numerical_derivative
from ...types import Tensor, Variable


@dataclass
class OpNode:
    """
    Represents a node in the computation graph, recording a primitive operation.
    
    Stores information about the function executed, its inputs, keyword arguments,
    and the resulting array. Maintains references to parent nodes for backpropagation.
    """
    func: Callable[..., Any]
    method: str
    inputs: tuple[Tensor, ...]
    kwargs: dict[str, Any]
    result: Tensor
    parents: list[OpNode] = field(default_factory=list)
    creation_index: int = 0
    last_visited: int = 0

    def __repr__(self) -> str:
        """Provides a concise string representation of the OpNode."""
        return (
            f"OpNode(func={self.func.__name__}, "
            f"result_id={id(self.result)}, "
            f"shape={self.result.shape}, "
            f"dtype={self.result.dtype})"
        )


@dataclass
class Gradient:
    """
    Stores holomorphic and anti-holomorphic components of a gradient.
    Essential for handling complex-differentiable operations.
    """
    holomorphic: Tensor
    antiholomorphic: Tensor


class GradientTape:
    """
    Records operations on Tensors/Variables to enable automatic differentiation.
    
    Builds a computation graph and performs reverse-mode gradient accumulation.
    Supports complex differentiation via holomorphic/anti-holomorphic components.
    """

    _GRADIENTS_TAPES: list[GradientTape] = []
    _primitive_registry: dict[str, None] = {}
    _visit_stamp_counter: int = 1

    def __init__(
        self,
        persistent: bool = False,
        watch_accessed_variables: bool = False,
        dtype: np.dtype | None = None,
    ) -> None:
        """
        Initialize a new gradient tape.
        
        Args:
            persistent: If True, allows multiple gradient computations
            watch_accessed_variables: If True, watches variables when accessed
            dtype: Force gradient computations to use specific dtype
        """
        self.result_to_node: dict[int, OpNode] = {}
        self.grads: dict[int, Gradient] = {}
        self.watched: set[int] = set()

        self.persistent = persistent
        self._used = False
        self.watch_on_read = watch_accessed_variables
        self.forced_dtype = dtype
        self._hooks: dict[int, Callable[[Tensor], Tensor]] = {}

        self._nodes_in_order: list[OpNode] = []
        self._next_creation_index = 0

        # Cache for gradient seeds to avoid repeated allocation
        self._ones_seed: Tensor | None = None
        self._zeros_seed: Tensor | None = None

        GradientTape._GRADIENTS_TAPES.append(self)

    def __enter__(self) -> GradientTape:
        """Context manager entry - reset state for non-persistent tapes."""
        if not self.persistent:
            self.result_to_node.clear()
            self.grads.clear()
            self.watched.clear()
            self._used = False
            self._hooks.clear()
            self._nodes_in_order.clear()
            self._next_creation_index = 0
            self._ones_seed = None
            self._zeros_seed = None
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Context manager exit - cleanup for non-persistent tapes."""
        if not self.persistent and self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def delete(self) -> None:
        """Manually remove tape from global registry."""
        if self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def watch(self, *objs: Tensor | Variable) -> None:
        """Mark tensors/variables for gradient tracking."""
        for obj in objs:
            if isinstance(obj, Variable):
                self._watch(obj.value)
            elif isinstance(obj, Tensor):
                self._watch(obj)

    def _watch(self, arr: Tensor) -> None:
        """Internal method to watch a specific tensor."""
        self.watched.add(id(arr))

    def stop_gradient(self, x: Tensor) -> Tensor:
        """Mark tensor to stop gradient flow through it."""
        setattr(x, "_stop_gradient", True)
        return x

    @classmethod
    def primitive(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a differentiable primitive."""
        name = func.__name__
        cls._primitive_registry[name] = None
        return func

    @classmethod
    def def_grad(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to define gradient function for a primitive."""
        target_name = func.__name__.replace("_grad", "")
        if target_name not in cls._primitive_registry:
            raise ValueError(f"Primitive '{target_name}' not registered before defining its gradient.")
        GRADIENTS[target_name] = func
        return func

    def register_hook(self, var: Tensor | Variable, hook: Callable[[Tensor], Tensor]) -> None:
        """Register a hook function to modify gradients during backpropagation."""
        arr = var.value if isinstance(var, Variable) else var
        self._hooks[id(arr)] = hook

    @staticmethod
    def _get_gradient_dtype(dtype: np.dtype, forced: np.dtype | None = None) -> np.dtype:
        """Determine appropriate dtype for gradient computation."""
        if forced is not None:
            return forced
        if np.issubdtype(dtype, np.complexfloating):
            return dtype
        return np.complex128 if np.issubdtype(dtype, np.number) else dtype

    def _normalize_inputs(
        self,
        inputs: tuple[Tensor | Variable, ...],
        kwargs: dict[str, Any]
    ) -> tuple[tuple[Tensor, ...], dict[str, Any]]:
        """Extract underlying tensors from Variables for computation."""
        normalized_inputs = tuple(i.value if isinstance(i, Variable) else i for i in inputs)
        normalized_kwargs = {k: v.value if isinstance(v, Variable) else v for k, v in kwargs.items()}
        return normalized_inputs, normalized_kwargs

    def _create_and_link_op_node(
        self,
        func: Callable[..., Any],
        method: str,
        inputs: tuple[Tensor, ...],
        kwargs: dict[str, Any],
        result: Tensor
    ) -> None:
        """Create OpNode and link it to parent nodes in computation graph."""
        if not isinstance(result, (Tensor, Variable)):
            return

        node = OpNode(
            func=func,
            method=method,
            inputs=inputs,
            kwargs=kwargs,
            result=result,
            creation_index=self._next_creation_index
        )
        self._next_creation_index += 1

        # Link to parent nodes
        for arr in inputs:
            parent = self.result_to_node.get(id(arr))
            if parent is not None:
                node.parents.append(parent)

        self.result_to_node[id(result)] = node
        self._nodes_in_order.append(node)

    def record(
        self,
        func: Callable[..., Any],
        method: str,
        inputs: tuple[Tensor | Variable, ...],
        kwargs: dict[str, Any],
        result: Tensor | Variable
    ) -> None:
        """Record an operation in all active gradient tapes."""
        if not isinstance(result, (Tensor, Variable)) or getattr(result, "_stop_gradient", False):
            return

        normalized_inputs, normalized_kwargs = self._normalize_inputs(inputs, kwargs)

        # Record in any tape that watches the inputs
        for tape in GradientTape._GRADIENTS_TAPES:
            if any(id(arr) in tape.watched for arr in normalized_inputs):
                tape._watch(result)
                tape._create_and_link_op_node(
                    func, method,
                    normalized_inputs,
                    normalized_kwargs,
                    result
                )
                break

    def _unbroadcast_gradient(self, grad: Tensor, original_shape: tuple[int, ...]) -> Tensor:
        """Reverse broadcasting effects in gradient by summing over broadcasted dimensions."""
        if grad.shape == original_shape:
            return grad
        if grad.ndim == 0:
            return np.full(original_shape, grad.item(), dtype=grad.dtype)

        # Sum over prepended dimensions
        ndim_diff = grad.ndim - len(original_shape)
        axes_to_sum = list(range(ndim_diff)) if ndim_diff > 0 else []

        # Sum over dimensions that were broadcast from size 1
        for i, dim_grad in enumerate(grad.shape[::-1]):
            if i < len(original_shape) and original_shape[::-1][i] == 1 and dim_grad > 1:
                axes_to_sum.append(grad.ndim - 1 - i)

        if axes_to_sum:
            grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)

        return grad.reshape(original_shape)

    def _compute_raw_gradients(
        self,
        node: OpNode,
        grad_res: Gradient
    ) -> list[tuple[Tensor, Tensor]]:
        """Compute raw gradients using analytical or numerical methods."""
        name = node.func.__name__
        if gradient_func := GRADIENTS.get(
            name
        ) or GradientTape._primitive_registry.get(name):
            try:
                raw_grads = gradient_func(
                    (grad_res.holomorphic, grad_res.antiholomorphic),
                    node.inputs,
                    **node.kwargs
                )
            except TypeError:
                raw_grads = gradient_func(
                    (grad_res.holomorphic, grad_res.antiholomorphic),
                    node.inputs
                )
        else:
            # Fall back to numerical differentiation
            warnings.warn(
                f"No analytical gradient for '{name}', using numerical approximation. "
                "Consider defining a gradient function for better performance and accuracy."
            )
            approx_jacobians = numerical_derivative(
                node.func,
                node.inputs,
                node.kwargs,
                high_precision=True,
                max_order=8,
                verbose=False
            )
            raw_grads = []
            for J_h, J_ah in approx_jacobians:
                gh = grad_res.holomorphic.dot(J_h) + grad_res.antiholomorphic.dot(np.conj(J_ah))
                gah = grad_res.holomorphic.dot(J_ah) + grad_res.antiholomorphic.dot(np.conj(J_h))
                raw_grads.append((gh, gah))

        return raw_grads

    def _process_raw_gradients_format(
        self,
        raw_grads: Any,
        num_inputs: int
    ) -> tuple[tuple[Tensor | None, Tensor | None], ...]:
        """Normalize gradient format to consistent tuple structure."""
        # Handle single input case
        if not isinstance(raw_grads, tuple) or not all(isinstance(g, tuple) for g in raw_grads):
            if num_inputs == 1 and isinstance(raw_grads, tuple) and len(raw_grads) == 2 and isinstance(raw_grads[0], (Tensor, Variable)):
                raw_grads = (raw_grads,)
            elif isinstance(raw_grads, list):
                raw_grads = tuple(raw_grads)
            else:
                raw_grads = ((raw_grads, np.zeros_like(raw_grads)),)

        processed_grads = []
        for grad_pair in raw_grads:
            if isinstance(grad_pair, tuple) and len(grad_pair) == 2:
                processed_grads.append(grad_pair)
            elif isinstance(grad_pair, Tensor):
                processed_grads.append((grad_pair, np.zeros_like(grad_pair)))
            else:
                warnings.warn(f"Unexpected gradient format: {type(grad_pair)}. Skipping gradient for this input.")
                processed_grads.append((None, None))
        
        return tuple(processed_grads)

    def _initialize_input_gradient(self, inp: Tensor) -> None:
        """Initialize gradient storage for an input tensor."""
        if id(inp) not in self.grads:
            store_dtype = self._get_gradient_dtype(inp.dtype, self.forced_dtype)
            self.grads[id(inp)] = Gradient(
                holomorphic=np.zeros(inp.shape, dtype=store_dtype),
                antiholomorphic=np.zeros(inp.shape, dtype=store_dtype)
            )

    def _accumulate_and_apply_hook(
        self,
        inp: Tensor,
        g_h: Tensor,
        g_ah: Tensor
    ) -> None:
        """Accumulate gradients and apply any registered hooks."""
        entry = self.grads[id(inp)]
        entry.holomorphic += g_h
        entry.antiholomorphic += g_ah

        if hook := self._hooks.get(id(inp)):
            entry.holomorphic = hook(entry.holomorphic)

    def _gradient_recursive(self, node: OpNode, grad_res: Gradient) -> None:
        """Recursively compute and accumulate gradients for a node's inputs."""
        raw_grads_for_inputs = self._compute_raw_gradients(node, grad_res)
        processed_grads = self._process_raw_gradients_format(raw_grads_for_inputs, len(node.inputs))

        for i, inp in enumerate(node.inputs):
            if i >= len(processed_grads):
                continue

            gh, gah = processed_grads[i]
            if gh is None:
                continue

            if not isinstance(inp, Tensor):
                inp = Tensor(inp)

            # Reverse broadcasting effects
            gh = self._unbroadcast_gradient(gh, inp.shape)
            gah = self._unbroadcast_gradient(gah, inp.shape)

            self._initialize_input_gradient(inp)
            self._accumulate_and_apply_hook(inp, gh, gah)

    def _backpropagate(self, target_arr: Tensor) -> None:
        """Perform reverse-mode automatic differentiation from target."""
        root_node = self.result_to_node.get(id(target_arr))
        if root_node is None:
            return

        # Mark all reachable nodes using topological ordering
        stamp = GradientTape._visit_stamp_counter
        GradientTape._visit_stamp_counter += 1

        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.last_visited == stamp:
                continue
            node.last_visited = stamp
            stack.extend(p for p in node.parents if p.last_visited != stamp)

        # Process nodes in reverse topological order
        for node in reversed(self._nodes_in_order):
            if node.last_visited != stamp:
                continue

            res_id = id(node.result)
            if res_id not in self.grads:
                continue

            grad_res = self.grads[res_id]
            self._gradient_recursive(node, grad_res)

    def _get_final_gradient_for_source(
        self,
        source_arr: Tensor,
        unconnected_gradients: str
    ) -> tuple[Tensor, Tensor] | None:
        """Get final gradient result for a source tensor."""
        grad = self.grads.get(id(source_arr))
        if grad is None:
            if unconnected_gradients == "zero":
                store_dtype = self._get_gradient_dtype(source_arr.dtype, self.forced_dtype)
                return (
                    np.zeros_like(source_arr, dtype=store_dtype),
                    np.zeros_like(source_arr, dtype=store_dtype)
                )
            return None
        return (grad.holomorphic, grad.antiholomorphic)

    def gradient(
        self,
        target: Tensor,
        sources: Tensor | Variable | Iterable[Tensor | Variable],
        output_gradients: Tensor | Variable | None = None,
        unconnected_gradients: str = "none"
    ) -> tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor] | None] | None:
        """
        Compute gradients of target with respect to sources.
        
        Args:
            target: Tensor to differentiate
            sources: Source tensor(s) to compute gradients for
            output_gradients: Gradient of some scalar w.r.t target
            unconnected_gradients: How to handle unconnected variables ("none" or "zero")
            
        Returns:
            Gradient tuple(s) (holomorphic, antiholomorphic) or None for unconnected
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "GradientTape has already been used and is not persistent. "
                "Set `persistent=True` during initialization for reuse."
            )
        self._used = True

        target_arr = target.value if isinstance(target, Variable) else target

        # Handle single vs multiple sources
        is_single_source = not isinstance(sources, Iterable)
        sources_raw = []
        sources_to_process = [sources] if is_single_source else list(sources)
        for s in sources_to_process:
            if isinstance(s, Variable):
                sources_raw.append(s.value)
            else:
                sources_raw.append(s)

        # Filter to only watched sources
        filtered_sources = [s for s in sources_raw if id(s) in self.watched]

        if not filtered_sources:
            if unconnected_gradients == "zero":
                result_list = []
                for s in sources_raw:
                    dtype_s = self._get_gradient_dtype(s.dtype, self.forced_dtype)
                    result_list.append((np.zeros_like(s, dtype=dtype_s), np.zeros_like(s, dtype=dtype_s)))
            else:
                result_list = [None] * len(sources_raw)
            return result_list[0] if is_single_source and result_list else result_list

        # Set up initial gradient
        if output_gradients is not None:
            og_arr = output_gradients.value if isinstance(output_gradients, Variable) else output_gradients
            if og_arr.shape != target_arr.shape:
                raise ValueError(f"Output gradients shape {og_arr.shape} must match target shape {target_arr.shape}.")
            gh_init = og_arr
            gah_init = np.zeros_like(og_arr)
        else:
            dtype_seed = self._get_gradient_dtype(target_arr.dtype, self.forced_dtype)
            if self._ones_seed is None or self._ones_seed.shape != target_arr.shape:
                self._ones_seed = np.ones_like(target_arr, dtype=dtype_seed)
            if self._zeros_seed is None or self._zeros_seed.shape != target_arr.shape:
                self._zeros_seed = np.zeros_like(target_arr, dtype=dtype_seed)
            gh_init = self._ones_seed
            gah_init = self._zeros_seed

        # Initialize gradients
        self.grads.clear()
        for src in filtered_sources:
            self.grads[id(src)] = Gradient(
                holomorphic=np.zeros_like(src),
                antiholomorphic=np.zeros_like(src)
            )

        self.grads[id(target_arr)] = Gradient(holomorphic=gh_init, antiholomorphic=gah_init)

        # Perform backpropagation
        self._backpropagate(target_arr)

        # Collect final gradients
        final_gradients = []
        for src in sources_raw:
            grad_pair = self._get_final_gradient_for_source(src, unconnected_gradients)
            final_gradients.append(grad_pair)

        return final_gradients[0] if is_single_source else final_gradients

    @staticmethod
    def _jac_row_worker(args: tuple[Any, ...]) -> Tensor:
        """Worker function for parallel Jacobian computation."""
        (target_arr, source_arr, idx, shape_target, shape_source, 
         output_gradients, unconnected_gradients, forced_dtype) = args

        # Create unit vector for this row
        flat_t_seed = np.zeros(shape_target, dtype=forced_dtype).reshape(-1)
        flat_t_seed[idx] = 1.0
        seed = flat_t_seed.reshape(shape_target)

        if output_gradients is not None:
            seed = seed * output_gradients

        # Compute gradient for this output component
        with GradientTape(persistent=False, watch_accessed_variables=False, dtype=forced_dtype) as tape:
            tape.watch(source_arr)
            grad_pair = tape._grad_scalar_or_tensor(target_arr, source_arr, seed, unconnected_gradients)

        if grad_pair is None:
            return np.zeros(np.prod(shape_source), dtype=forced_dtype)

        # Combine holomorphic and antiholomorphic parts
        combined = grad_pair[0] + np.conj(grad_pair[1])
        if np.isrealobj(combined) or np.allclose(combined.imag, 0):
            return combined.real.reshape(-1)
        return combined.reshape(-1)

    def jacobian(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        unconnected_gradients: str = "none",
        output_gradients: Tensor | Variable | None = None
    ) -> Tensor:
        """
        Compute full Jacobian matrix using parallel processing.
        
        Returns tensor of shape (target.shape + source.shape).
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "GradientTape has already been used and is not persistent. "
                "Set `persistent=True` during initialization for reuse."
            )
        self._used = True

        target_arr = target.value if isinstance(target, Variable) else target
        source_arr = source.value if isinstance(source, Variable) else source

        flat_target_size = target_arr.size
        shape_target = target_arr.shape
        shape_source = source_arr.shape
        forced_dtype = self._get_gradient_dtype(source_arr.dtype, self.forced_dtype)

        og_arr = None
        if output_gradients is not None:
            og_arr = output_gradients.value if isinstance(output_gradients, Variable) else output_gradients
            if og_arr.shape != shape_target:
                raise ValueError(f"Output gradients shape {og_arr.shape} must match target shape {shape_target}.")

        # Prepare arguments for parallel computation
        args_list = [
            (
                target_arr, source_arr, idx, shape_target, shape_source,
                og_arr, unconnected_gradients, forced_dtype
            )
            for idx in range(flat_target_size)
        ]

        # Compute Jacobian rows in parallel
        with mp.Pool() as pool:
            rows = pool.map(GradientTape._jac_row_worker, args_list)

        jacobian_matrix = np.stack(rows, axis=0)
        return jacobian_matrix.reshape(shape_target + shape_source)

    def _prepare_target_function(
        self,
        target: Tensor | Variable
    ) -> Callable[[Tensor], Tensor]:
        """Prepare target for JVP/VJP computation."""
        if isinstance(target, Variable):
            return lambda x: target.value
        else:
            return lambda x: target

    def _get_combined_gradient_from_tape_output(
        self,
        grad_output_from_tape: Any,
        source_arr: Tensor
    ) -> Tensor:
        """Combine holomorphic and antiholomorphic gradients."""
        grad_pair = None

        if isinstance(grad_output_from_tape, list):
            if grad_output_from_tape:
                grad_pair = grad_output_from_tape[0]
        else:
            grad_pair = grad_output_from_tape

        dtype_src = self._get_gradient_dtype(source_arr.dtype, self.forced_dtype)
        if grad_pair is None:
            return np.zeros_like(source_arr, dtype=dtype_src)
        else:
            return grad_pair[0] + np.conj(grad_pair[1])

    def jvp(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        vector: Tensor | Variable
    ) -> Tensor:
        """
        Compute Jacobian-vector product (forward-mode AD).
        
        For scalar targets, computes directional derivative.
        For vector targets, warns about current VJP-like behavior.
        """
        source_arr = source.value if isinstance(source, Variable) else source
        vec_arr = vector.value if isinstance(vector, Variable) else vector
        f = self._prepare_target_function(target)

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as inner_tape:
            inner_tape.watch(source_arr)
            y = f(source_arr)

        gy_result = inner_tape.gradient(y, source_arr)
        combined_grad_y = self._get_combined_gradient_from_tape_output(gy_result, source_arr)

        if isinstance(y, Tensor) and (y.ndim == 0 or y.size == 1):
            return np.sum(combined_grad_y * vec_arr)

        warnings.warn(
            "JVP for vectorial target is not implemented as a direct Jacobian-vector product. "
            "It currently computes a VJP-like result (vector^T @ J(f)). "
            "Consider using `jacobian` and then matrix multiplication for explicit JVP."
        )
        vjp_result = inner_tape.gradient(y, source_arr, output_gradients=vec_arr, unconnected_gradients="zero")
        return self._get_combined_gradient_from_tape_output(vjp_result, source_arr)

    def vjp(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        vector: Tensor | Variable
    ) -> Tensor:
        """Compute vector-Jacobian product (reverse-mode AD)."""
        source_arr = source.value if isinstance(source, Variable) else source
        vec_arr = vector.value if isinstance(vector, Variable) else vector
        f = self._prepare_target_function(target)

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as tape:
            tape.watch(source_arr)
            y = f(source_arr)

        vjp_result = tape.gradient(y, source_arr, output_gradients=vec_arr, unconnected_gradients="zero")
        return self._get_combined_gradient_from_tape_output(vjp_result, source_arr)

    def derivative(
        self,
        f: Callable[[Tensor], Tensor],
        x: Tensor | Variable,
        order: int
    ) -> Tensor:
        """Compute higher-order derivatives recursively."""
        x_arr = x.value if isinstance(x, Variable) else x

        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        if order == 0:
            return f(x_arr)

        # Recursive computation of higher-order derivatives
        inner_value = self.derivative(f, x_arr, order - 1)

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as tape:
            tape.watch(x_arr)
            grad_pair = tape.gradient(inner_value, x_arr)

        return self._get_combined_gradient_from_tape_output(grad_pair, x_arr)

    def hessian(self, f: Callable[[Tensor], Tensor], x: Tensor | Variable) -> Tensor:
        """Compute Hessian matrix using parallel processing."""
        x_arr = x.value if isinstance(x, Variable) else x

        n = x_arr.size
        hessian_dtype = self._get_gradient_dtype(x_arr.dtype, self.forced_dtype)

        # Compute first-order gradient
        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as g1:
            g1.watch(x_arr)
            y = f(x_arr)
            grad_f_pair = g1.gradient(y, x_arr)

        if not grad_f_pair:
            raise ValueError(
                "Could not compute first gradient for Hessian calculation. "
                "Ensure `f` is differentiable with respect to `x`."
            )

        def _hess_worker(args: tuple[Any, ...]) -> Tensor:
            """Worker for parallel Hessian computation."""
            i, original_x_arr, current_f_val, tape_watch_on_read, tape_forced_dtype = args

            # Create unit vector for i-th component
            seed_i = np.zeros_like(current_f_val)
            seed_i.flat[i] = 1.0

            with GradientTape(persistent=False, watch_accessed_variables=tape_watch_on_read, dtype=tape_forced_dtype) as tape:
                tape.watch(original_x_arr)
                grad_pair_inner = tape.gradient(
                    f(original_x_arr),
                    original_x_arr,
                    output_gradients=seed_i.reshape(original_x_arr.shape),
                    unconnected_gradients="zero"
                )
            
            if grad_pair_inner is None:
                return np.zeros(n, dtype=hessian_dtype)

            combined_inner = grad_pair_inner[0] + np.conj(grad_pair_inner[1])
            return combined_inner.reshape(-1)

        # Prepare arguments for parallel computation
        args_list = [
            (i, x_arr, f(x_arr), self.watch_on_read, hessian_dtype)
            for i in range(n)
        ]

        # Compute Hessian columns in parallel
        with mp.Pool() as pool:
            columns = pool.map(_hess_worker, args_list)

        H = np.zeros((n, n), dtype=hessian_dtype)

        for i in range(n):
            H[:, i] = columns[i]
        return H

    def print_graph(self, target: Tensor | Variable) -> None:
        """
        Prints the computation graph backwards from the target node using DFS.
        
        Args:
            target: The Tensor or Variable to start graph traversal from.
        """
        # Extract underlying tensor from Variable wrapper
        target_tensor = target.value if isinstance(target, Variable) else target
        visited_nodes: set[int] = set()

        def _dfs(node: Any, indent_level: int = 0) -> None:
            """Recursive DFS traversal of computation graph."""
            if node is None or id(node) in visited_nodes:
                return

            visited_nodes.add(id(node))
            indent = "  " * indent_level

            if isinstance(node, Tensor):
                print(f"{indent}Tensor(ID={id(node)}, shape={node.shape}, dtype={node.dtype})")
                
                # Find the OpNode that produced this tensor (if any)
                producer_node = self.result_to_node.get(id(node))
                if producer_node:
                    _dfs(producer_node, indent_level)

            elif isinstance(node, OpNode):
                print(f"{indent}OpNode(func={node.func.__name__}, "
                    f"result_id={id(node.result)}, "
                    f"shape={node.result.shape}, dtype={node.result.dtype})")
                
                # Recurse on input tensors (parents in computation graph)
                for parent_tensor in node.inputs:
                    _dfs(parent_tensor, indent_level + 1)
            else:
                print(f"{indent}Unknown node type: {type(node)} (ID={id(node)})")

        # Start traversal from root OpNode or handle input tensor case
        root_node = self.result_to_node.get(id(target_tensor))
        
        if root_node is None:
            # Target is an input tensor, not a computation result
            if id(target_tensor) in self.watched:
                print(f"Graph for watched input Tensor (ID={id(target_tensor)}):")
                
                # Find operations that consume this input
                consumers_found = False
                for op_node in self._nodes_in_order:
                    if any(id(inp) == id(target_tensor) for inp in op_node.inputs) and id(op_node) not in visited_nodes:
                        print("  --> Consumed by:")
                        _dfs(op_node, indent_level=2)
                        consumers_found = True
                
                if not consumers_found:
                    print("  No consuming operations found in recorded graph.")
            else:
                print(f"Target {id(target_tensor)} not found in computation graph. "
                    "Neither operation result nor watched input.")
            return

        # Target is result of an operation - start backward traversal
        print(f"Graph for result Tensor (ID={id(target_tensor)}):")
        _dfs(root_node)