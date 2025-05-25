import numpy as np
import multiprocessing as _mp
from .funcs import GRADIENTS, numerical_derivative
from typing import Any, List, Callable, Tuple, Dict, Union, Optional
import warnings
from dataclasses import dataclass, field

@dataclass
class OpNode:
    func: Callable
    method: str
    inputs: Tuple[np.ndarray, ...]
    kwargs: Dict[str, Any]
    result: np.ndarray
    parents: List['OpNode'] = field(default_factory=list)
    creation_index: int = 0      # Helps us keep creation order
    last_visited: int = 0        # Visit‐stamp for DFS marking

    def __repr__(self):
        return (
            f"OpNode(func={self.func.__name__}, "
            f"result_id={id(self.result)}, "
            f"shape={self.result.shape}, "
            f"dtype={self.result.dtype})"
        )

@dataclass
class Gradient:
    holomorphic: np.ndarray
    antiholomorphic: np.ndarray


class GradientTape:
    _GRADIENTS_TAPES: List['GradientTape'] = []
    _primitive_registry: Dict[str, Callable] = {}
    _visit_stamp_counter: int = 1  # For marking ancestors without a Python set

    def __init__(
        self,
        persistent: bool = False,
        watch_accessed_variables: bool = False,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        # Original fields
        self.result_to_node: Dict[int, OpNode] = {}
        self.grads: Dict[int, Gradient] = {}
        self.watched: set[int] = set()
        self.persistent = persistent
        self._used = False
        self.watch_on_read = watch_accessed_variables
        self.forced_dtype = dtype
        self._hooks: Dict[int, Callable[[np.ndarray], np.ndarray]] = {}

        # NEW: keep a flat list of OpNodes in creation order
        self._nodes_in_order: List[OpNode] = []
        self._next_creation_index = 0

        # Reusable seeds
        self._ones_seed: Optional[np.ndarray] = None
        self._zeros_seed: Optional[np.ndarray] = None

        GradientTape._GRADIENTS_TAPES.append(self)

    def __enter__(self) -> 'GradientTape':
        """Context manager entry: Clears tape state for a new computation."""
        self.result_to_node.clear()
        self.grads.clear()
        self.watched.clear()
        self._used = False
        self._hooks.clear()

        # Reset the node list & index counter
        self._nodes_in_order.clear()
        self._next_creation_index = 0

        # Reset seeds
        self._ones_seed = None
        self._zeros_seed = None
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Context manager exit: Deregisters tape if not persistent."""
        if not self.persistent and self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def delete(self) -> None:
        """Removes the tape from the global active tapes list."""
        if self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def watch(self, *objs: np.ndarray) -> None:
        """Adds numpy arrays to the set of watched variables."""
        for o in objs:
            if not isinstance(o, (np.ndarray, np.generic)):
                warnings.warn(
                    f"Cannot watch object of type {type(o)}. Only numpy arrays are supported."
                )
                continue
            self._watch(o)

    def _watch(self, obj: np.ndarray) -> None:
        """Internal helper to add an object's ID to the watched set."""
        self.watched.add(id(obj))

    def stop_gradient(self, x: np.ndarray) -> np.ndarray:
        """Marks a tensor to stop gradient propagation."""
        y = x.view()
        setattr(y, "_stop_gradient", True)
        return y

    @classmethod
    def primitive(cls, func: Callable) -> Callable:
        """Decorator to register a function as a primitive operation."""
        name = func.__name__
        cls._primitive_registry[name] = None  # Value can be None or a placeholder
        return func

    @classmethod
    def def_grad(cls, func: Callable) -> Callable:
        """Decorator to define the gradient for a primitive operation."""
        target_name = func.__name__.replace("_grad", "")
        if target_name not in cls._primitive_registry:
            raise ValueError(f"Primitive {target_name} not registered before defining grad.")
        GRADIENTS[target_name] = func
        return func

    def register_hook(self, var: np.ndarray, hook: Callable[[np.ndarray], np.ndarray]) -> None:
        """Registers a hook function to be applied to the gradient of a variable."""
        self._hooks[id(var)] = hook

    @staticmethod
    def _get_gradient_dtype(dtype: np.dtype, forced: Optional[np.dtype] = None) -> np.dtype:
        """Determines the appropriate dtype for gradients based on input and forced dtype."""
        if forced is not None:
            return forced
        if np.issubdtype(dtype, np.complexfloating):
            return dtype
        return np.complex128 if np.issubdtype(dtype, np.number) else dtype

    def _normalize_inputs(
        self,
        inputs: Tuple[Any, ...],
        kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]:
        """Normalizes inputs and kwargs to numpy arrays."""
        normalized_inputs = tuple(i if isinstance(i, np.ndarray) else np.array(i) for i in inputs)
        normalized_kwargs = {
            k: (v if isinstance(v, np.ndarray) else np.array(v))
            for k, v in kwargs.items()
        }
        return normalized_inputs, normalized_kwargs

    def _create_and_link_op_node(
        self,
        func: Callable,
        method: str,
        inputs: Tuple[np.ndarray, ...],
        kwargs: Dict[str, Any],
        result: np.ndarray
    ) -> None:
        """
        Creates an OpNode and links it to its parents in the computation graph.
        Also attaches the newly created node directly to the array, so we can
        skip dictionary lookups later.
        """
        # Skip anything that is not a plain ndarray:
        if not isinstance(result, np.ndarray):
            return

        idx = self._next_creation_index
        self._next_creation_index += 1

        node = OpNode(
            func=func,
            method=method,
            inputs=inputs,
            kwargs=kwargs,
            result=result,
            creation_index=idx
        )

        # Link parents (we rely on result_to_node for initial construction)
        for inp in inputs:
            parent = getattr(inp, "_tape_node", None)
            if parent is None:
                parent = self.result_to_node.get(id(inp))
            if parent is not None:
                node.parents.append(parent)

        # Register this node
        self.result_to_node[id(result)] = node
        self._nodes_in_order.append(node)

        # Attach node to array for O(1) access
        setattr(result, "_tape_node", node)

    def record(
        self,
        func: Callable,
        method: str,
        inputs: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        result: Any
    ) -> None:
        """Records an operation in the gradient tape if any input is watched."""
        # Skip if “result” isn’t a NumPy array
        if not isinstance(result, np.ndarray):
            return
        # Skip if stop_gradient was set
        if hasattr(result, "_stop_gradient") and getattr(result, "_stop_gradient", False):
            return

        normalized_inputs, normalized_kwargs = self._normalize_inputs(inputs, kwargs)
        tapes = GradientTape._GRADIENTS_TAPES  # local ref

        for tape in tapes:
            # As soon as we find one watched input, record to that tape:
            for x in normalized_inputs:
                if id(x) in tape.watched:
                    tape._watch(result)
                    tape._create_and_link_op_node(
                        func, method,
                        normalized_inputs,
                        normalized_kwargs,
                        result
                    )
                    break

    def _unbroadcast_gradient(self, grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Unbroadcasts a gradient to match the original input shape."""
        if grad.shape == shape:
            return grad
        if grad.ndim == 0:
            return np.full(shape, grad.item(), dtype=grad.dtype)

        # Inline unbroadcast logic for speed
        ndim_diff = len(grad.shape) - len(shape)
        axes_to_sum = list(range(ndim_diff)) if ndim_diff > 0 else []
        for i, dim_grad in enumerate(grad.shape[::-1]):
            if i < len(shape) and shape[::-1][i] == 1 and dim_grad > 1:
                axis_to_reduce = len(grad.shape) - 1 - i
                axes_to_sum.append(axis_to_reduce)

        if axes_to_sum:
            grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
        return grad.reshape(shape)

    def _compute_raw_gradients(
        self,
        node: OpNode,
        grad_res: Gradient
    ) -> Union[Tuple[Tuple[np.ndarray, np.ndarray], ...], List[Tuple[np.ndarray, np.ndarray]]]:
        """Computes raw gradients for a given operation node."""
        name = node.func.__name__
        if dfunc := GRADIENTS.get(name) or GradientTape._primitive_registry.get(name):
            try:
                raw_grads = dfunc(
                    (grad_res.holomorphic, grad_res.antiholomorphic),
                    node.inputs,
                    **node.kwargs
                )
            except TypeError:
                raw_grads = dfunc(
                    (grad_res.holomorphic, grad_res.antiholomorphic),
                    node.inputs
                )
        else:
            warnings.warn(f"No gradient for '{name}', using numerical approximation.")
            approx = numerical_derivative(
                node.func,
                node.inputs,
                node.kwargs,
                high_precision=True,
                max_order=8,
                verbose=False
            )
            raw_grads = []
            for J_h, J_ah in approx:
                gh = grad_res.holomorphic.dot(J_h) + grad_res.antiholomorphic.dot(np.conj(J_ah))
                gah = grad_res.holomorphic.dot(J_ah) + grad_res.antiholomorphic.dot(np.conj(J_h))
                raw_grads.append((gh, gah))
        return raw_grads

    def _process_raw_gradients_format(
        self,
        raw_grads: Any,
        num_inputs: int
    ) -> Tuple[Tuple[np.ndarray, Optional[np.ndarray]], ...]:
        """Normalizes the raw gradient format to tuples of (holomorphic, antiholomorphic)."""
        if not isinstance(raw_grads, tuple) or not all(isinstance(g, tuple) for g in raw_grads):
            if (
                num_inputs == 1
                and isinstance(raw_grads, tuple)
                and len(raw_grads) == 2
                and isinstance(raw_grads[0], np.ndarray)
            ):
                raw_grads = (raw_grads,)
            elif isinstance(raw_grads, list):
                raw_grads = tuple(raw_grads)
            else:
                raw_grads = ((raw_grads, np.zeros_like(raw_grads)),)

        processed_grads: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        for grad_pair in raw_grads:
            if isinstance(grad_pair, tuple) and len(grad_pair) == 2:
                processed_grads.append(grad_pair)
            elif isinstance(grad_pair, np.ndarray):
                processed_grads.append((grad_pair, np.zeros_like(grad_pair)))
            else:
                warnings.warn(f"Unexpected gradient format: {type(grad_pair)}. Skipping.")
                processed_grads.append((None, None))
        return tuple(processed_grads)

    def _initialize_input_gradient(self, inp: np.ndarray) -> None:
        """Initializes the gradient storage for a given input if not already present."""
        if id(inp) not in self.grads:
            store_dtype = self._get_gradient_dtype(inp.dtype, self.forced_dtype)
            self.grads[id(inp)] = Gradient(
                holomorphic=np.zeros(inp.shape, dtype=store_dtype),
                antiholomorphic=np.zeros(inp.shape, dtype=store_dtype)
            )

    def _accumulate_and_apply_hook(
        self,
        inp: np.ndarray,
        g_h: np.ndarray,
        g_ah: np.ndarray
    ) -> None:
        """Accumulates gradients and applies registered hooks."""
        entry = self.grads[id(inp)]
        entry.holomorphic += g_h
        entry.antiholomorphic += g_ah

        hook = self._hooks.get(id(inp))
        if hook is not None:
            entry.holomorphic = hook(entry.holomorphic)

    def _gradient_recursive(self, node: OpNode, grad_res: Gradient) -> None:
        """Recursively computes and propagates gradients through the computation graph."""
        raw_grads = self._compute_raw_gradients(node, grad_res)
        processed = self._process_raw_gradients_format(raw_grads, len(node.inputs))

        for i, inp in enumerate(node.inputs):
            if i >= len(processed):
                continue
            gh, gah = processed[i]
            if gh is None:
                continue

            # Ensure arrays
            if not isinstance(gh, np.ndarray):
                gh = np.array(gh, dtype=grad_res.holomorphic.dtype)
            if not isinstance(gah, np.ndarray):
                gah = np.array(gah, dtype=grad_res.antiholomorphic.dtype)

            gh = self._unbroadcast_gradient(gh, inp.shape)
            gah = self._unbroadcast_gradient(gah, inp.shape)

            self._initialize_input_gradient(inp)
            self._accumulate_and_apply_hook(inp, gh, gah)

    def _backpropagate(self, target: np.ndarray) -> None:
        """
        Backpropagate from `target` by:
          1. Marking all ancestors of target (via visit‐stamp DFS),
          2. Iterating reversed(self._nodes_in_order) and only processing marked nodes.
        """
        root_node = getattr(target, "_tape_node", None)
        if root_node is None:
            root_node = self.result_to_node.get(id(target))
        if root_node is None:
            return

        # 1) Mark ancestors using visit-stamp
        stamp = GradientTape._visit_stamp_counter
        GradientTape._visit_stamp_counter += 1

        stack = [root_node]
        while stack:
            n = stack.pop()
            if n.last_visited == stamp:
                continue
            n.last_visited = stamp
            stack.extend(p for p in n.parents if p.last_visited != stamp)
        # 2) Iterate all nodes in reverse creation order—only process if last_visited == stamp
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
        source: np.ndarray,
        unconnected_gradients: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieves the final (holomorphic, antiholomorphic) gradient for a single source."""
        grad = self.grads.get(id(source))
        if grad is None:
            if unconnected_gradients == "zero":
                store_dtype = self._get_gradient_dtype(source.dtype, self.forced_dtype)
                return (
                    np.zeros_like(source, dtype=store_dtype),
                    np.zeros_like(source, dtype=store_dtype)
                )
            return None
        return (grad.holomorphic, grad.antiholomorphic)

    def _grad_scalar_or_tensor(
        self,
        target: np.ndarray,
        source: np.ndarray,
        output_gradient_seed: np.ndarray,
        unconnected_gradients: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Computes the gradient for a single target-source pair."""
        gh_init = output_gradient_seed.copy()
        gah_init = np.zeros_like(output_gradient_seed)

        self.grads.clear()
        # Pre‐allocate gradients for the target (and any watched sources will be re‐initialized in `gradient` if necessary)
        self.grads[id(target)] = Gradient(holomorphic=gh_init, antiholomorphic=gah_init)
        self._backpropagate(target)
        return self._get_final_gradient_for_source(source, unconnected_gradients)

    def gradient(
        self,
        target: np.ndarray,
        sources: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
        output_gradients=None,
        unconnected_gradients="none"
    ) -> Union[Optional[Tuple[np.ndarray, np.ndarray]], List[Optional[Tuple[np.ndarray, np.ndarray]]]]:
        """Computes the gradient of target with respect to sources."""
        if self._used and not self.persistent:
            raise RuntimeError("GradientTape has already been used and is not persistent.")
        self._used = True

        # Normalize sources_list
        if isinstance(sources, (list, tuple)):
            sources_list = list(sources)
            return_list = True
        else:
            sources_list = [sources]
            return_list = False

        # Only keep watched sources
        sources_list = [s for s in sources_list if id(s) in self.watched]
        if not sources_list:
            if unconnected_gradients == "zero":
                result = [
                    (
                        np.zeros_like(s, dtype=self._get_gradient_dtype(s.dtype, self.forced_dtype)),
                        np.zeros_like(s, dtype=self._get_gradient_dtype(s.dtype, self.forced_dtype))
                    )
                    for s in sources_list
                ]
            else:
                result = [None for _ in sources_list]
            return result if return_list else (result[0] if result else None)


        # Decide on seed arrays
        if output_gradients is not None:
            if output_gradients.shape != target.shape:
                raise ValueError("output_gradients must have same shape as target")
            gh_init = output_gradients
            gah_init = np.zeros_like(output_gradients)
        else:
            dtype_seed = self._get_gradient_dtype(target.dtype, self.forced_dtype)
            if self._ones_seed is None or self._ones_seed.shape != target.shape:
                self._ones_seed = np.ones_like(target, dtype=dtype_seed)
            if self._zeros_seed is None or self._zeros_seed.shape != target.shape:
                self._zeros_seed = np.zeros_like(target, dtype=dtype_seed)
            gh_init = self._ones_seed
            gah_init = self._zeros_seed

        # Clear and pre‐allocate gradients for all watched sources
        self.grads.clear()
        for src in sources_list:
            dtype_src = self._get_gradient_dtype(src.dtype, self.forced_dtype)
            self.grads[id(src)] = Gradient(
                holomorphic=np.zeros(src.shape, dtype=dtype_src),
                antiholomorphic=np.zeros(src.shape, dtype=dtype_src)
            )
        # Override the target’s gradient
        self.grads[id(target)] = Gradient(holomorphic=gh_init.copy(), antiholomorphic=gah_init.copy())

        self._backpropagate(target)

        final_gradients = [
            self._get_final_gradient_for_source(s, unconnected_gradients)
            for s in sources_list
        ]
        return final_gradients if return_list else (final_gradients[0] if final_gradients else None)

    @staticmethod
    def _jac_row_worker(args) -> np.ndarray:
        """Helper for multiprocessing in jacobian."""
        (
            target,
            source,
            idx,
            shape_target,
            shape_source,
            output_gradients,
            unconnected_gradients,
            forced_dtype
        ) = args

        # Build a seed vector of shape `shape_target` with a 1 at `idx`
        flat_t = np.zeros(shape_target, dtype=forced_dtype).reshape(-1)
        flat_t[:] = 0
        flat_t[idx] = 1.0
        seed = flat_t.reshape(shape_target)
        if output_gradients is not None:
            seed = seed * output_gradients

        # In a separate process, we need a fresh tape to compute this “row”
        tape = GradientTape(persistent=False, watch_accessed_variables=False, dtype=forced_dtype)
        tape.watch(source)
        grad_pair = tape._grad_scalar_or_tensor(target, source, seed, unconnected_gradients)
        if grad_pair is None:
            return np.zeros(np.prod(shape_source), dtype=forced_dtype)

        combined = grad_pair[0] + np.conj(grad_pair[1])
        if np.isrealobj(combined) or np.allclose(combined.imag, 0):
            return combined.real.reshape(-1)
        return combined.reshape(-1)

    def jacobian(
        self,
        target: np.ndarray,
        source: np.ndarray,
        unconnected_gradients: str = "none",
        output_gradients: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes the Jacobian matrix of target with respect to source, in parallel."""
        if self._used and not self.persistent:
            raise RuntimeError("GradientTape has already been used and is not persistent.")
        self._used = True

        flat_t = target.reshape(-1)
        M = flat_t.shape[0]
        shape_target = target.shape
        shape_source = source.shape
        forced_dtype = self._get_gradient_dtype(source.dtype, self.forced_dtype)

        if output_gradients is not None and output_gradients.shape != shape_target:
            raise ValueError("output_gradients must have same shape as target")

        # Build argument list for each row
        args_list = [
            (
                target,
                source,
                idx,
                shape_target,
                shape_source,
                output_gradients,
                unconnected_gradients,
                forced_dtype
            )
            for idx in range(M)
        ]

        # Use multiprocessing Pool to compute each “row” in parallel
        with _mp.Pool() as pool:
            rows = pool.map(GradientTape._jac_row_worker, args_list)

        mat = np.stack(rows, axis=0)
        return mat.reshape(shape_target + shape_source)

    def jvp(
        self,
        target: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        source: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        """Computes the Jacobian-vector product."""
        f = (lambda x: target) if isinstance(target, np.ndarray) else target

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as inner_tape:
            inner_tape.watch(source)
            y = f(source)

        # gy_result will be Optional[Tuple[np.ndarray, np.ndarray]] because 'source' is a single np.ndarray
        gy_result = inner_tape.gradient(y, source)

        dtype_src = self._get_gradient_dtype(source.dtype, self.forced_dtype)

        if gy_result is None:
            combined = np.zeros_like(source, dtype=dtype_src)
        else:
            combined = gy_result[0] + np.conj(gy_result[1])

        if y.ndim == 0 or y.size == 1:
            return np.sum(combined * vector)

        warnings.warn(
            "JVP for vectorial target is not implemented correctly with this approach. Returning VJP‐like result."
        )
        # For the VJP-like result, we need the gradient of y wrt source with output_gradients=vector.
        # This is essentially the VJP.
        vjp_result = inner_tape.gradient(y, source, output_gradients=vector, unconnected_gradients="zero")
        # vjp_result will be Optional[Tuple[np.ndarray, np.ndarray]]
        if vjp_result is None:
            return np.zeros_like(source, dtype=dtype_src)
        return vjp_result[0] + np.conj(vjp_result[1])


    def vjp(
        self,
        target: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        source: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        """Computes the vector-Jacobian product."""
        f = (lambda x: target) if isinstance(target, np.ndarray) else target

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as tape:
            tape.watch(source)
            y = f(source)

        # vjp_result will be Optional[Tuple[np.ndarray, np.ndarray]] because 'source' is a single np.ndarray
        vjp_result = tape.gradient(y, source, output_gradients=vector, unconnected_gradients="zero")
        return self._get_combined_gradient_from_tape_output(vjp_result, source)

    def derivative(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        order: int
    ) -> np.ndarray:
        """Computes the n-th order derivative of a function."""
        if order < 0:
            raise ValueError("Order must be non-negative.")
        if order == 0:
            return f(x)
        
        # Recursively compute the (order-1)-th derivative.
        # This 'inner_derivative_value' will be a np.ndarray.
        inner_derivative_value = self.derivative(f, x, order - 1)

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as tape:
            tape.watch(x)
            # Compute the gradient of the (order-1)-th derivative value with respect to x.
            # Since 'x' is a single source, tape.gradient will return Optional[Tuple[np.ndarray, np.ndarray]].
            grad_result_pair = tape.gradient(inner_derivative_value, x)

        # Combine the holomorphic and anti-holomorphic parts using the helper.
        return self._get_combined_gradient_from_tape_output(grad_result_pair, x)

    # Renamed and updated helper for vjp/derivative
    def _get_combined_gradient_from_tape_output(self, grad_output_from_tape: Union[Optional[Tuple[np.ndarray, np.ndarray]], List[Optional[Tuple[np.ndarray, np.ndarray]]]], source_tensor: np.ndarray) -> np.ndarray:
        """
        Helper to extract and combine holomorphic and anti-holomorphic gradients
        from the output of the gradient method.
        """
        grad_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None

        if isinstance(grad_output_from_tape, list):
            # If the output is a list (e.g., from multiple sources), take the first element.
            if grad_output_from_tape:
                grad_pair = grad_output_from_tape[0]
        else:
            # Otherwise, it's already the single Optional[Tuple[np.ndarray, np.ndarray]]
            grad_pair = grad_output_from_tape

        dtype_src = self._get_gradient_dtype(source_tensor.dtype, self.forced_dtype)

        if grad_pair is None:
            # If no gradient is found, return zeros of the appropriate type and shape.
            return np.zeros_like(source_tensor, dtype=dtype_src)
        else:
            # Combine holomorphic and anti-holomorphic parts.
            return grad_pair[0] + np.conj(grad_pair[1])

    def hessian(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        """Computes the Hessian matrix of a scalar-valued function."""
        n = x.size
        hessian_dtype = self._get_gradient_dtype(x.dtype, self.forced_dtype)
        H = np.zeros((n, n), dtype=hessian_dtype)

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as g1:
            g1.watch(x)
            y = f(x)

        # grad_f_pair will be Optional[Tuple[np.ndarray, np.ndarray]]
        grad_f_pair = g1.gradient(y, x)

        if not grad_f_pair:
            raise ValueError("Could not compute first gradient for Hessian calculation.")
        # Flattened gradient
        grad_f_combined = grad_f_pair[0] + np.conj(grad_f_pair[1])
        grad_flat = grad_f_combined.flatten()

        def _hess_worker(args):
            i, flat_grad = args
            seed_i = np.zeros_like(flat_grad)
            seed_i[i] = 1.0
            seed_mat = seed_i.reshape(x.shape)

            tape = GradientTape(persistent=False, watch_accessed_variables=False, dtype=hessian_dtype)
            tape.watch(x)
            # grad_pair_inner will be Optional[Tuple[np.ndarray, np.ndarray]]
            grad_pair_inner = tape.gradient(f(x), x, output_gradients=seed_mat, unconnected_gradients="zero")
            if grad_pair_inner is None:
                return np.zeros(n, dtype=hessian_dtype)
            combined_inner = grad_pair_inner[0] + np.conj(grad_pair_inner[1])
            return combined_inner.reshape(-1)

        args = [(i, grad_flat) for i in range(n)]
        with _mp.Pool() as pool:
            columns = pool.map(_hess_worker, args)

        for i in range(n):
            H[:, i] = columns[i]
        return H

    def print_graph(self, target: np.ndarray) -> None:
        """Prints the computation graph leading to the target node."""
        visited = set()

        def _dfs(node: OpNode, indent=0):
            if node is None or id(node) in visited:
                return
            visited.add(id(node))
            print("    " * indent + repr(node))
            for parent in node.parents:
                _dfs(parent, indent + 1)

        root = getattr(target, "_tape_node", None)
        if root is None:
            root = self.result_to_node.get(id(target))
        _dfs(root)

    def _calculate_numerical_gradient(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float) -> np.ndarray:
        """Calculates the numerical gradient of f with respect to x."""
        x_copy = x.copy()
        num_grad = np.zeros_like(x_copy, dtype=self._get_gradient_dtype(x_copy.dtype, self.forced_dtype))
        it = np.nditer(x_copy, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = x_copy[idx].copy()

            x_copy[idx] = orig + eps
            fplus = f(x_copy)

            x_copy[idx] = orig - eps
            fminus = f(x_copy)

            x_copy[idx] = orig
            num_grad[idx] = (fplus - fminus) / (2 * eps)
            it.iternext()
        return num_grad

    def check_gradient(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-5) -> float:
        """Compares the analytical gradient with a numerical approximation."""
        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as g:
            g.watch(x)
            y = f(x)

        # grad_pair will be Optional[Tuple[np.ndarray, np.ndarray]]
        grad_pair = g.gradient(y, x)
        if not grad_pair:
            raise ValueError("Analytical gradient could not be computed for check_gradient.")

        analytical_grad = grad_pair[0] + np.conj(grad_pair[1])
        numerical_grad = self._calculate_numerical_gradient(f, x, eps)
        return np.max(np.abs(numerical_grad - analytical_grad))

