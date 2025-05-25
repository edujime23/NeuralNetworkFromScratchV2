import numpy as np
from .gradients import GRADIENTS, numerical_derivative
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
    creation_index: int = 0  # Helps us keep creation order

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
        """Creates an OpNode and links it to its parents in the computation graph."""
        idx = self._next_creation_index
        self._next_creation_index += 1

        node = OpNode(func=func, method=method, inputs=inputs, kwargs=kwargs, result=result, creation_index=idx)

        # Link parents
        for inp in inputs:
            parent = self.result_to_node.get(id(inp))
            if parent is not None:
                node.parents.append(parent)

        # Register this node
        self.result_to_node[id(result)] = node
        self._nodes_in_order.append(node)  # ALSO append to our flat list

    def record(
        self,
        func: Callable,
        method: str,
        inputs: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        result: np.ndarray
    ) -> None:
        """Records an operation in the gradient tape if any input is watched."""
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

        reduce_axes = []
        ndim_diff = len(grad.shape) - len(shape)
        if ndim_diff > 0:
            reduce_axes.extend(range(ndim_diff))

        for i, dim_grad in enumerate(grad.shape[::-1]):
            if i < len(shape) and shape[::-1][i] == 1 and dim_grad > 1:
                axis_to_reduce = len(grad.shape) - 1 - i
                reduce_axes.append(axis_to_reduce)

        if reduce_axes:
            grad = np.sum(grad, axis=tuple(reduce_axes), keepdims=True)

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
            # Only modify holomorphic part via hook, as before
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
          1. Marking all ancestors of target (via DFS),
          2. Iterating reversed(self._nodes_in_order) and only processing marked nodes.
        """
        # 1) Mark ancestors:
        visited = set()
        root_node = self.result_to_node.get(id(target))
        if root_node is None:
            return

        stack = [root_node]
        while stack:
            n = stack.pop()
            if id(n) in visited:
                continue
            visited.add(id(n))
            stack.extend(p for p in n.parents if id(p) not in visited)
        # 2) Iterate all nodes in reverse creation order—only process if visited
        for node in reversed(self._nodes_in_order):
            nid = id(node)
            if nid not in visited:
                continue
            if id(node.result) not in self.grads:
                continue
            grad_res = self.grads[id(node.result)]
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
        self.grads[id(target)] = Gradient(holomorphic=gh_init, antiholomorphic=gah_init)
        self._backpropagate(target)
        return self._get_final_gradient_for_source(source, unconnected_gradients)

    def gradient(
        self,
        target: np.ndarray,
        sources: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
        output_gradients=None,
        unconnected_gradients="none"
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Computes the gradient of target with respect to sources."""
        if self._used and not self.persistent:
            raise RuntimeError("GradientTape has already been used and is not persistent.")
        self._used = True

        if isinstance(sources, (list, tuple)):
            sources_list = list(sources)
        else:
            sources_list = [sources]

        sources_list = [s for s in sources_list if id(s) in self.watched]
        if not sources_list:
            if unconnected_gradients == "zero":
                return [
                    (
                        np.zeros_like(s, dtype=self._get_gradient_dtype(s.dtype, self.forced_dtype)),
                        np.zeros_like(s, dtype=self._get_gradient_dtype(s.dtype, self.forced_dtype))
                    )
                    for s in sources_list
                ]
            return [None for _ in sources_list]

        if output_gradients is not None:
            if output_gradients.shape != target.shape:
                raise ValueError("output_gradients must have same shape as target")
            gh_init = output_gradients
            gah_init = np.zeros_like(output_gradients)
        else:
            gh_init = np.ones_like(target, dtype=self._get_gradient_dtype(target.dtype, self.forced_dtype))
            gah_init = np.zeros_like(target, dtype=self._get_gradient_dtype(target.dtype, self.forced_dtype))

        self.grads.clear()
        self.grads[id(target)] = Gradient(holomorphic=gh_init.copy(), antiholomorphic=gah_init.copy())

        self._backpropagate(target)

        final_gradients = [
            self._get_final_gradient_for_source(s, unconnected_gradients)
            for s in sources_list
        ]
        return final_gradients if len(final_gradients) > 1 else final_gradients[0]

    def jacobian(
        self,
        target: np.ndarray,
        source: np.ndarray,
        unconnected_gradients: str = "none",
        output_gradients: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Computes the Jacobian matrix of target with respect to source."""
        if self._used and not self.persistent:
            raise RuntimeError("GradientTape has already been used and is not persistent.")
        self._used = True

        flat_t = target.reshape(-1)
        M = flat_t.shape[0]
        jac_rows = []

        for idx in range(M):
            seed = np.zeros_like(flat_t)
            seed[idx] = 1.0
            seed = seed.reshape(target.shape)

            if output_gradients is not None:
                if output_gradients.shape != target.shape:
                    raise ValueError("output_gradients must have same shape as target")
                seed = seed * output_gradients

            grad_pair = self._grad_scalar_or_tensor(target, source, seed, unconnected_gradients)
            store_dtype = self._get_gradient_dtype(source.dtype, self.forced_dtype)

            if grad_pair is None:
                combined = np.zeros_like(source, dtype=store_dtype)
            else:
                combined = grad_pair[0] + np.conj(grad_pair[1])

            if np.isrealobj(combined) or np.allclose(combined.imag, 0):
                jac_rows.append(combined.real.reshape(-1))
            else:
                jac_rows.append(combined.reshape(-1))

        mat = np.stack(jac_rows, axis=0)
        return mat.reshape(target.shape + source.shape)

    def jvp(
        self,
        target: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        source: np.ndarray,
        vector: np.ndarray
    ) -> np.ndarray:
        """Computes the Jacobian-vector product."""
        f = (lambda x: target) if isinstance(target, np.ndarray) else target

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as inner:
            inner.watch(source)
            y = f(source)

        gy_list = inner.gradient(y, source)
        gy_pair = gy_list[0] if gy_list else None
        store_dtype = self._get_gradient_dtype(source.dtype, self.forced_dtype)

        if gy_pair is None:
            combined = np.zeros_like(source, dtype=store_dtype)
        else:
            combined = gy_pair[0] + np.conj(gy_pair[1])

        if y.ndim == 0 or y.size == 1:
            return np.sum(combined * vector)

        warnings.warn(
            "JVP for vectorial target is not implemented correctly with this approach. Returning VJP‐like result."
        )
        vjp_list = inner.gradient(y, source, output_gradients=vector, unconnected_gradients="zero")
        vjp_pair = vjp_list[0] if vjp_list else None

        if vjp_pair is None:
            return np.zeros_like(source, dtype=store_dtype)
        return vjp_pair[0] + np.conj(vjp_pair[1])

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

        vjp_list = tape.gradient(y, source, output_gradients=vector, unconnected_gradients="zero")
        vjp_pair = vjp_list[0] if vjp_list else None
        store_dtype = self._get_gradient_dtype(source.dtype, self.forced_dtype)

        if vjp_pair is None:
            return np.zeros_like(source, dtype=store_dtype)
        return vjp_pair[0] + np.conj(vjp_pair[1])

    def derivative(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        order: int
    ) -> np.ndarray:
        """Computes the n-th order derivative of a function."""
        if order < 1:
            return f(x)
        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as tape:
            tape.watch(x)
            inner = self.derivative(f, x, order - 1)

        grad_list = tape.gradient(inner, x)
        grad_pair = grad_list[0] if grad_list else None
        store_dtype = self._get_gradient_dtype(x.dtype, self.forced_dtype)

        if grad_pair is None:
            return np.zeros_like(x, dtype=store_dtype)
        return grad_pair[0] + np.conj(grad_pair[1])

    def hessian(self, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
        """Computes the Hessian matrix of a scalar-valued function."""
        n = x.size
        hessian_dtype = self._get_gradient_dtype(x.dtype, self.forced_dtype)
        H = np.zeros((n, n), dtype=hessian_dtype)

        with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as g1:
            g1.watch(x)
            y = f(x)

        grad_f_list = g1.gradient(y, x)
        grad_f_pair = grad_f_list[0] if grad_f_list else None

        if not grad_f_pair:
            raise ValueError("Could not compute first gradient for Hessian calculation.")
        grad_f_combined = grad_f_pair[0] + np.conj(grad_f_pair[1])

        for i in range(n):
            def ith_comp_func(u):
                with GradientTape(persistent=False, watch_accessed_variables=self.watch_on_read, dtype=self.forced_dtype) as g2:
                    g2.watch(u)
                    y2 = f(u)
                grad2_list = g2.gradient(y2, u)
                grad2_pair = grad2_list[0] if grad2_list else None
                if not grad2_pair:
                    raise ValueError(f"Could not compute inner gradient for Hessian component {i}.")
                combined_inner = grad2_pair[0] + np.conj(grad2_pair[1])
                return combined_inner.flatten()[i]

            H[:, i] = GradientTape().derivative(ith_comp_func, x.copy(), 1).flatten()

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

        grad_list = g.gradient(y, x)
        grad_pair = grad_list[0] if grad_list else None
        if not grad_pair:
            raise ValueError("Analytical gradient could not be computed for check_gradient.")

        analytical_grad = grad_pair[0] + np.conj(grad_pair[1])
        numerical_grad = self._calculate_numerical_gradient(f, x, eps)
        return np.max(np.abs(numerical_grad - analytical_grad))
