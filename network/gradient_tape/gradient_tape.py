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

    def __repr__(self):
        return f"OpNode(func={self.func.__name__}, result_id={id(self.result)})"


@dataclass
class Gradient:
    holomorphic: np.ndarray
    antiholomorphic: np.ndarray


class GradientTape:
    _GRADIENTS_TAPES: List['GradientTape'] = []

    def __init__(
        self,
        persistent: Optional[bool] = False,
        watch_accessed_variables: bool = False,
    ) -> None:
        """
        If persistent=False (default), the tape can only be used once, and then its resources are freed.
        If persistent=True, you may call gradient()/jacobian() multiple times, but must delete the tape manually to free memory.
        watch_accessed_variables is reserved for compatibility; here, user must still call watch() manually on NumPy arrays.
        """
        self.result_to_node: Dict[int, OpNode] = {}
        self.grads: Dict[int, Gradient] = {}
        self.watched: set[int] = set()
        self.persistent = persistent
        self._used = False  # to enforce single‐use semantics if persistent=False

    def __enter__(self) -> 'GradientTape':
        self.result_to_node.clear()
        self.grads.clear()
        self.watched.clear()
        self._used = False
        GradientTape._GRADIENTS_TAPES.append(self)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if not self.persistent and self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def watch(self, *objs: Tuple[np.ndarray, ...]) -> None:
        """
        Manually mark one or more NumPy arrays to be watched by this tape.
        """
        for o in objs:
            if not isinstance(o, (np.ndarray, np.generic)):
                warnings.warn(f"Cannot watch object of type {type(o)}. Only numpy arrays/numbers are supported.")
                continue
            self._watch(o)

    def _watch(self, obj: np.ndarray) -> None:
        """
        Internal helper to add to watched set.
        """
        self.watched.add(id(obj))

    def record(
        self,
        func: Callable,
        method: str,
        inputs: Tuple[np.ndarray, ...],
        kwargs: Dict[str, Any],
        result: np.ndarray,
    ) -> None:
        """
        Record an operation for all active tapes. If any input was being watched, we watch the output.
        If result has attribute _stop_gradient=True, we skip recording that node entirely.
        """
        # If user explicitly marked the result to stop gradients, skip.
        if hasattr(result, "_stop_gradient") and getattr(result, "_stop_gradient", False):
            return

        # For each active tape, if any input is watched by that tape, record a node.
        for tape in list(GradientTape._GRADIENTS_TAPES):
            if any(id(i) in tape.watched for i in inputs):
                tape._watch(result)
                node = OpNode(func, method, inputs, kwargs, result)

                # link parents: any input that itself was produced by a recorded node
                for inp in inputs:
                    parent = tape.result_to_node.get(id(inp))
                    if parent is not None:
                        node.parents.append(parent)

                tape.result_to_node[id(result)] = node

    def gradient(
        self,
        target: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...], Dict[Any, np.ndarray]],
        sources: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...], Dict[Any, np.ndarray]],
        output_gradients: Optional[Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]] = None,
        unconnected_gradients: str = "none",  # "none" or "zero"
    ) -> Union[
        None,
        np.ndarray,
        Tuple[np.ndarray, ...],
        List[np.ndarray],
        Dict[Any, Union[None, np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]],
    ]:
        """
        Compute gradients of `target` w.r.t. `sources`.
        - output_gradients: if provided, must have same shape as target; used as the initial seed (instead of ones).
        - unconnected_gradients: if "none", return None for any source that is not connected to target; if "zero", return zero array of same shape.
        """
        if self._used and not self.persistent:
            raise RuntimeError("GradientTape has already been used and is not persistent.")
        self._used = True

        # Recursive handling for nested structures
        if isinstance(target, (list, tuple)):
            return [self.gradient(t, sources, output_gradients, unconnected_gradients) for t in target]

        if isinstance(target, dict):
            return {k: self.gradient(v, sources, output_gradients, unconnected_gradients) for k, v in target.items()}

        if isinstance(sources, dict):
            return {k: self.gradient(target, v, output_gradients, unconnected_gradients) for k, v in sources.items()}

        if isinstance(sources, (list, tuple)):
            return [self.gradient(target, s, output_gradients[i] if isinstance(output_gradients, (list, tuple)) else output_gradients, unconnected_gradients) for i, s in enumerate(sources)]

        # At this point, target and source are both single NumPy arrays
        return self._grad_scalar_or_tensor(target, sources, output_gradients, unconnected_gradients)

    def _grad_scalar_or_tensor(
        self,
        target: np.ndarray,
        source: np.ndarray,
        output_gradient: Optional[np.ndarray],
        unconnected_gradients: str,
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute ∂target/∂source. Returns:
         - None (if unconnected and unconnected_gradients="none")
         - zero-array (if unconnected and unconnected_gradients="zero")
         - For real source: a single ndarray
         - For complex source: a tuple (holomorphic, antiholomorphic)
        """
        if id(source) not in self.watched:
            return np.zeros_like(source) if unconnected_gradients == "zero" else None
        # Initialize gradient of target w.r.t itself
        if output_gradient is not None:
            if output_gradient.shape != target.shape:
                raise ValueError("output_gradients must have same shape as target")
            gh_init = output_gradient
            gah_init = np.zeros_like(output_gradient)
        else:
            gh_init = np.ones_like(target)
            gah_init = np.zeros_like(target)

        self.grads[id(target)] = Gradient(holomorphic=gh_init.copy(), antiholomorphic=gah_init.copy())

        # (Optional) Prune graph: mark only nodes reachable from target
        visited_nodes = set()
        stack = [self.result_to_node.get(id(target))]
        while stack:
            node = stack.pop()
            if node is None or id(node.result) in visited_nodes:
                continue
            visited_nodes.add(id(node.result))
            stack.extend(iter(node.parents))
        # Remove any nodes not in visited_nodes
        for key in list(self.result_to_node.keys()):
            if key not in visited_nodes:
                del self.result_to_node[key]

        # Backward traversal
        visited = set()
        stack = [self.result_to_node.get(id(target))]
        while stack:
            node = stack.pop()
            if node is None or id(node.result) in visited:
                continue
            visited.add(id(node.result))

            grad_res = self.grads.get(id(node.result), Gradient(
                holomorphic=np.zeros_like(node.result),
                antiholomorphic=np.zeros_like(node.result),
            ))

            # Find analytic gradient function
            dfunc = GRADIENTS.get(node.func.__name__)
            if not dfunc:
                if aliases := getattr(node.func, "__aliases__", None):
                    for alias in aliases:
                        dfunc = GRADIENTS.get(alias)
                        if dfunc:
                            break

            if dfunc:
                try:
                    raw_grads = dfunc((grad_res.holomorphic, grad_res.antiholomorphic), node.inputs, **node.kwargs)
                except TypeError:
                    raw_grads = dfunc((grad_res.holomorphic, grad_res.antiholomorphic), node.inputs)
            else:
                warnings.warn(f"No gradient function for '{node.func.__name__}'. Using numerical approximation.")
                approx = numerical_derivative(node.func, node.inputs, node.kwargs)
                # Convert to (holomorphic, antiholomorphic) pairs
                raw_grads = []
                raw_grads.extend(
                    (
                        gh * grad_res.holomorphic + gah * grad_res.antiholomorphic,
                        np.zeros_like(gh),
                    )
                    for gh, gah in approx
                )
            # Distribute to inputs, handling broadcasting
            for inp, grad_pair in zip(node.inputs, raw_grads):
                if isinstance(grad_pair, tuple):
                    g_h, g_ah = grad_pair
                else:
                    g_h = grad_pair
                    g_ah = np.zeros_like(g_h)

                # If shapes mismatch due to broadcasting, sum over broadcasted dims
                if g_h.shape != inp.shape:
                    g_h = self._unbroadcast(g_h, inp.shape)
                if g_ah.shape != inp.shape:
                    g_ah = self._unbroadcast(g_ah, inp.shape)

                if id(inp) not in self.grads:
                    self.grads[id(inp)] = Gradient(
                        holomorphic=np.zeros_like(inp),
                        antiholomorphic=np.zeros_like(inp),
                    )
                self.grads[id(inp)].holomorphic += g_h
                self.grads[id(inp)].antiholomorphic += g_ah

            # Proceed upward
            stack.extend(iter(node.parents))
        grad = self.grads.get(id(source))
        if grad is None:
            return np.zeros_like(source) if unconnected_gradients == "zero" else None
        if np.isrealobj(source):
            # Return a single real-valued gradient array
            return np.real(grad.holomorphic + np.conj(grad.antiholomorphic))
        # For complex inputs, return (holomorphic, antiholomorphic)
        return (grad.holomorphic, grad.antiholomorphic)

    def jacobian(
        self,
        target: np.ndarray,
        source: np.ndarray,
        unconnected_gradients: str = "none",
        output_gradients: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute full Jacobian of `target` w.r.t. `source`.
        Returns an array of shape `target.shape + source.shape`.
        """
        if self._used and not self.persistent:
            raise RuntimeError("GradientTape has already been used and is not persistent.")
        self._used = True

        # Flatten target to a vector of size N
        flat_t = target.reshape(-1)
        jac_list = []

        # For each scalar element in flat_t, compute gradient w.r.t. source
        for idx in range(flat_t.shape[0]):
            # Create a seed array with 1 at position idx, 0 elsewhere
            seed = np.zeros_like(flat_t)
            seed[idx] = 1.0
            seed = seed.reshape(target.shape)

            # If overall output_gradients is given, multiply by that
            seed_to_use = seed
            if output_gradients is not None:
                if output_gradients.shape != target.shape:
                    raise ValueError("output_gradients must have same shape as target")
                seed_to_use = seed * output_gradients

            # Reset internal state except watched variables
            self.grads.clear()
            # Reuse the same graph: call _grad_scalar_or_tensor with this seed
            grad_wrt_src = self._grad_scalar_or_tensor(target, source, seed_to_use, unconnected_gradients)
            if grad_wrt_src is None:
                jac_list.append(np.zeros_like(source).reshape(-1))
            elif isinstance(grad_wrt_src, tuple):
                # Complex-valued: only holomorphic part for Jacobian
                jac_list.append(grad_wrt_src[0].reshape(-1))
            else:
                jac_list.append(grad_wrt_src.reshape(-1))

        # Stack to form shape (N, M) where N = target.size, M = source.size
        mat = np.stack(jac_list, axis=0)
        # Reshape to target.shape + source.shape
        return mat.reshape(target.shape + source.shape)

    def _unbroadcast(self, grad: np.ndarray, shape_inp: Tuple[int, ...]) -> np.ndarray:
        """
        Sum over broadcasted dimensions to make `grad` match `shape_inp`.
        """
        if grad.shape == shape_inp:
            return grad
        # Align rightmost dimensions
        ndim_diff = len(grad.shape) - len(shape_inp)
        # For leading extra dims, sum them out
        for _ in range(ndim_diff):
            grad = grad.sum(axis=0)
        # Now same number of dims: for any axis where shape_inp is 1 and grad is >1, sum over that axis
        for axis, (dim_grad, dim_inp) in enumerate(zip(grad.shape, shape_inp)):
            if dim_inp == 1 and dim_grad > 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def reset(self, watched: bool = True) -> None:
        """
        Clear all recorded nodes and gradients. If watched=True, also clear the watched set.
        """
        self.result_to_node.clear()
        self.grads.clear()
        self._used = False
        if watched:
            self.watched.clear()

    def delete(self) -> None:
        """
        Explicitly remove this tape from the active stack, if present.
        """
        if self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def __del__(self):
        if self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)


def stop_gradient(x: np.ndarray) -> np.ndarray:
    """
    Mark `x` so that any subsequent operations producing a result based on x will not record a gradient.
    Use this to block gradient flow through `x`.
    """
    # Create a view of x (so id is different, but underlying data same), and tag it.
    y = x.view()
    setattr(y, "_stop_gradient", True)
    return y
