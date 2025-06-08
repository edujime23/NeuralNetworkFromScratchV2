from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...types import Tensor, Variable
from .funcs import GRADIENTS, numerical_derivative
from ...queues import tapes


@dataclass
class OpNode:
    """
    Represents a node in the computation graph, recording a primitive operation.

    Stores information about the function executed, its inputs, keyword arguments,
    and the resulting Tensor. Maintains references to parent nodes for backpropagation.
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
        """Concise string representation of the OpNode."""
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
    Both fields are guaranteed to be Tensors.
    """

    holomorphic: Tensor
    antiholomorphic: Tensor


class GradientTape:
    """
    Records operations on Tensors/Variables to enable automatic differentiation.

    Builds a computation graph and performs reverse-mode gradient accumulation.
    Supports complex differentiation via separate slots for ∂/∂w and ∂/∂conj(w).
    """

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
            persistent: If True, allows multiple gradient() calls on the same tape.
            watch_accessed_variables: If True, auto-watch any Variable/Tensor when read.
            dtype: Force gradient computations to use specific dtype (e.g. np.float32 or np.complex128).
        """
        # Maps Tensor-id → OpNode
        self.result_to_node: dict[int, OpNode] = {}
        # Maps Tensor-id → Gradient (two Tensors)
        self.grads: dict[int, Gradient] = {}
        # Set of Tensor ids we explicitly watch
        self.watched: set[int] = set()

        self.persistent = persistent
        self._used = False
        self.watch_on_read = watch_accessed_variables
        self.forced_dtype = dtype
        self._hooks: dict[int, Callable[[Tensor], Tensor]] = {}

        # A flat list of all OpNodes in creation order
        self._nodes_in_order: list[OpNode] = []
        self._next_creation_index = 0

        # Cache seeds for ones and zeros so we don't re-allocate repeatedly
        self._ones_seed: Tensor | None = None
        self._zeros_seed: Tensor | None = None

        tapes.append(self)

    def __enter__(self) -> GradientTape:
        """Context manager entry—clear state if non-persistent."""
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
        """Context manager exit—remove tape if non-persistent."""
        if not self.persistent and self in tapes:
            tapes.remove(self)

    def delete(self) -> None:
        """Manually remove this tape from the global registry."""
        if self in tapes:
            tapes.remove(self)

    def watch(self, *objs: Tensor | Variable) -> None:
        """Explicitly mark Tensors/Variables for gradient tracking."""
        for obj in objs:
            if isinstance(obj, Variable):
                self._watch(obj.value)
            elif isinstance(obj, Tensor):
                self._watch(obj)

    def _watch(self, arr: Tensor) -> None:
        """Internal helper to add this Tensor's id to the watched set."""
        self.watched.add(id(arr))

    def stop_gradient(self, x: Tensor) -> Tensor:
        """Mark a Tensor so that no gradient flows through it."""
        x._stop_gradient = True
        return x

    @classmethod
    def primitive(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to register a function as a differentiable primitive.
        The function name is used as the key in GRADIENTS.
        """
        name = func.__name__
        cls._primitive_registry[name] = None
        return func

    @classmethod
    def def_grad(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to define a gradient function for a previously-registered primitive.
        The gradient function name must be '<primitive>_grad'.
        """
        target_name = func.__name__.replace("_grad", "")
        if target_name not in cls._primitive_registry:
            raise ValueError(
                f"Primitive '{target_name}' not registered before defining gradient."
            )
        GRADIENTS[target_name] = func
        return func

    def register_hook(
        self, var: Tensor | Variable, hook: Callable[[Tensor], Tensor]
    ) -> None:
        """
        Register a hook that transforms the accumulated gradient in backprop.
        The hook will be applied to both holomorphic and antiholomorphic slots.
        """
        arr = var.value if isinstance(var, Variable) else var
        self._hooks[id(arr)] = hook

    @staticmethod
    def _get_gradient_dtype(
        dtype: np.dtype, forced: np.dtype | None = None
    ) -> np.dtype:
        """
        Determine the dtype to use when allocating gradient Tensors.
        If forced is provided, always use that. Otherwise:
        - Integers → upcast to float64
        - Floats → same dtype
        - Complex → same dtype
        - Others → float64
        """
        if forced is not None:
            return forced

        if np.issubdtype(dtype, np.integer):
            return np.float64  # always upcast integers to float
        if np.issubdtype(dtype, np.complexfloating):
            return dtype  # keep complex dtype
        return dtype if np.issubdtype(dtype, np.floating) else np.float64

    def _normalize_inputs(
        self, inputs: tuple[Tensor | Variable, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Tensor, ...], dict[str, Any]]:
        """
        Convert any Variables in inputs/kwargs to their underlying Tensors.
        Returns (tuple_of_Tensors, dict_of_Tensors).
        """
        normalized_inputs: list[Tensor] = []
        for i in inputs:
            if isinstance(i, Variable):
                normalized_inputs.append(i.value)
            elif isinstance(i, Tensor):
                normalized_inputs.append(i)
            else:
                normalized_inputs.append(Tensor(i))
        normalized_kwargs: dict[str, Any] = {
            k: v.value if isinstance(v, Variable) else v for k, v in kwargs.items()
        }
        return tuple(normalized_inputs), normalized_kwargs

    def _create_and_link_op_node(
        self,
        func: Callable[..., Any],
        method: str,
        inputs: tuple[Tensor, ...],
        kwargs: dict[str, Any],
        result: Tensor,
    ) -> None:
        """
        Create an OpNode for this primitive call, link its parent nodes
        based on 'inputs', and store it in self.result_to_node.
        """
        # If result is not a Tensor or user flagged stop_gradient, do nothing
        if not isinstance(result, Tensor) or getattr(result, "_stop_gradient", False):
            return

        node = OpNode(
            func=func,
            method=method,
            inputs=inputs,
            kwargs=kwargs,
            result=result,
            creation_index=self._next_creation_index,
        )
        self._next_creation_index += 1

        # Link to parent nodes based on inputs
        for arr in inputs:
            parent_node = self.result_to_node.get(id(arr))
            if parent_node is not None:
                node.parents.append(parent_node)

        self.result_to_node[id(result)] = node
        self._nodes_in_order.append(node)

    def record(
        self,
        func: Callable[..., Any],
        method: str,
        inputs: tuple[Tensor | Variable, ...],
        kwargs: dict[str, Any],
        result: Tensor | Variable,
    ) -> None:
        """
        Record an operation in all active gradient tapes that are watching any input.
        Normalizes Variables to Tensors before linking.
        """
        # If result is a Variable, extract its .value; if stop_gradient on result, skip.
        if isinstance(result, Variable):
            result_tensor = result.value
            if getattr(result_tensor, "_stop_gradient", False):
                return
        elif isinstance(result, Tensor):
            result_tensor = result
            if getattr(result_tensor, "_stop_gradient", False):
                return
        else:
            return  # not tracking non-Tensor outputs

        # Normalize inputs to Tensors
        normalized_inputs, normalized_kwargs = self._normalize_inputs(inputs, kwargs)

        # For each active tape, if any input is watched, record the new node
        if tapes:
            if any(id(tensor) in (tape := tapes[-1]).watched for tensor in normalized_inputs):
                tape.watch(result_tensor, *normalized_inputs)
                tape._create_and_link_op_node(
                    func, method, normalized_inputs, normalized_kwargs, result_tensor
                )

    def _unbroadcast_gradient(self, grad: Tensor, original_shape: tuple[int, ...]) -> Tensor:
        """
            Reverse NumPy-style broadcasting so that `grad` ends up with exactly `original_shape`.
        """
        if grad.shape == original_shape:
            return grad

        if grad.ndim == 0 and original_shape:
            fill_value = grad.item()
            return Tensor(
                np.full(original_shape, fill_value, dtype=grad.dtype)
            )

        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)

        for axis, orig_dim in enumerate(original_shape):
            if orig_dim == 1 and grad.shape[axis] > 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad.reshape(original_shape)

    def _compute_raw_gradients(
        self, node: OpNode, grad_res: Gradient
    ) -> list[tuple[Tensor, Tensor]]:
        """
        Compute “raw” (holomorphic, antiholomorphic) gradients for a single OpNode.
        If an analytic gradient function is registered in GRADIENTS, call it.
        Otherwise, fall back to numerical derivatives.
        Returns a list of (gh, gah) for each input of this node.
        All returned objects are guaranteed to be Tensors.
        """
        name = node.func.__name__
        if grad_func := GRADIENTS.get(name) or GradientTape._primitive_registry.get(
            name
        ):
            # Try attached analytic gradient
            try:
                raw_grads = grad_func(
                    (grad_res.holomorphic, grad_res.antiholomorphic),
                    node.inputs,
                    **node.kwargs,
                )
            except TypeError:
                raw_grads = grad_func(
                    (grad_res.holomorphic, grad_res.antiholomorphic), node.inputs
                )
        else:
            # No analytic gradient: perform a numerical derivative
            warnings.warn(
                f"No analytical gradient for '{name}', using numerical approximation. "
                "Consider defining a gradient function for better performance and accuracy.",
                stacklevel=2,
            )
            approximations = numerical_derivative(
                node.func,
                node.inputs,
                node.kwargs,
                high_precision=True,
                max_order=8,
                verbose=False,
            )
            raw_grads = []
            for J_h, J_ah in approximations:
                # Each J_h, J_ah is a NumPy array (Jacobian) for holomorphic/antiholomorphic parts
                # We need to turn them into Tensors before dot-product
                J_h_t = Tensor(J_h)
                J_ah_t = Tensor(J_ah)
                gh = grad_res.holomorphic.dot(J_h_t) + grad_res.antiholomorphic.dot(
                    J_ah_t.conj()
                )
                gah = grad_res.holomorphic.dot(J_ah_t) + grad_res.antiholomorphic.dot(
                    J_h_t.conj()
                )
                raw_grads.append((gh, gah))
        # raw_grads is now a list of tuples, but each component might be a NumPy array or a Tensor.
        # We'll normalize in the next step.
        return raw_grads

    def _process_raw_gradients_format(
        self, raw_grads: Any, num_inputs: int
    ) -> tuple[tuple[Tensor | None, Tensor | None], ...]:
        """
        Normalize gradient format so that each input's gradient is a pair (gh, gah),
        both of which are Tensors (or None). Returns a tuple of length num_inputs.
        """
        # 1) If raw_grads is a single tuple with two Tensors, wrap in a one-element tuple
        if not isinstance(raw_grads, tuple) or not all(
            isinstance(x, tuple) for x in raw_grads
        ):
            # Could be (gh, gah) for single input
            if (
                num_inputs == 1
                and isinstance(raw_grads, tuple)
                and len(raw_grads) == 2
                and isinstance(raw_grads[0], (Tensor, np.ndarray))
            ):
                raw_grads = (raw_grads,)
            elif isinstance(raw_grads, list):
                raw_grads = tuple(raw_grads)
            else:
                # Unexpected format—wrap as (None, None) to skip
                warnings.warn(
                    f"Unexpected gradient format {type(raw_grads)}. Skipping these inputs.",
                    stacklevel=2,
                )
                raw_grads = tuple(((None, None),) * num_inputs)

        processed: list[tuple[Tensor | None, Tensor | None]] = []
        for grad_pair in raw_grads:
            if isinstance(grad_pair, tuple) and len(grad_pair) == 2:
                gh, gah = grad_pair
                # Wrap each in a Tensor if it's a raw NumPy array
                if isinstance(gh, np.ndarray):
                    gh = Tensor(gh)
                if isinstance(gah, np.ndarray):
                    gah = Tensor(gah)
                # If either is None, leave as None
                processed.append((gh, gah))
            elif isinstance(grad_pair, Tensor):
                # Single-component gradient → treat as holomorphic, zero-fill antiholo
                processed.append((grad_pair, np.zeros_like(grad_pair)))
            else:
                # Anything else → no gradient for that input
                processed.append((None, None))

        # If raw_grads had fewer entries than num_inputs, pad with (None,None)
        while len(processed) < num_inputs:
            processed.append((None, None))
        return tuple(processed)

    def _initialize_input_gradient(self, inp: Tensor) -> None:
        """
        Ensure that self.grads[id(inp)] exists, initializing both holomorphic and
        antiholomorphic to zeros with the correct dtype.
        """
        if id(inp) not in self.grads:
            store_dtype = self._get_gradient_dtype(inp.dtype, self.forced_dtype)
            zero_h = Tensor(np.zeros(inp.shape, dtype=store_dtype))
            zero_ah = Tensor(np.zeros(inp.shape, dtype=store_dtype))
            self.grads[id(inp)] = Gradient(holomorphic=zero_h, antiholomorphic=zero_ah)

    def _accumulate_and_apply_hook(
        self, inp: Tensor, g_h: Tensor, g_ah: Tensor
    ) -> None:
        """
        Add the newly computed (g_h, g_ah) to the existing slots in self.grads, then run any hook.
        Both g_h and g_ah are guaranteed to be Tensors.
        """
        entry = self.grads[id(inp)]
        entry.holomorphic = entry.holomorphic + g_h
        entry.antiholomorphic = entry.antiholomorphic + g_ah

        if hook := self._hooks.get(id(inp)):
            # Apply hook to both components (hook returns a Tensor)
            entry.holomorphic = hook(entry.holomorphic)
            entry.antiholomorphic = hook(entry.antiholomorphic)

    def _gradient_recursive(self, node: OpNode, grad_res: Gradient) -> None:
        """
        Recursively compute and propagate gradients from 'node.result' back to its inputs.
        'grad_res' is the Gradient(holomorphic, antiholomorphic) at node.result.
        """
        raw_grads_for_inputs = self._compute_raw_gradients(node, grad_res)
        processed = self._process_raw_gradients_format(
            raw_grads_for_inputs, len(node.inputs)
        )

        for i, inp in enumerate(node.inputs):
            if i >= len(processed):
                break
            gh, gah = processed[i]
            if gh is None and gah is None:
                continue

            # Force inp to be a Tensor (in case someone passed a NumPy array directly)
            if not isinstance(inp, Tensor):
                inp = Tensor(inp)

            # Reverse any broadcasting that happened during the forward pass
            gh = self._unbroadcast_gradient(gh, inp.shape) if gh is not None else None
            gah = (
                self._unbroadcast_gradient(gah, inp.shape) if gah is not None else None
            )

            # Initialize storage slots for this input if needed
            self._initialize_input_gradient(inp)

            # Accumulate into the global gradient table, applying any hooks
            if gh is not None and gah is not None:
                self._accumulate_and_apply_hook(inp, gh, gah)

    def _backpropagate(self, target_arr: Tensor) -> None:
        """
        Perform reverse-mode automatic differentiation from 'target_arr' downwards.
        Only nodes reachable from target_arr (via parent pointers) receive nonzero grads.
        """
        root_node = self.result_to_node.get(id(target_arr))
        if root_node is None:
            return

        # 1) Mark all reachable nodes using a DFS, tagging with a “stamp”
        stamp = GradientTape._visit_stamp_counter
        GradientTape._visit_stamp_counter += 1

        stack = [root_node]
        while stack:
            node = stack.pop()
            if node.last_visited == stamp:
                continue
            node.last_visited = stamp
            stack.extend(parent for parent in node.parents if parent.last_visited != stamp)
        # 2) Traverse nodes in reverse-creation order, propagating gradients
        for node in reversed(self._nodes_in_order):
            if node.last_visited != stamp:
                continue
            res_id = id(node.result)
            if res_id not in self.grads:
                continue
            grad_res = self.grads[res_id]
            self._gradient_recursive(node, grad_res)

    def _get_final_gradient_for_source(
        self, source_arr: Tensor, unconnected_gradients: str
    ) -> tuple[Tensor, Tensor] | None:
        """
        After backpropagation, extract (holomorphic, antiholomorphic) for source_arr.
        If no entry in self.grads and unconnected_gradients="zero", return zero Tensors.
        Otherwise return None.
        """
        grad_pair = self.grads.get(id(source_arr))
        if grad_pair is None:
            if unconnected_gradients == "zero":
                dtype_src = self._get_gradient_dtype(
                    source_arr.dtype, self.forced_dtype
                )
                zero_h = Tensor(np.zeros(source_arr.shape, dtype=dtype_src))
                zero_ah = Tensor(np.zeros(source_arr.shape, dtype=dtype_src))
                return (zero_h, zero_ah)
            return None
        return (grad_pair.holomorphic, grad_pair.antiholomorphic)

    def gradient(
        self,
        target: Tensor | Variable,
        sources: Tensor | Variable | Iterable[Tensor | Variable],
        output_gradients: Tensor | Variable | None = None,
        unconnected_gradients: str = "none",
    ) -> tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor] | None] | None:
        """
        Compute gradients of 'target' w.r.t. 'sources', returning (holomorphic, antiholomorphic) pairs.
        If multiple sources, returns a list of pairs (or None for unconnected).
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "GradientTape has already been used and is not persistent. "
                "Set `persistent=True` for multiple gradient() calls."
            )
        self._used = True

        # Extract raw Tensor for target
        target_arr = target.value if isinstance(target, Variable) else target
        if not isinstance(target_arr, Tensor):
            target_arr = Tensor(target_arr)

        # Normalize sources to raw Tensors
        is_single = not isinstance(sources, Iterable)
        sources_list = [sources] if is_single else list(sources)
        raw_sources: list[Tensor] = []
        for s in sources_list:
            if isinstance(s, Variable):
                raw_sources.append(s.value)
            elif isinstance(s, Tensor):
                raw_sources.append(s)
            else:
                raw_sources.append(Tensor(s))

        # Filter only those the user explicitly watched
        filtered_sources = [s for s in raw_sources if id(s) in self.watched]

        if not filtered_sources:
            # If “zero,” return zero-gradients for each requested source.
            if unconnected_gradients == "zero":
                result_list: list[tuple[Tensor, Tensor]] = []
                for s in raw_sources:
                    dtype_s = self._get_gradient_dtype(s.dtype, self.forced_dtype)
                    zero1 = Tensor(np.zeros(s.shape, dtype=dtype_s))
                    zero2 = Tensor(np.zeros(s.shape, dtype=dtype_s))
                    result_list.append((zero1, zero2))
                return result_list[0] if is_single else result_list
            # Otherwise return a list of None
            return None if is_single else [None] * len(raw_sources)

        # Determine initial gradient “seed” on target: either user-provided or all-ones
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

        # Clear any existing gradients, then initialize for filtered_sources
        self.grads.clear()
        for src in filtered_sources:
            dtype_src = self._get_gradient_dtype(src.dtype, self.forced_dtype)
            zero_h = Tensor(np.zeros(src.shape, dtype=dtype_src))
            zero_ah = Tensor(np.zeros(src.shape, dtype=dtype_src))
            self.grads[id(src)] = Gradient(holomorphic=zero_h, antiholomorphic=zero_ah)

        # Set the gradient at the target to (gh_init, gah_init)
        self.grads[id(target_arr)] = Gradient(
            holomorphic=gh_init, antiholomorphic=gah_init
        )

        # Backpropagate through the graph
        self._backpropagate(target_arr)

        # Gather final gradients for *all* requested sources (return None or a pair)
        final_list: list[tuple[Tensor, Tensor] | None] = []
        for s in raw_sources:
            grad_pair = self._get_final_gradient_for_source(s, unconnected_gradients)
            final_list.append(grad_pair)

        return final_list[0] if is_single else final_list

    def _grad_scalar_or_tensor(
        self,
        target_arr: Tensor,
        source_arr: Tensor,
        seed: Tensor,
        unconnected_gradients: str,
    ) -> tuple[Tensor, Tensor] | None:
        """
        Internal helper for JVP/Jacobian: compute the gradient of target_arr wrt source_arr,
        using 'seed' as ∂L/∂(target). Clears old grads, initializes the seed, backpropagates,
        and returns (holomorphic, antiholomorphic) pair for source_arr (or None).
        """
        self.grads.clear()
        dtype = self._get_gradient_dtype(target_arr.dtype, self.forced_dtype)
        zero_ah = Tensor(np.zeros_like(seed.data, dtype=dtype))
        self.grads[id(target_arr)] = Gradient(holomorphic=seed, antiholomorphic=zero_ah)

        self._initialize_input_gradient(source_arr)
        self._backpropagate(target_arr)
        return self._get_final_gradient_for_source(source_arr, unconnected_gradients)

    @staticmethod
    def _jac_row_worker(args: tuple[Any, ...]) -> Tensor:
        """
        Worker function for parallel Jacobian row computation. Should only see numpy-pickleable args.
        Returns a flattened 1D numpy array representing one row of the Jacobian.
        """
        (
            target_np,  # raw NumPy array (not a Tensor) for target
            source_np,  # raw NumPy array for source
            idx,  # which row in flattened target to compute
            shape_target,
            shape_source,
            og_np,  # raw numpy output_gradients, or None
            unconnected_gradients,
            forced_dtype,
        ) = args

        # Create a unit-vector “seed” in NumPy, then wrap as a Tensor
        flat_t_seed = np.zeros(shape_target, dtype=forced_dtype).reshape(-1)
        flat_t_seed[idx] = 1.0
        seed_np = flat_t_seed.reshape(shape_target)
        seed = Tensor(seed_np)

        # If user provided output_gradients, multiply
        if og_np is not None:
            seed = seed * Tensor(og_np)

        # Re-wrap the raw NumPy buffers as Tensors for the inner tape
        target_tensor = Tensor(target_np)
        source_tensor = Tensor(source_np)

        # Compute the gradient for this single scalar component
        with GradientTape(
            persistent=False, watch_accessed_variables=False, dtype=forced_dtype
        ) as tape:
            tape.watch(source_tensor)
            # We want the gradient of target_tensor (which is a function of source_tensor)
            # at “seed.”  Internally, we only need to call backprop from that one scalar component.
            grad_pair = tape._grad_scalar_or_tensor(
                target_tensor, source_tensor, seed, unconnected_gradients
            )

        if grad_pair is None:
            return np.zeros(np.prod(shape_source), dtype=forced_dtype)

        # Combine the two parts into a single complex (or real) array
        combined = grad_pair[0] + grad_pair[1].conj()
        combined_np = combined.data  # convert back to a raw NumPy array
        if np.isrealobj(combined_np):
            return combined_np.reshape(-1)
        # If complex but imag is effectively zero, return real only
        if np.allclose(combined_np.imag, 0):
            return combined_np.real.reshape(-1)
        return combined_np.reshape(-1)

    def jacobian(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        unconnected_gradients: str = "none",
        output_gradients: Tensor | Variable | None = None,
    ) -> Tensor:
        """
        Compute full Jacobian matrix using parallel processes.
        Returns a Tensor of shape (target.shape + source.shape).
        """
        if self._used and not self.persistent:
            raise RuntimeError(
                "GradientTape has already been used and is not persistent. "
                "Set `persistent=True` during initialization for reuse."
            )
        self._used = True

        target_np = self._pickle_tensor(target)
        source_np = self._pickle_tensor(source)
        flat_target_size = target_np.size
        shape_target = target_np.shape
        shape_source = source_np.shape
        forced_dtype = self._get_gradient_dtype(source_np.dtype, self.forced_dtype)

        og_np = None
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

        # Build argument list for each row
        args_list: list[tuple[Any, ...]] = []
        args_list.extend(
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
        )
        # Map in parallel (each worker pickles the raw NumPy arrays once)
        with mp.Pool() as pool:
            rows = pool.map(GradientTape._jac_row_worker, args_list)

        jac_np = np.stack(rows, axis=0)  # shape = (flat_target_size, flat_source_size)
        full_shape = shape_target + shape_source
        jac_np = jac_np.reshape(full_shape)
        return Tensor(jac_np)

    # TODO Rename this here and in `jacobian`
    def _pickle_tensor(self, arg0):
        # Extract raw NumPy buffers from Tensors/Variables for pickling to workers
        target_arr = arg0.value if isinstance(arg0, Variable) else arg0
        if not isinstance(target_arr, Tensor):
            target_arr = Tensor(target_arr)
        return target_arr.data

    def _prepare_target_function(
        self, target: Tensor | Variable
    ) -> Callable[[Tensor], Tensor]:
        """
        Return a function f(x) that, when given a Tensor x, returns target.value (if Variable)
        or target (if already a Tensor). Used by jvp and vjp to isolate the “f” of source → target.
        """
        if isinstance(target, Variable):
            return lambda x: target.value
        else:
            return lambda x: target

    def _get_combined_gradient_from_tape_output(
        self, grad_output_from_tape: Any, source_arr: Tensor
    ) -> Tensor:
        """
        Given the raw output of tape.gradient(...), combine (holomorphic + conj(antiholomorphic))
        into a single Tensor. If None, return zero Tensor.
        """
        # grad_output_from_tape might be a list, a single pair, or None
        grad_pair = None
        if isinstance(grad_output_from_tape, list):
            if grad_output_from_tape:
                grad_pair = grad_output_from_tape[0]
        else:
            grad_pair = grad_output_from_tape

        dtype_src = self._get_gradient_dtype(source_arr.dtype, self.forced_dtype)
        if grad_pair is None:
            return Tensor(np.zeros_like(source_arr.data, dtype=dtype_src))
        gh, gah = grad_pair
        # Wrap raw arrays if needed
        if isinstance(gh, np.ndarray):
            gh = Tensor(gh)
        if isinstance(gah, np.ndarray):
            gah = Tensor(gah)
        return gh + gah.conj()

    def jvp(
        self,
        target: Tensor | Variable,
        source: Tensor | Variable,
        vector: Tensor | Variable,
    ) -> Tensor:
        """
        Jacobian-vector product (forward-mode AD). For scalar targets, this is the directional derivative.
        For vectorial targets, warns and computes a VJP-like result instead.
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
        ) as inner_tape:
            inner_tape.watch(source_arr)
            y = f(source_arr)

        gy_result = inner_tape.gradient(y, source_arr)
        combined_grad_y = self._get_combined_gradient_from_tape_output(
            gy_result, source_arr
        )

        # If y is a scalar Tensor, we can form uk·(∂f/∂x)
        if isinstance(y, Tensor) and (y.ndim == 0 or y.size == 1):
            return (combined_grad_y * vec_arr).sum()

        warnings.warn(
            "JVP for vector-valued targets is not implemented as a true Jacobian-vector product. "
            "This call returns (∂f/∂x)^T @ vector instead. "
            "To compute an explicit JVP, use jacobian() and multiply.",
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
        """
        Vector-Jacobian product (reverse-mode AD). Returns Vector^T @ (∂f/∂x).
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
        Compute higher-order derivative d^order f / dx^order recursively.
        Returns a Tensor of the same shape as x (for scalar functions) or shape+shape for vector functions.
        """
        x_arr = x.value if isinstance(x, Variable) else x
        if not isinstance(x_arr, Tensor):
            x_arr = Tensor(x_arr)

        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        if order == 0:
            return f(x_arr)

        inner = self.derivative(f, x_arr, order - 1)
        with GradientTape(
            persistent=False,
            watch_accessed_variables=self.watch_on_read,
            dtype=self.forced_dtype,
        ) as tape:
            tape.watch(x_arr)
            grad_pair = tape.gradient(inner, x_arr)
        return self._get_combined_gradient_from_tape_output(grad_pair, x_arr)

    def hessian(self, f: Callable[[Tensor], Tensor], x: Tensor | Variable) -> Tensor:
        """
        Compute the Hessian matrix ∂^2 f / ∂x_i ∂x_j using parallel processes.
        Returns a Tensor of shape (n, n) where n = x.size.
        """
        x_arr = x.value if isinstance(x, Variable) else x
        if not isinstance(x_arr, Tensor):
            x_arr = Tensor(x_arr)

        n = x_arr.size
        hess_dtype = self._get_gradient_dtype(x_arr.dtype, self.forced_dtype)

        # 1) Compute first-order gradient g_i = ∂f/∂x_i
        with GradientTape(
            persistent=False,
            watch_accessed_variables=self.watch_on_read,
            dtype=self.forced_dtype,
        ) as g1:
            g1.watch(x_arr)
            y = f(x_arr)
            grad_f_pair = g1.gradient(y, x_arr)

        if grad_f_pair is None:
            raise ValueError(
                "Could not compute first gradient for Hessian. "
                "Ensure f is differentiable wrt x."
            )

        # Worker function to compute the second derivative for one component
        def _hess_worker(args: tuple[Any, ...]) -> np.ndarray:
            (i, x_np, f_np, watch_on_read, forced_dtype_np) = args
            # x_np is raw NumPy buffer for x_arr, f_np is raw NumPy buffer for f(x)
            seed_i_np = np.zeros_like(f_np, dtype=forced_dtype_np).reshape(-1)
            seed_i_np[i] = 1.0
            seed_i = Tensor(seed_i_np.reshape(f_np.shape))

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
            if grad_inner_pair is None:
                return np.zeros(n, dtype=forced_dtype_np)

            combined_inner = grad_inner_pair[0] + grad_inner_pair[1].conj()
            return combined_inner.data.reshape(-1)

        # Prepare raw buffers for parallelization
        x_np = x_arr.data
        f_np = f(x_arr).data
        args_list = [(i, x_np, f_np, self.watch_on_read, hess_dtype) for i in range(n)]

        with mp.Pool() as pool:
            columns = pool.map(_hess_worker, args_list)

        H_np = np.zeros((n, n), dtype=hess_dtype)
        for i in range(n):
            H_np[:, i] = columns[i]
        return Tensor(H_np)

    def print_graph(self, target: Tensor | Variable) -> None:
        """
        Print the computation graph backward from the target node, using DFS.
        """
        target_tensor = target.value if isinstance(target, Variable) else target
        if not isinstance(target_tensor, Tensor):
            target_tensor = Tensor(target_tensor)

        visited: set[int] = set()

        def _dfs(node: Any, indent: int = 0) -> None:
            if node is None or id(node) in visited:
                return
            visited.add(id(node))

            pad = "  " * indent
            if isinstance(node, Tensor):
                print(
                    f"{pad}Tensor(ID={id(node)}, shape={node.shape}, dtype={node.dtype})"
                )
                producer = self.result_to_node.get(id(node))
                if producer:
                    _dfs(producer, indent + 1)

            elif isinstance(node, OpNode):
                print(
                    f"{pad}OpNode(func={node.func.__name__}, "
                    f"result_id={id(node.result)}, "
                    f"shape={node.result.shape}, dtype={node.result.dtype})"
                )
                for parent_tensor in node.inputs:
                    _dfs(parent_tensor, indent + 1)

            else:
                print(f"{pad}Unknown node {type(node)} (ID={id(node)})")

        root = self.result_to_node.get(id(target_tensor))
        if root is None:
            if id(target_tensor) in self.watched:
                print(f"Graph for watched input Tensor (ID={id(target_tensor)}):")
                consumers = [
                    opn
                    for opn in self._nodes_in_order
                    if any(id(inp) == id(target_tensor) for inp in opn.inputs)
                ]
                if not consumers:
                    print("  (no consumers)")
                else:
                    for c in consumers:
                        _dfs(c, 2)
            else:
                print(f"Target {id(target_tensor)} not found in graph.")
            return

        print(f"Graph for result Tensor (ID={id(target_tensor)}):")
        _dfs(root, 0)
