# autodiff/tape/tape_core.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...queues import tapes  # Assuming tapes is a global list of active tapes
from ...types import Tensor, Variable  # Assuming Tensor and Variable are defined
from .types import Gradient, OpNode

# Global counter for graph traversal stamps
_visit_stamp_counter: int = 1


class GradientTapeCore:
    """
    Core functionalities for recording operations and managing the computation graph.
    This class handles graph building, state management, and utility methods.
    It does NOT contain the public gradient computation methods (like .gradient, .jacobian, etc.).
    """

    def __init__(
        self,
        persistent: bool = False,
        watch_accessed_variables: bool = False,
        dtype: np.dtype | None = None,
    ) -> None:
        """
        Initialize a new gradient tape.

        Args:
            persistent: If True, the tape's recorded operations and accumulated
                        gradients persist after a `gradient()` call, allowing
                        multiple gradient computations on the same tape. If False,
                        the tape is cleared after the first `gradient()` call.
            watch_accessed_variables: If True, any `Variable` or `Tensor` that is
                                      accessed (read) during the forward pass within
                                      this tape's context will be automatically
                                      marked for gradient tracking.
            dtype: An optional `numpy.dtype` to force all gradient computations
                   to use this specific data type (e.g., `np.float32`, `np.complex128`).
                   If None, dtype is inferred based on the input Tensors.
        """
        self.result_to_node: dict[int, OpNode] = {}
        self.grads: dict[int, Gradient] = {}  # Accumulated gradients
        self.watched: set[int] = set()  # Set of Tensor IDs explicitly watched

        self.persistent = persistent
        self._used: bool = False  # Flag to track if `gradient()` has been called
        self.watch_on_read = watch_accessed_variables
        self.forced_dtype = dtype
        self._hooks: dict[int, Callable[[Tensor], Tensor]] = {}

        self._nodes_in_order: list[OpNode] = []
        self._next_creation_index: int = 0

        self._ones_seed: Tensor | None = None
        self._zeros_seed: Tensor | None = None

        tapes.append(self)

    def __enter__(self) -> "GradientTapeCore":
        """
        Context manager entry point. Clears state if not persistent.
        """
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
        """
        Context manager exit point. Removes tape from global registry if not persistent.
        """
        if not self.persistent and self in tapes:
            tapes.remove(self)

    def delete(self) -> None:
        """Manually removes this tape from the global registry of active tapes."""
        if self in tapes:
            tapes.remove(self)

    def watch(self, *objs: Tensor | Variable) -> None:
        """Explicitly marks one or more Tensors or Variables for gradient tracking."""
        for obj in objs:
            if isinstance(obj, Variable):
                self._watch(obj.value)
            elif isinstance(obj, Tensor):
                self._watch(obj)

    def _watch(self, arr: Tensor) -> None:
        """Internal helper to add a Tensor's ID to the set of watched Tensors."""
        self.watched.add(id(arr))

    def stop_gradient(self, x: Tensor) -> Tensor:
        """Marks a Tensor so that no gradient flows through it during backpropagation."""
        x._stop_gradient = True
        return x

    def register_hook(
        self, var: Tensor | Variable, hook: Callable[[Tensor], Tensor]
    ) -> None:
        """
        Registers a hook function that transforms the accumulated gradient
        for a specific Tensor or Variable during backpropagation.
        """
        arr = var.value if isinstance(var, Variable) else var
        self._hooks[id(arr)] = hook

    @staticmethod
    def _get_gradient_dtype(
        dtype: np.dtype, forced: np.dtype | None = None
    ) -> np.dtype:
        """
        Determines the appropriate `numpy.dtype` for allocating gradient Tensors.
        """
        if forced is not None:
            return forced
        if np.issubdtype(dtype, np.integer):
            return np.float64
        if np.issubdtype(dtype, np.complexfloating):
            return dtype
        return dtype if np.issubdtype(dtype, np.floating) else np.float64

    def _normalize_inputs(
        self, inputs: tuple[Tensor | Variable, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Tensor, ...], dict[str, Any]]:
        """
        Converts any `Variable` objects within the inputs and keyword arguments
        to their underlying `Tensor` values.
        """
        normalized_inputs: list[Tensor] = []
        for i in inputs:
            if isinstance(i, Variable):
                normalized_inputs.append(i.value)
            elif isinstance(i, Tensor):
                normalized_inputs.append(i)
            else:
                normalized_inputs.append(Tensor(i))  # Attempt conversion

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
        Creates an `OpNode` for a primitive operation, links it to its parent nodes
        in the computation graph, and stores it in the tape's internal structures.
        """
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
        Records an operation in all currently active gradient tapes that are
        "watching" any of the input Tensors.
        """
        result_tensor: Tensor | None = None
        if isinstance(result, Variable):
            result_tensor = result.value
        elif isinstance(result, Tensor):
            result_tensor = result

        if result_tensor is None or getattr(result_tensor, "_stop_gradient", False):
            return

        normalized_inputs, normalized_kwargs = self._normalize_inputs(inputs, kwargs)

        for tape in tapes:  # tapes is a global list of active tapes
            if any(id(tensor) in tape.watched for tensor in normalized_inputs):
                tape.watch(result_tensor, *normalized_inputs)
                tape._create_and_link_op_node(
                    func, method, normalized_inputs, normalized_kwargs, result_tensor
                )

    def _unbroadcast_gradient(
        self, grad: Tensor, original_shape: tuple[int, ...]
    ) -> Tensor:
        """
        Reverses NumPy-style broadcasting on a gradient Tensor.
        """
        if grad.shape == original_shape:
            return grad

        if grad.ndim == 0 and original_shape:
            return Tensor(np.full(original_shape, grad.item(), dtype=grad.dtype))

        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)

        for axis, orig_dim in enumerate(original_shape):
            if orig_dim == 1 and grad.shape[axis] > 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad.reshape(original_shape)

    def _initialize_input_gradient(self, inp: Tensor) -> None:
        """
        Ensures that storage slots for the gradient of `inp` exist in `self.grads`.
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
        Adds the newly computed gradient components to existing ones and applies hook.
        """
        entry = self.grads[id(inp)]
        entry.holomorphic = entry.holomorphic + g_h
        entry.antiholomorphic = entry.antiholomorphic + g_ah

        if hook := self._hooks.get(id(inp)):
            entry.holomorphic = hook(entry.holomorphic)
            entry.antiholomorphic = hook(entry.antiholomorphic)

    def _extract_raw_numpy_data(self, obj: Tensor | Variable) -> np.ndarray:
        """
        Helper method to extract the raw NumPy array data from a Tensor or Variable.
        """
        arr = obj.value if isinstance(obj, Variable) else obj
        if not isinstance(arr, Tensor):
            arr = Tensor(arr)
        return arr.data

    def _prepare_target_function(
        self, target: Tensor | Variable
    ) -> Callable[[Tensor], Tensor]:
        """
        Returns a function `f(x)` that returns the value of the `target` Tensor.
        """
        if isinstance(target, Variable):
            return lambda _: target.value
        else:
            return lambda _: target

    def _get_combined_gradient_from_tape_output(
        self,
        grad_output_from_tape: (
            tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor] | None] | None
        ),
        source_arr: Tensor,
    ) -> Tensor:
        """
        Combines the holomorphic and anti-holomorphic gradient components from
        `tape.gradient` output into a single `Tensor`.
        """
        grad_pair: tuple[Tensor | None, Tensor | None] | None = None

        if isinstance(grad_output_from_tape, list):
            if grad_output_from_tape:
                grad_pair = grad_output_from_tape[0]
        else:
            grad_pair = grad_output_from_tape

        dtype_src = self._get_gradient_dtype(source_arr.dtype, self.forced_dtype)

        if grad_pair is None or grad_pair[0] is None or grad_pair[1] is None:
            return Tensor(np.zeros_like(source_arr.data, dtype=dtype_src))

        gh, gah = grad_pair
        if isinstance(gh, np.ndarray):
            gh = Tensor(gh)
        if isinstance(gah, np.ndarray):
            gah = Tensor(gah)

        return gh + gah.conj()

    def print_graph(self, target: Tensor | Variable) -> None:
        """
        Prints a text-based representation of the computation graph.
        """
        target_tensor = target.value if isinstance(target, Variable) else target
        if not isinstance(target_tensor, Tensor):
            target_tensor = Tensor(target_tensor)

        visited_node_ids: set[int] = set()

        def _dfs_print(node: Any, indent: int = 0) -> None:
            if node is None or id(node) in visited_node_ids:
                return
            visited_node_ids.add(id(node))

            pad = "  " * indent

            if isinstance(node, Tensor):
                print(
                    f"{pad}Tensor(ID={id(node)}, shape={node.shape}, dtype={node.dtype})"
                )
                producer_node = self.result_to_node.get(id(node))
                if producer_node:
                    _dfs_print(producer_node, indent + 1)

            elif isinstance(node, OpNode):
                print(
                    f"{pad}OpNode(func={node.func.__name__}, "
                    f"method='{node.method}', "
                    f"result_id={id(node.result)}, "
                    f"shape={node.result.shape}, dtype={node.result.dtype})"
                )
                for parent_tensor in node.inputs:
                    _dfs_print(parent_tensor, indent + 1)

            else:
                print(f"{pad}Unknown node type {type(node)} (ID={id(node)})")

        root_op_node = self.result_to_node.get(id(target_tensor))

        if root_op_node is None:
            if id(target_tensor) in self.watched:
                print(
                    f"Target Tensor (ID={id(target_tensor)}) is a watched input. "
                    "Searching for its consumers:"
                )
                consumers = [
                    opn
                    for opn in self._nodes_in_order
                    if any(id(inp) == id(target_tensor) for inp in opn.inputs)
                ]
                if not consumers:
                    print("  (no operations consumed this input)")
                else:
                    for consumer_node in consumers:
                        print("  Consumed by:")
                        _dfs_print(consumer_node, 2)
            else:
                print(
                    f"Target Tensor (ID={id(target_tensor)}) not found in the computation graph "
                    "and was not explicitly watched."
                )
            return

        print(f"Computation Graph for target Tensor (ID={id(target_tensor)}):")
        _dfs_print(root_op_node, 0)
