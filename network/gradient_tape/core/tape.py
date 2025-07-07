from __future__ import annotations

import warnings

import numpy as np

from network.gradient_tape.types import Gradient, OpNode
from network.queues.tapes import tapes
from network.types.tensor import Tensor
from network.types.variable import Variable

from .registry import registry


class GradientTapeCore:
    """The core machinery for graph recording and backpropagation."""

    def __init__(self, persistent: bool = False, dtype: np.dtype | None = None):
        self.persistent = persistent
        self.forced_dtype = dtype
        self._watched: set[int] = set()
        self._grads: dict[int, Gradient] = {}
        self._nodes: dict[int, OpNode] = {}
        self._is_used: bool = False

    def _watch(self, *tensors: Tensor):
        """Internal watch method."""
        for t in tensors:
            self._watched.add(id(t))

    @staticmethod
    def _record_operation(op_name: str, inputs: tuple, kwargs: dict, result: Tensor):
        if not tapes:
            return

        tape = tapes[-1]

        normalized_inputs = tuple(
            inp.value if isinstance(inp, Variable) else inp for inp in inputs
        )

        should_record = False
        for inp_tensor in normalized_inputs:
            if id(inp_tensor) in tape._watched or id(inp_tensor) in tape._nodes:
                should_record = True
                break

        normalized_kwargs_tensors = []
        normalized_kwargs_tensors.extend(
            v.value if isinstance(v, Variable) else v
            for v in kwargs.values()
            if isinstance(v, (Tensor, Variable))
        )
        for kwarg_tensor in normalized_kwargs_tensors:
            if id(kwarg_tensor) in tape._watched or id(kwarg_tensor) in tape._nodes:
                should_record = True
                break

        if should_record:
            tape._is_used = True
            temp_parents = []
            for p_tensor in normalized_inputs:
                is_node = id(p_tensor) in tape._nodes
                if is_node:
                    temp_parents.append(tape._nodes[id(p_tensor)])
            parents = temp_parents

            node = OpNode(op_name, normalized_inputs, kwargs, result, parents)
            tape._nodes[id(result)] = node
            tape._watched.add(id(result))

    def _clear_state(self):
        """Resets tape state if not persistent."""
        if not self.persistent:
            self._watched.clear()
            self._grads.clear()
            self._nodes.clear()
            self._is_used = False

    def _backpropagate(self, target: Tensor, output_gradients: Gradient | None):
        self._grads.clear()

        if output_gradients is None:
            h_grad = Tensor(
                np.ones(target.shape, dtype=self.forced_dtype or target.dtype)
            )
            ah_grad = Tensor(
                np.ones(target.shape, dtype=self.forced_dtype or target.dtype)
            )
            output_gradients = Gradient(h=h_grad, ah=ah_grad)

        self._grads[id(target)] = output_gradients

        sorted_nodes = self._topological_sort(target)

        for node in reversed(sorted_nodes):
            upstream_grad = self._grads.get(id(node.result))
            if upstream_grad is None:
                warnings.warn(
                    f"No upstream gradient found for node result {node.op_name} (ID: {id(node.result)}). Backprop may be broken.",
                    stacklevel=1,
                )
                continue

            input_grads = self._compute_vjp(node, upstream_grad)

            for i, inp in enumerate(node.inputs):
                inp_id = id(inp)
                if inp_id in self._watched or inp_id in self._nodes:
                    self._accumulate_gradient(inp, input_grads[i])

    @staticmethod
    def _unbroadcast(grad: Tensor, target_shape: tuple[int]) -> Tensor:
        """
        Reduce the broadcasted gradient to match the shape of the target tensor.
        Handles complex broadcasting across all axes robustly.
        """
        if not target_shape:
            return grad.sum()  # Scalar case

        g_shape = grad.shape

        # Early return if shapes already match
        if g_shape == target_shape:
            return grad

        # Handle empty tensor case
        if grad.numel() == 0:
            return grad.reshape(target_shape)

        # Calculate dimensions and validate
        g_ndim = len(g_shape)
        t_ndim = len(target_shape)

        # Handle case where grad has fewer dimensions than target
        if g_ndim < t_ndim:
            # Pad grad shape with leading 1s
            pad_shape = (1,) * (t_ndim - g_ndim) + g_shape
            grad = grad.reshape(pad_shape)
            g_shape = pad_shape
            g_ndim = t_ndim

        # Collect axes for different reduction operations
        leading_axes = []
        keepdim_axes = []

        # Process dimensions from left to right
        for i in range(g_ndim):
            target_idx = i - (g_ndim - t_ndim)

            if target_idx < 0:
                # Leading dimension that needs to be summed away
                leading_axes.append(i)
            else:
                # Check if this dimension was broadcasted
                g_dim = g_shape[i]
                t_dim = target_shape[target_idx]

                if t_dim == 1 and g_dim > 1:
                    keepdim_axes.append(i)
                elif t_dim not in [g_dim, 1]:
                    # Shape mismatch that can't be resolved by broadcasting
                    raise ValueError(
                        f"Cannot unbroadcast: incompatible shapes {g_shape} -> {target_shape}"
                    )

        # Apply reductions in optimal order
        if leading_axes:
            grad = grad.sum(axis=tuple(leading_axes))

        if keepdim_axes:
            # Adjust indices after leading dimension reduction
            if leading_axes:
                keepdim_axes = [ax - len(leading_axes) for ax in keepdim_axes]
            grad = grad.sum(axis=tuple(keepdim_axes), keepdims=True)

        # Final reshape if needed (handles size-1 dimension adjustments)
        if grad.shape != target_shape:
            grad = grad.reshape(target_shape)

        return grad

    def _compute_vjp(
        self, node: OpNode, upstream_grad: Gradient
    ) -> list[Gradient | None]:
        """Calls the user-defined gradient function with the full Gradient object."""
        grad_func = registry.get(node.op_name)
        if not grad_func:
            warnings.warn(
                f"No gradient for '{node.op_name}'. Treating as constant.", stacklevel=1
            )
            return [None] * len(node.inputs)

        return grad_func(upstream_grad, node.result, *node.inputs, **node.kwargs)

    def _accumulate_gradient(self, tensor: Tensor, grad: Gradient | None) -> None:
        """Adds a new gradient to a tensor's accumulator, with unbroadcasting."""
        if grad is None:
            return

        tensor_id = id(tensor)

        # Initialize accumulator if not exists
        if tensor_id not in self._grads:
            init_dtype = self.forced_dtype or tensor.dtype
            zeros = np.zeros(tensor.shape, dtype=init_dtype)
            zero_tensor = Tensor(zeros)
            self._grads[tensor_id] = Gradient(h=zero_tensor, ah=zero_tensor)

        current = self._grads[tensor_id]

        # Extract and validate gradient components
        h_component = self._unbroadcast(grad.h, tensor.shape)
        ah_component = self._unbroadcast(grad.ah, tensor.shape)

        # Accumulate gradients
        new_h = current.h + h_component
        new_ah = current.ah + ah_component

        # Apply numeric stabilization for floating point tensors
        if np.issubdtype(tensor.dtype, np.floating):
            new_h = self._stabilize_floating_point(new_h)
            new_ah = self._stabilize_floating_point(new_ah)

        self._grads[tensor_id] = Gradient(h=new_h, ah=new_ah)

    def _stabilize_floating_point(self, tensor) -> Tensor:
        """Apply numeric stabilization to floating point data."""

        # Apply real_if_close with configurable tolerance
        tolerance = getattr(self, "_real_tolerance", 1e-13)
        return np.real_if_close(tensor, tol=tolerance)

    def _topological_sort(self, target: Tensor) -> list[OpNode]:
        """
        Returns a topologically sorted list of nodes for backpropagation.
        Uses Kahn's algorithm for better cycle detection and performance.
        """
        target_id = id(target)
        if target_id not in self._nodes:
            return []

        # Build adjacency list and compute in-degrees
        adjacency: dict[int, list[OpNode]] = {}
        in_degree: dict[int, int] = {}
        all_nodes: dict[int, OpNode] = {}

        # BFS to discover all reachable nodes from target
        queue = [self._nodes[target_id]]
        visited_discovery = {target_id}

        while queue:
            current = queue.pop(0)
            current_id = id(current)
            all_nodes[current_id] = current

            # Initialize structures
            if current_id not in adjacency:
                adjacency[current_id] = []
            if current_id not in in_degree:
                in_degree[current_id] = 0

            # Process parents (predecessors in computation graph)
            for parent in current.parents:
                parent_id = id(parent)

                # Add edge: parent -> current
                if parent_id not in adjacency:
                    adjacency[parent_id] = []
                adjacency[parent_id].append(current)
                in_degree[current_id] += 1

                # Continue discovery if not visited
                if parent_id not in visited_discovery:
                    visited_discovery.add(parent_id)
                    queue.append(parent)
                    all_nodes[parent_id] = parent

        # Kahn's algorithm for topological sorting
        zero_in_degree = [
            node
            for node_id, node in all_nodes.items()
            if in_degree.get(node_id, 0) == 0
        ]
        result = []

        while zero_in_degree:
            current = zero_in_degree.pop(0)
            result.append(current)
            current_id = id(current)

            # Remove edges from current node
            for neighbor in adjacency.get(current_id, []):
                neighbor_id = id(neighbor)
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    zero_in_degree.append(neighbor)

        # Cycle detection
        if len(result) != len(all_nodes):
            remaining_nodes = [
                node for node_id, node in all_nodes.items() if node not in result
            ]
            node_names = [
                f"{node.op_name}(id:{id(node)})" for node in remaining_nodes[:3]
            ]
            if len(remaining_nodes) > 3:
                node_names.append(f"... and {len(remaining_nodes) - 3} more")

            raise RuntimeError(
                f"Cycle detected in computation graph. "
                f"Nodes involved: {', '.join(node_names)}"
            )

        return result
