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

    @classmethod
    def _record_operation(
        cls, op_name: str, inputs: tuple, kwargs: dict, result: Tensor
    ):
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
        for _, v in kwargs.items():
            if isinstance(v, (Tensor, Variable)):
                normalized_kwargs_tensors.append(
                    v.value if isinstance(v, Variable) else v
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
        g = grad
        g_shape = g.shape
        t_shape = target_shape

        # Prepend 1s to target shape if needed (to match rank)
        if len(g_shape) > len(t_shape):
            t_shape = (1,) * (len(g_shape) - len(t_shape)) + t_shape

        for axis, (g_dim, t_dim) in enumerate(zip(g_shape, t_shape)):
            if t_dim == 1 and g_dim > 1:
                g = g.sum(axis=axis, keepdims=True)

        # Finally reshape to exact target shape if needed
        if g.shape != target_shape:
            g = g.reshape(target_shape)

        return g

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

    def _accumulate_gradient(self, tensor: Tensor, grad: Gradient | None):
        """Adds a new gradient to a tensor's accumulator, with unbroadcasting."""
        if grad is None:
            return

        tensor_id = id(tensor)
        if tensor_id not in self._grads:
            init_dtype = self.forced_dtype or tensor.dtype
            init_zero = Tensor(np.zeros(tensor.shape, dtype=init_dtype))
            self._grads[tensor_id] = Gradient(h=init_zero, ah=init_zero)
        current = self._grads[tensor_id]

        # Ensure h and ah are Tensor instances
        h_to_add = grad.h if isinstance(grad.h, Tensor) else Tensor(grad.h)
        ah_to_add = grad.ah if isinstance(grad.ah, Tensor) else Tensor(grad.ah)

        # Unbroadcast before adding
        h_data = self._unbroadcast(h_to_add.data, tensor.shape)
        ah_data = self._unbroadcast(ah_to_add.data, tensor.shape)

        new_h = current.h + h_data
        new_ah = current.ah + ah_data

        if np.issubdtype(tensor.dtype, np.floating):
            new_h = np.real_if_close(new_h)
            new_ah = np.real_if_close(new_ah)

        self._grads[tensor_id] = Gradient(h=new_h, ah=new_ah)

    def _topological_sort(self, target: Tensor) -> list[OpNode]:
        """Returns a topologically sorted list of nodes for backpropagation."""
        sorted_nodes, visited = [], set()

        def visit(node: OpNode):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            sorted_nodes.append(node)

        if id(target) in self._nodes:
            visit(self._nodes[id(target)])
        return sorted_nodes
