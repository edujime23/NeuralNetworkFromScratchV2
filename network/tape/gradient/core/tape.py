from __future__ import annotations

import warnings

import numpy as np

from ....queues.tapes import tapes
from ....types.tensor import Tensor
from ....types.variable import Variable
from ..types import Gradient, OpNode
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

    def watch(self, *tensors: Tensor):
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
        normalized_inputs = [
            inp.value if isinstance(inp, Variable) else inp for inp in inputs
        ]

        if relevant_inputs := [
            inp
            for inp in normalized_inputs
            if id(inp) in tape._watched or id(inp) in tape._nodes
        ]:
            tape.watch(result)
            parents = [
                tape._nodes[id(p)] for p in relevant_inputs if id(p) in tape._nodes
            ]
            node = OpNode(op_name, normalized_inputs, kwargs, result, parents)
            tape._nodes[id(result)] = node

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
        """Adds a new gradient to a tensor's accumulator."""
        if grad is None:
            return
        tensor_id = id(tensor)
        if tensor_id not in self._grads:
            init_dtype = self.forced_dtype or tensor.dtype
            init_zero = Tensor(np.zeros(tensor.shape, dtype=init_dtype))
            self._grads[tensor_id] = Gradient(h=init_zero, ah=init_zero)
        current = self._grads[tensor_id]

        h_to_add = grad.h if isinstance(grad.h, Tensor) else Tensor(grad.h)
        ah_to_add = grad.ah if isinstance(grad.ah, Tensor) else Tensor(grad.ah)

        new_h = current.h + h_to_add
        new_ah = current.ah + ah_to_add

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
