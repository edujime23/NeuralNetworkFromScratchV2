from abc import ABC
from network.tape.base.types import OpNode
from network.types.tensor import Tensor
from network.types.variable import Variable
from network.queues.tapes import tapes
import numpy as np

class TapeCore(ABC):
    def __init__(self, persistent: bool = False, dtype: np.dtype | None = None):
        self.persistent = persistent
        self.forced_dtype = dtype
        self._watched: set[int] = set()
        self._nodes: dict[int, OpNode] = {}
        self._is_used: bool = False

    def _watch(self, *tensors: Tensor):
        """Internal watch method."""
        if isinstance(tensors, Tensor):
            tensors = (tensors,)
        for t in tensors:
            self._watched.add(id(t))

    def _record_operation(
        self, op_name: str, inputs: tuple, kwargs: dict, result: Tensor
    ):
        if not tapes:
            return

        normalized_inputs = tuple(
            inp.value if isinstance(inp, Variable) else inp for inp in inputs
        )

        should_record = False
        for inp_tensor in normalized_inputs:
            if id(inp_tensor) in self._watched or id(inp_tensor) in self._nodes:
                should_record = True
                break

        normalized_kwargs_tensors = []
        normalized_kwargs_tensors.extend(
            v.value if isinstance(v, Variable) else v
            for v in kwargs.values()
            if isinstance(v, (Tensor, Variable))
        )
        for kwarg_tensor in normalized_kwargs_tensors:
            if id(kwarg_tensor) in self._watched or id(kwarg_tensor) in self._nodes:
                should_record = True
                break

        if should_record:
            self._add_node_to_graph(
                normalized_inputs, op_name, kwargs, result
            )

    def _add_node_to_graph(self, normalized_inputs, op_name, kwargs, result):
        self._is_used = True
        temp_parents = []
        for p_tensor in normalized_inputs:
            is_node = id(p_tensor) in self._nodes
            if is_node:
                temp_parents.append(self._nodes[id(p_tensor)])
        parents = temp_parents

        node = OpNode(op_name, normalized_inputs, kwargs, result, parents)
        self._nodes[id(result)] = node
        self._watched.add(id(result))

    def _topological_sort(self, target: Tensor) -> list[OpNode]:
        """Returns a topologically sorted list of nodes for a target Tensor."""
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