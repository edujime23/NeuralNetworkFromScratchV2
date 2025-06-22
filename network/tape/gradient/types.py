# autodiff/tape/types.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...types import Tensor


@dataclass
class OpNode:
    r"""
    Represents a node in the computation graph, recording a primitive operation.

    Each `OpNode` stores crucial information about a mathematical operation performed
    on Tensors. This information is later used during backpropagation to compute
    gradients. It maintains references to its parent nodes, which are the inputs
    to the operation, forming a directed acyclic graph (DAG).

    Attributes:
        func: The callable function (e.g., `np.add`, `np.multiply`) that was executed.
        method: A string indicating the method name on the Tensor that triggered this
                operation (e.g., "add", "matmul"). This is primarily for debugging
                and identifying the gradient function.
        inputs: A tuple of `Tensor` objects that were used as inputs to `func`.
        kwargs: A dictionary of keyword arguments passed to `func`.
        result: The `Tensor` object produced as the output of `func`.
        parents: A list of `OpNode` instances that are direct parents in the
                 computation graph (i.e., nodes whose results were inputs to this node).
                 This is crucial for traversing the graph during backpropagation.
        creation_index: An integer indicating the order in which this node was created.
                        Used for reverse-mode traversal.
        last_visited: A timestamp used during graph traversal (e.g., DFS) to avoid
                      reprocessing nodes and detect cycles.
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
        """Concise string representation of the OpNode for debugging."""
        return (
            f"OpNode(func={self.func.__name__}, "
            f"result_id={id(self.result)}, "
            f"shape={self.result.shape}, "
            f"dtype={self.result.dtype})"
        )


@dataclass
class Gradient:
    r"""
    Stores holomorphic and anti-holomorphic components of a gradient.

    In complex analysis, for a function $f(w)$, the gradient can be split into
    two components: $\frac{\partial f}{\partial w}$ (holomorphic) and
    $\frac{\partial f}{\partial \bar{w}}$ (anti-holomorphic). This class
    encapsulates these two components, both guaranteed to be `Tensor` objects.

    Attributes:
        holomorphic: The holomorphic component of the gradient ($\partial f/\partial w$).
        antiholomorphic: The anti-holomorphic component of the gradient ($\partial f/\partial \conj{w}$).
    """

    holomorphic: Tensor
    antiholomorphic: Tensor
