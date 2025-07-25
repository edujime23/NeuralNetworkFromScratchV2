from dataclasses import dataclass, field
from typing import Any, Self

from network.types.tensor import Tensor

@dataclass(frozen=True, slots=True)
class OpNode:
    """
    Internal: Represents a recorded operation in the computation graph.

    This class captures the essential information about an operation performed
    during a forward pass, allowing for the construction of a computation graph
    for backpropagation.

    Attributes:
        op_name: The name of the operation (e.g., "add", "mul", "matmul").
        inputs: A tuple of input Tensors to this operation.
        kwargs: Keyword arguments passed to the operation, providing additional
                context or parameters for the operation (e.g., 'axis' for sum).
        result: The Tensor produced as the output of this operation.
        parents: A list of OpNode instances that are direct predecessors in the
                 computation graph (i.e., whose results were inputs to this op).
                 This helps in tracing the graph backwards for gradient computation.
    """

    op_name: str
    inputs: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Tensor
    parents: list[Self] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.result, Tensor):
            raise ValueError(f"Result must be a Tensor. Got {type(self.result)}")

        if not isinstance(self.inputs, tuple):
            raise ValueError(f"Inputs must be a tuple. Got {type(self.inputs)}")

        # For future me, else block only execs when the loop runs normally with no breaks, also if the obj thats iterated has len 0 ;D
        arg_n_inp = 0
        for i, input in enumerate(self.inputs):
            arg_n_inp = (i, input)
            if not isinstance(input, Tensor):
                continue
            else:
                break
        else:
            if len(self.inputs) > 0:
                raise ValueError(
                    f"Input #{arg_n_inp[0]} must be a Tensor. Got {type(arg_n_inp[1])}"
                )

        del arg_n_inp