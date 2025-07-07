from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from network.types.tensor import Tensor


# Internal-facing types. Users defining gradients won't need these.
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
    parents: list[OpNode] = field(default_factory=list)

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

        current_key_value = None
        for key, value in self.kwargs.items():
            current_key_value = (key, value)
            if not isinstance(value, Tensor):
                continue
            else:
                break
        else:
            if len(self.kwargs) > 0:
                raise ValueError(
                    f"Keyword argument {current_key_value[0]} must be a Tensor. Got {type(current_key_value[1])}"
                )

        del current_key_value, arg_n_inp


@dataclass(frozen=True, slots=True)
class Gradient:
    """
    Internal: Stores holomorphic and anti-holomorphic gradient components.

    This class is used to represent gradients in a complex-differentiable
    context, utilizing Wirtinger derivatives. It separates the gradient
    into its holomorphic (df/dz) and anti-holomorphic (df/dconj(z)) parts.
    """

    h: Tensor  # Holomorphic component: df/dz
    ah: Tensor  # Anti-holomorphic component: df/dconj(z)

    def __post_init__(self):
        if self.h.shape != self.ah.shape:
            raise ValueError(
                f"Holomorphic and anti-holomorphic shapes mismatch. h={self.h.shape} != ah={self.ah.shape}"
            )

        if not isinstance(self.h, Tensor) or not isinstance(self.ah, Tensor):
            raise ValueError(
                f"Holomorphic and anti-holomorphic must be Tensors. h={type(self.h)} != ah={type(self.ah)}"
            )

        if self.h.dtype != self.ah.dtype:
            raise ValueError(
                f"Holomorphic and anti-holomorphic dtype mismatch. h={self.h.dtype} != ah={self.ah.dtype}"
            )

    @property
    def total(self):
        """
        Returns the sum of the holomorphic and anti-holomorphic gradient components.

        This property is particularly useful when the gradient is for a real-valued
        function with real-valued inputs.

        For functions where both components contribute (e.g., in Wirtinger derivatives
        for real functions where both are 1/2 of the total gradient), this sum
        represents the complete gradient.

        Returns:
            Tensor: The sum of Gradient.h and Gradient.ah.
        """
        return self.h + self.ah
