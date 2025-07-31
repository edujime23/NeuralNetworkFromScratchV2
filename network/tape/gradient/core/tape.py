from __future__ import annotations

import warnings
import numpy as np

from network.tape.gradient.types import Gradient
from network.queues.tapes import tapes
from network.types.tensor import Tensor

from .registry import registry
from network.tape.base.core import TapeCore
from network.tape.base.types import OpNode


class GradientTapeCore(TapeCore):
    """
    Core machinery for automatic differentiation using Wirtinger calculus.

    Implements pure Wirtinger derivatives:
    - d/dz = (1/2)(d/dx - i*d/dy)
    - d/dz_conj = (1/2)(d/dx + i*d/dy)

    Mathematical conventions:
    - For complex functions: Economic derivative dL/df = 1, dL/df_conj = 0
    - For real functions: f = f_conj, so both derivatives contribute equally
    - For real variables: df/dz = df/dz_conj = (1/2) * df/dx
    - For complex variables in holomorphic functions: df/dz_conj = 0
    """

    def __init__(self, persistent: bool = False, dtype: np.dtype | None = None):
        super().__init__(persistent, dtype)
        self._grads: dict[int, Gradient] = {}

    def _clear_state(self):
        """Resets tape state when not in persistent mode."""
        if not self.persistent:
            self._watched.clear()
            self._grads.clear()
            self._nodes.clear()
            self._is_used = False

    def _backpropagate(self, target: Tensor, output_gradients: Gradient | None = None):
        """
        Backpropagates gradients using mathematically correct Wirtinger calculus.

        The key insight: initialization must respect the mathematical nature of the target.
        """
        # Check if target was recorded
        if id(target) not in self._nodes and id(target) not in self._watched:
            raise ValueError("Target tensor was not recorded on this tape")

        # Warn if tape has no operations
        if not self._is_used:
            warnings.warn("Tape has no recorded operations", stacklevel=2)
            return

        # Clear gradients for fresh backpropagation
        self._grads.clear()

        # Initialize output gradients if not provided
        if output_gradients is None:
            # Mathematical correctness: proper initialization based on target type
            if np.iscomplexobj(target):
                # Complex target: Use economic derivative
                # For L = f (complex), we have dL/df = 1, dL/df_conj = 0
                ones = np.ones(target.shape, dtype=target.dtype)
                zeros = np.zeros(target.shape, dtype=target.dtype)
                output_gradients = Gradient(h=Tensor(ones), ah=Tensor(zeros))
            else:
                # Real target: f = f_conj, so both Wirtinger derivatives contribute
                # For L = f (real), we have dL/df = dL/df_conj = 1
                ones = np.ones(target.shape, dtype=target.dtype)
                output_gradients = Gradient(h=Tensor(ones), ah=Tensor(ones))
        elif (output_gradients.h.shape != target.shape or
              output_gradients.ah.shape != target.shape):
            raise ValueError(
                f"Output gradients shape {output_gradients.h.shape} "
                f"must match target shape {target.shape}"
            )

        # Set initial gradient for target
        self._grads[id(target)] = output_gradients

        # Get nodes in topological order
        sorted_nodes = self._topological_sort(target)

        # Backpropagate through computational graph
        for node in reversed(sorted_nodes):
            # Get gradient for current node
            upstream_grad = self._grads.get(id(node.result))
            if upstream_grad is None:
                continue

            # Compute gradients for inputs
            input_grads = self._compute_vjp(node, upstream_grad)

            # Accumulate gradients for each input
            for i, inp in enumerate(node.inputs):
                inp_id = id(inp)
                if ((inp_id in self._watched or inp_id in self._nodes) and
                    input_grads[i] is not None):
                    self._accumulate_gradient(inp, input_grads[i])

    def _compute_vjp(self, node: OpNode, upstream_grad: Gradient) -> list[Gradient | None]:
        """Computes vector-Jacobian product using registered gradient functions."""
        # Get gradient function for operation
        grad_func = registry.get(node.op_name)
        if not grad_func:
            warnings.warn(
                f"No gradient for '{node.op_name}'. Treating as constant.",
                stacklevel=1
            )
            return [None] * len(node.inputs)

        # Compute gradients using registered function
        return grad_func(upstream_grad, node.result, *node.inputs, **node.kwargs)

    def _accumulate_gradient(self, tensor: Tensor, grad: Gradient):
        """
        Accumulates gradients for a tensor using proper Wirtinger arithmetic.

        Maintains real gradients for real tensors when possible.
        """
        tensor_id = id(tensor)

        if tensor_id not in self._grads:
            # Initialize gradient storage based on tensor type
            shape = tensor.shape

            # For real tensors, try to keep gradients real
            if not np.iscomplexobj(tensor):
                # Check if incoming gradients are purely real
                h_is_real = (not np.iscomplexobj(grad.h) or
                            np.allclose(np.imag(grad.h), 0, atol=1e-12))
                ah_is_real = (not np.iscomplexobj(grad.ah) or
                             np.allclose(np.imag(grad.ah), 0, atol=1e-12))

                if h_is_real and ah_is_real:
                    # Can use real storage
                    dtype = np.float32 if tensor.dtype == np.float32 else np.float64

                    # Extract real parts if needed
                    h_data = np.real(grad.h) if np.iscomplexobj(grad.h) else grad.h
                    ah_data = np.real(grad.ah) if np.iscomplexobj(grad.ah) else grad.ah

                    # Initialize with zeros
                    self._grads[tensor_id] = Gradient(
                        h=Tensor(np.zeros(shape, dtype=dtype)),
                        ah=Tensor(np.zeros(shape, dtype=dtype))
                    )

                    # Accumulate real parts
                    self._grads[tensor_id] = Gradient(
                        h=Tensor(h_data),
                        ah=Tensor(ah_data)
                    )
                    return
                else:
                    # Need complex storage
                    dtype = np.complex64 if tensor.dtype == np.float32 else np.complex128
            else:
                # Complex tensor: use complex storage
                dtype = np.complex64 if tensor.dtype == np.complex64 else np.complex128

            # Initialize gradient storage
            self._grads[tensor_id] = Gradient(
                h=Tensor(np.zeros(shape, dtype=dtype)),
                ah=Tensor(np.zeros(shape, dtype=dtype))
            )

        # Get current gradient
        current = self._grads[tensor_id]

        # For real tensors with real gradients, keep them real
        if (not np.iscomplexobj(tensor) and
            not np.iscomplexobj(current.h) and
            np.iscomplexobj(grad.h) and
            (np.allclose(np.imag(grad.h), 0, atol=1e-12) and
             np.allclose(np.imag(grad.ah), 0, atol=1e-12))):
            # Keep real by extracting real parts
            new_h = current.h + np.real(grad.h)
            new_ah = current.ah + np.real(grad.ah)
            self._grads[tensor_id] = Gradient(h=new_h, ah=new_ah)
            return

        # Standard accumulation (may promote to complex if needed)
        new_h = current.h + grad.h
        new_ah = current.ah + grad.ah
        self._grads[tensor_id] = Gradient(h=new_h, ah=new_ah)

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Gradient]:
        """
        Computes gradients using pure Wirtinger calculus.

        Returns mathematically correct Wirtinger derivatives for all variables.
        No dtype conversion is performed to preserve mathematical correctness.

        Args:
            target: The tensor to differentiate
            sources: List of tensors to compute gradients with respect to

        Returns:
            List of Gradient objects containing Wirtinger derivatives.
            Each Gradient has:
            - h: d_target/dz
            - ah: d_target/dz_conj

        Note: For real variables z=x, both derivatives are equal and represent
              half the real derivative. For complex variables in holomorphic
              functions, df/dz_conj = 0.
        """
        # Run backpropagation
        self._backpropagate(target)

        # Collect gradients for requested sources
        gradients = []
        for source in sources:
            source_id = id(source)
            if source_id in self._grads:
                # Return gradients as-is, maintaining mathematical correctness
                gradients.append(self._grads[source_id])
            else:
                # Source was not involved in computation
                gradients.append(None)

        return gradients

    def _get_gradient_info(self, grad: Gradient, source: Tensor) -> dict:
        """
        Utility method for debugging/inspection.
        Returns human-readable gradient information.
        """
        # Check if gradients are real
        h_is_real = (np.allclose(np.imag(grad.h), 0, atol=1e-12)
                    if np.iscomplexobj(grad.h) else True)
        ah_is_real = (np.allclose(np.imag(grad.ah), 0, atol=1e-12)
                     if np.iscomplexobj(grad.ah) else True)

        # Collect basic info
        info = {
            'h_shape': grad.h.shape,
            'ah_shape': grad.ah.shape,
            'h_is_real': h_is_real,
            'ah_is_real': ah_is_real,
            'h_dtype': grad.h.dtype,
            'ah_dtype': grad.ah.dtype,
            'source_dtype': source.dtype,
            'source_is_complex': np.iscomplexobj(source)
        }

        # Determine function type
        if np.iscomplexobj(source):
            # Complex variable
            info['function_type'] = (
                'holomorphic'
                if h_is_real and ah_is_real and np.allclose(grad.ah, 0, atol=1e-12)
                else 'non_holomorphic'
            )
        elif h_is_real and ah_is_real and np.allclose(grad.h, grad.ah, atol=1e-12):
            # Real variable with equal Wirtinger derivatives
            info['function_type'] = 'real_valued'
            info['real_derivative'] = (
                np.real(grad.h + grad.ah) if np.iscomplexobj(grad.h)
                else (grad.h + grad.ah)
            )
        else:
            info['function_type'] = 'complex_valued_from_real'

        return info