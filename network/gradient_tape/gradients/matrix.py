from typing import Tuple, Any
import numpy as np
from .util import ensure_shape
import warnings

class MatrixGradients:
    @staticmethod
    def dot(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        A, B = inputs
        grad_A = np.dot(grad_output, B.T)
        grad_B = np.dot(A.T, grad_output)

        return [
            ensure_shape(grad_A, A.shape),
            ensure_shape(grad_B, B.shape),
        ]
        
    @staticmethod
    def matmul(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        if np.iscomplexobj(grad_output) or np.iscomplexobj(inputs[0]):
            warnings.warn("Complex gradient for matmul might not align with standard 2*dL/d(conj(X)) definition.")
        
        a, b = inputs

        # Ensure that inputs and grad_output are numpy arrays for consistency
        a_val = np.array(a)
        b_val = np.array(b)
        grad_out_val = np.array(grad_output)

        a_orig_shape = a.shape if hasattr(a, 'shape') else ()
        b_orig_shape = b.shape if hasattr(b, 'shape') else ()

        # Reshape for 1D inputs
        reshape_a = False
        if a_val.ndim == 1:
            a_val = a_val.reshape(1, -1)
            if grad_out_val.ndim == 1:
                grad_out_val = grad_out_val.reshape(-1, 1)
            elif grad_out_val.ndim > 1:
                grad_out_val = np.expand_dims(grad_out_val, axis=-2)
            reshape_a = True

        reshape_b = False
        if b_val.ndim == 1:
            b_val = b_val.reshape(-1, 1)
            if grad_out_val.ndim == 1:
                grad_out_val = grad_out_val.reshape(1, -1)
            elif grad_out_val.ndim > 1:
                grad_out_val = np.expand_dims(grad_out_val, axis=-1)
            reshape_b = True

        # Compute gradients, handle complex inputs using conjugates
        if np.iscomplexobj(a_val) or np.iscomplexobj(b_val) or np.iscomplexobj(grad_out_val):
            grad_a = np.matmul(grad_out_val, np.conjugate(np.swapaxes(b_val, -1, -2)))
            grad_b = np.matmul(np.conjugate(np.swapaxes(a_val, -1, -2)), grad_out_val)
        else:
            grad_a = np.matmul(grad_out_val, np.swapaxes(b_val, -1, -2))
            grad_b = np.matmul(np.swapaxes(a_val, -1, -2), grad_out_val)

        # Reshape back if needed
        if reshape_a:
            grad_a = np.squeeze(grad_a, axis=-2)
        if reshape_b:
            grad_b = np.squeeze(grad_b, axis=-1)

        return [
            ensure_shape(grad_a, a_orig_shape),
            ensure_shape(grad_b, b_orig_shape)
        ]