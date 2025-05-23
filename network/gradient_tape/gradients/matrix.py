from typing import Tuple, Any
import numpy as np
from .util import ensure_shape
import warnings

class MatrixGradients:
    @staticmethod
    def dot(
        grad_output: Any,  # could be ndarray or tuple for (h, ah)
        inputs: Tuple[np.ndarray, np.ndarray]
    ):
        A, B = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(A)

        grad_A_h = np.dot(grad_output_h, B.T)
        grad_B_h = np.dot(A.T, grad_output_h)

        grad_A_ah = np.dot(grad_output_ah, B.T)
        grad_B_ah = np.dot(A.T, grad_output_ah)

        return [
            (ensure_shape(grad_A_h, np.shape(A)), ensure_shape(grad_A_ah, np.shape(A))),
            (ensure_shape(grad_B_h, np.shape(B)), ensure_shape(grad_B_ah, np.shape(B))),
        ]
        
    @staticmethod
    def matmul(
        grad_output: Any,  # could be ndarray or tuple for (h, ah)
        inputs: Tuple[np.ndarray, np.ndarray]
    ):
        if np.iscomplexobj(grad_output) or np.iscomplexobj(inputs[0]):
            warnings.warn("Complex gradient for matmul might not align with standard 2*dL/d(conj(X)) definition.")

        a, b = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(a)

        a_val = np.array(a)
        b_val = np.array(b)
        grad_out_h = np.array(grad_output_h)
        grad_out_ah = np.array(grad_output_ah)

        a_orig_shape = a.shape if hasattr(a, 'shape') else ()
        b_orig_shape = b.shape if hasattr(b, 'shape') else ()

        # Handle reshaping for 1D inputs, same as original
        reshape_a = False
        if a_val.ndim == 1:
            a_val = a_val.reshape(1, -1)
            grad_out_h = grad_out_h.reshape((-1, 1)) if grad_out_h.ndim == 1 else np.expand_dims(grad_out_h, axis=-2)
            grad_out_ah = grad_out_ah.reshape((-1, 1)) if grad_out_ah.ndim == 1 else np.expand_dims(grad_out_ah, axis=-2)
            reshape_a = True

        reshape_b = False
        if b_val.ndim == 1:
            b_val = b_val.reshape(-1, 1)
            grad_out_h = grad_out_h.reshape((1, -1)) if grad_out_h.ndim == 1 else np.expand_dims(grad_out_h, axis=-1)
            grad_out_ah = grad_out_ah.reshape((1, -1)) if grad_out_ah.ndim == 1 else np.expand_dims(grad_out_ah, axis=-1)
            reshape_b = True

        # Compute holomorphic grads
        if np.iscomplexobj(a_val) or np.iscomplexobj(b_val) or np.iscomplexobj(grad_out_h):
            grad_a_h = np.matmul(grad_out_h, np.conjugate(np.swapaxes(b_val, -1, -2)))
            grad_b_h = np.matmul(np.conjugate(np.swapaxes(a_val, -1, -2)), grad_out_h)
        else:
            grad_a_h = np.matmul(grad_out_h, np.swapaxes(b_val, -1, -2))
            grad_b_h = np.matmul(np.swapaxes(a_val, -1, -2), grad_out_h)

        # Compute anti-holomorphic grads similarly
        if np.iscomplexobj(a_val) or np.iscomplexobj(b_val) or np.iscomplexobj(grad_out_ah):
            grad_a_ah = np.matmul(grad_out_ah, np.conjugate(np.swapaxes(b_val, -1, -2)))
            grad_b_ah = np.matmul(np.conjugate(np.swapaxes(a_val, -1, -2)), grad_out_ah)
        else:
            grad_a_ah = np.matmul(grad_out_ah, np.swapaxes(b_val, -1, -2))
            grad_b_ah = np.matmul(np.swapaxes(a_val, -1, -2), grad_out_ah)

        # Reshape back if needed
        if reshape_a:
            grad_a_h = np.squeeze(grad_a_h, axis=-2)
            grad_a_ah = np.squeeze(grad_a_ah, axis=-2)
        if reshape_b:
            grad_b_h = np.squeeze(grad_b_h, axis=-1)
            grad_b_ah = np.squeeze(grad_b_ah, axis=-1)

        return [
            (ensure_shape(grad_a_h, np.shape(a)), ensure_shape(grad_a_ah, np.shape(a))),
            (ensure_shape(grad_b_h, np.shape(b)), ensure_shape(grad_b_ah, np.shape(b)))
        ]