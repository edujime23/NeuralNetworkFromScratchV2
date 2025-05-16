from typing import Optional, Tuple, Any, Union, List
import numpy as np
from .util import ensure_shape
import warnings

class AggregationGradients:
    @staticmethod
    def sum(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        inp = inputs[0]
        grad: np.typing.NDArray[Any] = np.broadcast_to(grad_output, inp.shape)
        
        if axis is not None:
            # If summing over a specific axis, we need to replicate grad_output along that axis
            # To match the input shape post-sum along the given axis
            grad: np.typing.NDArray[Any] = np.sum(grad, axis=axis, keepdims=bool(keepdims) or False)
            
        return [ensure_shape(grad_output, inp.shape)]
    
    @staticmethod
    def mean(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False
    ) -> List[Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]]:
        inp = inputs[0]
        shape = inp.shape if hasattr(inp, 'shape') else ()

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        if axis is None:
            count = np.prod(shape) if shape else 1
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = np.prod([shape[a] for a in axes])
            if not keepdims:
                grad_output_h = np.expand_dims(grad_output_h, axes)
        grad = np.broadcast_to(grad_output_h, shape) / count
        grad_ah = np.zeros_like(grad)

        return [(ensure_shape(grad, shape), ensure_shape(grad_ah, shape))]
    
    @staticmethod
    def nanmean(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False):
        x = inputs[0]
        
        # Create a mask that is True for non-NaN values in the input
        mask = ~np.isnan(x)
        
        # Sum the mask (number of valid entries) along the specified axis, or across the whole array
        if axis is not None:
            mask_sum: np.typing.NDArray[Any] = np.sum(mask, axis=axis, keepdims=bool(keepdims))
        else:
            mask_sum = np.sum(mask)

        # Compute the gradient for non-NaN elements only
        grad = grad_output * mask / (mask_sum + np.finfo(grad_output.dtype).eps)  # Add epsilon for numerical stability

        # If the input is complex, we take the conjugate of the gradient
        if np.iscomplexobj(x):
            grad = np.conj(grad)

        # Ensure the gradient has the same shape as the input
        return [ensure_shape(grad, x.shape)]
    
    @staticmethod
    def prod(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        inp = inputs[0]
        
        # Calculate the product of the conjugates of the input along the given axis
        prod_conj_val = np.prod(np.conjugate(inp), axis=axis, keepdims=True)
        
        # Broadcast the product result to match the shape of the input
        prod_conj_broadcasted = np.broadcast_to(prod_conj_val, inp.shape)
        
        # Calculate the gradient of the product using the chain rule
        grad = grad_output * prod_conj_broadcasted / (np.conjugate(inp) + np.finfo(inputs[0].dtype).eps)
        
        # Ensure the gradient has the correct shape
        return [ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def max(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        inp = inputs[0]
        
        # If input contains complex numbers, return zero gradients
        if np.iscomplexobj(inp):
            warnings.warn("Gradient of max is not well-defined for complex inputs. Returning zero gradients.")
            return [np.zeros_like(inp, dtype=grad_output.dtype)]
        
        # Find the maximum value in the input along the specified axis
        max_val = np.max(inp, axis=axis, keepdims=True)
        
        # Create a mask where elements are equal to the max value
        mask = (inp == max_val)
        
        # Count the number of maximums along the specified axis (for normalization)
        num_max = np.sum(mask, axis=axis, keepdims=True)
        
        # Broadcast the grad_output to match the number of maximums
        grad_output_broadcasted = np.broadcast_to(grad_output, num_max.shape)
        
        # Compute the gradient for each element
        grad = grad_output_broadcasted * mask / (num_max + np.finfo(inputs[0].dtype).eps)

        return [ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def maximum(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        a, b = inputs
        
        # Handle complex case: Gradients of maximum are not well-defined for complex numbers
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]
        
        # For real numbers: Compute gradients
        if axis is None:
            # No reduction, handle element-wise comparison
            grad_a = grad_output * (a >= b)
            grad_b = grad_output * (b > a)
        else:
            # Reduction along axis, handle the maximum operation with axis
            grad_a = np.zeros_like(a)
            grad_b = np.zeros_like(b)
            
            # Identify which elements contributed to the maximum in each axis
            max_a = (a >= b)
            max_b = (b > a)
            
            # Gradients only where the maximum occurs
            grad_a = np.where(max_a, grad_output, 0)
            grad_b = np.where(max_b, grad_output, 0)
            
            # If axis is specified, reduce along that axis
            if keepdims:
                grad_a = np.sum(grad_a, axis=axis, keepdims=True)
                grad_b = np.sum(grad_b, axis=axis, keepdims=True)
            else:
                grad_a = np.sum(grad_a, axis=axis, keepdims=False)
                grad_b = np.sum(grad_b, axis=axis, keepdims=False)
        
        return [
            ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
            ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())
        ]

    @staticmethod
    def min(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        inp = inputs[0]

        # If input contains complex numbers, return zero gradients
        if np.iscomplexobj(inp):
            warnings.warn("Gradient of min is not well-defined for complex inputs. Returning zero gradients.")
            return [np.zeros_like(inp, dtype=grad_output.dtype)]

        # Find the minimum value in the input along the specified axis
        min_val = np.min(inp, axis=axis, keepdims=True)

        # Create a mask where elements are equal to the min value
        mask = (inp == min_val)

        # Count the number of minimums along the specified axis (for normalization)
        num_min = np.sum(mask, axis=axis, keepdims=True)

        # Broadcast the grad_output to match the number of minimums
        grad_output_broadcasted = np.broadcast_to(grad_output, num_min.shape)

        # Compute the gradient for each element
        grad = grad_output_broadcasted * mask / (num_min + np.finfo(inputs[0].dtype).eps)

        return [ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def minimum(
        grad_output: np.typing.NDArray[Any],
        inputs: np.typing.NDArray[Any],
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False
    ):
        a, b = inputs

        # For complex numbers, gradient is undefined for minimum
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            return [
                np.zeros_like(a, dtype=grad_output.dtype),
                np.zeros_like(b, dtype=grad_output.dtype)
            ]

        # Elementwise comparison mask
        mask_a = a <= b
        mask_b = b < a

        grad_a = np.where(mask_a, grad_output, 0)
        grad_b = np.where(mask_b, grad_output, 0)

        # If reduction over axis is specified
        if axis is not None:
            if not keepdims:
                grad_a = np.expand_dims(grad_a, axis=axis)
                grad_b = np.expand_dims(grad_b, axis=axis)
                grad_output = np.expand_dims(grad_output, axis=axis)

            # Broadcast gradients to input shape
            grad_a = np.where(mask_a, grad_output, 0)
            grad_b = np.where(mask_b, grad_output, 0)

        return [
            ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
            ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())
        ]
    
    @staticmethod
    def std(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False) -> List[np.typing.NDArray[Any]]:
        x = inputs[0]
        
        # If axis is None, we operate over the whole array, otherwise along a specific axis
        n = x.size if axis is None else x.shape[axis]
        
        # Compute the mean and standard deviation over the specified axis
        xm: np.typing.NDArray[Any] = np.mean(x, axis=axis, keepdims=bool(keepdims))
        std_x = np.std(x, axis=axis, keepdims=bool(keepdims))
        
        # Ensure we don't divide by zero (or a very small number)
        std_x_safe = std_x + np.finfo(x.dtype).eps  # Add small epsilon for stability

        # Compute the gradient with respect to the standard deviation
        grad = grad_output * (x - xm) / (std_x_safe * n)

        # If the input is complex, we take the conjugate of the gradient
        if np.iscomplexobj(x):
            grad = np.conj(grad)

        # Ensure the gradient matches the shape of the input
        return [ensure_shape(grad, x.shape)]
    
    @staticmethod
    def nanstd(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        x = inputs[0]
        
        # Create a mask for non-NaN values
        mask = ~np.isnan(x)

        # Sum of non-NaN values
        if axis is not None:
            mask_sum = np.sum(mask, axis=axis, keepdims=bool(keepdims))
        else:
            mask_sum = np.sum(mask)

        # Compute the mean of the valid values
        xm = np.nansum(x) / mask_sum

        # Compute the standard deviation of the valid values
        std_x = np.nanstd(x)

        # Gradient computation (we ensure to handle division by zero or small values)
        grad = grad_output * (x - xm) * mask / (mask_sum * (std_x + np.finfo(grad_output.dtype).eps))

        # If the input is complex, take the conjugate of the gradient
        if np.iscomplexobj(x):
            grad = np.conj(grad)

        return [ensure_shape(grad, x.shape)]
    
    @staticmethod
    def var(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], axis: Optional[int] = None, keepdims: Optional[bool] = False):
        x = inputs[0]
        
        # Compute the mean over the specified axis
        xm = np.mean(x, axis=axis, keepdims=bool(keepdims))
        
        # Determine the number of elements based on axis or full array
        n = x.size if axis is None else x.shape[axis]
        
        # Compute the gradient of the variance
        grad = grad_output * 2 * (x - xm) / n

        # If the input is complex, take the conjugate of the gradient
        if np.iscomplexobj(x):
            grad = np.conj(grad)

        # Ensure the gradient has the same shape as the input
        return [ensure_shape(grad, x.shape)]
    
    @staticmethod
    def nanvar(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)], **kwargs: dict[str, Any]):
        x = inputs[0]
        
        # Extract any extra keyword arguments, such as 'axis' and 'keepdims'
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)

        # Mask for non-NaN values
        mask = ~np.isnan(x)
        
        # Calculate the number of non-NaN elements
        n: np.typing.NDArray[Any] = np.sum(mask, axis=axis if isinstance(axis, (int, tuple, type(None))) else None, keepdims=bool(keepdims))
        
        # Calculate the mean (ignoring NaNs)
        xm = np.nansum(x, axis=axis if isinstance(axis, (int, tuple, type(None))) else None, keepdims=bool(keepdims)) / n
        
        # Compute the gradient of the variance
        grad = grad_output * 2 * (x - xm) * mask / n

        # If the input is complex, take the conjugate of the gradient
        if np.iscomplexobj(x):
            grad = np.conj(grad)

        # Ensure the gradient has the same shape as the input
        return [ensure_shape(grad, x.shape)]