from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class AngleGradients:
    @staticmethod
    def degrees(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x: np.typing.NDArray[Any] = inputs[0]
        
        # Convert the gradient output from radians to degrees
        grad = grad_output * (180 / np.pi)

        # Ensure the gradient has the same shape as the input x
        return [ensure_shape(grad, x.shape)]

    @staticmethod
    def radians(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # Convert the gradient output from degrees to radians
        grad = grad_output * (np.pi / 180)

        # Ensure the gradient has the same shape as the input x
        return [ensure_shape(grad, x.shape)]
    
    @staticmethod
    def deg2rad(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        return [ensure_shape(grad_output * np.pi / 180, inputs[0].shape if hasattr(inputs[0], 'shape') else ())]

    @staticmethod
    def rad2deg(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        return [ensure_shape(grad_output * 180 / np.pi, inputs[0].shape if hasattr(inputs[0], 'shape') else ())]
    
    @staticmethod
    def angle(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # Calculate the squared magnitude of x using conjugate
        abs_x_squared = np.conj(x) * x
        
        # Gradient computation for the angle of a complex number
        grad_val = 1j * x / abs_x_squared
        
        # Apply the incoming gradient
        grad = grad_output * grad_val

        return [ensure_shape(grad, x.shape)]