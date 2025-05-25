from network.tape import GradientTape, numerical_derivative, GRADIENTS
from network.types import Variable
import numpy as np
from numba import vectorize

del GRADIENTS['conjugate']

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1+1j, 2+2j, 3+3j], dtype=np.complex128)

def func(x):
    return x**2

dx = GradientTape().derivative(func, x, 2)

print(dx)
