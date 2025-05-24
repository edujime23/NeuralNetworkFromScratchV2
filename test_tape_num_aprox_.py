from network.gradient_tape import GradientTape, numerical_derivative, GRADIENTS
from network.types import Variable
import numpy as np
from numba import vectorize

del GRADIENTS['conjugate']

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1+1j, 2+2j, 3+3j], dtype=np.complex128)

def func(u):
    val = np.conj(u)
    return val

with GradientTape() as tape:
    tape.watch(x, y)
    a = func(x)
    b = func(y)
    z = a + b

dx, dy = tape.gradient(z, [x, y])
print("∂z/∂x =", np.round(dx, 3))
print("∂z/∂y =", np.round(dy, 3))


