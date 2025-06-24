import numpy as np

from network.gradient_tape import GradientTape
from network.types import Variable

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)


def func(u):
    val = np.abs(u)
    val = val**2
    return val


with GradientTape() as tape:
    tape.watch(x, y)
    z = func(y)

dx, dy = tape.gradient(z, [x, y])
print("∂z/∂x =", dx)
print("∂z/∂y =", dy)
