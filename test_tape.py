import numpy as np

from network.gradient_tape import GradientTape
from network.types import Tensor, Variable

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1 + 3j, 2 + 1j, 3 + 2j], dtype=np.complex128)
z = Tensor([1 + 2j, 2 + 3j, 3 + 1j], dtype=np.complex128)


def func(u):
    return np.abs(u) ** 2


with GradientTape() as tape:
    tape.watch(x, y, z)
    r = func(x) + func(y) + func(z)

dx, dy, dz = tape.gradient(r, [x, y, z])
print(r)
print("∂r/∂x =", dx.total)
print("∂r/∂y =", dy)
print("∂r/∂z =", dz)
