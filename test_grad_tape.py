import numpy as np

from network.tape.gradient import GradientTape
from network.types import Tensor, Variable

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1 + 3j, 2 + 1j, 3 + 2j], dtype=np.complex128)
z = Tensor([1 + 2j, 2 + 3j, 3 + 1j], dtype=np.complex128)


def func(u):
    return u**2


with GradientTape() as tape:
    tape.watch(x, y, z)
    r = func(x) + func(y) + func(z)
    print(tape._grads)
    print("r =", r)

dx, dy, dz = tape.gradient(r, [x, y, z])
print("∂r/∂x =", dx)
print("∂r/∂y =", dy)
print("∂r/∂z =", dz)
