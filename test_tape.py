import numpy as np

from network.tape import GradientTape
from network.types import Tensor, Variable

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
z = Tensor([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)


def func(u):
    return np.abs(u) ** 2


with GradientTape() as tape:
    tape.watch(x, y, z)
    r = func(x) + func(y) + func(z)
    # print("Recorded ops:")
    # for node in tape._nodes_in_order:
    #     print("   func is:", node.func, "   name:", node.func.__name__, "   result:", node.result, "   inputs:", node.inputs)

dx, dy, dz = tape.gradient(r, [x, y, z])
print(r)
print("∂r/∂x =", np.round(dx, 3))
print("∂r/∂y =", np.round(dy, 3))
print("∂r/∂z =", np.round(dz, 3))
