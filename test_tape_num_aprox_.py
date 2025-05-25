from network.tape import GradientTape, GRADIENTS
from network.types import Variable
import numpy as np

del GRADIENTS['conjugate']
GRADIENTS.clear()

x = Variable([1.0, 2.0, 3.0], dtype=np.float64)
y = Variable([1+1j, 2+2j, 3+3j], dtype=np.complex128)

def func(u):
    val = u**2
    return val

with GradientTape() as tape:
    tape.watch(x, y)
    z = func(x)
    z += func(y)

dx, dy = tape.gradient(z, [x, y])
print("∂z/∂x =", np.round(dx, 3))
print("∂z/∂y =", np.round(dy, 3))

tape.print_graph(z)


