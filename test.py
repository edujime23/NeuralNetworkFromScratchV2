from network.types import Variable
from network.tape import GradientTape
import numpy as np

x = Variable([1+1j, 2+2j, 3+3j], dtype=np.complex128, name="x")

def func(x):
    res = Variable()
    res += x**2
    return res

with GradientTape() as tape:
    tape.watch(x)
    z = func(x)

grads = tape.gradient(z, x)

print(grads)