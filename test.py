from network.types import Variable
from network.tape import GradientTape
import numpy as np

x = Variable([1.0, 2.0, 3.0], dtype=np.float32, name="x")

def func(x):
    res = x ** 2
    return res

with GradientTape() as tape:
    tape.watch(x)
    z = func(x)

grads = tape.gradient(z, x)

print(grads)