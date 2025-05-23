from network.gradient_tape import GradientTape
from network.types import Variable
import numpy as np

# Testing on f(w) = (w - 3)^2
x = Variable(value=[1, 2,3], shape=(3,), dtype=np.float64, trainable=True, name='x', initializer='zeros')
y = Variable(value=[1+1j, 2+2j, 3+3j], shape=(3,), dtype=np.complex128, trainable=True, name='y', initializer='zeros')

def func(x):
    return x

with GradientTape() as tape:
    tape.watch(x, y)
    z = func(x)
    z += func(y)
    
print(f"Deriv: {tape.gradient(z, [x, y])}")
