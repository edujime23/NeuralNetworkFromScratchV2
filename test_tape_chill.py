import cProfile
import pstats
from network.optimizers import AdamOptimizer
from network.gradient_tape import GradientTape
from network.types import Variable
import numpy as np

# Testing on f(w) = (w - 3)^2
x = Variable(value=[1, 2,3], shape=(3,), dtype=np.float32, trainable=True, name='w', initializer='zeros')
y = Variable(value=[1j, 2j, 3j], shape=(3,), dtype=np.complex64, trainable=True, name='w', initializer='zeros')

def func(x):
    return np.mean(x)

with GradientTape() as tape:
    tape.watch(x, y)
    z = func(x)
    z += func(y)
    
print(f"Deriv: {tape.gradient(z, [x,y])}")
