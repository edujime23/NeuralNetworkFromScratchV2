from network.tensor.variable import Variable
from network.gradient_tape.gradient_tape import GradientTape
import numpy as np

# Create test Variables
x = Variable([1, 2, 3], (3,), np.float32, True, 'x', 'custom')
y = Variable([3, 2, 1], (3,), np.float32, False, 'y', 'custom')

with GradientTape(persistent=True) as tape:
    tape.watch(x)
    z = x * y

dz_dx, = tape.gradient(z, [x])
print("dz/dx after sin and multiply:", dz_dx)  # Should match the expected derivative


