from network.types.variable import Variable
from network.gradientTape import Tape
import numpy as np

x = Variable([1, 2, 3], (3,), np.float32, True, 'x', 'custom')
y = Variable([3, 2, 1], (3,), np.float32, False, 'y', 'custom')

with Tape(persistent=True) as tape:
    tape.watch(x)

    # Composite expression using many ufuncs
    z = (
        np.sin(x) +
        np.cos(x * y) +
        np.tanh(x - y) +
        np.exp(x) +
        np.log(np.abs(x) + 1e-5) +
        np.negative(x) +
        np.maximum(x, y) +
        np.minimum(x, y) +
        np.floor(x) +
        np.ceil(x) +
        np.round(x) +
        np.mod(x, y) +
        np.remainder(x, y) +
        np.fmod(x, y) +
        np.clip(x, -1, 1)
    )

dz_dx, = tape.gradient(z, [x])
print("dz/dx:", dz_dx)
