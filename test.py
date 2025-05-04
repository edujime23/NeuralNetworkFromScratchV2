from network.tensor.variable import Variable
from network.gradient_tape.gradient_tape import GradientTape
import numpy as np

# Create test Variables
x = Variable([1, 2, 3], (3,), np.float32, True, 'x', 'none')
y = Variable([3,2,1], (3,), np.float32, False, 'y', 'ones')

print(y)
print(x)

def special(x):
    return x ** 2

with GradientTape() as third:
    with GradientTape() as second:
        with GradientTape() as first:
            first.watch(x, y)
            second.watch(x, y)
            third.watch(x, y)
            z = special(x)
            z *= special(y)
        deriv = first.gradient(z, [x, y])
    deriv2 = second.gradient(deriv, [x, y])
deriv3 = third.gradient(deriv2, [x, y])

print("first order:", deriv)
print("second order:", deriv2)
print("third order:", deriv3)