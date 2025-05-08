from network.types.variable import Variable
from network.gradient_tape.gradient_tape import GradientTape
import numpy as np

# Create test Variables
x = Variable([1+1j, 2+2j, 3+3j], (3,), np.complex64, True, 'x', 'none')
y = Variable([3+3j,2+2j,1+1j], (3,), np.complex64, False, 'y', 'ones')

print(y)
print(x)
print()

# def special(x):
#     return x ** 2

# with GradientTape() as third:
#     with GradientTape() as second:
#         with GradientTape() as first:
#             first.watch(x, y)
#             second.watch(x, y)
#             third.watch(x, y)
#             z = special(x)
#             z *= special(y)
#         deriv = first.gradient(z, [x, y])
#     deriv2 = second.gradient(deriv, [x, y])
# deriv3 = third.gradient(deriv2, [x, y])

# print("first order:", deriv)
# print("second order:", deriv2)
# print("third order:", deriv3)

with GradientTape() as tape:
    tape.watch(x, y)
    z = x
    print(tape.gradient(z, x))
    z *= np.conj(x)
    print(tape.gradient(z, x))
    z = z.real
    print(tape.gradient(z, x))