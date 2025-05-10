from network.types.variable import Variable
from network.gradient_tape.gradient_tape import GradientTape
import numpy as np

# Create test Variables
x = Variable([1+1j, 2+2j, 3+3j], (3,), np.complex64, True, 'x', 'none').real / 32
y = Variable([3+3j, 2+2j, 1+1j], (3,), np.complex64, False, 'y', 'ones').real / 32

print("x:", x)
print("y:", y)
print()

np.random.seed(0)

def func(x, y):
    # Ensure x and y are 2D arrays for matmul
    x = np.reshape(x, (-1, 1))  # Reshape to a 2D column vector (3, 1)
    y = np.reshape(y, (-1, 1))  # Reshape to a 2D column vector (3, 1)
    
    # Apply every single numpy ufunc and operation to test the gradient tape
    z = np.add(x, y)
    z += np.subtract(x, y)
    z -= np.multiply(x, y)
    z *= np.divide(x, y)
    z /= np.add(x, y)
    z **= np.subtract(x, y)
    z += np.sqrt(x)
    z += np.cbrt(x)
    z += np.sin(x)
    z += np.cos(x)
    z += np.tan(x)
    z += np.exp(x)
    z += np.log(x)
    z += np.log10(x)
    z += np.log2(x)
    z += np.arcsin(x)
    z += np.arccos(x)
    z += np.arctan(x)
    z += np.arctan2(x, y)
    z += np.sinh(x)
    z += np.cosh(x)
    z += np.tanh(x)
    z += np.arcsinh(x)
    z += np.arccosh(x + 1)
    z += np.arctanh(x)
    z += np.deg2rad(x)
    z += np.rad2deg(x)
    z += np.clip(x, 0, 1)
    z += np.floor(x)
    z += np.ceil(x)
    z += np.round(x)
    z += np.trunc(x)
    z += np.abs(x)
    z += np.conjugate(x)
    z += np.real(x)
    z += np.imag(x)
    z += np.mean(x)
    z += np.std(x)
    z += np.var(x)
    z += np.sum(x)
    z += np.prod(x)
    z += np.min(x)
    z += np.max(x)
    z += np.sign(x)
    z += np.maximum(x, y)
    z += np.minimum(x, y)
    z += np.power(x, y)
    z += np.mod(x, y)
    z += np.fmod(x, y)
    z += np.divmod(x, y)[0]  # divmod returns a tuple, so we use the first element
    z += np.remainder(x, y)
    z += np.floor_divide(x, y)
    z += np.true_divide(x, y)
    
    # Special functions
    z += np.sinc(x)
    z += np.angle(x)
    z += np.radians(x)
    z += np.degrees(x)
    
    # Linear algebra functions
    z += np.dot(x.T, y)  # dot product (use x.T to transpose)
    z += np.matmul(x.T, y)  # matrix multiplication (use x.T to transpose)
    
    # Random number generation functions
    z += np.random.normal(0, 1, x.shape)  # Normal distribution
    z += np.random.rand(*x.shape)  # Uniform distribution
    
    return z

with GradientTape() as tape:
    tape.watch(x)  # Watch x for gradient calculation
    tape.watch(y)  # Watch y for gradient calculation
    z = func(x, y)

print("Output of func:", z)

# Compute the gradients of z with respect to x and y
gradients = tape.gradient(z, [x, y])
print("Gradients:", gradients)

