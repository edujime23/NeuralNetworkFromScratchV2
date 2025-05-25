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
    print("z:", tape.gradient(z, [x, y]))
    z += np.subtract(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z -= np.multiply(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z *= np.divide(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z /= np.add(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z **= np.subtract(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.sqrt(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.cbrt(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.sin(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.cos(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.tan(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.exp(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.log(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.log10(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.log2(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arcsin(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arccos(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arctan(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arctan2(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.sinh(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.cosh(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.tanh(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arcsinh(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arccosh(x + 1)
    print("z:", tape.gradient(z, [x, y]))
    z += np.arctanh(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.deg2rad(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.rad2deg(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.clip(x, 0, 1)
    print("z:", tape.gradient(z, [x, y]))
    z += np.floor(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.ceil(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.round(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.trunc(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.abs(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.conjugate(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.real(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.imag(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.mean(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.std(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.var(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.sum(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.prod(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.min(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.max(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.sign(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.maximum(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.minimum(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.power(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.mod(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.fmod(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.divmod(x, y)[0]  # divmod returns a tuple, so we use the first element
    print("z:", tape.gradient(z, [x, y]))
    z += np.remainder(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.floor_divide(x, y)
    print("z:", tape.gradient(z, [x, y]))
    z += np.true_divide(x, y)
    print("z:", tape.gradient(z, [x, y]))
    
    # Special functions
    z += np.sinc(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.angle(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.radians(x)
    print("z:", tape.gradient(z, [x, y]))
    z += np.degrees(x)
    print("z:", tape.gradient(z, [x, y]))
    
    # Linear algebra functions
    z += np.dot(x.T, y)  # dot product (use x.T to transpose)
    print("z:", tape.gradient(z, [x, y]))
    z += np.matmul(x.T, y)  # matrix multiplication (use x.T to transpose)
    print("z:", tape.gradient(z, [x, y]))
    
    # Random number generation functions
    # z += np.random.normal(0, 1, x.shape)  # Normal distribution
    # z += np.random.rand(*x.shape)  # Uniform distribution
    
    return z

with GradientTape(persistent=True) as tape:
    tape.watch(x)  # Watch x for gradient calculation
    tape.watch(y)  # Watch y for gradient calculation
    z = func(x, y)
 
print("Output of func:", z)

# Compute the gradients of z with respect to x and y
gradients = tape.gradient(z, [x, y])
print("Gradients:", gradients)

del tape

