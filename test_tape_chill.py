import numpy as np
from network.gradient_tape import GradientTape
from network.types import Variable

# Utility: numeric derivative via complex‐step method
def complex_step_grad(f, z, h=1e-6):
    # Use higher-precision arithmetic for the complex-step
    z_hp = z.astype(np.complex128)
    grad = np.zeros_like(z_hp, dtype=np.complex128)
    for idx in np.ndindex(*z_hp.shape):
        z_h = z_hp.copy()
        z_h[idx] = z_h[idx] + (h * 1j)
        f1 = f(z_h)
        # ensure f1 is complex128
        f1 = np.array(f1, dtype=np.complex128)
        grad[idx] = np.imag(f1[idx]) / h
    return grad

# Test functions and their analytic gradients (where holomorphic)
def square(z):
    return z ** 2

def grad_square(z):
    # The true complex‐derivative of z^2 is 2*z
    return 2 * z


def exp_fn(z):
    return np.exp(z)

def grad_exp(z):
    return np.exp(z)

# Non-holomorphic functions

def conj_fn(z):
    return np.conjugate(z)

def abs_fn(z):
    return np.abs(z)

# Helper to wrap variables through GradientTape

def compute_grad(f, z_data):
    # Wrap input in Variable and let operator overloading record ops
    x = Variable(z_data, z_data.shape, np.complex64, True, 'x', 'none')
    with GradientTape() as tape:
        tape.watch(x)
        out_var = f(x)        
    # Assumes f returns a Variable or numpy array; if array, Variable overrides pow etc. so out_var is Variable
    return tape.gradient(out_var, x)

# Run tests and report

def run_tests():
    data = np.array([1+2j, 3+4j, -1-1j], dtype=np.complex64)
    tests = [
        ('square', square, grad_square, True),
        ('exp_fn', exp_fn, grad_exp, True),
        # For non-holomorphic, compare against custom analytic partial wrt z
        ('conj_fn', conj_fn, lambda z: np.zeros_like(z), False),
        ('abs_fn', abs_fn, lambda z: z / np.abs(z), False),
    ]

    for name, fn, analytic, is_holo in tests:
        print(f"Testing {name}...")
        grad = compute_grad(fn, data)
        numeric = complex_step_grad(fn, data)
        if is_holo:
            expected = analytic(data)
            if not np.allclose(grad, expected, atol=1e-4, rtol=1e-4):
                raise AssertionError(f"{name} failed holomorphic check.\nGot {grad}\nExpected {expected}")
        else:
            # Compare against analytic partial for non-holomorphic
            expected = analytic(data)
            if not np.allclose(grad, expected, atol=1e-4, rtol=1e-4):
                raise AssertionError(f"{name} failed non-holomorphic check. Got {grad} Expected {expected}")
        print(f"  {name} passed.")

if __name__ == '__main__':
    run_tests()
    print("All tests passed successfully.")
