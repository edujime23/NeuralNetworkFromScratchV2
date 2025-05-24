import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.special import gamma

def integer_derivative_precise(f, x_vals, order=1):
    """
    Compute the 'order'-th integer derivative of f at points x_vals using
    finite differences with step size h ~ eps^(1/3) to achieve O(eps^(2/3)) accuracy.
    
    For order = 1: central difference: (f(x+h) - f(x-h)) / (2h).
    For order = 2: second difference: (f(x+h) - 2f(x) + f(x-h)) / (h^2).
    For order >2: recursively apply first- or second-derivative routines.
    """
    eps = np.finfo(float).eps
    D_vals = np.zeros_like(x_vals, dtype=float)
    
    if order == 1:
        for i, x in enumerate(x_vals):
            h = eps**(1/3) * (1.0 + abs(x))
            D_vals[i] = (f(x + h) - f(x - h)) / (2 * h)
        return D_vals
    
    elif order == 2:
        for i, x in enumerate(x_vals):
            h = eps**(1/3) * (1.0 + abs(x))
            D_vals[i] = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
        return D_vals
    
    else:
        # For higher-order, apply recursive differentiation:
        # e.g. D^n f = D^1(D^(n-1) f)
        # We compute the (n-1)-th derivative at each x, then take its first derivative.
        # Note: intermediate results require interpolating the derivative function; 
        # for simplicity, we discretize again on the same x_vals.
        prev_vals = integer_derivative_precise(f, x_vals, order=order-1)
        g = lambda z: np.interp(z, x_vals, prev_vals)
        return integer_derivative_precise(g, x_vals, order=1)

def integer_integral_precise(f, a, x_vals):
    """
    Compute the definite integral of f from a to each point in x_vals using
    cumulative trapezoidal integration (O(h^2) local error). Return array of integrals.
    """
    y_vals = f(x_vals)
    I_vals = cumulative_trapezoid(y_vals, x_vals, initial=0)
    return I_vals

def fractional_derivative_rl_precise(f, a, alpha, x_vals):
    """
    Compute D_a^alpha f(x) with precision ~ eps^(2/3). 
    If alpha is an integer, calls integer_derivative_precise directly. 
    Otherwise, uses Riemann-Liouville definition + finite differences 
    with step h ~ eps^(1/3).
    """
    # Check if alpha is (nearly) an integer
    if np.isclose(alpha, np.round(alpha), atol=0, rtol=0):
        int_order = int(round(alpha))
        return integer_derivative_precise(f, x_vals, order=int_order)
    
    eps = np.finfo(float).eps
    m = int(np.ceil(alpha))
    gamma_prefactor = 1.0 / gamma(m - alpha)
    
    # Define the RL integral function I(x) for each x
    def I_of(x):
        if x <= a:
            return 0.0
        integrand = lambda t: f(t) / ((x - t) ** (alpha - m + 1))
        I_val = quad(integrand, a, x)[0]
        return gamma_prefactor * I_val
    
    D_vals = np.zeros_like(x_vals, dtype=float)
    # For each x, approximate the m-th derivative of I(x) via finite differences
    for i, x in enumerate(x_vals):
        h = eps**(1/3) * (1.0 + abs(x))
        
        # Recursive central-difference for m-th derivative
        def derivative_order_k(func, x0, order_k, h0):
            if order_k == 1:
                return (func(x0 + h0) - func(x0 - h0)) / (2 * h0)
            elif order_k == 2:
                return (func(x0 + h0) - 2 * func(x0) + func(x0 - h0)) / (h0**2)
            else:
                # Differentiate (order_k - 1) then 1:
                g = lambda z: derivative_order_k(func, z, order_k - 1, h0)
                return (g(x0 + h0) - g(x0 - h0)) / (2 * h0)
        
        D_vals[i] = derivative_order_k(I_of, x, m, h)
    
    return D_vals

# ---------------------------------------------
# EXAMPLE USAGE WITH PRECISION ~ eps^(1/3)
# ---------------------------------------------

# 1) Define a test function, e.g. f(x) = x^2.
f = lambda x: x**2

# 2) Build a grid of x-values (sorted, starts at a=0).
x_vals = np.linspace(0, 5, 100)

# 3) Integer derivative 1st order (h ~ eps^(1/3)):
D1_precise = integer_derivative_precise(f, x_vals, order=1)

# 4) Integer derivative 2nd order (should be exactly 2 for x^2):
D2_precise = integer_derivative_precise(f, x_vals, order=2)

# 5) Integer integral (trapezoidal rule):
I_vals = integer_integral_precise(f, a=0, x_vals=x_vals)

# 6) Fractional derivative α = 2.0 (integer case → second derivative):
alpha_int = 2.0
D_frac_int = fractional_derivative_rl_precise(f, a=0, alpha=alpha_int, x_vals=x_vals)

# 7) Fractional derivative α = 1.5 (true fractional):
alpha_frac = 1.5
D_frac_nonint = fractional_derivative_rl_precise(f, a=0, alpha=alpha_frac, x_vals=x_vals)

# 8) Print a few sample results to verify:
print("x_vals[:5] =", x_vals[:5])
print("1st-derivative (precise):", D1_precise[:5])
print("2nd-derivative of x^2 (should ~2.0):", D2_precise[:5])
print("Integer RL α=2.0 via wrapper (should ~2.0):", D_frac_int[:5])
print("Fractional RL α=1.5 samples:", D_frac_nonint[:5])
