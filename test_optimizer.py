from network.tape import GradientTape
from network.optimizers import AdaDelta, AdaGrad, AdamOptimizer, Momentum, RMSProp, SGD
from network.types import Variable
import numpy as np

np.random.seed(69)

target = np.array([1 + -1j], dtype=np.complex128)

w = Variable(value=[0.0 + 0.0j], trainable=True, name='w', initializer='ones')
w.initialize()

opt = AdamOptimizer(1e-3)
steps = int(1e4)

def func(x):
    res = 2 ** (x ** x)
    return res

def losss(x):
    return (np.abs(x-target)**2)

def run_optimization():
    print("Step\tw value\tLoss")
    for i in range(steps):
        with GradientTape() as tape:
            tape.watch(w)
            out = func(w)
            loss = losss(out)
            
        grads = tape.gradient(loss, w)

        holo, anti = grads

        grad = holo
        opt.apply_gradients([(grad, w)])

        if (i + 1) % (steps // 10) == 0:
            print(f"{i+1}\tw = {w}\tloss = {loss}\tgrad_ho = {holo}\tgrad_anti = {anti}")

        if np.all(loss < 1e-32):
            print(f"Converged at step {i+1}:  w = {w}  loss = {loss}")
            break

    print(f"\nFinal parameter value after {steps} steps: w = {w}")
    print(f"Final loss value after {steps} steps: loss = {loss}")
    print(f"Final Function value after {steps} steps: out = {out}")

run_optimization()