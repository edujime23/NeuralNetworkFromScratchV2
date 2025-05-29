from network.tape import GradientTape, GRADIENTS
from network.optimizers import *
from network.types import Variable
import numpy as np

np.random.seed(69)

target = np.array([1 + -1j], dtype=np.complex128)

w = Variable(value=[0.0 + 0.0j], trainable=True, name='w', initializer='ones')
w.initialize()

opt = AdamOptimizer(1e-3)
steps = int(1e4)

def func(x):
    res = x
    return res

def losss(x):
    return np.mean(np.sqrt(np.conj(target - x) * (target - x)))

def run_optimization():
    print("Step\tw value\tLoss")
    for i in range(steps):
        with GradientTape() as tape:
            tape.watch(w)
            out = func(w)
            loss = losss(out)

        holo, anti = tape.gradient(loss, w)
        # print(tape._nodes_in_order)

        grad = holo + anti
        opt.apply_gradients([(grad, w)])

        if (i + 1) % (steps // 10) == 0:
            print(f"grad: {grad}\tholo: {holo}\tanti: {anti}")
            print(f"{i+1}\tw = {w}\tloss = {loss}\tgrad_ho = {holo}\tgrad_anti = {anti}")

        if np.all(loss < 1e-32):
            print(f"Converged at step {i+1}:  w = {w}  loss = {loss}")
            break

    print(f"\nFinal parameter value after {steps} steps: w = {w}")
    print(f"Final loss value after {steps} steps: loss = {loss}")
    print(f"Final Function value after {steps} steps: out = {out}")

run_optimization()