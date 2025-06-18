import numpy as np

from network.optimizers import (
    Adam,
    AdaptiveGradientClippingAddon,
    AdaptiveNoiseAddon,
    L1L2RegularizationAddon,
    LookaheadAddon,
    NesterovMomentumAddon,
)
from network.tape import GradientTape
from network.types import Variable

np.random.seed(69)

target = np.array([2 + -1j], dtype=np.complex128)

w = Variable(value=[0.0 + 0.0j], trainable=True, name="w", initializer="ones")
# w = Variable(value=[0.0], trainable=True, name="w", initializer="ones")
w.initialize()
opt = Adam(1e-3)
steps = int(1e4)

opt.add_addon(NesterovMomentumAddon())
opt.add_addon(L1L2RegularizationAddon(1e-6, 1e-6))
opt.add_addon(AdaptiveGradientClippingAddon())
opt.add_addon(AdaptiveNoiseAddon())
opt.add_addon(LookaheadAddon())

print(opt.summary())


def func(x):
    return np.abs(np.sqrt(x)) ** (np.exp(x))


def losss(x):
    return np.abs(x - target) ** 2


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

        if (i + 1) % np.ceil(steps / 10) == 0:
            print(
                f"{i+1}\tw = {w}\tloss = {loss}\tgrad_ho = {holo}\tgrad_anti = {anti}"
            )

        if np.all(loss < 1e-32):
            print(f"Converged at step {i+1}:  w = {w}  loss = {loss}")
            break

    print(f"\nFinal parameter value after {steps} steps: w = {w}")
    print(f"Final loss value after {steps} steps: loss = {loss}")
    print(f"Final Function value after {steps} steps: out = {out}")


run_optimization()
