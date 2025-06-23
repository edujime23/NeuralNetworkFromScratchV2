import numpy as np

from network.optimizers import Adam
from network.plugins.optimizer import (
    AdaptiveGradientClippingPlugin,
)
from network.tape import GradientTape
from network.types import Variable

np.random.seed(69)

target = np.array([-1 + 0j], dtype=np.complex128)

w = Variable(value=[np.pi + np.e * 1j], trainable=True, name="w", initializer="ones")
# w = Variable(value=[0.0], trainable=True, name="w", initializer="ones")
opt = Adam(5e-3)
steps = int(1e4)

# opt.add_addon(NesterovMomentumAddon(momentum=0.99))
# opt.add_addon(L1L2RegularizationAddon(1e-6, 1e-6))
opt.add_plugin(AdaptiveGradientClippingPlugin())
# opt.add_addon(AdaptiveNoiseAddon())
# opt.add_addon(LookaheadAddon(k=5, alpha=0.5))

print(opt.summary())


def func(x):
    return x  # np.exp(x)


def losss(x):
    return np.abs(x - target) ** 2


def run_optimization():
    print("Step\tw value\tLoss")
    for i in range(steps):
        with GradientTape() as tape:
            tape.watch(w)
            out = func(w)
            loss = losss(out)

        grad = tape.gradient(loss, w)[0]

        holo, anti, total = grad.h, grad.ah, grad.total

        opt.apply_gradients([(anti, w)])

        if (i + 1) % np.ceil(steps / 10) == 0:
            print(
                f"{i+1}\tw = {w}\tloss = {loss}\n\tgrad_ho = {holo}\tgrad_anti = {anti}\tgrad_total = {total}"
            )

        if np.all(loss < 1e-32):
            print(f"Converged at step {i+1}:  w = {w}  loss = {loss}")
            break

    print(f"\nFinal parameter value after {steps} steps: w = {w}")
    print(f"Final loss value after {steps} steps: loss = {loss}")
    print(f"Final Function value after {steps} steps: out = {out}")


run_optimization()
