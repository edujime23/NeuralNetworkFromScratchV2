import numpy as np

from network.gradient_tape import GradientTape
from network.layers import Dense
from network.optimizers.adam import Adam
from network.plugins.optimizer import StochasticGradientClippingPlugin, LookaheadPlugin
from network.types.tensor import Tensor


def relu(x):
    return np.maximum(x, 0)


np.random.seed(0)
dense = Dense(units=32, activation=relu, name="linear", dtype=None)

X = Tensor(np.arange(0, 5), dtype=None).reshape(-1, 1)
Y = 2 * X + 1

optimizer = Adam(lr=1e-3)

optimizer.add_plugins([StochasticGradientClippingPlugin(), LookaheadPlugin()])

dense(X, training=True)


def mse(pred, target):
    return np.mean(np.abs((pred - target) ** 2))


epochs = 1e4

for step in range(int(epochs)):
    with GradientTape() as tape:
        tape.watch(dense.kernel, dense.bias)

        preds = dense(X, training=True)
        loss = mse(preds, Y)

    grad_k, grad_b = tape.gradient(loss, [dense.kernel, dense.bias])
    optimizer.apply_gradients([(grad_k.ah, dense.kernel), (grad_b.ah, dense.bias)])

    if step % 1000 == 0:
        print(f"grad_k: {grad_k} grad_b: {grad_b}")
        print(
            f"Step {step}: loss={loss}, w={dense.kernel.value.squeeze()}, b={dense.bias.value.squeeze()}"
        )

print("Final w:", dense.kernel.value)
print("Final b:", dense.bias.value)
print("Final loss:", mse(dense(X), Y))
