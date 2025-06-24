import numpy as np

from network.gradient_tape import GradientTape
from network.layers import Dense
from network.optimizers.adam import Adam
from network.types.tensor import Tensor


def relu(x):
    return np.max(x, 0)


np.random.seed(0)
dense = Dense(units=1, activation=lambda x: x, name="linear")

# Create a toy dataset: y = 3x + 2
X = Tensor(np.random.randn(1, 2).astype(np.float32))
Y = X + 1

optimizer = Adam(lr=1e-3)

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
            f"Step {step}: loss={loss}, w={dense.kernel.value.squeeze()}, b={dense.bias.value}"
        )

print("Final w:", dense.kernel.value)
print("Final b:", dense.bias.value)
print("Final loss:", mse(dense(X), Y))
