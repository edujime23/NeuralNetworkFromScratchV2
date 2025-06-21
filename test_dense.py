import numpy as np

from network.layers import Dense
from network.optimizers.adam import Adam
from network.tape import GradientTape


def relu(x):
    return np.max(x, 0)


np.random.seed(0)
dense = Dense(units=1, activation=lambda x: x, name="linear")

# Create a toy dataset: y = 3x + 2
X = np.random.randn(100, 1).astype(np.float32)
Y = X + 1

optimizer = Adam(lr=1e-3)


def mse(pred, target):
    return np.mean((pred - target) ** 2)


for step in range(10000):
    with GradientTape() as tape:
        tape.watch(dense.kernel, dense.bias)

        preds = dense(X, training=True)
        loss = mse(preds, Y)

    grad_k, grad_b = tape.gradient(loss, [dense.kernel, dense.bias])
    optimizer.apply_gradients(
        [(grad_k[0] + grad_k[1], dense.kernel), (grad_b[0] + grad_b[1], dense.bias)]
    )

    if step % 1000 == 0:
        print(f"grad_k: {grad_k[0]} grad_b: {grad_b[0]}")
        print(
            f"Step {step}: loss={loss}, w={dense.kernel.value.squeeze()}, b={dense.bias.value}"
        )

print("Final w:", dense.kernel.value)
print("Final b:", dense.bias.value)
print("Final loss:", mse(dense(X), Y))
