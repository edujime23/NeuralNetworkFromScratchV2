import numpy as np

from network.functions.activation import softmax, leaky_relu, bent_identity, tanh
from network.functions.loss import categorical_crossentropy
from network.layers import Dense, Input
from network.models import Sequential
from network.optimizers.adam import Adam
from network.plugins.optimizer.clipping import StochasticGradientClippingPlugin
from network.plugins.optimizer.look_ahead import LookaheadPlugin
from network.plugins.model.lr import AdaptiveLRPlugin

# -- Generate challenging dataset with 100 samples --
n = 10
X = np.linspace(0, 4, n).reshape(-1, 1)


def soft_targets(x):
    centers = np.array([0, 1, 2, 3])
    sigma = 0.7
    diffs = x - centers  # broadcasts: (10,1) - (4,) -> (10,4)
    exps = np.exp(-0.5 * (diffs / sigma) ** 2)
    return exps / np.sum(exps, axis=1, keepdims=True)


Y = soft_targets(X)
# -- Model architecture unchanged, just larger layers as you wrote --

layers = [
    Input(input_shape=(1,)),
    Dense(units=1024, activation=leaky_relu, name="l-0"),
    Dense(units=512, activation=leaky_relu, name="l-1"),
    Dense(units=256, activation=bent_identity, name="l-2"),
    Dense(units=128, activation=tanh, name="l-3"),
    Dense(units=64, activation=tanh, name="l-4"),
    Dense(units=32, activation=leaky_relu, name="l-5"),
    Dense(units=16, activation=leaky_relu, name="l-6"),
    Dense(units=8, activation=leaky_relu, name="l-7"),
    Dense(units=4, activation=softmax, name="l-8"),
]

model = Sequential(layers, name="f_approx_model")

optimizer = Adam(lr=1e-3)
optimizer.add_plugins([LookaheadPlugin(), StochasticGradientClippingPlugin()])
model.add_plugins([AdaptiveLRPlugin(256)])

model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=["mse"])

EPOCHS = 1000
BATCH_SIZE = Y.size  # minibatch size to help training stability and speed

print(f"Starting training for {EPOCHS} epochs...")
model.fit(x=X, y=Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("\nTraining complete.")
model.summary()

print(f"Example inputs:\n{X[:5].flatten()}")
print(f"Example targets:\n{np.round(Y[:5], 3)}")
print(f"Model predictions:\n{np.round(model(X[:5]), 3)}")
