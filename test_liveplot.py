import numpy as np

from network.layers import Dense, Input
from network.models import Sequential
from network.optimizers.adam import Adam
from network.plugins.model.plotting import LivePlotPlugin
from network.plugins.optimizer.clipping import StochasticGradientClippingPlugin
from network.plugins.optimizer.look_ahead import LookaheadPlugin
from network.plugins.model.lr import AdaptiveLRPlugin
from network.functions.activation import bent_identity, tanh, leaky_relu, softmax, relu, sigmoid, swish, elu, gelu, mish


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def func(x):
    return np.exp2(-x**5)
    return np.tan(x**7 * np.abs(x)**np.abs(x))


n = 300
X = np.linspace(0, 5, n).reshape(-1, 1)
Y = func(X).reshape(-1, 1)

layers = [
    Input(input_shape=(1,)),
    Dense(units=64, activation=tanh),
    Dense(units=64, activation=softmax),
    Dense(units=64, activation=bent_identity),
    Dense(units=32, activation=elu),
    Dense(units=1, activation=None),
]

model = Sequential(layers, name="f_approx_model")

optimizer = Adam(lr=1e-2)

optimizer.add_plugins([LookaheadPlugin(), StochasticGradientClippingPlugin()])

model.add_plugins([AdaptiveLRPlugin(1024), LivePlotPlugin(metrics=["mse"], max_points=512)])

model.compile(optimizer=optimizer, loss=mse, metrics=["mse"])

EPOCHS = int(1e8)
BATCH_SIZE = X.shape[0]

print(f"Starting training for {EPOCHS} epochs...")
model.fit(x=X, y=Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("\nTraining complete.")

model.summary()
