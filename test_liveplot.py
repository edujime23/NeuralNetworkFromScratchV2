import numpy as np

from network.layers import Dense, Input
from network.models import Sequential
from network.optimizers.adam import Adam
from network.plugins.model.plotting import LivePlotPlugin
from network.plugins.optimizer.clipping import StochasticGradientClippingPlugin
from network.plugins.optimizer.look_ahead import LookaheadPlugin
from network.plugins.model.lr import AdaptiveLRPlugin


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def func(x):
    return x ** np.exp(-x)


n = 300
X = np.linspace(0, 5, n).reshape(-1, 1)
Y = func(X).reshape(-1, 1)

layers = [
    Input(input_shape=(1,)),
    Dense(units=256, activation=np.tanh, name="l-0"),
    Dense(units=128, activation=np.tanh, name="l-1"),
    Dense(units=64, activation=np.tanh, name="l-2"),
    Dense(units=32, activation=np.tanh, name="l-3"),
    Dense(units=16, activation=np.tanh, name="l-4"),
    Dense(units=8, activation=np.tanh, name="l-5"),
    Dense(units=4, activation=None, name="l-6"),
    Dense(units=1, activation=None, name="l-7"),
]

model = Sequential(layers, name="f_approx_model")

optimizer = Adam(lr=1e-3)

optimizer.add_plugins([LookaheadPlugin(), StochasticGradientClippingPlugin()])

model.add_plugins([AdaptiveLRPlugin(256), LivePlotPlugin(metrics=["mse"])])

model.compile(optimizer=optimizer, loss=mse, metrics=["mse"])

EPOCHS = int(1e8)
BATCH_SIZE = X.shape[0]

print(f"Starting training for {EPOCHS} epochs...")
model.fit(x=X, y=Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

print("\nTraining complete.")

model.summary()
