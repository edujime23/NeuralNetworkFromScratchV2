import numpy as np

from network.layers import Dense
from network.models import Sequential

# Seed for reproducibility
np.random.seed(0)

# Toy dataset: y = 3x + 2 + noise
X = np.random.randn(100, 1).astype(np.float32)
Y = 3 * X + 2 + 0.1 * np.random.randn(100, 1).astype(np.float32)


# Define loss function
def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


# Build the model
model = Sequential(
    [Dense(units=1, activation=None, name="linear")], name="linear_model"
)

# Compile the model
model.compile(optimizer="adam", loss=mse, metrics=["mse"])

# Train the model
model.fit(x=X, y=Y, epochs=50, batch_size=16)

# Summary
model.summary()

# Predict
predictions = model(X)
print("\nSample predictions:\n", predictions[:5])

#Commit de TigerWavvee full hd 4k free VBucks 2025 HOW TO