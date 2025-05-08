import numpy as np
from network.models import Sequential
from network.layers import Dense

# Define a simple activation function
def linear_activation(x):
    return x

# Generate synthetic data for a regression task
np.random.seed(42)
x_train = np.random.rand(1000, 100)  # 1000 samples, 100 features
true_weights = np.random.randn(100, 1)
y_train = x_train @ true_weights + np.random.randn(1000, 1) * 0.1  # Add some noise

# Define the model
model = Sequential([
    Dense(units=64, activation=linear_activation, use_bias=True),
    Dense(units=1, activation=linear_activation, use_bias=True)
])

# Compile the model
model.compile(optimizer='AdamOptimizer', loss=lambda x, y: x - y, metrics=['mae'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
