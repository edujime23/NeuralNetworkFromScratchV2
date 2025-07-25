import numpy as np
from scipy.special import erf


from network.types.tensor import Tensor


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation: maps real values into (0, 1).
    Useful for binary classification outputs or gating.
    Formula: 1 / (1 + exp(-x)).
    """
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: Tensor) -> Tensor:
    """
    Hyperbolic tangent: maps real values into (-1, 1).
    Zero-centered, often converges faster than sigmoid.
    """
    return np.tanh(x)


def relu(x: Tensor) -> Tensor:
    """
    Rectified Linear Unit: sets negative inputs to zero.
    Simple, efficient, and mitigates vanishing gradients.
    """
    return np.maximum(0, x)


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Leaky ReLU: allows a small gradient for negative inputs.
    Prevents "dying ReLU" by using alpha * x when x < 0.
    """
    return np.where(x >= 0, x, alpha * x)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Exponential Linear Unit: smooths negative part.
    For x >= 0, returns x; else alpha * (exp(x) - 1).
    """
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def selu(x: Tensor) -> Tensor:
    """
    Scaled ELU: self-normalizing activation.
    Uses predefined alpha and lambda for mean-zero, unit-variance.
    """
    # According to original paper
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def swish(x: Tensor) -> Tensor:
    """
    Swish activation: x * sigmoid(x).
    Smooth, non-monotonic, often outperforms ReLU variants.
    """
    sig = 1.0 / (1.0 + np.exp(-x))
    return x * sig


def softplus(x: Tensor) -> Tensor:
    """
    Softplus: smooth approximation to ReLU.
    Formula: log(1 + exp(x)). Useful for numerical stability.
    """
    return np.log1p(np.exp(x))


def softsign(x: Tensor) -> Tensor:
    """
    Softsign: alternative to tanh, rational form.
    Formula: x / (1 + |x|).
    """
    return x / (1.0 + np.abs(x))


def softmax(x: Tensor) -> Tensor:
    """
    Softmax: converts vector to probability distribution.
    Numerically stable: subtract max before exponent.
    Operates along last axis.
    """
    # Shift inputs by max for numerical stability
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def gelu(x: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit (GELU): x * Φ(x)
    Smooth activation that approximates dropout behavior.
    Uses erf for exact formulation.
    """
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def mish(x: Tensor) -> Tensor:
    """
    Mish activation: x * tanh(softplus(x))
    Smooth, non-monotonic, often yields better performance.
    """
    return x * np.tanh(np.log1p(np.exp(x)))


def hard_sigmoid(x: Tensor) -> Tensor:
    """
    Hard Sigmoid: piecewise linear approximation of sigmoid.
    Faster to compute, less precise.
    Formula: clip(0.2 * x + 0.5, 0, 1)
    """
    return np.clip(0.2 * x + 0.5, 0.0, 1.0)


def hard_swish(x: Tensor) -> Tensor:
    """
    Hard Swish: x * hard_sigmoid(x)
    Faster approximation to swish.
    """
    return x * np.clip(0.2 * x + 0.5, 0.0, 1.0)


def thresholded_relu(x: Tensor, theta: float = 1.0) -> Tensor:
    """
    Thresholded ReLU: x if x > theta else 0.
    Introduces a learnable threshold for activation.
    """
    return np.where(x > theta, x, 0.0)


def bent_identity(x: Tensor) -> Tensor:
    """
    Bent Identity: (sqrt(x^2 + 1) - 1)/2 + x
    Smooth, combines linear and non-linear behavior.
    """
    return ((np.sqrt(x * x + 1.0) - 1.0) / 2.0) + x


def gaussian(x: Tensor, sigma: float = 1.0) -> Tensor:
    """
    Gaussian activation: exp(-x^2 / (2*sigma^2)).
    Localized bell-shaped function.
    """
    return np.exp(-(x * x) / (2.0 * sigma * sigma))
