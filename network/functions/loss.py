import numpy as np
from network.types.tensor import Tensor
from numba import vectorize

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Mean Squared Error (MSE): average of squared differences.
    Penalizes larger errors more. Used in regression tasks.
    """
    diff = y_pred - y_true
    return np.mean(diff * diff)

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Mean Absolute Error (MAE): average of absolute differences.
    More robust to outliers than MSE.
    """
    return np.mean(np.abs(y_pred - y_true))

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def huber(y_true: Tensor, y_pred: Tensor, delta: float = 1.0) -> Tensor:
    """
    Huber Loss: quadratic for small errors, linear for large ones.
    Smooth alternative between MSE and MAE.
    """
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic = 0.5 * error * error
    linear = delta * (abs_error - 0.5 * delta)
    return np.mean(np.where(abs_error <= delta, quadratic, linear))

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def log_cosh(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Log-Cosh Loss: log(cosh(pred - true)), like MSE but less sensitive to outliers.
    """
    error = y_pred - y_true
    return np.mean(np.log(np.cosh(error)))

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def binary_crossentropy(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Binary Cross-Entropy: for binary classification (sigmoid outputs).
    Assumes y_true in {0, 1}. Applies epsilon for numerical stability.
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def categorical_crossentropy(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Categorical Cross-Entropy: for multi-class classification (softmax outputs).
    Assumes one-hot `y_true` and `y_pred` summing to 1.
    """
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -np.sum(y_true * np.log(y_pred), axis=-1).mean()

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def sparse_categorical_crossentropy(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Sparse Categorical Cross-Entropy: like categorical_crossentropy, but y_true contains integer class indices.
    """
    y_pred = np.clip(y_pred, epsilon, 1.0)
    # y_true must be broadcastable to index the last axis
    log_probs = -np.log(y_pred[np.arange(len(y_pred)), y_true.astype(int)])
    return np.mean(log_probs)

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def kullback_leibler_divergence(p: Tensor, q: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    Kullback-Leibler Divergence: KL(P || Q), measures divergence from `q` to `p`.
    Not symmetric.
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1).mean()

# @vectorize(["float64(float64, float64)", "complex128(complex128, complex128)", "float32(float32, float32)", "complex64(complex64, complex64)"])
def cosine_similarity(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Cosine Similarity: measures angle (not magnitude) between vectors.
    Returns 1 for perfectly aligned vectors, 0 for orthogonal.
    """
    y_true_norm = y_true / np.linalg.norm(y_true, axis=-1, keepdims=True)
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=-1, keepdims=True)
    return np.mean(np.sum(y_true_norm * y_pred_norm, axis=-1))


def cosine_distance(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Cosine Distance: 1 - cosine similarity.
    Often used as a loss when alignment matters more than magnitude.
    """
    return 1.0 - cosine_similarity(y_true, y_pred)
