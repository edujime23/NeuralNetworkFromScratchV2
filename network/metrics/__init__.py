import numpy as np

class Metric:
    def from_string(initializer_name: str):
        pass
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        pass