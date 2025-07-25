from __future__ import annotations

from dataclasses import dataclass

from network.types.tensor import Tensor


@dataclass(slots=True)
class Gradient:
    """
    Internal: Stores holomorphic and anti-holomorphic gradient components.

    This class is used to represent gradients in a complex-differentiable
    context, utilizing Wirtinger derivatives. It separates the gradient
    into its holomorphic (df/dz) and anti-holomorphic (df/dconj(z)) parts.
    """

    h: Tensor  # Holomorphic component: df/dz
    ah: Tensor  # Anti-holomorphic component: df/dconj(z)

    def __post_init__(self):
        if self.h.shape != self.ah.shape:
            raise ValueError(
                f"Holomorphic and anti-holomorphic shapes mismatch. h={self.h.shape} != ah={self.ah.shape}"
            )

        if not isinstance(self.h, Tensor) or not isinstance(self.ah, Tensor):
            raise ValueError(
                f"Holomorphic and anti-holomorphic must be Tensors. h={type(self.h)} != ah={type(self.ah)}"
            )

        if self.h.dtype != self.ah.dtype:
            raise ValueError(
                f"Holomorphic and anti-holomorphic dtype mismatch. h={self.h.dtype} != ah={self.ah.dtype}"
            )

    @property
    def total(self):
        """
        Returns the sum of the holomorphic and anti-holomorphic gradient components.

        This property is particularly useful when the gradient is for a real-valued
        function with real-valued inputs.

        For functions where both components contribute (e.g., in Wirtinger derivatives
        for real functions where both are 1/2 of the total gradient), this sum
        represents the complete gradient.

        Returns:
            Tensor: The sum of Gradient.h and Gradient.ah.
        """
        return self.h + self.ah
