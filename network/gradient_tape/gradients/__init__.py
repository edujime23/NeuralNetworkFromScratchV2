from .aggregation import AggregationGradients
from .angle import AngleGradients
from .arithmetic import ArithmeticGradients
from .exponential import ExponentialGradients
from .hyperbolic import HyperbolicGradients
from .logarithmic import LogarithmicGradients
from .matrix import MatrixGradients
from .rounding import RoundingGradients
from .shape import ShapeGradients
from .special import SpecialGradients
from .trigonometric import TrigonometricGradients
from .util import ensure_shape
from .numerical import numerical_derivative

GRADIENTS: dict[str, object] = {ensure_shape.__name__: ensure_shape}

for gradients in (
    AggregationGradients,
    AngleGradients,
    ArithmeticGradients,
    ExponentialGradients,
    HyperbolicGradients,
    LogarithmicGradients,
    MatrixGradients,
    RoundingGradients,
    ShapeGradients,
    SpecialGradients,
    TrigonometricGradients,
):
    for name, func in gradients.__dict__.items():
        if name.startswith('__') or not callable(func):
            continue
        GRADIENTS[name] = func   