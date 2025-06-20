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
from .numerical import numerical_derivative, WirtingerDifferentiator, DerivativeConfig
from .util import complex_log, epsilon

GRADIENTS: dict[str, object] = {}

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

__all__ = [
    'GRADIENTS',
    'numerical_derivative',
    'WirtingerDifferentiator',
    'DerivativeConfig'
]

__all__.extend(GRADIENTS.keys())