from ramsey._src.gaussian_process.kernel.non_stationary import Linear
from ramsey._src.gaussian_process.kernel.stationary import (
    ExponentiatedQuadratic,
    Periodic,
    exponentiated_quadratic,
)

SquaredExponential = ExponentiatedQuadratic
RBF = ExponentiatedQuadratic

__all__ = [
    "ExponentiatedQuadratic",
    "exponentiated_quadratic",
    "RBF",
    "SquaredExponential",
    "Periodic",
    "Linear",
]
