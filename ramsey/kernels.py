from ramsey._src.contrib.gaussian_process.kernel.non_stationary import Linear#, linear
from ramsey._src.contrib.gaussian_process.kernel.stationary import (
    ExponentiatedQuadratic,
    Periodic,
    exponentiated_quadratic,
   # periodic,
)

SquaredExponential = ExponentiatedQuadratic
RBF = ExponentiatedQuadratic

__all__ = [
    "ExponentiatedQuadratic",
    "exponentiated_quadratic",
    "RBF",
    "SquaredExponential",
    "Periodic",
    #"periodic",
    "Linear",
    #"linear"
]
