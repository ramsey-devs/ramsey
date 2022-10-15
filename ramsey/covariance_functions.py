from ramsey._src.gaussian_process.kernel.stationary import (
    ExponentiatedQuadratic,
    exponentiated_quadratic
)

SquaredExponential = ExponentiatedQuadratic
RBF = ExponentiatedQuadratic
__all__ = [
    "ExponentiatedQuadratic",
    "exponentiated_quadratic",
    "RBF",
    "SquaredExponential",
]
