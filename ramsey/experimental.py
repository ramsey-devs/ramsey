"""Experimental and recently contributed methods."""

from ramsey._src.experimental.kernel.non_stationary import Linear, linear
from ramsey._src.experimental.kernel.stationary import (
  ExponentiatedQuadratic,
  Periodic,
  exponentiated_quadratic,
  periodic,
)

SquaredExponential = ExponentiatedQuadratic


__all__ = [
  "ExponentiatedQuadratic",
  "Linear",
  "Periodic",
  "SquaredExponential",
  "exponentiated_quadratic",
  "linear",
  "periodic",
]
