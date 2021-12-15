import jax.numpy as np
from pax.gaussian_process.covariance.base import squared_distance


def exponentiated_quadratic(x: np.ndarray, y: np.ndarray, sigma=1.0, rho=1.0):
    x = x / rho
    y = y / rho
    dist = squared_distance(x, y)
    cov = sigma * np.exp(-0.5 * dist)
    return cov


rbf = exponentiated_quadratic
squared_exponential = exponentiated_quadratic
