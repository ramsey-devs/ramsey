from typing import Union

import jax.numpy as np

from ramsey._src.gaussian_process.covariance.base import (
    covariance,
    squared_distance,
)


# pylint: disable=invalid-name
def exponentiated_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    sigma=1.0,
    rho: Union[float, np.ndarray] = 1.0,
):
    """
    Exponentiated-quadratic convariance function

    Computes the cross-covariance between two-sets of data points X and Y

    Parameters
    -----------
    x: np.ndarray
        (`n x p`)-dimensional set of data points
    y: np.ndarray
        (`m x p`)-dimensional set of data points
    sigma: float
        the standard deviation of the covariance function
    rho: Union[float, np.ndarray]
        the lengthscale of the covariance function. Can be a float or a
        :math:`p`-dimensional vector if ARD-behaviour is desired

    Returns
    -------
    np.ndarray
        returns a (`n x m`)-dimensional covariance matrix
    """

    def _exponentiated_quadratic(x, y, sigma, rho):
        x = x / rho
        y = y / rho
        dist = squared_distance(x, y)
        cov = sigma * np.exp(-0.5 * dist)
        return cov

    return covariance(
        _exponentiated_quadratic, {"sigma": sigma, "rho": rho}, x, y
    )


rbf = exponentiated_quadratic
squared_exponential = exponentiated_quadratic
