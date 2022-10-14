import haiku as hk
from jax import numpy as jnp
from ramsey.covariance_functions import exponentiated_quadratic as rbf
import abc

class Kernel(abc.ABC):
    """
    Kernel Interface
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray):
        pass

class RBFKernel(hk.Module, Kernel):

  def __init__(self):
    super().__init__()

  def __call__(self, x1 : jnp.ndarray, x2 : jnp.ndarray):
        
        rho = hk.get_parameter("rho", [],dtype=jnp.float_, init=hk.initializers.RandomUniform(minval=jnp.log(1), maxval=jnp.log(5)))
        sigma = hk.get_parameter("sigma", [], init=hk.initializers.RandomUniform(minval=jnp.log(0.1), maxval=jnp.log(3)))

        cov = rbf(x1,x2, sigma = jnp.exp(sigma), rho = jnp.exp(rho))

        return cov