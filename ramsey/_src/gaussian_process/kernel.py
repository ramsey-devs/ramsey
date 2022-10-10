import haiku as hk
from jax import numpy as jnp
from ramsey.covariance_functions import exponentiated_quadratic as rbf
import chex
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

class KernelUtils():


    def shape_input(x : jnp.ndarray):
        
        shape = jnp.shape(x)
        
        if shape == (len(x),):

            x = x.reshape((len(x),1))
            
        return x

    def check_input_dim(x1 : jnp.ndarray, x2 : jnp.ndarray):

        shape_x1 = jnp.shape(x1)
        shape_x2 = jnp.shape(x2)

        chex.assert_equal(shape_x1[1], shape_x2[1])

    def check_cov_dim(x1 : jnp.ndarray, x2 : jnp.ndarray, cov : jnp.ndarray):

        n = jnp.shape(x1)[0]
        m = jnp.shape(x2)[0]

        chex.assert_shape(cov, (n,m))



class LinearKernel(hk.Module, Kernel):

  def __init__(self):
    super().__init__()

  def __call__(self, x1 : jnp.ndarray, x2 : jnp.ndarray):
        
    x1 = KernelUtils.shape_input(x1)
    x2 = KernelUtils.shape_input(x2)

    KernelUtils.check_input_dim(x1, x2)

    a = hk.get_parameter("a", [], init=jnp.ones)
    b = hk.get_parameter("b", [], init=jnp.ones)

    cov = a * jnp.dot(x1,x2.T) + b

    KernelUtils.check_cov_dim(x1, x2, cov)

    return cov

class RBFKernel(hk.Module, Kernel):

  def __init__(self):
    super().__init__()

  def __call__(self, x1 : jnp.ndarray, x2 : jnp.ndarray):
        
        x1 = KernelUtils.shape_input(x1)
        x2 = KernelUtils.shape_input(x2)

        KernelUtils.check_input_dim(x1, x2)

        rho = hk.get_parameter("rho", [], init=hk.initializers.RandomUniform(minval=0, maxval=10))
        # sigma = hk.get_parameter("sigma", [], init=hk.initializers.RandomUniform(minval=1, maxval=5))
        sigma = 1

        cov = rbf(x1,x2, sigma = sigma, rho = rho)

        KernelUtils.check_cov_dim(x1, x2, cov)

        return cov