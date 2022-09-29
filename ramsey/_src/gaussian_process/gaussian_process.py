import haiku as hk
import jax
import jax.numpy as np
import numpyro.distributions as dist
from chex import assert_axis_dimension, assert_rank
from ramsey._src.family import Family, Gaussian

__all__ = ["GP"]


# pylint: disable=too-many-instance-attributes,duplicate-code
class GP(hk.Module):


    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray
    ):
        assert_axis_dimension(mu, 1, 1)
        assert_axis_dimension(mu, 0, np.shape(cov)[0])
        assert_axis_dimension(mu, 0, np.shape(cov)[1])        

        super().__init__()
        self.mu = mu
        self.cov = cov

    def __call__(self):
        print('I am the _call__ method')

    def sample(self):


        if(np.all(np.linalg.eigvals(self.cov)>0)):
            print('Covariance matirx is positive definite.')
        else:
            print('Warning: Covariance matirx is not positive definite.')

        key = hk.next_rng_key()

        ys = jax.random.multivariate_normal(key, self.mu[:,0], self.cov, method='svd')

        return ys