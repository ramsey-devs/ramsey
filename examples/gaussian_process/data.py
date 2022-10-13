import numpyro.distributions as dist
from jax import numpy as jnp
from jax import random

from ramsey.covariance_functions import exponentiated_quadratic


def sample_from_sine(key, n_samples, sigma_noise, frequency = 1, amplitude=1, offset=0, x_min = -2*jnp.pi, x_max = +2*jnp.pi):

    print('Sample from sine wave')
    print('  n_samples = %d' % (n_samples))
    print('  amplitude = %.3f' % (amplitude))
    print('  frequency = %.3f' % (frequency))
    print('  offset = %.3f' % (offset))
    print('  noise stddev = %.3f' % (sigma_noise**2))

    x = random.uniform(key, (n_samples,1)) * (x_max - x_min) + x_min

    f = amplitude*jnp.sin(2*jnp.pi*frequency*x)+offset

    noise = random.normal(key, (n_samples,1))*sigma_noise

    y = f + noise

    return x,y,f


def sample_from_gp_with_rbf_kernel( 
    key, 
    n_samples : jnp.int_, 
    sigma_noise : jnp.float_, 
    sigma_rbf : jnp.float_ = 1.0, 
    rho_rbf : jnp.float_ = 1.0, 
    x_min : jnp.float_ = -10.0, 
    x_max : jnp.float_ = 10.0):

    """Samples from a GP with RBF Kernel

    Args:
        key: random key for sampling
        n_samples (int): number of samples to draw from the distribution
        sigma_noise (float): std. deviation of the noise added to the samples
        sigma_rbf (float): RBF kernel std. deviation parameter
        rho_rbf (float): RBF kernel length scale parameter.
        x_min (float): lower bound of the sampling interval
        x_max (float): upper bound of the sampling interval

    Returns:
        jnp.ndarray: x (n_samples,1) array with sample locations
        jnp.ndarray: y (n_samples,1) array with noisy samples drawn from GP
        jnp.ndarray: f (n_samples,1) array with sample drawn form GP
    """

    print('  Sample from GP with RBF Kernel')
    print('    n_samples = %d' % (n_samples))
    print('    sigma_rbf = %.3f' % (sigma_rbf))
    print('    rho_rbf = %.3f' % (rho_rbf))
    print('    sigma_noise = %.3f' % (sigma_noise))

    x = jnp.linspace(x_min, x_max, n_samples)

    K_f = exponentiated_quadratic(x, x, sigma = sigma_rbf, rho = rho_rbf)

    f = random.multivariate_normal(key, mean=jnp.zeros(n_samples), cov=K_f, method='svd')
    f = jnp.reshape(f, (n_samples, 1))

    y = random.normal(key, (n_samples,1))*sigma_noise + f
    
    return x,y,f