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


def sample_from_gp_with_rbf_kernel(key, n_samples, sigma_noise, sigma = 1, rho = 1, x_min = -10, x_max = 10):

    print('  Sample from GP with RBF Kernel')
    print('    n_samples = %d' % (n_samples))
    print('    sigma = %.3f' % (sigma))
    print('    rho = %.3f' % (rho))
    print('    noise stddev = %.3f' % (sigma_noise**2))


    x = random.uniform(key, (n_samples,1)) * (x_max - x_min) + x_min

    K = exponentiated_quadratic(x, x, sigma = sigma, rho = rho)

    f = random.multivariate_normal(key, mean=jnp.zeros(n_samples), cov=K)

    f = jnp.reshape(f, (n_samples, 1))

    noise = random.normal(key, (n_samples,1))*sigma_noise

    y = f + noise

    return x,y,f