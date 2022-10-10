import haiku as hk
import time
from jax import numpy as jnp
import matplotlib.pyplot as plt

from ramsey._src.gaussian_process.kernel import RBFKernel

from ramsey._src.gaussian_process.data import sample_from_sine, sample_from_gp_with_rbf_kernel

from ramsey.models.low_level import GP

def main():

  key = hk.PRNGSequence(6)

  print('\n--------------------------------------------------')
  print('Load Dataset')
  n_samples = 10
  sigma_noise = 0.1
  #x, y, f = sample_from_sine(next(key), n_samples, sigma_noise, frequency=0.25)
  x, y, f = sample_from_gp_with_rbf_kernel(next(key), n_samples, sigma_noise, sigma=1, rho=2.5)

  print('\n--------------------------------------------------')
  print('Create GP')
  kernel = RBFKernel()
  gp = GP(key, kernel, x, y, sigma_noise)

  print('\n--------------------------------------------------')
  print('Train GP')
  start = time.time()
  gp(method='train', n_iter = 10000, n_restart = 5, init_lr = 1e-3)
  end = time.time()
  print('  Training Duration: %.3fs' % (end - start))

  print('\n--------------------------------------------------')
  print('Predict')
  N = 200
  start = time.time()
  x_s = jnp.linspace(jnp.min(x), jnp.max(x), num = N)
  mu, cov = gp(method='predict', x_s=x_s)
  std = jnp.reshape(jnp.diagonal(cov), (N, 1))
  end = time.time()
  print('  Prediction Duration: %.3fs' % (end - start))

  print('\n--------------------------------------------------')
  print('Plot Results')
  plt.scatter(x, y, color='blue', marker='+', label='y')
  plt.scatter(x, f, color='green', marker='+', label='f')
  plt.plot(x_s, mu, color='orange', label='fit')

  lower_conf_bound = jnp.subtract(mu, 1.96*std)
  upper_conf_bound = jnp.add(mu, 1.96*std)

  plt.fill_between(
    x_s,
    lower_conf_bound.ravel(),
    upper_conf_bound.ravel(),
    alpha=0.5,
    label=r'95% confidence interval',
  )

  plt.legend()
  plt.grid()
  plt.show(block = True)


if __name__ == "__main__":
  main()