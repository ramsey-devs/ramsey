import jax
import haiku as hk
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
from ramsey._src.gaussian_process.data import sample_from_sine, sample_from_gp_with_rbf_kernel
from ramsey._src.gaussian_process.kernel import RBFKernel

from ramsey.models.high_level import GP


def main():

  key = hk.PRNGSequence(6)

  print('Load Dataset')
  n_samples = 10
  sigma_noise = 0.1
  #x, y, f = sample_from_sine(next(key), n_samples, sigma_noise, frequency=0.25)
  x, y, f = sample_from_gp_with_rbf_kernel(next(key), n_samples, sigma_noise, sigma=1, rho=2.5)

  print('Create GP')

  kernel = RBFKernel()
  gp = GP(key, kernel, sigma_noise)
  
  print('Start Training')
  start = time.time()
  gp.train(x, y, n_iter = 1000, n_restart = 20)
  end = time.time()
  print('  Training Duration: %.3fs' % (end - start))

  print('Start Prediction')
  N = 200
  start = time.time()
  x_s = jnp.linspace(jnp.min(x), jnp.max(x), num = N)
  mu, cov = gp.predict(x_s)
  std = jnp.reshape(jnp.diagonal(cov), (N, 1))
  end = time.time()
  print('  Prediction Duration: %.3fs' % (end - start))
  

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