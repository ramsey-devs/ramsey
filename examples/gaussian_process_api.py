import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time
from ramsey._src.gaussian_process.data import sample_from_sine, sample_from_gp_with_rbf_kernel


from ramsey.models import GP


def main():

  key = jax.random.PRNGKey(18)

  print('Load Dataset')
  n_samples = 10
  sigma_noise = 0.2
  # x, y, f = sample_from_sine(key, n_samples, sigma_noise, f=0.25)
  x, y, f = sample_from_gp_with_rbf_kernel(key, n_samples, sigma_noise, sigma=2, rho=4)

  print('Create GP')
  gp = GP(sigma_noise)
  
  print('Start Training')
  start = time.time()
  gp.train(x, y)
  end = time.time()
  print('  Training Duration: %.3fs' % (end - start))

  print('Start Prediction')
  start = time.time()

  x_s = jnp.linspace(jnp.min(x), jnp.max(x), num = 200)
  mu, cov = gp.predict(x_s)
  end = time.time()
  print('  Prediction Duration: %.3fs' % (end - start))
  

  plt.scatter(x, y, color='blue', marker='+', label='y')
  plt.scatter(x, f, color='green', marker='+', label='f')
  plt.plot(x_s, mu, color='orange', label='fit')

  plt.legend()
  plt.grid()
  plt.show(block = True)



if __name__ == "__main__":
  main()