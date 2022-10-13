import haiku as hk
import time
import jax
from jax import numpy as jnp
from jax.numpy.linalg import inv, det
import matplotlib.pyplot as plt
import optax
import chex
from typing import Tuple

from ramsey._src.gaussian_process.kernel import RBFKernel

from data import sample_from_sine, sample_from_gp_with_rbf_kernel

from ramsey.models.low_level import GP

def main():

  key = hk.PRNGSequence(2)

  print('\n--------------------------------------------------')
  print('Load Dataset')
  n_samples = 20
  rho_rbf = 1
  sigma_rbf = 1
  sigma_noise = 0.1
  #x, y, f = sample_from_sine(next(key), n_samples, sigma_noise, frequency=0.25)
  x, y, f = sample_from_gp_with_rbf_kernel(next(key), n_samples, sigma_noise, sigma_rbf, rho_rbf)
  n_predict = 200
  x_s = jnp.linspace(jnp.min(x), jnp.max(x), num = n_predict)

  print('\n--------------------------------------------------')
  print('Create GP')

  def _gaussian_process(**kwargs) -> hk.Transformed:

    gp = GP(
      kernel=RBFKernel(),
      x_train=x,
      y_train=y
    )
  
    return gp(**kwargs)

  gaussian_process = hk.transform(_gaussian_process)
  gaussian_process = hk.without_apply_rng(gaussian_process)

  print('\n--------------------------------------------------')
  print('Train GP')

  start = time.time()

  n_iter = 10000
  n_restart = 10
  init_lr = 1e-3

  def _mll_loss(params : hk.Params, x : jnp.ndarray, y : jnp.ndarray) -> jnp.ndarray:
    K = gaussian_process.apply(params, method='covariance')
    data_fit = y.T.dot(inv(K)).dot(y)
    complexity_penalty = jnp.log(det(K))
    loss = data_fit + complexity_penalty
    loss = jnp.reshape(loss, ())
    return loss

  opt = optax.adam(init_lr)

  objective = _mll_loss

  @jax.jit
  def update(params : hk.Params, state : chex.ArrayTree, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[hk.Params, chex.ArrayTree]:
    objective_grad = jax.grad(objective)
    grads = objective_grad(params, x, y)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    return params, state

  @jax.jit
  def evaluate(params: hk.Params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      return objective(params, x, y)

  opt_loss = 1e3
  opt_params = {}

  for i in range(n_restart):

    print('  Training Loop %d ...' % (i))

    params = gaussian_process.init(next(key), x_s = x_s)

    print('    Start Parameter: ' + str(params))

    state = opt.init(params)

    for step in range(n_iter):

        params, state = update(params, state, x, y)

        if (step % 500 == 0):
            loss = evaluate(params, x, y)
            #K = gaussian_process.apply(params, method='covariance')
            print('  step=%d, loss=%.3f' % (step, loss))
            # print('  cond(K)=%.2f' % (jnp.linalg.cond(K)))
            # print(' ' + str(params))
            
    if loss < opt_loss:
        opt_loss = loss
        opt_params = params

    print('    End Parameter:   ' + str(params))
    
  params = opt_params

  end = time.time()

  print('\n')
  print('  Best Parameter: ' + str(opt_params))
  print('  Min. Loss: ' + str(opt_loss))

  print('\n')
  print('  Training Duration: %.3fs' % (end - start))

  print('\n--------------------------------------------------')
  print('Predict')
  start = time.time()
  mu, cov = gaussian_process.apply(params, x_s=x_s)
  std = jnp.reshape(jnp.diagonal(cov), (n_predict, 1))
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