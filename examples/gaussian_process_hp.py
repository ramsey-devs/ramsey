import haiku as hk
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import optax
from typing import Iterator, NamedTuple
import time



from ramsey.data import sample_from_sinus_function

from ramsey.covariance_functions import exponentiated_quadratic as rbf

from ramsey.models import GP

def mse(labels, predictions):
  return jnp.square(jnp.subtract(labels,predictions)).mean()


class Model(hk.Module):
  def __call__(self, x):
    a = hk.get_parameter("a", [], init=jnp.ones)
    b = hk.get_parameter("b", [], init=jnp.ones)
    c = hk.get_parameter("c", [], init=jnp.ones)
    d = hk.get_parameter("d", [], init=jnp.ones)

    return a*jnp.sin(b*x+c)+d


def model_fn(x):
  m = Model()
  return m(x)

class Dataset():
  def __init__(self, x, y) -> None:
    self.x = x
    self.y = y

def load_dataset(key, num_samples, train_split = 1):
    (x, y), f = sample_from_sinus_function(key, batch_size = 1, num_observations=num_samples)

    n_train = int(num_samples * train_split)

    print('  Training Set: %d/%d Samples (%.1f%%)' % (n_train, num_samples, train_split*100))
    print('  Test Set:     %d/%d Samples (%.1f%%)' % (num_samples-n_train, num_samples, (1-train_split)*100))

    x = jnp.squeeze(x)
    y = jnp.squeeze(y)
    f = jnp.squeeze(f)

    idx = jnp.arange(0, num_samples)
    idx = jax.random.permutation(key, idx, independent=True)

    idx_train = idx[:n_train]
    
    x_train = x[idx_train]
    y_train = y[idx_train]

    x_test = jnp.delete(x,idx_train)
    y_test = jnp.delete(y,idx_train)

    train_data = Dataset(x_train,y_train)
    test_data = Dataset(x_test, y_test)


    return train_data, test_data
    


def main():

  key = jax.random.PRNGKey(23)

  m = hk.transform(model_fn)
  m = hk.without_apply_rng(m)

  opt = optax.adam(1e-3)

  @jax.jit
  def loss(params: hk.Params, x, y):
    y_est = m.apply(params, x)
    return mse(y,y_est)

  @jax.jit
  def update(params, state, x, y):
    grads = jax.grad(loss)(params, x, y)
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    return params, state

  @jax.jit
  def evaluate(params, x, y):
    y_est = m.apply(params, x)
    error = mse(y_est, y)
    return error

  print('Load Dataset')
  train_data, test_data = load_dataset(key, 200, train_split = 0.8)

  print('Start Training')
  start = time.time()

  params = m.init(key, train_data.x)
  state = opt.init(params)
  
  for step in range(5000):

    params, state = update(params, state, train_data.x, train_data.y)

    if step % 100 == 0:
      mse_train = evaluate(params, train_data.x, train_data.y)
      mse_test= evaluate(params, test_data.x, test_data.y)
      print('  step=%d, mse_train=%.3f, mse_test=%.3f' % (step, mse_train, mse_test))


  end = time.time()


  print('  Training Duration: %.3fs' % (end - start))

  x = jnp.concatenate((train_data.x, test_data.x))
  x = jnp.linspace(jnp.min(x), jnp.max(x), num = 200)
  y = m.apply(params, x)
  

  plt.scatter(train_data.x, train_data.y, color='blue', marker='+', label='y_train')
  plt.scatter(test_data.x, test_data.y, color='green', marker='+', label='y_test')
  plt.plot(x, y, color='orange', label='fit')

  plt.legend()
  plt.grid()
  plt.show(block = True)



if __name__ == "__main__":
  main()