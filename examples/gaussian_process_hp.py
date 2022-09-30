import haiku as hk
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import optax
from typing import Iterator, NamedTuple



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


def loss_fn(x, y):
  m = Model() 
  loss = mse(y,m(x))
  return loss

class Dataset():
  def __init__(self, x, y) -> None:
    self.x = x
    self.y = y

def load_dataset(key, num_samples, train_split = 1):
    (x, y), f = sample_from_sinus_function(key, batch_size = 1, num_observations=num_samples)

    n_train = int(num_samples * train_split)

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

    print(jnp.shape(x_train))
    print(jnp.shape(x_test))

    train_data = Dataset(x_train,y_train)
    test_data = Dataset(x_test, y_test)


    return train_data, test_data
    

def update_rule(param, update):
  return param - 0.01 * update



key = jax.random.PRNGKey(23)

loss = hk.transform(loss_fn)
loss = hk.without_apply_rng(loss)

train_data, test_data = load_dataset(key, 200, train_split = 0.8)


params = loss.init(key, train_data.x, train_data.y)


# optimiser = optax.adam(1e-3)


for step in range(1000):

  
  grads = jax.grad(loss.apply)(params, train_data.x, train_data.y)
  params = jax.tree_util.tree_map(update_rule, params, grads)

  if step % 100 == 0:

     y_est_train = loss.apply(params, train_data.x, train_data.y)
     y_est_test = loss.apply(params, test_data.x, test_data.y)
     
     mse_train = mse(y_est_train, train_data.y)
     mse_test= mse(y_est_test, test_data.y)

     print('step=%d, mse_train=%.2f, mse_test=%.2f' % (step, mse_train, mse_test))



plt.scatter(train_data.x, train_data.y, color='blue', marker='+', label='y_train')
plt.scatter(test_data.x, test_data.y, color='green', marker='+', label='y_test')

plt.legend()
plt.grid()
plt.show(block = True)