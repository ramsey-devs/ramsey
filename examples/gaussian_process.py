
import haiku as hk
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from ramsey.data import sample_from_sinus_function

from ramsey.covariance_functions import exponentiated_quadratic as rbf


from ramsey.models import GP


def _gaussian_process(mu, cov):

    gp = GP(mu, cov)
    return gp


def _sample(gp):
    ys = gp.sample()
    return ys


key = jax.random.PRNGKey(23)
(x, y), f = sample_from_sinus_function(key, batch_size = 1, num_observations=300)

n = 50

x = jnp.squeeze(x)
y = jnp.squeeze(y)
f = jnp.squeeze(f)



# create datapoints to train GP
idx = jax.random.randint(key, shape=(n,1), minval=0, maxval=jnp.shape(x)[0])
x_train = x[idx]
y_train = y[idx]

# define the prior GP
m = 50
xs = jnp.linspace(jnp.min(x), jnp.max(x), m)
mu = jnp.zeros((m,1))
cov = rbf(xs, xs, sigma=1, rho=1)


gaussian_process = hk.transform(_gaussian_process)
params = gaussian_process.init(key, mu, cov)
gp = gaussian_process.apply(params, key, mu, cov)

sample = hk.transform(_sample)
params = sample.init(key, gp)

plt.scatter(x_train,y_train, color="orange", marker="+")
plt.plot(x, f)

m = 3
keys = jax.random.split(key, m)

for i in range(m):
    ys = sample.apply(params, keys[i], gp)
    plt.scatter(xs, ys, marker="x", s=20, linewidth= 0.5)


plt.grid()
plt.show(block = True)