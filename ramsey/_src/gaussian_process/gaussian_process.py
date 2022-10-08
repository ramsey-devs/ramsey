from re import A
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax
from ramsey.covariance_functions import exponentiated_quadratic as rbf
import chex

__all__ = ["GP"]


class LinearKernel(hk.Module):

  def __init__(self, name = None):

    super().__init__(name=name)

  def __call__(self, x1 : jnp.ndarray, x2 : jnp.ndarray):
        
    shape_x1 = jnp.shape(x1)
    shape_x2 = jnp.shape(x2)

    if shape_x1 == (len(x1),):

        x1 = x1.reshape((len(x1),1))
        shape_x1 = jnp.shape(x1)

    if shape_x2 == (len(x2),):

        x2 = x2.reshape((len(x2),1))
        shape_x2 = jnp.shape(x2)

    n = shape_x1[0]
    m = shape_x2[0]

    chex.assert_equal(shape_x1[1], shape_x2[1])

    a = hk.get_parameter("a", [], init=jnp.ones)
    b = hk.get_parameter("b", [], init=jnp.ones)

    cov = a * jnp.dot(x1,x2.T) + b

    chex.assert_shape(cov, (n,m))

    return cov

class RBFKernel(hk.Module):

  def __init__(self, name = None):

    super().__init__(name=name)

  def __call__(self, x1 : jnp.ndarray, x2 : jnp.ndarray):
        
        shape_x1 = jnp.shape(x1)
        shape_x2 = jnp.shape(x2)

        if shape_x1 == (len(x1),):

            x1 = x1.reshape((len(x1),1))
            shape_x1 = jnp.shape(x1)

        if shape_x2 == (len(x2),):

            x2 = x2.reshape((len(x2),1))
            shape_x2 = jnp.shape(x2)

        n = shape_x1[0]
        m = shape_x2[0]

        chex.assert_equal(shape_x1[1], shape_x2[1])

        rho = hk.get_parameter("rho", [], init=jnp.ones)
        sigma = hk.get_parameter("sigma", [], init=jnp.ones)

        cov = rbf(x1,x2, sigma = sigma, rho = rho)

        chex.assert_shape(cov, (n,m))

        return cov

class GP:

    def __init__(self, sigma_noise) -> None:
        self.key = jax.random.PRNGKey(23)

        self.stddev_noise = sigma_noise**2

        # transform kernel
        k = hk.transform(self._kernel_fn)
        self.k = hk.without_apply_rng(k)

    def _kernel_fn(self, x1, x2):
        k = RBFKernel()
        # k = LinearKernel()
        return k(x1, x2)

    def _mll_loss(self, params: hk.Params, x, y):
        
        K = self.k.apply(params, x, x) + self.stddev_noise * jnp.eye(len(x))
        
        data_fit = jnp.dot(y.T, jnp.linalg.inv(K))
        data_fit = jnp.dot(data_fit, y)

        complexity_penalty = jnp.log(jnp.linalg.det(K))

        loss = data_fit + complexity_penalty

        loss = jnp.reshape(loss, ())
        return loss

    def train(self, x, y):

        self.x_train = x
        self.y_train = y

        opt = optax.adam(1e-3)

        objective = self._mll_loss

        @jax.jit
        def update(params, state, x, y):
            objective_grad = jax.grad(objective)
            grads = objective_grad(params, x, y)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state

        @jax.jit
        def evaluate(params, x, y):
            return objective(params, x, y)

    
        params = self.k.init(self.key, x, x)

        state = opt.init(params)
    
        for step in range(10000+1):

            params, state = update(params, state, x, y)

            # loss = evaluate(params, x, y)
            # print('  step=%d, loss=%.3f' % (step, loss))

            # K = self.k.apply(params, self.x_train, self.x_train)
            # print(K)

            if step % 100 == 0:
                loss = evaluate(params, x, y)
                print('  step=%d, loss=%.3f' % (step, loss))
                print(' ' + str(params))

        self.params = params

    def predict(self, x_s):

        print(' ' + str(self.params))

        K_tt = self.k.apply(self.params, self.x_train, self.x_train) + self.stddev_noise* jnp.eye(len(self.x_train))
        K_ts = self.k.apply(self.params, self.x_train, x_s)
        K_ss = self.k.apply(self.params, x_s, x_s)

        K_tt_inv = jnp.linalg.inv(K_tt)

         # Equation (7)
        mu_s = K_ts.T.dot(K_tt_inv).dot(self.y_train)

        # Equation (8)
        cov_s = K_ss - K_ts.T.dot(K_tt_inv).dot(K_ts)
    
        return mu_s, cov_s
