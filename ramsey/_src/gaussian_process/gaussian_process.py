import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.numpy.linalg import cholesky, det, inv
from ramsey.covariance_functions import exponentiated_quadratic as rbf
import chex

__all__ = ["GP"]


class RBFKernel(hk.Module):

  def __init__(self, name = None):

    super().__init__(name=name)

    self.rho = hk.get_parameter("rho", [], init=jnp.ones)
    self.sigma = hk.get_parameter("sigma", [], init=jnp.ones)

  def __call__(self, x1 : jnp.ndarray, x2 : jnp.ndarray):
    
        print('RBFKernel __call__()')
        
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

        cov = rbf(x1,x2, self.sigma, self.rho)
                
        chex.assert_shape(cov, (n,m))

        return cov

class GP:

    def __init__(self) -> None:
        self.key = jax.random.PRNGKey(23)

        # transform kernel
        k = hk.transform(self._kernel_fn)
        self.k = hk.without_apply_rng(k)

    def _kernel_fn(self, x1, x2):
        k = RBFKernel()
        return k(x1, x2)

    # http://krasserm.github.io/2018/03/19/gaussian-processes/
    def _mll_loss(self, params: hk.Params, x, y):

        noise = 0.1
        K = self.k.apply(params, x, x) + noise**2 * jnp.eye(len(x))
        
        data_fit = -0.5 * y.dot(inv(K).dot(y))
        complexity_penalty = -0.5 * jnp.log(det(K))
        norm_const = -0.5 * len(x) * jnp.log(2*jnp.pi)

        loss = data_fit + complexity_penalty + norm_const

        return -loss

    def train(self, x, y):

        self.x_train = x
        self.y_train = y

        opt = optax.adam(1e-3)

        loss = self._mll_loss

        @jax.jit
        def update(params, state, x, y):
            grads = jax.grad(loss)(params, x, y)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state

        @jax.jit
        def evaluate(params, x, y):
            error = loss(params, x, y)
            return error
    
        params = self.k.init(self.key, x[0].reshape((1,1)), x[0].reshape((1,1)))

        print('----------- Initial Parameter ------------')
        print(params)

        state = opt.init(params)
    
        for step in range(1000):

            params, state = update(params, state, x, y)

            if step % 100 == 0:
                print(params)
                loss = evaluate(params, x, y)
                print('  step=%d, loss=%.3f' % (step, loss))

        self.params = params

    def predict(self, x_s):
        noise = 0.1
        K_tt = self.k.apply(self.params, self.x_train, self.x_train) + noise**2 * jnp.eye(len(self.x_train))
        K_ts = self.k.apply(self.params, self.x_train, x_s)
        K_ss = self.k.apply(self.params, x_s, x_s) + 1e-8 * jnp.eye(len(x_s))

        K_tt_inv = inv(K_tt)

         # Equation (7)
        mu_s = K_ts.T.dot(K_tt_inv).dot(self.y_train)

        # Equation (8)
        cov_s = K_ss - K_ts.T.dot(K_tt_inv).dot(K_ts)
    
        return mu_s, cov_s
