from cmath import nan
import haiku as hk
import jax
from jax import numpy as jnp
from jax.numpy import dot
from jax.numpy.linalg import inv, det
from .kernel import Kernel, KernelModule

import optax


class GP():

    def __init__(self, key : hk.PRNGSequence, kernel : Kernel, x_train : jnp.ndarray, y_train : jnp.ndarray, sigma_noise) -> None:

        self._key = key
        self._kernel_core = kernel
        self._x = x_train
        self._y = y_train
        self._stddev_noise = sigma_noise**2
        self._params = {}

        k = hk.transform(self._kernel_fn)
        self._kernel = hk.without_apply_rng(k)

    def _kernel_fn(self, x1, x2):
        k = KernelModule(self._kernel_core)
        return k(x1,x2)

    def __call__(self, method, **kwargs):
        return getattr(self, method)(**kwargs)

    def _mll_loss(self, params: hk.Params, x, y):
        
        K = self._kernel.apply(params, x, x) + self._stddev_noise * jnp.eye(len(x))
        data_fit = y.T.dot(inv(K)).dot(y)
        complexity_penalty = jnp.log(det(K))
        loss = data_fit + complexity_penalty
        loss = jnp.reshape(loss, ())
        return loss

    def train(self, n_iter = 10000, n_restart = 5, init_lr = 1e-3):

        opt = optax.adam(init_lr)

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
   

        for i in range(n_restart):

            print('  Training Loop %d ...' % (i))

            opt_loss = 1e3
            opt_params = {}

            params = self._kernel.init(next(self._key),self._x, self._x)

            print('   ' + str(params))

            state = opt.init(params)
        
            for step in range(n_iter):

                params, state = update(params, state, self._x, self._y)

                if (step % 250 == 0):
                    loss = evaluate(params, self._x, self._y)
                    K = self._kernel.apply(params, self._x, self._x) + self._stddev_noise * jnp.eye(len(self._x))
                    # print('  step=%d, loss=%.3f' % (step, loss))
                    # print('  cond(K)=%.2f' % (jnp.linalg.cond(K)))
                    # print(' ' + str(params))
                    

            if loss < opt_loss:
                opt_loss = loss
                opt_params = params

            print('   ' + str(params))

        self._params = opt_params

        return opt_params, opt_loss

    def predict(self, x_s):

        K_tt = self._kernel.apply(self._params, self._x, self._x) + self._stddev_noise* jnp.eye(len(self._x))
        K_ts = self._kernel.apply(self._params, self._x, x_s)
        K_ss = self._kernel.apply(self._params, x_s, x_s)

        K_tt_inv = jnp.linalg.inv(K_tt)

        mu_s = K_ts.T.dot(K_tt_inv).dot(self._y)

        cov_s = K_ss - K_ts.T.dot(K_tt_inv).dot(K_ts)
    
        return mu_s, cov_s
