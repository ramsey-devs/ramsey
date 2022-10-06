import haiku as hk
import jax
import jax.numpy as jnp
import optax

__all__ = ["GP"]


class GPModel(hk.Module):

  def __init__(self, name = None):

    super().__init__(name=name)

    self.a = hk.get_parameter("a", [], init=jnp.ones)
    self.b = hk.get_parameter("b", [], init=jnp.ones)
    self.c = hk.get_parameter("c", [], init=jnp.ones)
    self.d = hk.get_parameter("d", [], init=jnp.ones)

  def __call__(self, x):
    return self.a*jnp.sin(self.b*x+self.c)+self.d

class GP:

    def __init__(self) -> None:
        self.key = jax.random.PRNGKey(23)
        m = hk.transform(self._model_fn)
        self.m = hk.without_apply_rng(m)

    def _model_fn(self, x):
        m = GPModel()
        return m(x)

    def _mse(self, labels, predictions):
        return jnp.square(jnp.subtract(labels,predictions)).mean()

    def train(self, x, y):
        
        print('Start Training')

        opt = optax.adam(1e-3)

        @jax.jit
        def loss(params: hk.Params, x, y):
            y_est = self.m.apply(params, x)
            return self._mse(y,y_est)

        @jax.jit
        def update(params, state, x, y):
            grads = jax.grad(loss)(params, x, y)
            updates, state = opt.update(grads, state)
            params = optax.apply_updates(params, updates)
            return params, state

        @jax.jit
        def evaluate(params, x, y):
            y_est = self.m.apply(params, x)
            error = self._mse(y_est, y)
            return error
    
        params = self.m.init(self.key, x)
        state = opt.init(params)
    
        for step in range(5000):

            params, state = update(params, state, x, y)

            if step % 100 == 0:
                mse_train = evaluate(params, x, y)
                mse_test= evaluate(params, x, y)
                print('  step=%d, mse_train=%.3f, mse_test=%.3f' % (step, mse_train, mse_test))

        self.params = params

    def predict(self, x):
        return self.m.apply(self.params, x)