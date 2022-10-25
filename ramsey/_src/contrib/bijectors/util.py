import haiku as hk
from jax import lax
from jax import numpy as jnp


def make_degrees(p, hidden_dims):
    m = [jnp.arange(1, p + 1)]
    for dim in hidden_dims:
        n_min = jnp.minimum(jnp.min(m[-1]), p - 1)
        degrees = jnp.maximum(
            n_min, (jnp.arange(dim) % max(1, p - 1) + min(1, p - 1))
        )
        m.append(degrees)
    return m


def make_masks(degrees):
    masks = [None] * len(degrees)
    for i, (ind, outd) in enumerate(zip(degrees[:-1], degrees[1:])):
        masks[i] = (ind[:, jnp.newaxis] <= outd).astype(jnp.float32)
    masks[-1] = (degrees[-1][:, jnp.newaxis] < degrees[0]).astype(jnp.float32)
    return masks


def make_network(p, hidden_dims, params, ctor):
    masks = make_masks(make_degrees(p, hidden_dims))
    masks[-1] = jnp.tile(masks[-1][..., jnp.newaxis], [1, 1, params])
    masks[-1] = jnp.reshape(masks[-1], [masks[-1].shape[0], p * params])
    layers = []
    for mask in masks:
        layer = ctor(mask)
        layers.append(layer)
    layers.append(hk.Reshape((p, params)))
    return hk.Sequential(layers)


def unstack(x, axis=0):
    return [
        lax.index_in_dim(x, i, axis, keepdims=False)
        for i in range(x.shape[axis])
    ]
