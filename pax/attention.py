import jax.numpy as np


def uniform_attention(
    x_context : np.DeviceArray,
    x_target: np.DeviceArray,
    latent: np.DeviceArray
):
    rep = np.mean(latent, axis=1, keepdims=True)
    return rep


class Attention:
    def __call__(self, x_context, x_target, latent):
        _, q = x_context, x_target
        rep = uniform_attention(q, latent)
        return rep
