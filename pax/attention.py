import jax.numpy as np


def uniform_attention(
    x_context: np.DeviceArray,
    x_target: np.DeviceArray,
    latent: np.DeviceArray
):
    rep = np.mean(latent, axis=1, keepdims=True)
    return rep
