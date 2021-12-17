"""
Attention functions
"""

import jax.numpy as np


def uniform_attention(
    x_context: np.DeviceArray,  # pylint: disable=unused-argument
    x_target: np.DeviceArray,  # pylint: disable=unused-argument
    latent: np.DeviceArray,
):
    rep = np.mean(latent, axis=1, keepdims=True)
    return rep
