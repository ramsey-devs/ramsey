import haiku as hk
import jax.numpy as np
import jax.random as random

import matplotlib.pyplot as plt

from pax import train_neural_process
from pax.attention import MultiHeadAttention
from pax.data import sample_from_gaussian_process
from pax.models import ANP

key = random.PRNGKey(12)
key, sample_key = random.split(key, 2)

(x_target, y_target), f_target = sample_from_gaussian_process(
    sample_key, batch_size=10, num_observations=200
)


def neural_process(**kwargs):
    dim = 128
    np = ANP(
        decoder=hk.nets.MLP([dim] * 3 + [2]),
        latent_encoder=(
            hk.nets.MLP([dim] * 3),
            hk.nets.MLP([dim, dim * 2])
        ),
        deterministic_encoder=(
            hk.nets.MLP([dim] * 3),
            MultiHeadAttention(num_heads=8, head_size=8)
        )
    )
    return np(**kwargs)


neural_process = hk.transform(neural_process)

key, init_key = random.split(key)
params = neural_process.init(
    init_key, x_context=x_target, y_context=y_target, x_target=x_target
)

yy = neural_process.apply(
    params=params, rng=init_key, x_context=x_target, y_context=y_target, x_target=x_target
)

key, train_key = random.split(key, 2)
n_context, n_target = 30, 40
params, objectives = train_neural_process(
    neural_process,
    params,
    train_key,
    x=x_target,
    y=y_target,
    n_context=n_context,
    n_target=n_target,
    n_iter=10000,
)

key, sample_key = random.split(key, 2)
sample_idxs = random.choice(
    sample_key, x_target.shape[1], shape=(n_context + n_target,), replace=False
)

idxs = [0, 2, 5, 7]

_, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
for _, (idx, ax) in enumerate(zip(idxs, axes.flatten())):
    x = np.squeeze(x_target[idx, :, :])
    f = np.squeeze(f_target[idx, :, :])
    y = np.squeeze(y_target[idx, :, :])

    srt_idxs = np.argsort(x)
    ax.plot(x[srt_idxs], f[srt_idxs], color="blue", alpha=0.75)
    ax.scatter(
        x[sample_idxs[:n_context]],
        y[sample_idxs[:n_context]],
        color="blue",
        marker="+",
        alpha=0.75,
    )

    for i in range(20):
        key, apply_key = random.split(key, 2)
        y_star = neural_process.apply(
            params=params,
            rng=apply_key,
            x_context=x[np.newaxis, sample_idxs, np.newaxis],
            y_context=y[np.newaxis, sample_idxs, np.newaxis],
            x_target=x_target[[idx], :, :],
        ).mean
        x_star = np.squeeze(x_target[[idx], :, :])
        y_star = np.squeeze(y_star)
        ax.plot(x_star[srt_idxs], y_star[srt_idxs], color="black", alpha=0.1)
plt.show()
