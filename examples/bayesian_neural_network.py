"""
Bayesian Neural Network
=======================

This example implements the training and prediction of a
Bayesian Neural Network.
Predictions from a Haiku MLP fro the same data are shown
as a reference.

References
----------

[1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
    "Weight Uncertainty in Neural Networks".
    ICML, 2015.
"""

from jax.config import config
config.update("jax_enable_x64", True)

import haiku as hk
import jax
from jax import numpy as jnp, random
from distrax import Normal
from ramsey.data import sample_from_gaussian_process
from ramsey.contrib.models import BayesianLinear, BayesianNeuralNetwork
from ramsey.contrib.train import train_model
import matplotlib.pyplot as plt

rng = hk.PRNGSequence(12356)

def data(key, rho, sigma, n=1000):

    (x_target, y_target), f_target = sample_from_gaussian_process(
        key, batch_size=1, num_observations=n, rho=rho, sigma=sigma
    )

    return (x_target.reshape(n,1), y_target.reshape(n,1)), f_target.reshape(n,1)

def _bayesian_nn(**kwargs):

    def _inv_softplus(x):
        return jnp.log(jnp.exp(x)-1)

    w_prior = Normal(loc=0, scale=1.05)
    b_prior = Normal(loc=0, scale=1.05)

    w_rho_init = hk.initializers.TruncatedNormal(
        mean=_inv_softplus(6.71e-3),
        stddev=0.1)

    b_rho_init = hk.initializers.TruncatedNormal(
        mean=_inv_softplus(6.71e-3),
        stddev=0.1)

    bayes_1 = BayesianLinear(8, w_prior=w_prior, b_prior=b_prior,
                                w_rho_init=w_rho_init, b_rho_init=b_rho_init,
                                activation=jax.nn.leaky_relu)

    bayes_2 = BayesianLinear(8, w_prior=w_prior, b_prior=b_prior,
                                w_rho_init=w_rho_init, b_rho_init=b_rho_init,
                                activation=jax.nn.leaky_relu)

    bayes_3 = BayesianLinear(1, w_prior=w_prior, b_prior=b_prior,
                                w_rho_init=w_rho_init, b_rho_init=b_rho_init,
                                activation=None)

    layers = [bayes_1, bayes_2, bayes_3]

    lklh_log_sigma_init = hk.initializers.RandomUniform(jnp.log(0.1), jnp.log(1.0))

    bayesian_nn = BayesianNeuralNetwork(layers, lklh_log_sigma_init)

    return bayesian_nn(**kwargs)

def _std_nn(x, **kwargs):

    standard_nn = hk.nets.MLP([8,8,1], with_bias=True)
    return standard_nn(x, **kwargs)

def _plot_loss(bayesian_nn_loss, std_nn_loss):

    _, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6))

    losses = [bayesian_nn_loss, std_nn_loss]
    labels = ['neg. ELBO','MSE']
    titles = ['Bayesian NN', 'Standard NN']

    [(ax.plot(
            jnp.arange(len(losses[_])),
            jnp.squeeze(losses[_]),
            # color="black",
            alpha=0.5,
            label=labels[_]
        ),
    ax.set_title(titles[_]),
    ax.set_xlabel('iteration'),
    ax.grid(),
    ax.legend()) for (_,ax) in enumerate(axes)]

def _plot_bayesian_nn_samples(rng, ax, nn, nn_params, x, n_samples = 10):

    y_s = [nn.apply(params=nn_params,rng=next(rng),x=x) for _ in range(n_samples)]

    srt_idxs = jnp.argsort(jnp.squeeze(x))

    [ax.plot(
        jnp.squeeze(x)[srt_idxs], jnp.squeeze(y)[srt_idxs], color="blue", alpha = 0.1
    ) for y in y_s]

def _plot_bayesian_nn_mean_std(rng, ax, nn, nn_params, x, n_samples = 100):

    y_s = [nn.apply(params=nn_params,rng=next(rng),x=x) for _ in range(n_samples)]

    y_s = jnp.squeeze(jnp.stack([y for y in y_s]))

    mean = jnp.mean(y_s, axis=0)
    sigma = jnp.std(y_s, axis=0)

    srt_idxs = jnp.argsort(jnp.squeeze(x))

    ax.plot(    jnp.squeeze(x)[srt_idxs], jnp.squeeze(mean)[srt_idxs],
                color="blue", alpha = 0.5,
                label='Posterior Mean')

    ax.fill_between(jnp.squeeze(x)[srt_idxs],
                 jnp.squeeze(mean+1.644854*sigma)[srt_idxs],
                 jnp.squeeze(mean-1.644854*sigma)[srt_idxs],
                 color="blue", alpha=0.05, label=r'90% Posterior Interval')


def _plot_standard_nn(rng, ax, standard_nn, standard_nn_params, x):

    y_s = standard_nn.apply(params=standard_nn_params,rng=next(rng),x=x)

    srt_idxs = jnp.argsort(jnp.squeeze(x))

    ax.plot(    jnp.squeeze(x)[srt_idxs], jnp.squeeze(y_s)[srt_idxs],
                color="green", alpha = 0.5,
                label='Predictions Standard NN')


def plot(   rng,
            bayesian_nn, bayesian_nn_params, bayesian_nn_loss,
            standard_nn, standard_nn_params, standard_nn_loss,
            x, f, x_train, y_train):

    _plot_loss(bayesian_nn_loss, standard_nn_loss)

    _, ax = plt.subplots(figsize=(8, 3))
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(f)[srt_idxs],
        color="black",
        alpha=0.5,
        label="Latent function " + r"$f \sim GP$"
    )
    ax.scatter(
        jnp.squeeze(x_train),
        jnp.squeeze(y_train),
        color="red",
        marker="+",
        alpha=0.2,
        label="Training data"
    )

    _plot_bayesian_nn_mean_std(rng, ax, bayesian_nn, bayesian_nn_params, x)
    _plot_bayesian_nn_samples(rng, ax, bayesian_nn, bayesian_nn_params, x)
    _plot_standard_nn(rng, ax, standard_nn, standard_nn_params, x)

    ax.legend(
        loc="best",
        frameon=False,
    )
    ax.grid()

    ax.set_frame_on(False)
    plt.show()

def bayesian_nn_objective(par, key, model, x, y):
    """
    The BNN training objective is the approx. ELBO (see [1]).
    The number m controls how many samples are used for this
    approximation.
    """

    m = 5

    keys = random.split(key, m)

    obj_list = []

    for _ in range(m):
        obj = model.apply(
            params=par,
            rng=keys[_],
            x=x,
            y=y
        )

        obj_list.append(obj)

    obj = jnp.mean(jnp.asarray(obj_list))

    return obj

def std_nn_objective(par, key, model, x, y):

    y_star = model.apply(
        params=par,
        rng=key,
        x=x
    )

    obj = jnp.mean(jnp.square(y_star - y))
    return obj

def choose_training_samples(key, x, y, n_train):

    train_idxs = random.choice(
        key, jnp.arange(x.shape[0]), shape=(n_train,1), replace=False
    )
    x_train, y_train = jnp.take(x,train_idxs), jnp.take(y, train_idxs)
    return x_train, y_train

def run():

    n_train = 100
    n_iter = 5000
    stepsize = 0.01

    (x, y), f = data(next(rng), 0.7, 1.0)
    x_train, y_train = choose_training_samples(next(rng), x, y, n_train)

    key = next(rng)

    print('\nTrain Bayesian Neural Network')
    bayesian_nn = hk.transform(_bayesian_nn)
    bayesian_nn_params = bayesian_nn.init(key, x=x_train, y=y_train)
    bayesian_nn_params, bayesian_nn_loss = train_model(
        bayesian_nn,
        bayesian_nn_objective,
        bayesian_nn_params,
        key,
        x=x_train,
        y=y_train,
        n_iter=n_iter,
        stepsize=stepsize
    )

    print('\nTrain Standard Neural Network')
    std_nn = hk.transform(_std_nn)
    std_nn_params = std_nn.init(key, x=x_train)
    std_nn_params, std_nn_loss = train_model(
        std_nn,
        std_nn_objective,
        std_nn_params,
        key,
        x=x_train,
        y=y_train,
        n_iter=n_iter,
        stepsize=stepsize
    )

    plot(
        rng,
        bayesian_nn,
        bayesian_nn_params,
        bayesian_nn_loss,
        std_nn,
        std_nn_params,
        std_nn_loss,
        x,
        f,
        x_train,
        y_train
    )

if __name__ == "__main__":
    run()
