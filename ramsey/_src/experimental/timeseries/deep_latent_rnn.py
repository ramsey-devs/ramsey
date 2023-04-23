# class _MVNegativeBinomialLSTM(hk.Module):
#     def __init__(self, name="mvnnb_lstm"):
#         super().__init__(name=name)
#         self._net = hk.DeepRNN(
#             [
#                 hk.LSTM(40),
#                 jax.nn.relu,
#                 hk.LSTM(40),
#                 jax.nn.relu,
#                 hk.Linear(1 + 1 + 10),
#             ]
#         )
#
#     def __call__(self, x):
#         p = x.shape[1]
#         outs, _ = hk.dynamic_unroll(self._net, x, self._net.initial_state(p))
#         mu, d, v = np.split(outs, [1, 2], axis=-1)
#         d, v = np.exp(d), v[:, :, None, :]
#         return mu, v, d
#
#
# def _mv_nb_lstm(x):
#     module = _MVNegativeBinomialLSTM()
#     return module(x)
#
#
# def model(y, x):
#     mv_lstm = hk.transform(_mv_nb_lstm)
#     nn = haiku_module("nn", mv_lstm, x=x)
#     m, v, d = nn(x)
#     f = numpyro.sample("f", dist.LowRankMultivariateNormal(m, v, d))
#     mu = numpyro.deterministic("mu", np.log(1.0 + np.exp(f)))
#     kappa = numpyro.param("kappa", 1.0, constraint=constraints.positive)
#     numpyro.sample("y", NegativeBinomial(mu, kappa), obs=y)
