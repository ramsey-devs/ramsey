import jax
import jax.numpy as np
import haiku as hk
import numpyro.distributions as dist


class NP:
    def __init__(
        self,
        deterministic_encoder: hk.Module,
        latent_encoder: hk.Module,
        decoder: hk.Module,
        do_attention=False
    ):

        self._deterministic_encoder = deterministic_encoder
        self._latent_encoder = latent_encoder
        self._decoder = decoder
        self._do_attention = do_attention
        

    def fit(
        self,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_target: np.ndarray,
        y_target: np.ndarray
    ):
        
        x = np.hstack([x_context, x_target])
        y = np.hstack([y_context, y_target])

        z_context = self._map_xy_to_z_params(x_context, y_context)
        z_all = self._map_xy_to_z_params(x, y)

    def _encode_deterministic(
        self,
        x_context: np.ndarray, 
        y_context: np.ndarray,
        x_target: np.ndarray = None
    ):
        
        input = np.vstack([x_context, y_context])
        z = self._deterministic_encoder(input)
        
        if self._attention:
            z = self._attend_to(z, x_context, x_target)

        z = np.mean(z, axis=-1)
        return z

    def _encode_latent(
        self, 
        x_context: np.ndarray,
        y_context: np.ndarray
    ):
        z = np.vstack([x_context, y_context])
        latent = self._latent_encoder(z)
        latent = np.mean(latent, axis=-1)
        mu, log_sigma = np.split(latent, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.sigmoid(log_sigma)
        
        return dist.Normal(mu, sigma)
    
    def _decode(
        self,
        representation: np.ndarray, 
        x_target:np.ndarray
    ):
        z = np.vstack([representation, x_target])
        z = self._decoder(z)
        mu, log_sigma = np.split(z, 2, axis=-1)
        sigma = 0.1 + 0.9 * jax.nn.softplus(log_sigma)
        
        return dist.Normal(mu, sigma)
    
    
    def _aggregate(z: np.ndarray):
        return np.mean(z, axis=-1)
