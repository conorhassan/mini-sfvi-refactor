import jax.numpy as jnp
from numpyro import distributions as dist

def client_model_log_density(data, z_L, z_G):
    z_G_exp = jnp.exp(z_G[3:]) 
    logp = jnp.sum(z_G[3:])
    logp += dist.Normal(z_G[0], z_G_exp[0]).log_prob(z_L[0])
    logp += dist.Normal(z_G[1], z_G_exp[1]).log_prob(z_L[1])
    logp += dist.Normal(z_G[2], z_G_exp[2]).log_prob(z_L[2])
    logp += dist.Poisson(jnp.exp(z_L[0] + z_L[1]*data["x1"] + z_L[2]*data["x2"])).log_prob(data["y"]).sum(axis=-1)
    return logp

def global_model_log_density(z_G):
    z_G_exp = jnp.exp(z_G[3:])
    logp = jnp.sum(z_G[3:])
    logp += dist.Normal(0, 1).log_prob(z_G[:3]).sum(axis=-1)
    logp += dist.HalfNormal(1).log_prob(z_G_exp).sum(axis=-1)
    return logp