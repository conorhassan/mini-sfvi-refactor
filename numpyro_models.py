import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def numpyro_multilevel_model(data, n_clients=5, n_obs_per_client=100):
    # Global-level parameters
    mu_0 = numpyro.sample("mu_0", dist.Normal(0.0, 1.0))
    sigma_0 = numpyro.sample("sigma_0", dist.HalfNormal(1.0))
    mu_1 = numpyro.sample("mu_1", dist.Normal(0.0, 1.0))
    sigma_1 = numpyro.sample("sigma_1", dist.HalfNormal(1.0))
    mu_2 = numpyro.sample("mu_2", dist.Normal(0.0, 1.0))
    sigma_2 = numpyro.sample("sigma_2", dist.HalfNormal(1.0))

    # Group-level random effects
    b0 = numpyro.sample("b0", dist.Normal(mu_0, sigma_0), sample_shape=(n_clients,))
    b1 = numpyro.sample("b1", dist.Normal(mu_1, sigma_1), sample_shape=(n_clients,))
    b2 = numpyro.sample("b2", dist.Normal(mu_2, sigma_2), sample_shape=(n_clients,))

    log_lambda_1 = numpyro.deterministic("log_lambda_1", b0[0] + b1[0] * data["client_0_data"]["x1"] + b2[0] * data["client_0_data"]["x2"] + jnp.log(data["client_0_data"]["E"])) 
    numpyro.sample("y_1", dist.Poisson(jnp.exp(log_lambda_1)), obs=data["client_0_data"]["y"])

    log_lambda_2 = numpyro.deterministic("log_lambda_2", b0[1] + b1[1] * data["client_1_data"]["x1"] + b2[1] * data["client_1_data"]["x2"] + jnp.log(data["client_1_data"]["E"]))
    numpyro.sample("y_2", dist.Poisson(jnp.exp(log_lambda_2)), obs=data["client_1_data"]["y"])

    log_lambda_3 = numpyro.deterministic("log_lambda_3", b0[2] + b1[2] * data["client_2_data"]["x1"] + b2[2] * data["client_2_data"]["x2"] + jnp.log(data["client_2_data"]["E"]))
    numpyro.sample("y_3", dist.Poisson(jnp.exp(log_lambda_3)), obs=data["client_2_data"]["y"])

    log_lambda_4 = numpyro.deterministic("log_lambda_4", b0[3] + b1[3] * data["client_3_data"]["x1"] + b2[3] * data["client_3_data"]["x2"] + jnp.log(data["client_3_data"]["E"]))
    numpyro.sample("y_4", dist.Poisson(jnp.exp(log_lambda_4)), obs=data["client_3_data"]["y"])

    log_lambda_5 = numpyro.deterministic("log_lambda_5", b0[4] + b1[4] * data["client_4_data"]["x1"] + b2[4] * data["client_4_data"]["x2"] + jnp.log(data["client_4_data"]["E"]))
    numpyro.sample("y_5", dist.Poisson(jnp.exp(log_lambda_5)), obs=data["client_4_data"]["y"])