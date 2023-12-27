import jax
import jax.numpy as jnp
import numpyro.distributions as dist

def generate_multilevel_params_and_data(seed, n_clients=5, n_obs_per_client=100):
    # Split the main PRNG key
    main_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(main_key, 13)  # Total 13 distinct operations requiring keys

    # Generate covariates
    x1 = dist.Normal(0.0, 1.0).sample(keys[0], (n_clients, n_obs_per_client))
    x2 = dist.Bernoulli(0.5).sample(keys[1], (n_clients, n_obs_per_client))
    E = dist.Uniform(500, 1500).sample(keys[2], (n_clients, n_obs_per_client))

    # Set global-level parameters
    mu_0 = dist.Normal(0.0, 1.0).sample(keys[3])
    # sigma_0 = dist.HalfNormal(1.0).sample(keys[4])
    sigma_0 = 0.1
    mu_1 = dist.Normal(0.0, 1.0).sample(keys[5])
    # sigma_1 = dist.HalfNormal(1.0).sample(keys[6])
    sigma_1 = 0.1
    mu_2 = dist.Normal(0.0, 1.0).sample(keys[7])
    # sigma_2 = dist.HalfNormal(1.0).sample(keys[8])
    sigma_2 = 0.1

    # Set group-level random effects
    b0 = dist.Normal(mu_0, sigma_0).sample(keys[9], (n_clients,))
    b1 = dist.Normal(mu_1, sigma_1).sample(keys[10], (n_clients,))
    b2 = dist.Normal(mu_2, sigma_2).sample(keys[11], (n_clients,))

    # Set observation-level random errors
    eps = dist.Normal(0, 0.5).sample(keys[12], (n_clients, n_obs_per_client))

    params = {
        'mu_0': mu_0, 'sigma_0': sigma_0, 
        'mu_1': mu_1, 'sigma_1': sigma_1, 
        'mu_2': mu_2, 'sigma_2': sigma_2, 
        'b0': b0, 'b1': b1, 'b2': b2, 
        'eps': eps
    }

    data = {'total_data': {'x1': x1, 'x2': x2, 'E': E, 'y': []}}
    # Generate outcome for each client
    for client in range(n_clients):
        client_key = jax.random.split(keys[12], n_clients)[client]
        log_lambda = b0[client] + b1[client] * x1[client] + b2[client] * x2[client] + eps[client] + jnp.log(E[client])
        y = dist.Poisson(jnp.exp(log_lambda)).sample(client_key)
        data[f"client_{client}_data"] = {'x1': x1[client], 'x2': x2[client], 'E': E[client], 'y': y}
        data['total_data']['y'].append(y)

    # Convert list of y arrays to a single array
    data['total_data']['y'] = jnp.concatenate(data['total_data']['y'], axis=0)

    return params, data