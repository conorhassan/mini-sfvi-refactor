import jax
import jax.numpy as jnp
import jax.scipy as jsp

from typing import NamedTuple
from optax import OptState

class State(NamedTuple): 
    mu: jnp.ndarray
    rho: jnp.ndarray
    opt_state: OptState

def init(position, 
         optimizer):
    mu = jax.tree_map(jnp.zeros_like, position)
    rho = jax.tree_map(lambda x: -2.0 * jnp.ones_like(x), position)
    opt_state = optimizer.init((mu, rho))
    return State(mu, rho, opt_state)

def sample_model_params(rng_key, mu, rho, num_samples):
    sigma = jax.tree_map(jnp.exp, rho)
    mu_flatten, unravel_fn = jax.flatten_util.ravel_pytree(mu)
    sigma_flat, _ = jax.flatten_util.ravel_pytree(sigma)
    flatten_sample = (
        jax.random.normal(rng_key, (num_samples,) + mu_flatten.shape) * sigma_flat
        + mu_flatten
    )
    return jax.vmap(unravel_fn)(flatten_sample)

def variational_logdensity(mu, rho):
    sigma_param = jax.tree_map(jnp.exp, rho)

    def variational_logdensity_fn(position):
        logq_pytree = jax.tree_map(jsp.stats.norm.logpdf, position, mu, sigma_param)
        logq = jax.tree_map(jnp.sum, logq_pytree)
        return jax.tree_util.tree_reduce(jnp.add, logq)

    return variational_logdensity_fn

def client_step(rng_key, 
                global_state,
                client_state, 
                client_logdensity_fn, 
                optimizer, 
                num_samples=5): 
    
    global_params = (global_state.mu, global_state.rho) # TODO: see the line below...
    client_params = (client_state.mu, client_state.rho) # TODO: unpack the parameters from something like a `namedtuple`

    def objective_fn(global_params, client_params):
        local_key, _ = jax.random.split(rng_key) 
        mu_G, rho_G = global_params
        mu_L, rho_L = client_params

        z_G = sample_model_params(rng_key, mu_G, rho_G, num_samples)    
        z_L = sample_model_params(local_key, mu_L, rho_L, num_samples)

        mu_G = jax.lax.stop_gradient(mu_G)
        rho_G = jax.lax.stop_gradient(rho_G)
        mu_L = jax.lax.stop_gradient(mu_L)
        rho_L = jax.lax.stop_gradient(rho_L)

        log_client_approx = jax.vmap(variational_logdensity(mu_L, rho_L))(z_L)
        log_client_model = jax.vmap(client_logdensity_fn)(z_L, z_G)

        return (log_client_approx - log_client_model).mean()

    client_objective = objective_fn(global_params, client_params)
    client_grad = jax.grad(objective_fn, argnums=1)(global_params, client_params)
    global_grad = jax.grad(objective_fn, argnums=0)(global_params, client_params)

    updates, new_opt_state = optimizer.update(client_grad, client_state.opt_state, client_params)
    new_client_params = jax.tree_map(lambda p, u: p + u, client_params, updates)
    new_client_state = State(new_client_params[0], new_client_params[1], new_opt_state)

    return new_client_state, (client_objective, global_grad)

def server_step(rng_key,
                global_state, 
                client_updates,
                global_prior_fn, 
                optimizer, 
                num_samples=5):

    client_objectives, client_grads = zip(*client_updates)
    sum_client_objectives = jax.tree_util.tree_map(lambda *xs: jnp.sum(jnp.stack(xs), axis=0), *client_objectives)
    sum_client_grads = jax.tree_util.tree_map(lambda *xs: jnp.sum(jnp.stack(xs), axis=0), *client_grads)

    global_params = (global_state.mu, global_state.rho)

    def objective_fn(global_params):
        mu_G, rho_G = global_params

        z_G = sample_model_params(rng_key, mu_G, rho_G, num_samples)

        mu_G = jax.lax.stop_gradient(mu_G)
        rho_G = jax.lax.stop_gradient(rho_G)

        log_approx = jax.vmap(variational_logdensity(mu_G, rho_G))(z_G)
        log_prior = jax.vmap(global_prior_fn)(z_G)

        return (log_approx - log_prior).mean()
    
    global_objective, global_grad = jax.value_and_grad(objective_fn)(global_params)

    objective = sum_client_objectives + global_objective
    grad = jax.tree_util.tree_map(jnp.add, *[sum_client_grads, global_grad])

    updates, new_opt_state = optimizer.update(grad, global_state.opt_state, global_params)
    new_global_params = jax.tree_map(lambda p, u: p + u, global_params, updates)
    new_global_state = State(new_global_params[0], new_global_params[1], new_opt_state)

    return new_global_state, objective

def fit(seed, 
        client_states, 
        global_state,
        client_logdensity_fns,
        global_prior_fn,
        optimizer,
        num_samples=5,
        num_steps=10):
    
    keys = jax.random.split(jax.random.PRNGKey(seed), num_steps)

    def step(rng_key,
             global_state,
             client_states,
             client_logdensity_fns,
             global_prior_fn,
             optimizer,
             num_samples=5):
            
            client_updates = []
            for client_state, client_logdensity_fn in zip(client_states, client_logdensity_fns):
                _, local_key = jax.random.split(rng_key)
                client_state, client_update = client_step(local_key, global_state, client_state, client_logdensity_fn, optimizer, num_samples=num_samples)
                client_updates.append(client_update)
            global_state, global_update = server_step(rng_key, global_state, client_updates, global_prior_fn, optimizer, num_samples=num_samples)
            return global_state, client_states, global_update
    
    for i in range(num_steps):
        global_state, client_states, global_update = step(keys[i], global_state, client_states, client_logdensity_fns, global_prior_fn, optimizer, num_samples=num_samples)
        print(f"Step {i+1}/{num_steps} | Global Objective: {global_update}")

    return global_state, client_states