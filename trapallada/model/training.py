from typing import Callable

import numpyro.distributions as dist

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import equinox as eqx
from jaxtyping import Array, Float, PyTree, PRNGKeyArray

import torch

import trapallada.model.nn

def evaluate(
        model: eqx.Module, dataloader: torch.utils.data.DataLoader, loss_function: Callable,
        x_sd: float, mask_non_gti_intervals: bool, key: PRNGKeyArray):

    # # in case, there are layers that behave differently in training and inference modes
    # model = eqx.nn.inference_mode(model)

    loss_acc = 0.
    log_likelihood_acc = 0.
    kl_value_acc = 0.

    for _, x, mask, _ in dataloader:

        key, subkey = jr.split(key)

        x = x.numpy()
        mask = mask.numpy()
    
        loss_value, (log_likelihood_value, kl_value) = loss_function(model, x, mask, x_sd, mask_non_gti_intervals, subkey)
        
        # adding up *averages*
        loss_acc += loss_value.item()
        log_likelihood_acc += log_likelihood_value.item()
        kl_value_acc += kl_value.item()

    # # reverting to training mode (most likely unnecessary since the `model` inside this function is a copy)
    # model = eqx.nn.inference_mode(model, value=False)

    return loss_acc / len(dataloader), log_likelihood_acc / len(dataloader), kl_value_acc / len(dataloader)

@eqx.filter_jit
def loss(
        model: eqx.Module,
        x: Float[Array, 'batch channel length'],
        mask: Float[Array, 'batch 1 length'],
        x_sd: float,
        mask_non_gti_intervals: bool,
        key: PRNGKeyArray
        ) -> tuple[Float[Array, ''], tuple[Float[Array, ''], Float[Array, '']]]:

    # for the sake of readability
    batch_size = len(x)
    overall_good_times = mask.sum()

    x_pred, z_mean, z_sd = jax.vmap(model)(x, jr.split(key, batch_size))

    # breakpoint()

    # if this is a Conditional VAE...
    if isinstance(model, trapallada.model.nn.CVAE):
        
        # ...it is assumed that the last channel is the condition, and hence must not be reconstructed
        x = x.at[:, :-1, :].get()

    assert mask.ndim == 2

    # reconstruction term
    if mask_non_gti_intervals:
        loglikelihood = (dist.Normal(loc=x_pred, scale=x_sd).log_prob(x) * mask[:, None, :]).sum() / overall_good_times
    else:
        loglikelihood = dist.Normal(loc=x_pred, scale=x_sd).log_prob(x).sum() / batch_size

    # KL divergence
    if mask_non_gti_intervals:
        kl = -0.5 * jnp.sum(1. + 2.*jnp.log(z_sd) - z_mean**2 - z_sd**2) / overall_good_times
    else:
        kl = -0.5 * jnp.sum(1. + 2.*jnp.log(z_sd) - z_mean**2 - z_sd**2) / batch_size

    return -loglikelihood + kl, (loglikelihood, kl)

@eqx.filter_jit
def take_step(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    x: Float[Array, 'batch channel length'],
    mask: Float[Array, 'batch 1 length'],
    x_sd: float,
    mask_non_gti_intervals: bool,
    key: PRNGKeyArray
) -> tuple[tuple[eqx.Module], PyTree, Float[Array, '']]:

    # the value (plus some auxiliary variables) and gradient of the model
    value_aux, grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, x, mask, x_sd, mask_non_gti_intervals, key)

    # it returns transformed/processed gradients, and the optimizer state
    updates, opt_state = optimizer.update(grads, opt_state, model)

    model = eqx.apply_updates(model, updates)

    return model, opt_state, value_aux