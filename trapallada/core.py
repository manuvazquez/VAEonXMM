import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
from jaxtyping import Array, Float, Int, PyTree

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset

import trapallada.data
import trapallada.model

def fit(data_base_filename: str | pathlib.Path, parameters: dict, rng_key: int | jax.Array):

    # if an `int` (as opposed to a jax `PRNGKey`) was passed...
    if isinstance(rng_key, int):
        rng_key = jr.PRNGKey(rng_key)

    data_base_filename = pathlib.Path(data_base_filename)
    
    ds = trapallada.data.ImageDataset(
        data_base_filename.with_stem(data_base_filename.stem + '_imgs').with_suffix('.pt'),
        data_base_filename.with_stem(data_base_filename.stem + '_labels').with_suffix('.parquet'),
    )
    
    # input_size = tuple(ds.imgs.shape[-2:])
    train_idx, validation_idx = train_test_split(
        np.arange(len(ds)), 
        **parameters['training']['splitting'], 
        shuffle=True,
        stratify=ds.labels
    )
    train_ds, validation_ds = Subset(ds, train_idx), Subset(ds, validation_idx)

    key, subkey = jr.split(rng_key)

    # torch_rng = torch.Generator().manual_seed(parameters['training']['data loader generator seed'])
    torch_seed = jr.bits(subkey, (1,), dtype=jnp.uint32).item()
    torch_rng = torch.Generator().manual_seed(torch_seed)
    train_dl = DataLoader(train_ds, batch_size=parameters['training']['batch_size'], shuffle=True, generator=torch_rng)
    validation_dl = DataLoader(validation_ds, batch_size=parameters['training']['batch_size'], shuffle=False, generator=torch_rng)

    key, subkey = jr.split(rng_key)

    # model = trapallada.model.vae(**parameters['model'])
    model = trapallada.model.vae(**parameters['model'], rng_key=subkey)

    optim = optax.adamw(parameters['training']['learning_rate'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def take_step(
        model,
        opt_state: PyTree,
        x: Float[Array, 'batch 1 height width'],
        dim_z: int,
        x_sd: float,
        rng_key
    ) -> tuple[tuple[eqx.Module], PyTree, Float[Array, '']]:
    
        # (loss_value, (loglikelihood, kl)), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, x, x_sd, rng_key)
        value_aux, grads = eqx.filter_value_and_grad(trapallada.model.loss, has_aux=True)(model, x, dim_z, x_sd, rng_key)
    
        # it returns transformed/processed gradients, and the optimizer state
        updates, opt_state = optim.update(grads, opt_state, model)
    
        model = eqx.apply_updates(model, updates)
    
        return model, opt_state, value_aux

    # key = rng_key
    
    # for the sake of readability
    n_epochs = parameters['training']['n_epochs']
    x_sd = parameters['loss']['x_sd']
    dim_z = parameters['model']['dim_z']
    
    train_loss = [None] * n_epochs
    log_likelihood = [None] * n_epochs
    kl = [None] * n_epochs
    
    valid_loss = [None] * n_epochs
    
    for epoch in range(n_epochs):
    
        train_loss[epoch] = 0.
        log_likelihood[epoch] = 0.
        kl[epoch] = 0.
    
        for step, (x, _) in enumerate(train_dl):
    
            key, subkey = jr.split(key)
    
            model, opt_state, (train_loss_value, (log_likelihood_value, kl_value)) = take_step(model, opt_state, x.numpy(), dim_z, x_sd, subkey)
    
            train_loss[epoch] += train_loss_value
            log_likelihood[epoch] += log_likelihood_value
            kl[epoch] += kl_value
    
            # print(step)
    
            # if (step % )
    
        # # validation
        # valid_loss[epoch] = 0.
        
        # for x, _ in validation_dl:
        
        #     key, subkey = jr.split(key)
        #     valid_loss[epoch] += loss(model, x.numpy(), x_sd, subkey)[0]
    
        # valid_loss[epoch] /= len(validation_dl)
    
        print(epoch, train_loss[epoch] / len(train_dl), valid_loss[epoch])