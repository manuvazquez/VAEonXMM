import pathlib
import argparse

import yaml
import pandas as pd
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

def fit(data_base_filename: str | pathlib.Path, parameters: dict, rng_key: int | jax.Array) -> tuple[eqx.Module, pd.DataFrame]:

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

    # a random key...
    key, subkey = jr.split(rng_key)

    # ...to compute a seed for a PyTorch `Generator`...
    torch_seed = jr.bits(subkey, (1,), dtype=jnp.uint32).item()
    torch_rng = torch.Generator().manual_seed(torch_seed)

    # ...meant for "batching" in the `DataLoader`'s
    train_dl = DataLoader(train_ds, batch_size=parameters['training']['batch_size'], shuffle=True, generator=torch_rng)
    validation_dl = DataLoader(validation_ds, batch_size=parameters['training']['batch_size'], shuffle=False, generator=torch_rng)

    # another random key is shed...
    key, subkey = jr.split(rng_key)
    
    # ...to initialize the model
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
    valid_log_likelihood = [None] * n_epochs
    valid_kl = [None] * n_epochs
    
    for epoch in range(n_epochs):
    
        train_loss[epoch] = 0.
        log_likelihood[epoch] = 0.
        kl[epoch] = 0.
    
        for x, _ in train_dl:
    
            key, subkey = jr.split(key)
    
            model, opt_state, (train_loss_value, (log_likelihood_value, kl_value)) = take_step(model, opt_state, x.numpy(), dim_z, x_sd, subkey)
    
            train_loss[epoch] += train_loss_value.item()
            log_likelihood[epoch] += log_likelihood_value.item()
            kl[epoch] += kl_value.item()
    
            # print(step)
    
            # if (step % )

        # a random key for validation
        validation_key = jr.fold_in(key, epoch)
    
        # validation
        valid_loss[epoch] = 0.
        valid_log_likelihood[epoch] = 0.
        valid_kl[epoch] = 0.
        
        for x, _ in validation_dl:
        
            validation_key, validation_subkey = jr.split(validation_key)
            (train_loss_value, (log_likelihood_value, kl_value)) = trapallada.model.loss(model, x.numpy(), dim_z, x_sd, validation_subkey)
            
            valid_loss[epoch] += train_loss_value.item()
            valid_log_likelihood[epoch] += log_likelihood_value.item()
            valid_kl[epoch] += kl_value.item()
    
        valid_loss[epoch] /= len(validation_dl)
        valid_log_likelihood[epoch] /= len(validation_dl)
        valid_kl[epoch] /= len(validation_dl)
    
        print(epoch, train_loss[epoch] / len(train_dl), valid_loss[epoch])

    df = pd.DataFrame({
        ('train', 'loss'): train_loss,
        ('train', 'log_likelihood'): log_likelihood,
        ('train', 'KL'): kl,
        ('valid', 'loss'): valid_loss,
        ('valid', 'log_likelihood'): valid_log_likelihood,
        ('valid', 'KL'): valid_kl,
    })

    return model, df
    # return model, (train_loss, log_likelihood, kl), (valid_loss, valid_log_likelihood, valid_kl)

def cl_fit(args: list = None):
    """
    Command line interface for the `fit` function.
    """

    parser = argparse.ArgumentParser(description='Fit a model to a dataset.')

    parser.add_argument(
        'parameters_file', type=argparse.FileType('r'), default='parameters.yaml', help='parameters file',
        nargs='?')
    
    parser.add_argument('database_filename', type=str, help='Database filename')
    
    parser.add_argument('-r', '--rng_key', type=int, default=42, help='Pseudo-random number generator key')

    command_line_arguments = parser.parse_args(args)

    with open(command_line_arguments.parameters_file.name, 'r') as f:
        parameters = yaml.safe_load(f)
    
    model, df = fit(command_line_arguments.database_filename, parameters, command_line_arguments.rng_key)
    
    output_filename = pathlib.Path(
        f'res_z={parameters["model"]["dim_z"]}_epochs={parameters["training"]["n_epochs"]}')

    trapallada.model.save_vae(output_filename.with_suffix('.eqx'), model)
    df.to_parquet(output_filename.with_suffix('.parquet'))