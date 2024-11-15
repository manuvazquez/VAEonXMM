import pathlib
import argparse
import shutil
import signal
import time
import datetime

import yaml
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import optax
import equinox as eqx
from jaxtyping import Array, Float, PyTree, PRNGKeyArray

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset

import trapallada.data
import trapallada.model
import trapallada.model.signal
import trapallada.model.image
import trapallada.util

import trapallada.logging as log

import mlflow

logger = log.setup_logging(__name__)

# --------

sigint_received = False

def signal_handler(sig, frame):

    global sigint_received
    
    print('Control-C pressed, exiting gracefully...')
    
    sigint_received = True

signal.signal(signal.SIGINT, signal_handler)

# --------

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

# --------

def fit_vae_on_signals(train_dl, validation_dl, parameters: dict) -> tuple[eqx.Module, pd.DataFrame]:

    global sigint_received

    @eqx.filter_jit
    def loss(
            model: eqx.Module,
            x: Float[Array, 'batch channel length'],
            x_sd: float,
            key: PRNGKeyArray
            ) -> tuple[Float[Array, ''], tuple[Float[Array, ''], Float[Array, '']]]:

        # for the sake of readability
        batch_size = len(x)

        x_pred, z_mean, z_sd = jax.vmap(model)(x, jr.split(key, batch_size))

        # reconstruction term
        loglikelihood = dist.Normal(loc=x_pred, scale=x_sd).log_prob(x).sum() / batch_size

        # KL divergence
        kl = -0.5 * jnp.sum(1. + 2.*jnp.log(z_sd) - z_mean**2 - z_sd**2) / batch_size

        return -loglikelihood + kl, (loglikelihood, kl)
    
    @eqx.filter_jit
    def take_step(
        model: eqx.Module,
        opt_state: PyTree,
        x: Float[Array, 'batch 1 length'],
        x_sd: float,
        key: PRNGKeyArray
    ) -> tuple[tuple[eqx.Module], PyTree, Float[Array, '']]:

        value_aux, grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, x, x_sd, key)

        # it returns transformed/processed gradients, and the optimizer state
        updates, opt_state = optim.update(grads, opt_state, model)

        model = eqx.apply_updates(model, updates)

        return model, opt_state, value_aux
    
    def evaluate(model: eqx.Module, dataloader: torch.utils.data.DataLoader, x_sd: float, key: PRNGKeyArray):

        # # in case, there are layers that behave differently in training and inference modes
        # model = eqx.nn.inference_mode(model)

        loss_acc = 0.
        log_likelihood_acc = 0.
        kl_value_acc = 0.

        for _, x, _ in dataloader:

            key, subkey = jr.split(key)

            x = x.numpy()
        
            loss_value, (log_likelihood_value, kl_value) = loss(model, x, x_sd, subkey)
            
            # adding up *averages*
            loss_acc += loss_value.item()
            log_likelihood_acc += log_likelihood_value.item()
            kl_value_acc += kl_value.item()

        # # reverting to training mode
        # model = eqx.nn.inference_mode(model, value=False)

        return loss_acc / len(dataloader), log_likelihood_acc / len(dataloader), kl_value_acc / len(dataloader)
    
    # the VAE is assembled according to the given parameters
    model = trapallada.model.signal.assemble_vae(parameters)
    
    # optimizer
    optim = optax.adamw(parameters['training']['learning_rate'])
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # a `DataFrame` to collect metrics during training
    results_df = trapallada.util.make_results_df(parameters['training']['n_epochs'], metrics=['loss', 'loglikelihood', 'kl'])

    # a random key for drawing samples in the training set...
    key = jr.PRNGKey(parameters['PRNG seeds']['sampling']['training set'])

    # ...a different one for the validation set
    evaluation_key = jr.PRNGKey(parameters['PRNG seeds']['sampling']['validation set'])

    epoch = 0

    while (epoch < parameters['training']['n_epochs']) and not sigint_received:
    # for epoch in range(parameters['training']['n_epochs']):

        for i_batch, (i_sample, x, _) in enumerate(train_dl):

            key, subkey = jr.split(key)

            x = x.numpy()

            model, opt_state, (loss_value, (log_likelihood_value, kl_value)) = take_step(model, opt_state, x, parameters['loss']['x_sd'], subkey)

            # adding up *averages* (of both the loss and the accuracy)
            results_df.loc[epoch, ('training', 'loss')] += loss_value.item()
            results_df.loc[epoch, ('training', 'loglikelihood')] += log_likelihood_value.item()
            results_df.loc[epoch, ('training', 'kl')] += kl_value.item()


        results_df.loc[
            epoch, [('training', 'loss'), ('training', 'loglikelihood'), ('training', 'kl')]
            ] /= len(train_dl)

        # CAVEAT: the same `evaluation_key` is used every time

        # validation
        results_df.loc[epoch, [('validation', 'loss'), ('validation', 'loglikelihood'), ('validation', 'kl')]] = evaluate(
            model, validation_dl, parameters['loss']['x_sd'], evaluation_key)

        print(results_df.loc[epoch].to_frame().T)
        # display(results_df.loc[epoch].to_frame().T)

        epoch += 1

    # if training was interrupted...
    if epoch < parameters['training']['n_epochs']:
        
        results_df = results_df.loc[:epoch-1]

    return model, results_df

def cl_fit_vae_on_signals(args: list = None):
    """
    Command line interface for the `fit_vae_on_signals` function.
    """

    starting_time = time.time()

    # TODO: should `signals_hyperparameters_file` be optional?

    parser = argparse.ArgumentParser(description='Fit a VAE model to a dataset.')

    parser.add_argument('parameters_file', type=pathlib.Path, help='parameters file')
    parser.add_argument('signals_file', type=pathlib.Path, help='signals file')
    parser.add_argument('metadata_file', type=pathlib.Path, help='associated metadata for signals file')
    parser.add_argument('signals_hyperparameters_file', type=pathlib.Path, nargs='?', default=None, help='hyperparameters used in assembling the signals')
    parser.add_argument('-o', '--output-dir', type=pathlib.Path, default=None, help='directory for output files',
        nargs='?')

    command_line_arguments = parser.parse_args(args)

    # for the sake of readability
    dataset_inferred_name = command_line_arguments.signals_file.parent.stem
    training_inferred_name = command_line_arguments.parameters_file.stem

    # if a directory for the output files was NOT passed...
    if not command_line_arguments.output_dir:

        if '.' in command_line_arguments.parameters_file.stem:
            
            print('The *name* of the parameters file should NOT include a dot (.)')

            raise SystemExit
        
        # ...it is inferred from that of the parameters file and the (name of the) dataset determines the output directory
        output_dir = command_line_arguments.parameters_file.parent / dataset_inferred_name / training_inferred_name
        
        # it is created if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f'output directory: {output_dir}')
        
    # parameters file is read
    with command_line_arguments.parameters_file.open() as f:
        
        parameters = yaml.safe_load(f)

    # the dataset is loaded...
    ds = trapallada.data.SignalsDataset(
        command_line_arguments.signals_file, command_line_arguments.metadata_file, **parameters['data']
    )
    
    # ...and split into training and validation sets (here in the form of `DataLoader`s)
    train_dl, validation_dl = trapallada.data.dataloaders_from_dataset(
        ds,
        parameters['training']['train_test_split'] | dict(random_state=parameters['PRNG seeds']['training-validation splitting']),
        parameters['training']['batch_size'], parameters['PRNG seeds']['training dataloader']
    )

    # the model is trained
    model, results_df = fit_vae_on_signals(train_dl, validation_dl, parameters)

    logger.info(f'model has {sum(trapallada.util.get_layers_n_parameters(model).values())} parameters')

    # for the sake of convenience
    eqx_filename = output_dir / 'model.eqx'
    parquet_filename = output_dir / 'in_training_metrics.parquet'
    yaml_filename = output_dir / 'extended_parameters.yaml'
    
    # the weights of the model are saved...
    trapallada.model.image.save_vae(eqx_filename, model)

    # ...and so are the results
    results_df.to_parquet(parquet_filename)

    end_time = time.time()

    logger.info(f'elapsed time: {datetime.timedelta(seconds=end_time - starting_time)} (hours:minutes:seconds)')

    # ----- MLflow logging -----

        # the URL for mlflow
    # mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
    # mlflow.set_tracking_uri(uri='http://meiga05:5000')

    # in order to obtain relative paths down below
    home_dir = pathlib.Path.home()

    # a section in the parameters file for everything related to the dataset used for training
    parameters['dataset'] = {}
    
    # the files used for training are recorded
    parameters['dataset']['files'] = dict(
        signals_file=command_line_arguments.signals_file.relative_to(home_dir).as_posix(),
        metadata_file=command_line_arguments.metadata_file.relative_to(home_dir).as_posix())

    # if a file with hyperparameters for the signals was passed...
    if command_line_arguments.signals_hyperparameters_file:
    
        # ...it is read...
        with command_line_arguments.signals_hyperparameters_file.open() as f:
            
            data_hyperparameters = yaml.safe_load(f)

        # ...and its content added to the `parameters` dictionary
        parameters['dataset']['hyperparameters'] = data_hyperparameters

        # another file used for training is recorded
        parameters['dataset']['files']['signals_hyperparameters_file'] = command_line_arguments.\
            signals_hyperparameters_file.relative_to(home_dir).as_posix()

    with open(yaml_filename, 'w') as f:
        
        yaml.dump(parameters, f, default_flow_style=False)

    mlflow.set_experiment("VAE")

    with mlflow.start_run():
            
        # mlflow.log_params(parameters)
        mlflow.log_params(trapallada.util.flatten_dict(parameters))

        # mlflow.set_tag("Vanilla VAE", "plain")
        mlflow.set_tags({
            'data': dataset_inferred_name,
            'training': training_inferred_name})

        # for every row (epoch) in the results `DataFrame`...
        for i_res, res in results_df.iterrows():
            
            mlflow.log_metrics({'_'.join(k):v for k,v in res.to_dict().items()}, step=i_res)

        mlflow.log_artifact(yaml_filename)
        mlflow.log_artifact(eqx_filename)
        mlflow.log_artifact(parquet_filename)
        # mlflow.log_input(mlflow.data.Dataset(name='XMM-Newton Science Archive', source='https://nxsa.esac.esa.int/nxsa-web/#home'))

        if command_line_arguments.signals_hyperparameters_file:
            
            mlflow.log_artifact(command_line_arguments.signals_hyperparameters_file)