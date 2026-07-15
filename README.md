Source code to reproduce the results in the paper:

> Vázquez, Manuel A., Cid-Sueiro, Jesús, & López-Santiago Javier (2026). Unsupervised sifting of XMM-Newton EPIC observations using Variational Autoencoders. RAS Techniques and Instruments.

# Install

## uv 

Using [uv](https://docs.astral.sh/uv/)
```
uv add --extra cpu git+https://github.com/manuvazquez/VAEonXMM
```
for CPU or
```
uv add --extra cuda git+https://github.com/manuvazquez/VAEonXMM
```
for GPU.

## pip

Allegedly (though not tested), you can also install it using [pip](https://pypi.org/project/pip/)

```
pip install 'trapallada[cpu] @ git+https://github.com/manuvazquez/VAEonXMM@main'
```
for CPU or
```
pip install 'trapallada[cuda] @ git+https://github.com/manuvazquez/VAEonXMM@main'
```
for GPU.

Notice the name of the library is `trapallada` (not that of the repo, which stems from the above paper).

# Commands

After installing the library, a few commands become available:

- `trap_download_obs.py`

- `trap_pack_obs.py`

- `trap_fit_vae_on_signals.py`

- `trap_grid_fit_vae_on_signals.py`