import torch
import matplotlib.pyplot as plt
import equinox as eqx

def img_and_pred(model: tuple[eqx.Module, eqx.Module], x: torch.Tensor, dim_z: int):
    """
    A function to plot an image and its decoded counterpart side by side.

    Args:
        model (tuple): A tuple containing the encoder and decoder models.
        x (torch.Tensor): The input image tensor.
        dim_z (int): The dimension of the latent space.
    """
    
    # axes
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # original image
    axes[0].imshow(x[0].numpy())

    # for the sake of readability
    encoder, decoder = model
    
    # decoded image
    axes[1].imshow(decoder(encoder(x.numpy())[:dim_z])[0])