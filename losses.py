import torch
import torch.nn.functional as F

def VAE_loss(y_pred, y, mean, log_var, latent_dim):
    batch_size = y.shape[0]
    bce = F.binary_cross_entropy(y_pred, y, reduction='sum')
    kl_divergence = 0.5 * torch.sum(log_var.exp() * mean ** 2 - log_var - 1)

    return (bce + kl_divergence) / batch_size
