import torch
from torch import nn
import torch.nn.functional as F

def VAE_loss(y_pred, y, mean, log_var):
    batch_size = y.shape[0]
    bce = F.binary_cross_entropy(y_pred, y, reduction='sum') / y.numel()
    kl_divergence = 0.5 * torch.sum(log_var.exp() * mean ** 2 - log_var - 1) / mean.numel()

    return (bce + kl_divergence)

def A_loss(logits, t_labels):
    loss = F.cross_entropy(logits, t_labels) 
    return loss