import torch
from torch import nn
import torch.nn.functional as F

def VAE_loss(y_pred, y, mean_genre, log_var_genre, mean_instance, log_var_instance):
    batch_size = y.shape[0]
    bce = F.binary_cross_entropy(y_pred, y, reduction='sum') / y.numel()
    kl_divergence_genre = 0.5 * torch.sum(log_var_genre.exp() + mean_genre ** 2 - log_var_genre - 1) / mean_genre.numel()
    kl_divergence_instance = 0.5 * torch.sum(log_var_instance.exp() + mean_instance ** 2 - log_var_instance - 1) / mean_instance.numel()

    return (bce + kl_divergence_genre + kl_divergence_instance)

def shuffle(mean_genre):
    n = mean_genre.shape[0]
    shuffle_indices = torch.randperm(n)
    shuffled_mean_genre = mean_genre.clone()
    shuffled_mean_genre = mean_genre[shuffle_indices]

    return shuffled_mean_genre

def adversarial_loss(logits, t_labels):
    loss = F.binary_cross_entropy_with_logits(logits, t_labels) 
    return loss

def classification_loss(logits, t_labels):
    loss = F.cross_entropy(logits, t_labels) 
    return loss