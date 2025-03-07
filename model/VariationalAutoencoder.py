import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
from .genre_classifier import GenreClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DownsampleBlock(nn.Module):
    """Basic residual CNN downsampling block for Encoder class"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownsampleBlock, self).__init__()

        self.left1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.left2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.right = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)

    def forward(self, x):
        x_left = F.relu(self.bn1(self.left1(x)))
        x_left = F.relu(self.bn2(self.left2(x_left)))

        x_right = self.right(x)

        return F.relu(x_left + x_right)


class UpsampleBlock(nn.Module):
    """Basic residual CNN upsampling block for Decoder class"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpsampleBlock, self).__init__()

        self.left1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.left2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.right = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1)

    def forward(self, x):
        x_left = F.relu(self.bn1(self.left1(x)))
        x_left = F.relu(self.bn2(self.left2(x_left)))

        x_right = self.right(x)

        return F.relu(x_left + x_right)



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.latent_dim = args.latent_dim

        self.proj = nn.Conv2d(5, 64, kernel_size=1)
        self.block1 = DownsampleBlock(64, 128, kernel_size=3)
        self.block2 = DownsampleBlock(128, 256, kernel_size=3)
        self.block3 = DownsampleBlock(256, 512, kernel_size=3)
        self.block4 = DownsampleBlock(512, 1024, kernel_size=3)

        self.global_max_pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc_mu = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim * 2)
        )
        self.fc_sigma = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim * 2)
        )

    def forward(self, x):
        x = F.relu(self.proj(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_max_pooling(x).reshape(-1, 1024)

        mu_genre, mu_instance = self.fc_mu(x).reshape(-1, 2, self.latent_dim).transpose(0, 1)
        sigma_genre, sigma_instance = self.fc_sigma(x).reshape(-1, 2, self.latent_dim).transpose(0, 1)

        return mu_genre, sigma_genre, mu_instance, sigma_instance


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.latent_dim = args.latent_dim

        self.proj1 = nn.Linear(2 * self.latent_dim, 8 * 32)#256)
        self.proj2 = nn.Conv2d(1, 1024, kernel_size=1)

        self.block1 = UpsampleBlock(1024, 512, kernel_size=3)
        self.block2 = UpsampleBlock(512, 256, kernel_size=3)
        self.block3 = UpsampleBlock(256, 128, kernel_size=3)
        self.block4 = UpsampleBlock(128, 64, kernel_size=3)

        self.out = nn.Conv2d(64, 5, kernel_size=1)

 
    def forward(self, x):
        x = F.relu(self.proj1(x)).reshape(-1, 1, 8, 32)#256)
        x = F.relu(self.proj2(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out(x)

        return torch.sigmoid(x)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dim = args.latent_dim
        self.encoder = Encoder(args)
        self.genre_classifier = GenreClassifier(num_genres=5, latent_dim=128, hidden_dim=128)
        self.decoder = Decoder(args)

    def forward(self, x):
        mean_genre, log_var_genre, mean_instance, log_var_instance = self.encoder(x)
        sample_genre = torch.randn(self.latent_dim).to(x.get_device())
        sample_instance = torch.randn(self.latent_dim).to(x.get_device())
        std_genre = (0.5 * log_var_genre).exp()
        std_instance = (0.5 * log_var_instance).exp()
        z_genre = mean_genre + std_genre * sample_genre
        z_instance = mean_instance + std_instance * sample_instance

        z = torch.cat((z_genre, z_instance), dim=-1)

        x = self.decoder(z)

        return x, mean_genre, log_var_genre, mean_instance, log_var_instance