import torch
from torch import nn
import torch.nn.functional as F
from adversarial_classifier import AdversarialClassifier
from losses import A_loss


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
        self.left2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.right = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)

    def forward(self, x):
        x_left = F.relu(self.bn1(self.left1(x)))
        x_left = F.relu(self.bn2(self.left2(x_left)))

        x_right = self.right(x)

        return F.relu(x_left + x_right)



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.latent_dim = args.latent_dim

        self.proj = nn.Conv2d(20, 64, kernel_size=1)
        self.block1 = DownsampleBlock(64, 128, kernel_size=3)
        self.block2 = DownsampleBlock(128, 256, kernel_size=3)
        self.block3 = DownsampleBlock(256, 512, kernel_size=3)
        self.block4 = DownsampleBlock(512, 1024, kernel_size=3)

        self.global_max_pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc_mu = nn.Linear(1024, self.latent_dim)
        self.fc_sigma = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.proj(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_max_pooling(x)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.latent_dim = args.latent_dim

        self.proj1 = nn.Linear(self.latent_dim, 16 * 16)
        self.proj2 = nn.Conv2d(1, 1024, kernel_size=1)

        self.block1 = UpsampleBlock(1024, 512, kernel_size=3)
        self.block2 = UpsampleBlock(512, 256, kernel_size=3)
        self.block3 = UpsampleBlock(256, 128, kernel_size=3)
        self.block4 = UpsampleBlock(128, 64, kernel_size=3)

        self.out = nn.Conv2d(64, 20, kernel_size=1)

 
    def forward(self, x):
        x = F.relu(self.proj1(x))
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
        self.adversarial = AdversarialClassifier()
        self.decoder = Decoder(args)

    def forward(self, x):
        A_loss(x)
        mean, log_var = self.encoder(x)

        sample = torch.randn(self.latent_dim).to(x.get_device())
        std = (0.5 * log_var).exp()
        z = mean + std * sample

        x = self.decoder(z)

        return x, mean, log_var
