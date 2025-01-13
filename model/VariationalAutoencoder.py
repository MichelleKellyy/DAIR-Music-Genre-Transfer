import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.latent_dim = args.latent_dim

        self.fc1 = nn.Linear(500, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_sigma = nn.Linear(256, self.latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.latent_dim = args.latent_dim

        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 500)

 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return torch.sigmoid(x)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dim = args.latent_dim

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x):
        mean, log_var = self.encoder(x)

        sample = torch.randn(self.latent_dim).to(x.get_device())
        std = (0.5 * log_var).exp()
        z = mean + std * sample

        x = self.decoder(z)

        return x, mean, log_var
