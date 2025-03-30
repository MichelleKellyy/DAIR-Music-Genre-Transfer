import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from .genre_classifier import GenreClassifier

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_directions = 2
        self.embedding = nn.Embedding(30000, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(2 * self.hidden_size, 2 * self.latent_dim)
        self.fc_sigma = nn.Linear(2 * self.hidden_size, 2 * self.latent_dim)

    def forward(self, x):
        input_embeddings = self.embedding(x)
        _, (hid_state, _) = self.lstm(input_embeddings)
        hid_state = hid_state.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
        hid_state = torch.cat((hid_state[-1, 0], hid_state[-1, 1]), dim=-1)

        mu_genre, mu_instance = self.fc_mu(hid_state).reshape(-1, 2, self.latent_dim).transpose(0, 1)
        sigma_genre, sigma_instance = self.fc_sigma(hid_state).reshape(-1, 2, self.latent_dim).transpose(0, 1)

        return mu_genre, sigma_genre, mu_instance, sigma_instance
    
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 30000)

    def forward(self,z, seq_len):
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(z)
        token_pred = self.fc(out)
        return token_pred
    
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dim = args.latent_dim

        self.encoder = EncoderLSTM(input_size=512, hidden_size=256, num_layers=1, latent_dim=self.latent_dim)
        self.genre_classifier = GenreClassifier(num_genres=4, latent_dim=self.latent_dim, hidden_dim=128)
        self.decoder = DecoderLSTM(input_size=2 * self.latent_dim, hidden_size=256, num_layers=1)

    def forward(self, x):
        seq_len = x.shape[1]
        mean_genre, log_var_genre, mean_instance, log_var_instance = self.encoder(x)
        sample_genre = torch.randn(self.latent_dim).to(x.get_device())
        sample_instance = torch.randn(self.latent_dim).to(x.get_device())
        std_genre = (0.5 * log_var_genre).exp()
        std_instance = (0.5 * log_var_instance).exp()
        z_genre = mean_genre + std_genre * sample_genre
        z_instance = mean_instance + std_instance * sample_instance

        z = torch.cat((z_genre, z_instance), dim=-1)

        x = self.decoder(z, seq_len)

        return x, mean_genre, log_var_genre, mean_instance, log_var_instance
