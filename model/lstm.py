import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn
from .adversarial_classifier import AdversarialClassifier
from losses import A_loss

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_directions = 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(256, self.latent_dim)
        self.fc_sigma = nn.Linear(256, self.latent_dim)

    def forward(self,z):
        _, (hid_state, _) = self.lstm(z)
        hid_state = hid_state.view(self.num_layers, self.num_directions, z.size(0), self.hidden_size)
        hid_state = torch.cat((hid_state[-1, 0], hid_state[-1, 1]), dim=-1)
        mu = self.fc_mu(hid_state)
        sigma = self.fc_sigma(hid_state)
        return mu, sigma
    
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len=4319):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self,z):
        #z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z)
        return out
    
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dim = args.latent_dim

        self.encoder = EncoderLSTM(input_size=500, hidden_size=128, num_layers=1, latent_dim=self.latent_dim)
        self.adversarial = AdversarialClassifier(num_genres=3, latent_dim=128, hidden_dim=128)
        self.decoder = DecoderLSTM(input_size=128, hidden_size=128, num_layers=1)

    def forward(self, x):
        #A_loss(x, t_labels)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        mean, log_var = self.encoder(x)
        sample = torch.randn(self.latent_dim).to(x.get_device())
        std = (0.5 * log_var).exp()
        z = mean + std * sample

        x = self.decoder(z)

        return x, mean, log_var
