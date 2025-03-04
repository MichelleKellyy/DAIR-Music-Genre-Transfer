import torch
from torch import nn
import torch.nn.functional as F

class GenreClassifier(nn.Module):
    def __init__(self, num_genres, latent_dim, hidden_dim):
        super(GenreClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(latent_dim, num_genres)
        self.dropout = nn.Dropout(0.3)

    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x