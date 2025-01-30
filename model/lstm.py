import torch
from torch import nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self,z):
        _, (hid_state, _) = self.lstm(z)
        hid_state = hid_state.view(self.num_layers, self.num_directions, z.size(0), self.hidden_size)
        hid_state = torch.cat((hid_state[-1, 0], hid_state[-1, 1]), dim=-1)
        return hid_state
    
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, seq_len):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self,z):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z)
        return out