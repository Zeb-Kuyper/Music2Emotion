import torch.nn as nn
import torch

from torch.autograd import Variable

# Define the LSTM model
class LatentLSTMClassifier(nn.Module):
    def __init__(self, input_size = 2136, hidden_size = 128, num_layers = 2, num_class = 56):
        super(LatentLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Set initial hidden and cell states
        # x = x.view(x.size(0), -1, 9)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_len, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        logit = torch.sigmoid(out)
        return logit
