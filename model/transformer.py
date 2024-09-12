import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
import random
from utilities.constants import *

from torch.autograd import Variable

# Define the Transformer model
class LatentTransformerClassifier(nn.Module):
    def __init__(self, feature_dim=8960, num_heads=8, num_layers=6, d_model=512, dropout=0.5):
        super(LatentTransformerClassifier, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout

        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 60, d_model))  # assuming maximum sequence length is 500
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)

        self.fc_mood = nn.Linear(d_model, MOOD_CLASS_SIZE)        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.feature_dim)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        x = torch.mean(x, dim=1)
        # x = x[-1, :, :]  # (batch_size, hidden_dim)
        x = self.dropout_layer(x)
        
        # logit_mood = nn.Sigmoid()(self.fc_mood(x))
        logit_mood = self.fc_mood(x)


        return logit_mood
