import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
import random
from utilities.constants import *

from torch.autograd import Variable

# Define the Transformer model
class MultitaskLatentTransformerClassifier(nn.Module):
    def __init__(self, feature_dim=8960, num_heads=8, num_layers=6, d_model=512, dropout=0.5):
        super(MultitaskLatentTransformerClassifier, self).__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout

        self.embedding = nn.Linear(feature_dim, d_model)
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, 60, d_model))  # assuming maximum sequence length is 500

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc_mood1 = nn.Linear(d_model, d_model//2)
        self.fc_mood2 = nn.Linear(d_model//2, MOOD_CLASS_SIZE)
        
        self.fc_genre1 = nn.Linear(d_model, d_model//2)
        self.fc_genre2 = nn.Linear(d_model//2, GENRE_CLASS_SIZE)

        self.fc_instr1 = nn.Linear(d_model, d_model//2)
        self.fc_instr2 = nn.Linear(d_model//2, INSTR_CLASS_SIZE)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embed the input
        x = x.view(x.size(0), -1, self.feature_dim)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)

        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        # Transformer's encoder expects input of shape (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)

        # Forward propagate through the Transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        x = x[-1, :, :]  # (batch_size, hidden_dim)

        x = self.dropout_layer(x)

        x_mood1 = self.relu(self.fc_mood1(x))  # Apply ReLU after the first fully connected layer
        logit_mood = self.fc_mood2(x_mood1)

        x_genre1 = self.relu(self.fc_genre1(x))  # Apply ReLU after the first fully connected layer
        logit_genre = self.fc_genre2(x_genre1)

        x_instr1 = self.relu(self.fc_instr1(x))  # Apply ReLU after the first fully connected layer
        logit_instr = self.fc_instr2(x_instr1)


        return logit_mood, logit_genre, logit_instr
