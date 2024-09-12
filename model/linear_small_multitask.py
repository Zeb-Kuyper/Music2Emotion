import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np

class MultitaskFeedforwardModelSmall(nn.Module):
    def __init__(self, input_size, mood_output_size, genre_output_size, instr_output_size):
        super(MultitaskFeedforwardModelSmall, self).__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 256),  # Increased the first layer's units
            nn.BatchNorm1d(256),  # Batch normalization after linear layer
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch normalization after linear layer
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Output branch for mood prediction
        self.mood_head = nn.Linear(128, mood_output_size)
        
        # Output branch for genre prediction
        self.genre_head = nn.Linear(128, genre_output_size)

        # Output branch for instr prediction
        self.instr_head = nn.Linear(128, instr_output_size)




    def forward(self, x):
        # Shared backbone
        x = self.backbone(x)
        
        # Separate outputs for mood and genre
        mood_logit = self.mood_head(x)
        genre_logit = self.genre_head(x)
        instr_logit = self.instr_head(x)

        # mood_logit = nn.Sigmoid()(self.mood_head(x))
        # genre_logit = nn.Sigmoid()(self.genre_head(x))
        
        return mood_logit, genre_logit, instr_logit
        # return mood_output, genre_output