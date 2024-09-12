import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np

class FeedforwardModelSmall(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedforwardModelSmall, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),  # Increased the first layer's units
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch normalization after linear layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        logit = self.model(x)
        # logit = nn.Sigmoid()(self.model(x))
        return logit
    
    
        # # classifier
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # logit = nn.Sigmoid()(self.dense(x))

        # return logit
