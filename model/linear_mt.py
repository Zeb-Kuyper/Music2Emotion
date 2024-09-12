import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np


class FeedforwardModelMT(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedforwardModelMT, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),  # Increased the first layer's units
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        logit = self.model(x)
        # logit = nn.Sigmoid()(self.model(x))
        return logit
    