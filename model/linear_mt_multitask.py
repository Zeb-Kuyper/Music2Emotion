import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np
from utilities.constants import *

class MultitaskFeedforwardModelMT(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultitaskFeedforwardModelMT, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),  # Increased the first layer's units
            nn.ReLU(),
        )
                # Output branch for mood prediction
        self.mood_head = nn.Linear(512, MOOD_CLASS_SIZE)
        
        # Output branch for genre prediction
        self.genre_head = nn.Linear(512, GENRE_CLASS_SIZE)

        # Output branch for instr prediction
        self.instr_head = nn.Linear(512, INSTR_CLASS_SIZE)


    def forward(self, x):
        x = self.model(x)

        # Separate outputs for mood and genre
        mood_logit = self.mood_head(x)
        genre_logit = self.genre_head(x)
        instr_logit = self.instr_head(x)

        # mood_logit = nn.Sigmoid()(self.mood_head(x))
        # genre_logit = nn.Sigmoid()(self.genre_head(x))
        
        return mood_logit, genre_logit, instr_logit

        # logit = nn.Sigmoid()(self.model(x))
        # return logit
    