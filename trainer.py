import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np
from collections import OrderedDict
from torchmetrics import MeanMetric, MaxMetric, Accuracy

from model.transformer import LatentTransformerClassifier
from model.transformer_multitask import MultitaskLatentTransformerClassifier

from model.linear import FeedforwardModel
from model.linear_small import FeedforwardModelSmall
from model.linear_multitask import MultitaskFeedforwardModel
from model.linear_small_multitask import MultitaskFeedforwardModelSmall
from model.linear_mt import FeedforwardModelMT
from model.linear_mt_multitask import MultitaskFeedforwardModelMT


from utilities.constants import *

class MusicClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MusicClassifier, self).__init__()
        self.config = config
        feature_dim_dict = {
            "MERT": 768,
            "M2L": 8192,
            "LIBROSA": 51
        }
        encoders = ENCODER.split("-")
        self.inputdim = sum(feature_dim_dict[encoder] for encoder in encoders)        

        if CLASSIFIER == "transformer":
            self.model = LatentTransformerClassifier( feature_dim= self.inputdim )
        elif CLASSIFIER == "transformer-multitask":
            self.model = MultitaskLatentTransformerClassifier( feature_dim=self.inputdim )
        elif CLASSIFIER == "linear":
            self.model = FeedforwardModel( self.inputdim ,56 )
        elif CLASSIFIER == "linear-small":
            self.model = FeedforwardModelSmall( self.inputdim ,56 )
        elif CLASSIFIER == "linear-multitask":
            self.model = MultitaskFeedforwardModel( self.inputdim , MOOD_CLASS_SIZE, GENRE_CLASS_SIZE, INSTR_CLASS_SIZE )
        elif CLASSIFIER == "linear-small-multitask":
            self.model = MultitaskFeedforwardModelSmall( self.inputdim , MOOD_CLASS_SIZE, GENRE_CLASS_SIZE, INSTR_CLASS_SIZE )
        elif CLASSIFIER == "linear-mt":
            self.model = FeedforwardModel( self.inputdim ,56 )
        elif CLASSIFIER == "linear-mt-multitask":
            self.model = MultitaskFeedforwardModelMT( self.inputdim ,56 )


        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.loss_fn = nn.BCELoss()

        self.prd_array = []
        self.gt_array = []
        self.song_array = []
        self.validation_predictions = []
        self.validation_targets = []

        self.trn_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def adapt_classes(self, subset):
        if subset == 'all':
            return 183
        elif subset == 'genre':
            return 87
        elif subset == 'instrument':
            return 40
        elif subset == 'moodtheme':
            return 56
        elif subset == 'top50tags':
            return 50
        else:
            raise ValueError(f"Unknown subset: {subset}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_mert = batch["x_mert"]
        x_m2l = batch["x_m2l"]
        x_librosa = batch["x_librosa"]
        feature_dict = {
            "MERT": x_mert,
            "M2L": x_m2l,
            "LIBROSA": x_librosa
        }
        encoders = ENCODER.split("-")
        x_combined = torch.cat([feature_dict[encoder] for encoder in encoders], dim=-1)

        if "multitask" in CLASSIFIER:
            y_mood = batch["y_mood"]
            y_genre = batch["y_genre"]
            y_instr = batch["y_instr"]
            logit_mood, logit_genre, logit_instr  = self(x_combined)
            
            loss_mood = self.loss_fn(logit_mood, y_mood)
            loss_genre = self.loss_fn(logit_genre, y_genre)
            loss_instr = self.loss_fn(logit_instr, y_instr)

            loss = loss_mood + loss_genre + loss_instr
        else:
            y = batch["y_mood"]
            logits = self(x_combined)
            loss = self.loss_fn(logits, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x_combined.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x_mert = batch["x_mert"]
        x_m2l = batch["x_m2l"]
        x_librosa = batch["x_librosa"]
        feature_dict = {
            "MERT": x_mert,
            "M2L": x_m2l,
            "LIBROSA": x_librosa
        }
        encoders = ENCODER.split("-")
        x_combined = torch.cat([feature_dict[encoder] for encoder in encoders], dim=-1)

        if "multitask" in CLASSIFIER:
            y_mood = batch["y_mood"]
            y_genre = batch["y_genre"]
            y_instr = batch["y_instr"]

            logit_mood, logit_genre, logit_instr  = self(x_combined)
            
            loss_mood = self.loss_fn(logit_mood, y_mood)
            loss_genre = self.loss_fn(logit_genre, y_genre)
            loss_instr = self.loss_fn(logit_instr, y_instr)

            loss = loss_mood + loss_genre + loss_instr
        else:
            y = batch["y_mood"]

            logits = self(x_combined)
            loss = self.loss_fn(logits, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x_combined.size(0))        
        return loss

    def test_step(self, batch, batch_idx):
        x_mert = batch["x_mert"]
        x_m2l = batch["x_m2l"]
        x_librosa = batch["x_librosa"]
        feature_dict = {
            "MERT": x_mert,
            "M2L": x_m2l,
            "LIBROSA": x_librosa
        }
        encoders = ENCODER.split("-")
        x_combined = torch.cat([feature_dict[encoder] for encoder in encoders], dim=-1)

        if "multitask" in CLASSIFIER:
            y_mood = batch["y_mood"]
            y_genre = batch["y_genre"]
            y_instr = batch["y_instr"]
            y = y_mood

            logit_mood, logit_genre, logit_instr  = self(x_combined)
            logits = logit_mood
            
            loss_mood = self.loss_fn(logit_mood, y_mood)
            loss_genre = self.loss_fn(logit_genre, y_genre)
            loss_instr = self.loss_fn(logit_instr, y_instr)

            loss = loss_mood + loss_genre + loss_instr
        else:
            y = batch["y_mood"]

            logits = self(x_combined)
            loss = self.loss_fn(logits, y)
                
        probs = torch.sigmoid(logits)
        self.prd_array.extend(probs.detach().cpu().numpy())
        self.gt_array.extend(y.detach().cpu().numpy())
        self.song_array.extend(batch["path"])

        return loss
    
    def on_test_end(self):
        roc_auc, pr_auc = self.get_auc(self.prd_array, self.gt_array)

        print('*** Display ROC_AUC_MACRO scores ***')
        print(roc_auc)

        print('*** Display PR_AUC_MACRO scores ***')
        print(pr_auc)

        self.prd_array.clear()
        self.gt_array.clear()
        self.song_array.clear()

    def get_auc(self, prd_array, gt_array):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)
        try:
            roc_auc = metrics.roc_auc_score(gt_array, prd_array, average='macro')
            pr_auc = metrics.average_precision_score(gt_array, prd_array, average='macro')
        except ValueError as e:
            print(f"Error computing metrics: {e}")
            roc_auc = None
            pr_auc = None
        
        return roc_auc, pr_auc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
