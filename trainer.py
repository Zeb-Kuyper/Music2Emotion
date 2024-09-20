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

import logging
import yaml

log = logging.getLogger(__name__)
# from utilities.constants import *
class MusicClassifier(pl.LightningModule):
    def __init__(self, **task_args):
        super(MusicClassifier, self).__init__()
        self.task_args = task_args
        self.encoder = task_args.get('encoder', "MERT")
        self.classifier = task_args.get('classifier', "linear-mt")
        self.lr = task_args.get('lr', 1e-4)
        self.output_file = task_args.get('output_file', None)
        
        # self.config = config
        feature_dim_dict = {
            "MERT": 768,
            "M2L": 8192,
            "LIBROSA": 51
        }
        genre_class_size = 87
        mood_class_size = 56
        instr_class_size = 40

        encoders = self.encoder.split("-")
        self.inputdim = sum(feature_dim_dict[encoder] for encoder in encoders)

        if self.classifier == "transformer":
            self.model = LatentTransformerClassifier( feature_dim= self.inputdim )
        elif self.classifier == "transformer-multitask":
            self.model = MultitaskLatentTransformerClassifier( feature_dim=self.inputdim )
        elif self.classifier == "linear":
            self.model = FeedforwardModel( self.inputdim ,mood_class_size )
        elif self.classifier == "linear-small":
            self.model = FeedforwardModelSmall( self.inputdim ,mood_class_size )
        elif self.classifier == "linear-multitask":
            self.model = MultitaskFeedforwardModel( self.inputdim , mood_class_size, genre_class_size, instr_class_size )
        elif self.classifier == "linear-small-multitask":
            self.model = MultitaskFeedforwardModelSmall( self.inputdim , mood_class_size, genre_class_size, instr_class_size )
        elif self.classifier == "linear-mt":
            self.model = FeedforwardModel( self.inputdim , mood_class_size )
        elif self.classifier == "linear-mt-multitask":
            self.model = MultitaskFeedforwardModelMT( self.inputdim , mood_class_size )

        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.loss_fn = nn.BCELoss()

        self.prd_array = []
        self.gt_array = []
        self.song_array = []
        self.validation_predictions = []
        self.validation_targets = []

        self.trn_loss = MeanMetric()
        self.val_loss = MeanMetric()

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
        encoders = self.encoder.split("-")
        x_combined = torch.cat([feature_dict[encoder] for encoder in encoders], dim=-1)

        if "multitask" in self.classifier:
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
        encoders = self.encoder.split("-")
        x_combined = torch.cat([feature_dict[encoder] for encoder in encoders], dim=-1)

        if "multitask" in self.classifier:
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
        encoders = self.encoder.split("-")
        x_combined = torch.cat([feature_dict[encoder] for encoder in encoders], dim=-1)

        if "multitask" in self.classifier:
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

        log.info('*** Display ROC_AUC_MACRO scores ***')
        log.info(roc_auc)
        log.info('*** Display PR_AUC_MACRO scores ***')
        log.info(pr_auc)
 
        self.prd_array.clear()
        self.gt_array.clear()
        self.song_array.clear()

        if self.output_file is not None:
            with open(self.output_file, 'w') as f:
                f.write(f"ROC_AUC_MACRO: {roc_auc}\n")
                f.write(f"PR_AUC_MACRO: {pr_auc}\n")

        return {"test_roc_auc": roc_auc, "test_pr_auc": pr_auc}
    
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    