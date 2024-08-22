import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification

import numpy as np

class AudioModel(pl.LightningModule):
    def __init__(self, config):
        super(AudioModel, self).__init__()
        self.config = config
        self.model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.loss_fn = nn.BCELoss()  # Define your loss function

        self.num_class = self.adapt_classes(config.subset)
        in_features = self.model.classifier.dense.in_features
        self.model.classifier.dense = nn.Linear(in_features, self.num_class)

        # Initialize lists to store predictions and targets during validation
        self.predictions = []
        self.targets = []
        self.tag_list = [f"tag_{i}" for i in range(self.num_class)]  # Replace with actual tag names if available

        # Initialize lists to accumulate predictions and ground truth labels
        self.prd_array = []
        self.gt_array = []

        self.song_array = []
        # self.start_t = time.time()

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

    def forward(self, x_amt):
        outputs = self.model(x_amt)
        logits = outputs.logits
        return torch.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        x_amt = batch["x_amt"]
        y_mood = batch["y_mood"]
        logits = self(x_amt)
        loss = self.loss_fn(logits, y_mood)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x_amt.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x_amt = batch["x_amt"]
        y_mood = batch["y_mood"]
        logits = self(x_amt)
        val_loss = self.loss_fn(logits, y_mood)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=x_amt.size(0))

        return {"val_loss": val_loss, "logits": logits, "y_mood": y_mood}
    
    def test_step(self, batch, batch_idx):
        x_amt = batch["x_amt"]
        y_mood = batch["y_mood"]
        song_paths = batch["path"]

        # Move inputs and labels to GPU if available
        x_amt = x_amt.to(self.device)
        y_mood = y_mood.to(self.device)

        logits = self(x_amt)
        loss = self.loss_fn(logits, y_mood)

        # Store predictions and ground truths
        self.prd_array.extend(logits.detach().cpu().numpy())
        self.gt_array.extend(y_mood.detach().cpu().numpy())

        self.song_array.extend(song_paths)

        return {"test_loss": loss}
    

    def on_test_end(self):
        # Calculate ROC AUC and PR AUC
        roc_auc, pr_auc, _, _ = self.get_auc(self.prd_array, self.gt_array)

        # Save the results
        # results_path = os.path.join(self.config.results_save_path, "results", "VERSION")
        # os.makedirs(results_path, exist_ok=True)

        print('*** Display ROC_AUC_MACRO scores ***')
        print(roc_auc)

        print('*** Display PR_AUC_MACRO scores ***')
        print(pr_auc)

    def get_auc(self, prd_array, gt_array):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        # Compute the metrics
        roc_auc = metrics.roc_auc_score(gt_array, prd_array, average='macro')
        pr_auc = metrics.average_precision_score(gt_array, prd_array, average='macro')

        try:
            roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
            pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)
        except Exception as e:
            print(f"Error computing detailed metrics: {e}")
            roc_auc_all, pr_auc_all = None, None

        return roc_auc, pr_auc, roc_auc_all, pr_auc_all

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)







