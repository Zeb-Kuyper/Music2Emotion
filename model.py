import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification
import torchaudio.transforms as T

from torch.utils.data import DataLoader


class AudioClassificationModel(LightningModule):
    def __init__(self, config):
        super(AudioClassificationModel, self).__init__()
        self.config = config
        # self.model = AutoModelForAudioClassification.from_pretrained(
        #     "MIT/ast-finetuned-audioset-10-10-0.4593",
        #     num_labels=56  # Set the number of classes
        # )
        self.model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )

        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['input_values']
        y = batch['labels']

        logits = self(x).logits
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input_values']
        y = batch['labels']
        logits = self(x).logits
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.batch_size)
