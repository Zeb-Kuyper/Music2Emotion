import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from transformers import ASTFeatureExtractor

from data_loader import JamendoDataModule

import torch
from trainer import AudioModel  # Import your Lightning module
from utilities.constants import *

def configure_logging():
    logging.basicConfig(
        filename='model_training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main(config):
    configure_logging()

    feature_extractor = ASTFeatureExtractor()
    data_module = JamendoDataModule(
        root=config.audio_path,
        subset=config.subset,
        batch_size=config.batch_size,
        split=config.split,
        feature_extractor=feature_extractor
    )

    data_module.setup()

    trainloader = data_module.train_dataloader()
    valloader = data_module.val_dataloader()
    testloader = data_module.test_dataloader()
    
    model = AudioModel(config)
    logger = TensorBoardLogger("tb_logs", name="train_audio_classification")

    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          filename=f"{MODEL}-" + "{epoch:02d}",
                                          save_top_k=1,
                                          mode="min",
                                          auto_insert_metric_name=False,
                                          save_last=True)
    
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        devices=[0, 1],  # Specify the GPUs you want to use
        accelerator='gpu',  # Use the GPU accelerator
        callbacks=[checkpoint_callback,],
        logger=logger
    )

    trainer.fit(model, trainloader, valloader)
    trainer.test(model, testloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a music mood classification model.')
    parser.add_argument('--audio_path', type=str, default='./dataset/jamendo')
    parser.add_argument('--subset', type=str, default='moodtheme', choices=['all', 'genre', 'instrument', 'moodtheme', 'top50tags'], help='Subset of tags to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--model_save_path', type=str, default='./saved_models/')
    parser.add_argument('--results_save_path', type=str, default='./results/')
    parser.add_argument('--log_step', type=int, default=1)  # Log step interval
    parser.add_argument('--split', type=int, default=0)
    
    config = parser.parse_args()
    main(config)
