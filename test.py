import os
import argparse
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import ASTFeatureExtractor
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loader import JamendoDataModule
from trainer import AudioModel  # Import your Lightning module

from utilities.constants import *


def configure_logging():
    logging.basicConfig(
        filename='model_testing.log',
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
    
    # model = AudioModel.load_from_checkpoint(CHECKPOINT)
    model = AudioModel.load_from_checkpoint(CHECKPOINT, config=config)

    logger = TensorBoardLogger("tb_logs", name="test_audio_classification")

    trainer = pl.Trainer(
        devices=[0],  # Specify the GPUs you want to use
        accelerator='gpu',  # Use the GPU accelerator
        logger=logger
    )

    testloader = data_module.test_dataloader()

    trainer.test(model, testloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained music mood classification model.')
    parser.add_argument('--audio_path', type=str, default='./dataset/jamendo')
    parser.add_argument('--subset', type=str, default='moodtheme', choices=['all', 'genre', 'instrument', 'moodtheme', 'top50tags'], help='Subset of tags to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing.')
    # parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the saved model checkpoint.')    
    parser.add_argument('--results_save_path', type=str, default='./results/')
    parser.add_argument('--split', type=int, default=0)
    
    config = parser.parse_args()
    main(config)
