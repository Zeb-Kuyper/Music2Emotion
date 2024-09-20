import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_loader import JamendoDataModule
from trainer import MusicClassifier
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping

log = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path="config", config_name="train_config")
def main(config: DictConfig):
    logger = TensorBoardLogger("tb_logs", name="train_audio_classification")
    logger.log_hyperparams(config)
    train_log_dir = logger.log_dir
    log.info("Training starts")

    data_module = JamendoDataModule( **config.dataset )
    data_module.setup()
    trainloader = data_module.train_dataloader()
    valloader = data_module.val_dataloader()

    model = MusicClassifier( **config.model )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          filename="{epoch:02d}-{val_loss:.4f}",
                                          save_top_k=2,
                                          mode="min",
                                          auto_insert_metric_name=False,
                                          save_last=True
                                          )
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, mode="min")
    trainer = pl.Trainer(**config.trainer,
                        callbacks=[checkpoint_callback, early_stop_callback],
                        logger=logger)
    trainer.fit(model, trainloader, valloader)

    if trainer.global_rank == 0:
        best_checkpoint_file = os.path.join(train_log_dir, 'best_checkpoint.txt')
        with open(best_checkpoint_file, 'w') as f:
            f.write(f"Best checkpoint: {checkpoint_callback.best_model_path}\n")
            f.write(f"Version: {logger.version}\n")

if __name__ == '__main__':
    main()
