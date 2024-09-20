import os
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data_loader import JamendoDataModule
from trainer import MusicClassifier
import yaml
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)

def get_latest_version(log_dir):
    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    version_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by version number
    return version_dirs[-1] if version_dirs else None

def save_metrics_and_checkpoint(metrics, checkpoint, output_file):
    data = {
        'checkpoint': checkpoint,
        'metrics': metrics
    }
    with open(output_file, 'w') as f:
        yaml.dump(data, f)

def read_best_checkpoint_info(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint info file not found: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        checkpoint = lines[0].split("Best checkpoint: ")[-1].strip()
    return checkpoint

@hydra.main(version_base=None, config_path="config", config_name="test_config")
def main(config: DictConfig):
    log.info("Testing starts")
    log_base_dir = 'tb_logs/train_audio_classification'
    # log_base_dir = to_absolute_path('tb_logs/train_audio_classification')

    latest_version = get_latest_version(log_base_dir)
    if not latest_version:
        raise FileNotFoundError("No version directories found in log base directory.")
    version_log_dir = os.path.join(log_base_dir, latest_version)
    output_file = os.path.join(version_log_dir, 'test_metrics.txt')

    if config.checkpoint_latest:
        best_checkpoint_file = os.path.join(version_log_dir, 'best_checkpoint.txt')
        ckpt = read_best_checkpoint_info(best_checkpoint_file)
    else:
        ckpt = config.checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    
    log.info(f"Using checkpoint: {ckpt}")
    data_module = JamendoDataModule( **config.dataset )
    data_module.setup()
    model = MusicClassifier.load_from_checkpoint(ckpt, **config.model, output_file=output_file)
    logger = TensorBoardLogger(save_dir=log_base_dir,  
                                name="",  
                                version=latest_version)
    trainer = pl.Trainer(**config.trainer,
                        logger=logger)
    testloader = data_module.test_dataloader()
    trainer.test(model, testloader)

    # metrics = test_results
    # log.info(f"Test metrics: {metrics}")
    # output_file = os.path.join(version_log_dir, 'test_metrics.yaml')
    # save_metrics_and_checkpoint(metrics, ckpt, output_file)

if __name__ == '__main__':
    main()
