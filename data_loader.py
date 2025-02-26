from torch.utils import data
import pytorch_lightning as pl

from omegaconf import DictConfig

from Music2Emotion.dataset_loaders.jamendo import JamendoDataset
from Music2Emotion.dataset_loaders.pmemo import PMEmoDataset
from Music2Emotion.dataset_loaders.deam import DEAMDataset
from Music2Emotion.dataset_loaders.emomusic import EmoMusicDataset


DATASET_REGISTRY = {
    "jamendo": JamendoDataset,
    "pmemo": PMEmoDataset,
    "deam": DEAMDataset,
    "emomusic": EmoMusicDataset
}

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
    def setup(self, stage=None):
        # Clear previous dataset lists
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        # Register the datasets and load them
        for dataset_name in self.cfg.datasets:
            dataset_cfg = self.cfg.dataset[dataset_name]

            if dataset_name in DATASET_REGISTRY:
                train_dataset = DATASET_REGISTRY[dataset_name](**dataset_cfg, cfg=self.cfg, tr_val='train')
                val_dataset = DATASET_REGISTRY[dataset_name](**dataset_cfg, cfg=self.cfg, tr_val='validation')
                test_dataset = DATASET_REGISTRY[dataset_name](**dataset_cfg, cfg=self.cfg, tr_val='test')

                self.train_datasets.append(train_dataset)
                self.val_datasets.append(val_dataset)
                self.test_datasets.append(test_dataset)
            else:
                raise ValueError(f"Dataset {dataset_name} not found in registry")

    def train_dataloader(self):
        return [data.DataLoader(ds, batch_size=self.cfg.dataset[ds_name].batch_size, 
                                shuffle=True, num_workers=self.cfg.dataset[ds_name].num_workers, 
                                persistent_workers=True)
                for ds, ds_name in zip(self.train_datasets, self.cfg.datasets)]

    def val_dataloader(self):
        return [data.DataLoader(ds, batch_size=self.cfg.dataset[ds_name].batch_size, 
                                shuffle=False, num_workers=self.cfg.dataset[ds_name].num_workers, 
                                persistent_workers=True)
                for ds, ds_name in zip(self.val_datasets, self.cfg.datasets)]

    def test_dataloader(self):
        return [data.DataLoader(ds, batch_size=self.cfg.dataset[ds_name].batch_size, 
                                shuffle=False, num_workers=self.cfg.dataset[ds_name].num_workers, 
                                persistent_workers=True)
                for ds, ds_name in zip(self.test_datasets, self.cfg.datasets)]



