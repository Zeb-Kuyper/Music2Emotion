import os
import numpy as np
import pickle
from torch.utils import data
import torchaudio.transforms as T
import torchaudio

import torch
import csv
import pytorch_lightning as pl

def preprocess(latent, seq_len=2136):
    if latent.shape[1] > seq_len:
        return latent[:, :seq_len]  # Clip to seq_len
    elif latent.shape[1] < seq_len:
        padding = torch.zeros(latent.shape[0], seq_len - latent.shape[1])
        return torch.cat((latent, padding), dim=1)  # Pad to seq_len
    return latent

def load_genre_info(tsv_file):
    genre_info = {}
    with open(tsv_file, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        headers = next(reader)  # Skip the header row
        for row in reader:
            track_id = row[0]
            tags = row[5:]  # Assuming 'TAGS' is the 6th column and beyond
            genres = [tag.strip() for tag in tags if '---' in tag]
            genre_info[track_id] = genres
    return genre_info

class AudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', split=0, seq_len=2136, feature_extractor=None, resample_rate=16000):
        self.trval = tr_val
        self.root = root
        self.seq_len = seq_len
        self.feature_extractor = feature_extractor
        self.resample_rate = resample_rate

        fn = f'dataset/jamendo/splits/split-{split}/{subset}_{tr_val}_dict.pickle'
        genre_path = 'dataset/jamendo/meta/autotagging_genre.tsv'
        
        self.tag_list = np.load('dataset/jamendo/meta/tag_list.npy')
        self.tag_list_genre = list(self.tag_list[:87])
        self.tag_list_instrument = list(self.tag_list[87:127])
        self.tag_list_moodtheme = list(self.tag_list[127:])
        
        self.genre_dic = load_genre_info(genre_path)
        
        with open(fn, 'rb') as pf:
            self.dictionary = pickle.load(pf)
    
    def __getitem__(self, index):
        path = self.dictionary[index]['path']
        fn_mp3 = os.path.join(self.root, 'mp3', path[:-3] + 'mp3')
        
        waveform, sampling_rate = torchaudio.load(fn_mp3)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)
        resampler = T.Resample(sampling_rate, self.resample_rate)
        waveform = resampler(waveform)
        
        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=self.resample_rate, padding="max_length", return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)
        
        tags = self.dictionary[index]['tags']
        tags_genre = np.zeros(87)
        track_num = int(path.split('/')[-1].split('.')[0])
        track_id = f'track_{track_num:07d}'
        
        genres = self.genre_dic.get(track_id, [])
        for genre in genres:
            if genre in self.tag_list_genre:
                tags_genre[self.tag_list_genre.index(genre)] = 1

        return {"x_amt": input_values, 
                "y_mood" : tags.astype('float32'), 
                "y_genre" : tags_genre.astype('float32'), 
                "path": self.dictionary[index]['path']
                }

    def __len__(self):
        return len(self.dictionary)

class JamendoDataModule(pl.LightningDataModule):
    def __init__(self, root, subset, batch_size, split=0, seq_len=2136, num_workers=19, feature_extractor=None):
        super().__init__()
        self.root = root
        self.subset = subset
        self.batch_size = batch_size
        self.split = split
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.feature_extractor = feature_extractor

    def setup(self, stage=None):
        self.train_dataset = AudioFolder(self.root, self.subset, tr_val='train', split=self.split, seq_len=self.seq_len, feature_extractor=self.feature_extractor)
        self.val_dataset = AudioFolder(self.root, self.subset, tr_val='validation', split=self.split, seq_len=self.seq_len, feature_extractor=self.feature_extractor)
        self.test_dataset = AudioFolder(self.root, self.subset, tr_val='test', split=self.split, seq_len=self.seq_len, feature_extractor=self.feature_extractor)




    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
