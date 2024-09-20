import os
import numpy as np
import pickle
from torch.utils import data
import torchaudio.transforms as T
import torchaudio
import torch
import csv
import pytorch_lightning as pl
from music2latent import EncoderDecoder
import json
import math
from sklearn.preprocessing import StandardScaler

# from utilities.constants import *
def preprocess(latent, seq_len=60):
    if latent.shape[0] > seq_len:
        return latent[:seq_len, :]  # Clip to seq_len
    elif latent.shape[0] < seq_len:
        padding = torch.zeros(seq_len - latent.shape[0], latent.shape[1])
        return torch.cat((latent, padding), dim=0)  # Pad to seq_len
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

def load_instr_info(tsv_file):
    instr_info = {}
    with open(tsv_file, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        headers = next(reader)  # Skip the header row
        for row in reader:
            track_id = row[0]
            tags = row[5:]  # Assuming 'TAGS' is the 6th column and beyond
            instrs = [tag.strip() for tag in tags if '---' in tag]
            instr_info[track_id] = instrs
    return instr_info

class AudioFolder(data.Dataset):
    def __init__(self, **task_args):
        self.task_args = task_args
        self.tr_val = task_args.get('tr_val', "train")
        self.root = task_args.get('root', "./dataset/jamendo")
        self.subset = task_args.get('subset', "moodtheme")
        self.split = task_args.get('split', 0)
        self.segment_type = task_args.get('segment_type', "all")

        fn = f'dataset/jamendo/splits/split-{self.split}/{self.subset}_{self.tr_val}_dict.pickle'
        
        genre_path = 'dataset/jamendo/meta/autotagging_genre.tsv'
        instr_path = 'dataset/jamendo/meta/autotagging_instrument.tsv'

        self.tag_list = np.load('dataset/jamendo/meta/tag_list.npy')
        self.tag_list_genre = list(self.tag_list[:87])
        self.tag_list_instrument = list(self.tag_list[87:127])
        self.tag_list_moodtheme = list(self.tag_list[127:])
        
        self.genre_dic = load_genre_info(genre_path)
        self.instr_dic = load_instr_info(instr_path)
        
        with open(fn, 'rb') as pf:
            self.dictionary = pickle.load(pf)
            # dictionary :
            # {0: {'path': '48/948.mp3', 'duration': 9968.0, 'tags': array([0., 0., 0., 1., ... ,  0.])}, 1: {'path': ... } }

    def __getitem__(self, index):
        path = self.dictionary[index]['path'] # e.g. path: "47/3347.mp3"
        # fn_mp3 = os.path.join(self.root, 'mp3', path[:-3] + 'mp3')

        # --- Mood (emotion) tag label --- 
        y_mood = self.dictionary[index]['tags'] # MOOD TAG LABEL
        y_mood = y_mood.astype('float32')
        y_mood = torch.from_numpy(y_mood)

        # --- Genre tag label (multi-tasking) --- 
        y_genre = np.zeros(87)
        track_num = int(path.split('/')[-1].split('.')[0])
        track_id = f'track_{track_num:07d}'
        genres = self.genre_dic.get(track_id, [])
        for genre in genres:
            if genre in self.tag_list_genre:
                y_genre[self.tag_list_genre.index(genre)] = 1
        y_genre = y_genre.astype('float32')
        y_genre = torch.from_numpy(y_genre)

        # --- Instrument tag label (multi-tasking) --- 
        y_instr = np.zeros(40)
        track_num = int(path.split('/')[-1].split('.')[0])
        track_id = f'track_{track_num:07d}'
        instrs = self.instr_dic.get(track_id, [])
        for instr in instrs:
            if instr in self.tag_list_instrument:
                y_instr[self.tag_list_instrument.index(instr)] = 1
        y_instr = y_instr.astype('float32')
        y_instr = torch.from_numpy(y_instr)

        # --- Librosa feature ---
        fn_librosa = os.path.join(self.root, 'librosa_norm', path[:-3] + 'json')
        if os.path.exists(fn_librosa):
            with open(fn_librosa, 'r') as file:
                data = json.load(file)
            # Replace NaN values with 0
            feature_librosa_list = [
                0 if (value is None or isinstance(value, float) and math.isnan(value)) else value 
                for value in data.values()
            ]
            # feature_librosa_list = list(data.values())
            final_embedding_librosa = torch.tensor(feature_librosa_list, dtype=torch.float32)  # len = 51
        else:
            final_embedding_librosa = torch.zeros(51)

        # --- MERT feature --- 
        fn_mert = os.path.join(self.root, 'mert_30s_all', path[:-4])
        embeddings = []
        
        for filename in os.listdir(fn_mert):
            file_path = os.path.join(fn_mert, filename)
            if os.path.isfile(file_path) and filename.endswith('.npy'):
                segment = np.load(file_path)
                layer_embedding = segment[:, 6, :]  # Shape: (1, 768)
                layer_embedding = np.squeeze(layer_embedding)  # Shape: (768,)
                embeddings.append(layer_embedding)
                if self.segment_type == "f30s":
                    break
        
        embeddings = np.array(embeddings)
        final_embedding_mert = np.mean(embeddings, axis=0)  # Shape: (768,)
        final_embedding_mert = torch.from_numpy(final_embedding_mert)

        # --- Music2latent feature --- 
        fn_music2latent = os.path.join(self.root, 'music2latent', path[:-4])
        latent_list_music2latent_10s = []
        for filename in os.listdir(fn_music2latent):
            file_path = os.path.join(fn_music2latent, filename)
            if os.path.isfile(file_path) and filename.endswith('.npy'):
                latent = np.load(file_path)
                latent_list_music2latent_10s.append(latent)
        latent_array_music2latent_10s = np.vstack(latent_list_music2latent_10s)  # Shape will be [num_segments, 8192]

        final_embedding_m2l = np.mean(latent_array_music2latent_10s, axis=0)  # Shape will be [8192]
        final_embedding_m2l = torch.from_numpy(final_embedding_m2l)

        return {
                "x_mert" : final_embedding_mert,
                "x_m2l" : final_embedding_m2l,
                "x_librosa" : final_embedding_librosa,
                "y_mood" : y_mood, 
                "y_genre" : y_genre, 
                "y_instr" : y_instr, 
                "path": self.dictionary[index]['path']
        }

    def __len__(self):
        return len(self.dictionary)
    
class JamendoDataModule(pl.LightningDataModule):
    def __init__(self, **task_args):
        super().__init__()
        self.task_args = task_args
        self.batch_size = task_args.get('batch_size', 8)
        self.num_workers = task_args.get('num_workers', 4)
        
    def setup(self, stage=None):
        self.train_dataset = AudioFolder(**self.task_args, tr_val='train')
        self.val_dataset = AudioFolder(**self.task_args, tr_val='validation')
        self.test_dataset = AudioFolder(**self.task_args, tr_val='test')

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)


# class DMDDDataModule(pl.LightningDataModule):
#     def __init__(self, root, subset, batch_size, split=0, seq_len=60, num_workers=4, feature_extractor=None):
#         super().__init__()
#         self.root = root
#         self.subset = subset
#         self.batch_size = batch_size
#         self.split = split
#         self.seq_len = seq_len
#         self.num_workers = num_workers
#         self.feature_extractor = feature_extractor

#     def setup(self, stage=None):
#         self.train_dataset = AudioFolder(self.root, self.subset, tr_val='train', split=self.split, seq_len=self.seq_len, feature_extractor=self.feature_extractor)
#         self.val_dataset = AudioFolder(self.root, self.subset, tr_val='validation', split=self.split, seq_len=self.seq_len, feature_extractor=self.feature_extractor)
#         self.test_dataset = AudioFolder(self.root, self.subset, tr_val='test', split=self.split, seq_len=self.seq_len, feature_extractor = self.feature_extractor)

#     def train_dataloader(self):
#         return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

#     def val_dataloader(self):
#         return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

#     def test_dataloader(self):
#         return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)




