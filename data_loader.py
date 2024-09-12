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

# def preprocess(latent, seq_len=60):
#     if latent.shape[1] > seq_len:
#         return latent[:, :seq_len]  # Clip to seq_len
#     elif latent.shape[1] < seq_len:
#         padding = torch.zeros(latent.shape[0], seq_len - latent.shape[1])
#         return torch.cat((latent, padding), dim=1)  # Pad to seq_len
#     return latent

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
    def __init__(self, root, subset, tr_val='train', split=0, seq_len=60, resample_rate=16000, feature_extractor = None):
        self.trval = tr_val
        self.root = root
        self.seq_len = seq_len
        self.resample_rate = resample_rate

        self.feature_extractor = feature_extractor
        # self.encdec = EncoderDecoder()

        fn = f'dataset/jamendo/splits/split-{split}/{subset}_{tr_val}_dict.pickle'
        
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

        # 30s stacking
        # latent_list_music2latent_30s = []
        # if len(latent_array_music2latent_10s) >= 3:
        #     for i in range(len(latent_array_music2latent_10s) - 2):
        #         averaged_segment = np.mean(latent_array_music2latent_10s[i:i+3], axis=0)
        #         latent_list_music2latent_30s.append(averaged_segment)
        # else:
        #     averaged_segment = np.mean(latent_array_music2latent_10s, axis=0)
        #     latent_list_music2latent_30s.append(averaged_segment)
        # latent_array_music2latent_30s = np.vstack(latent_list_music2latent_30s) # Shape will be [num_segments-2, 768]

        # average

        # --- Feature normalization ---  
        # mean = torch.mean(averaged_latent_mert)
        # std = torch.std(averaged_latent_mert)
        # if std != 0:  # Prevent division by zero
        #     averaged_latent_mert = (averaged_latent_mert - mean) / std
        
        # mean = torch.mean(averaged_latent_music2latent)
        # std = torch.std(averaged_latent_music2latent)
        # if std != 0:  # Prevent division by zero
        #     averaged_latent_music2latent = (averaged_latent_music2latent - mean) / std

        # mean = torch.mean(feature_librosa_tensor)
        # std = torch.std(feature_librosa_tensor)
        # if std != 0:  # Prevent division by zero
        #     feature_librosa_tensor = (feature_librosa_tensor - mean) / std

        # --- Feature concatenation ---  
        # combined_feature_all = torch.cat((averaged_latent_mert, averaged_latent_music2latent, feature_librosa_tensor))
        
        # 10s stacking
        # min_size = min(latent_array_mert_10s.shape[0], latent_array_music2latent_10s.shape[0])
        # latent_array_mert_10s = latent_array_mert_10s[:min_size]
        # latent_array_music2latent_10s = latent_array_music2latent_10s[:min_size]
        # combined_feature_10s = np.concatenate((latent_array_mert_10s, latent_array_music2latent_10s), axis=1)
        # combined_feature_10s = preprocess(torch.from_numpy(combined_feature_10s), seq_len=self.seq_len)

        # 30s stacking
        # min_size = min(latent_array_mert_30s.shape[0], latent_array_music2latent_30s.shape[0])
        # latent_array_mert_30s = latent_array_mert_30s[:min_size]
        # latent_array_music2latent_30s = latent_array_music2latent_30s[:min_size]
        # combined_feature_30s = np.concatenate((latent_array_mert_30s, latent_array_music2latent_30s), axis=1)
        # combined_feature_30s = preprocess(torch.from_numpy(combined_feature_30s), seq_len=self.seq_len)

        # min_size = min(latent_array_mert.shape[0], latent_array_music2latent.shape[0])
        # latent_array_mert = latent_array_mert[:min_size]
        # latent_array_music2latent = latent_array_music2latent[:min_size]
        # combined_feature = np.concatenate((latent_array_mert, latent_array_music2latent), axis=1)
        # combined_feature = preprocess(torch.from_numpy(combined_feature), seq_len=self.seq_len)
        # combined_feature = torch.from_numpy(combined_feature)

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
    def __init__(self, root, subset, batch_size, split=0, seq_len=60, num_workers=4, feature_extractor=None):
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
        self.test_dataset = AudioFolder(self.root, self.subset, tr_val='test', split=self.split, seq_len=self.seq_len, feature_extractor = self.feature_extractor)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)







# print(averaged_latent.shape)
# assert(False)
# waveform, sampling_rate = torchaudio.load(fn_mp3)
# if waveform.shape[0] > 1:
#     waveform = waveform.mean(dim=0).unsqueeze(0)
# features_music2latent = self.feature_extractor.encode(waveform, max_waveform_length=44100*1, extract_features=True)
# features_music2latent = self.encdec.encode(waveform, extract_features=True)
# features_music2latent = features_music2latent.squeeze().mean(axis=-1)
# resampler = T.Resample(sampling_rate, self.resample_rate)
# waveform = resampler(waveform)
# inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=self.resample_rate, padding="max_length", return_tensors="pt")
# input_values = inputs.input_values.squeeze(0)