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
import pandas as pd

class JamendoDataset(data.Dataset):
    def __init__(self, **task_args):
        self.task_args = task_args
        self.tr_val = task_args.get('tr_val', "train")
        self.root = task_args.get('root', "./dataset/jamendo")
        self.subset = task_args.get('subset', "moodtheme")
        self.split = task_args.get('split', 0)
        self.segment_type = task_args.get('segment_type', "all")
        self.cfg = task_args.get('cfg')

        fn = f'dataset/jamendo/splits/split-{self.split}/{self.subset}_{self.tr_val}_dict.pickle'
        
        genre_path = 'dataset/jamendo/meta/autotagging_genre.tsv'
        instr_path = 'dataset/jamendo/meta/autotagging_instrument.tsv'

        self.tag_list = np.load('dataset/jamendo/meta/tag_list.npy')
        self.tag_list_genre = list(self.tag_list[:87])
        self.tag_list_instrument = list(self.tag_list[87:127])
        self.tag_list_moodtheme = list(self.tag_list[127:])
        
        self.genre_dic = self.load_genre_info(genre_path)
        self.instr_dic = self.load_instr_info(instr_path)

        # key_signatures = [
        #     "A major", "A- major", "B major", "B- major", "C major", "C# major", "D major", "E major", 
        #     "E- major", "F major", "F# major", "G major", "None", "a minor", "b minor", "b- minor", 
        #     "c minor", "c# minor", "d minor", "e minor", "e- minor", "f minor", "f# minor", 
        #     "g minor", "g# minor"
        # ]

        # Separate tonic and mode
        tonic_signatures = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        mode_signatures = ["major", "minor"]  # Major and minor modes

        self.tonic_to_idx = {tonic: idx for idx, tonic in enumerate(tonic_signatures)}
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(mode_signatures)}

        self.idx_to_tonic = {idx: tonic for tonic, idx in self.tonic_to_idx.items()}
        self.idx_to_mode = {idx: mode for mode, idx in self.mode_to_idx.items()}

        # self.key_to_idx = {key: idx for idx, key in enumerate(key_signatures)}
        # self.idx_to_key = {idx: key for key, idx in self.key_to_idx.items()}


        # Load the CSV file
        file_path_m2va = 'dataset/jamendo/meta/moodtag_va_scores.csv'  # Replace with the path to your CSV file
        data_m2va = pd.read_csv(file_path_m2va)

        # Extract Valence and Arousal columns and convert them to NumPy arrays
        self.valence = data_m2va['Valence'].to_numpy()
        self.arousal = data_m2va['Arousal'].to_numpy()

        # print("Valence array:", valence)
        # print("Arousal array:", arousal)


        with open('dataset/jamendo/meta/chord.json', 'r') as f:
            self.chord_to_idx = json.load(f)
        with open('dataset/jamendo/meta/chord_inv.json', 'r') as f:
            self.idx_to_chord = json.load(f)
            self.idx_to_chord = {int(k): v for k, v in self.idx_to_chord.items()}  # Ensure keys are ints
        
        with open('dataset/emomusic/meta/chord_root.json') as json_file:
            self.chordRootDic = json.load(json_file)
        with open('dataset/emomusic/meta/chord_attr.json') as json_file:
            self.chordAttrDic = json.load(json_file)            

            
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

        v_score = y_mood*self.valence
        a_score = y_mood*self.arousal

        v_score = np.mean( v_score[v_score!=0] )
        a_score = np.mean( a_score[a_score!=0] )

        y_valence = torch.tensor(v_score, dtype=torch.float32)
        y_arousal = torch.tensor(a_score, dtype=torch.float32)

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

        # --- Chord feature ---
        fn_chord = os.path.join(self.root, 'chord', 'lab3', path[:-4] + ".lab")
        chords = []
        
        if not os.path.exists(fn_chord):
            chords.append((float(0), float(0), "N"))
        else:
            with open(fn_chord, 'r') as file:
                for line in file:
                    start, end, chord = line.strip().split()
                    chords.append((float(start), float(end), chord))

        encoded = []
        encoded_root= []
        encoded_attr=[]
        durations = []
        for start, end, chord in chords:
            chord_arr = chord.split(":")
            if len(chord_arr) == 1:
                chordRootID = self.chordRootDic[chord_arr[0]]
                if chord_arr[0] == "N" or chord_arr[0] == "X":
                    chordAttrID = 0
                else:
                    chordAttrID = 1
            elif len(chord_arr) == 2:
                chordRootID = self.chordRootDic[chord_arr[0]]
                chordAttrID = self.chordAttrDic[chord_arr[1]]
            encoded_root.append(chordRootID)
            encoded_attr.append(chordAttrID)

            if chord in self.chord_to_idx:
                encoded.append(self.chord_to_idx[chord])
            else:
                print(f"Warning: Chord {chord} not found in chord.json. Skipping.")
            
            durations.append(end - start)  # Compute duration
        
        encoded_chords = np.array(encoded)
        encoded_chords_root = np.array(encoded_root)
        encoded_chords_attr = np.array(encoded_attr)
        
        # Maximum sequence length for chords
        max_sequence_length = 100  # Define this globally or as a parameter

        # Truncate or pad chord sequences
        if len(encoded_chords) > max_sequence_length:
            # Truncate to max length
            encoded_chords = encoded_chords[:max_sequence_length]
            encoded_chords_root = encoded_chords_root[:max_sequence_length]
            encoded_chords_attr = encoded_chords_attr[:max_sequence_length]
        
        else:
            # Pad with zeros (padding value for chords)
            padding = [0] * (max_sequence_length - len(encoded_chords))
            encoded_chords = np.concatenate([encoded_chords, padding])
            encoded_chords_root = np.concatenate([encoded_chords_root, padding])
            encoded_chords_attr = np.concatenate([encoded_chords_attr, padding])
            
        # Convert to tensor
        chords_tensor = torch.tensor(encoded_chords, dtype=torch.long)  # Fixed length tensor
        chords_root_tensor = torch.tensor(encoded_chords_root, dtype=torch.long)  # Fixed length tensor
        chords_attr_tensor = torch.tensor(encoded_chords_attr, dtype=torch.long)  # Fixed length tensor

        # --- Key feature (Tonic and Mode separation) --- 
        fn_key = os.path.join(self.root, 'key', path[:-4] + ".lab")

        if not os.path.exists(fn_key):
            mode = "major"
        else:
            mode = "major"  # Default value
            with open(fn_key, 'r') as file:
                for line in file:
                    key = line.strip()
            if key == "None":
                mode = "major"
            else:
                mode = key.split()[-1]

        # Split the key into tonic and mode
        # tonic, mode = key.split() if key != "None" else ("None", "None")
        # Encode the tonic and mode separately
        # encoded_tonic = self.tonic_to_idx.get(tonic, -1)  # Use -1 for unknown tonic
        
        encoded_mode = self.mode_to_idx.get(mode, 0)
        #print(encoded_mode)
        mode_tensor = torch.tensor([encoded_mode], dtype=torch.long)

        # --- MERT feature --- 
        fn_mert = os.path.join(self.root, 'mert_30s', path[:-4])
        embeddings = []

        # Specify the layers to extract (3rd, 6th, 9th, and 12th layers)..
        layers_to_extract = self.cfg.model.layers
        
        #layers_to_extract = [5, 6, 7]
        for filename in os.listdir(fn_mert):
            file_path = os.path.join(fn_mert, filename)
            if os.path.isfile(file_path) and filename.endswith('.npy'):
                segment = np.load(file_path)
                
                # Extract and concatenate features for the specified layers
                concatenated_features = np.concatenate(
                    [segment[:, layer_idx, :] for layer_idx in layers_to_extract], axis=1
                ) 
                
                concatenated_features = np.squeeze(concatenated_features)  # Shape: 768 * 2 = 1536
                embeddings.append(concatenated_features)
                
                if self.segment_type == "f30s":
                    break

        # Calculate the final embedding
        embeddings = np.array(embeddings)
        final_embedding_mert = np.mean(embeddings, axis=0)  
        final_embedding_mert = torch.from_numpy(final_embedding_mert)

        # print(final_embedding_mert.shape)
        # assert(False)
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
                "x_chord" : chords_tensor,
                "x_chord_root" : chords_root_tensor,
                "x_chord_attr" : chords_attr_tensor,
                "x_key" : mode_tensor,
                "y_mood" : y_mood, 
                "y_va": torch.stack([y_valence, y_arousal], dim=0),
                "y_genre" : y_genre, 
                "y_instr" : y_instr, 
                "path": self.dictionary[index]['path']
        }

    def __len__(self):
        return len(self.dictionary)
    
    def load_genre_info(self, tsv_file):
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

    def load_instr_info(self, tsv_file):
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
    


# --- MERT feature --- 
# fn_mert = os.path.join(self.root, 'mert_30s', path[:-4])
# embeddings = []

# for filename in os.listdir(fn_mert):
#     file_path = os.path.join(fn_mert, filename)
#     if os.path.isfile(file_path) and filename.endswith('.npy'):
#         segment = np.load(file_path)
#         layer_embedding = segment[:, 6, :]  # Shape: (1, 768)
#         layer_embedding = np.squeeze(layer_embedding)  # Shape: (768,)
#         embeddings.append(layer_embedding)
#         if self.segment_type == "f30s":
#             break

# embeddings = np.array(embeddings)
# final_embedding_mert = np.mean(embeddings, axis=0)  # Shape: (768,)
# final_embedding_mert = torch.from_numpy(final_embedding_mert)

        # Optional: You can also process durations similarly if needed
        # durations_tensor = None
        # if durations:
        #     if len(durations) > max_sequence_length:
        #         durations = durations[:max_sequence_length]
        #     else:
        #         durations += [0.0] * (max_sequence_length - len(durations))  # Pad durations with 0.0
        #     durations_tensor = torch.tensor(durations, dtype=torch.float32)

        # --- chord feature --- 
        # fn_chord = os.path.join(self.root, 'chord', 'lab', path[:-4] + ".lab")

        # chords = []
        # with open(fn_chord, 'r') as file:
        #     for line in file:
        #         start, end, chord = line.strip().split()
        #         chords.append((float(start), float(end), chord))

        # encoded = []
        # durations = []
        # for start, end, chord in chords:
        #     if chord in self.chord_to_idx:
        #         encoded.append(self.chord_to_idx[chord])
        #     else:
        #         print(fn_chord)
        #         raise ValueError(f"Chord {chord} not found in chord.json!")
            
        #     durations.append(end - start)  # Compute duration
        
        # encoded_chords = np.array(encoded)
        # chords_tensor = torch.tensor(encoded_chords, dtype=torch.long)  # For embedding
        # durations_tensor = torch.tensor(durations, dtype=torch.float32)  # Optional


        # # --- Chord feature --- 
        # fn_chord = os.path.join(self.root, 'chord', 'lab3', path[:-4] + ".lab")
        
        # chords_list = []
        # current_sequence = []
        # ngram_list = []

        # if not os.path.exists(fn_chord):
        #     chords.append("N")  # Use "N" as a placeholder for no chords
        # else:
        #     with open(fn_chord, 'r') as file:
        #         for line in file:
        #             start, end, chord = line.strip().split()
        #             # if chord != "N":  # Ignore "N" chords
        #             #     chords.append(chord)
        #             if chord != 'N':  # Ignore 'N' (no chord)
        #                 current_sequence.append(chord)
        #             else:
        #                 if len(current_sequence) > 0:
        #                     chords_list.append(current_sequence)
        #                     current_sequence = []

        # # Generate n-grams for the chord sequence
        # n = 4  # You can choose a suitable value for n
        # for chord_sequence in current_sequence:
        #     # Generate n-grams for the current chord sequence
        #     ngrams = [tuple(chord_sequence[i:i + n]) for i in range(len(chord_sequence) - n + 1)]
        #     ngram_list.append(ngrams)

        # # Encode n-grams
        # encoded = []
        # durations = []
        # for ngrams in ngram_list:
        #     for ngram in ngrams:
        #         if ngram in self.ngram_vocab:
        #             encoded.append(self.ngram_vocab[ngram])  # Encode the n-gram
        #         else:
        #             print(f"Warning: N-gram {ngram} not found in ngram_vocab. Skipping.")
            
        # # If no valid n-grams, fallback to zeros
        # if len(encoded) == 0:
        #     encoded.append(0)  # Use a default encoding for empty sequences

        # encoded_ngrams = np.array(encoded)

        # # Maximum sequence length for n-grams
        # max_sequence_length = 100  # Define this globally or as a parameter

        # # Truncate or pad n-gram sequences
        # if len(encoded_ngrams) > max_sequence_length:
        #     # Truncate to max length
        #     encoded_ngrams = encoded_ngrams[:max_sequence_length]
        # else:
        #     # Pad with zeros (padding value for n-grams)
        #     padding = [0] * (max_sequence_length - len(encoded_ngrams))
        #     encoded_ngrams = np.concatenate([encoded_ngrams, padding])

        # # Convert to tensor
        # ngrams_tensor = torch.tensor(encoded_ngrams, dtype=torch.long)  # Fixed length tensor
