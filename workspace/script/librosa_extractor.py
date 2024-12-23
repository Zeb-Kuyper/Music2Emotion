import librosa
import numpy as np
import pandas as pd
import os 
from os import listdir, walk
from os.path import isfile, join
import time
import json
from tqdm import tqdm  # Import tqdm for progress bar

def extract_feature(path):
    all_files = []
    for root, dirs, files in walk(path):
        for file in files:
            if file.endswith('.mp3'):
                all_files.append(os.path.join(root, file))
                root_librosa = root.replace("mp3", "librosa")
                if not os.path.exists(root_librosa):
                    os.makedirs(root_librosa)

    # Iterate over all files with a progress bar
    for songname in tqdm(all_files, desc="Processing files", unit="file"):
        y, sr = librosa.load(songname, duration=30)
        
        #root_librosa = songname.replace("mp3", "librosa")
        output_path = songname[0:-3] + "json"
        output_path = output_path.replace("mp3", "librosa")

        if not os.path.exists(output_path):
            print(output_path)
            S = np.abs(librosa.stft(y))

            # Extracting Features
            features = {}
            features['tempo'], beats = librosa.beat.beat_track(y=y, sr=sr)

            features['total_beats'] = int(sum(beats))
            features['average_beats'] = float(np.average(beats))
            
            features['chroma_stft_mean'] = float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
            features['chroma_stft_std'] = float(np.std(librosa.feature.chroma_stft(y=y, sr=sr)))
            features['chroma_stft_var'] = float(np.var(librosa.feature.chroma_stft(y=y, sr=sr)))
            
            features['chroma_cq_mean'] = float(np.mean(librosa.feature.chroma_cqt(y=y, sr=sr)))
            features['chroma_cq_std'] = float(np.std(librosa.feature.chroma_cqt(y=y, sr=sr)))
            features['chroma_cq_var'] = float(np.var(librosa.feature.chroma_cqt(y=y, sr=sr)))
            
            features['chroma_cens_mean'] = float(np.mean(librosa.feature.chroma_cens(y=y, sr=sr)))
            features['chroma_cens_std'] = float(np.std(librosa.feature.chroma_cens(y=y, sr=sr)))
            features['chroma_cens_var'] = float(np.var(librosa.feature.chroma_cens(y=y, sr=sr)))
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            features['mfcc_mean'] = float(np.mean(mfcc))
            features['mfcc_std'] = float(np.std(mfcc))
            features['mfcc_var'] = float(np.var(mfcc))
            
            mfcc_delta = librosa.feature.delta(mfcc)
            features['mfcc_delta_mean'] = float(np.mean(mfcc_delta))
            features['mfcc_delta_std'] = float(np.std(mfcc_delta))
            features['mfcc_delta_var'] = float(np.var(mfcc_delta))
            
            features['rms_mean'] = float(np.mean(librosa.feature.rms(y=y)))
            features['rms_std'] = float(np.std(librosa.feature.rms(y=y)))
            features['rms_var'] = float(np.var(librosa.feature.rms(y=y)))
            
            features['cent_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features['cent_std'] = float(np.std(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features['cent_var'] = float(np.var(librosa.feature.spectral_centroid(y=y, sr=sr)))
            
            features['spec_bw_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            features['spec_bw_std'] = float(np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            features['spec_bw_var'] = float(np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            
            features['contrast_mean'] = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
            features['contrast_std'] = float(np.std(librosa.feature.spectral_contrast(S=S, sr=sr)))
            features['contrast_var'] = float(np.var(librosa.feature.spectral_contrast(S=S, sr=sr)))
            
            features['rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features['rolloff_std'] = float(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            features['rolloff_var'] = float(np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            
            poly_features = librosa.feature.poly_features(S=S, sr=sr)
            features['poly_mean'] = float(np.mean(poly_features))
            features['poly_std'] = float(np.std(poly_features))
            features['poly_var'] = float(np.var(poly_features))
            
            features['tonnetz_mean'] = float(np.mean(librosa.feature.tonnetz(y=y, sr=sr)))
            features['tonnetz_std'] = float(np.std(librosa.feature.tonnetz(y=y, sr=sr)))
            features['tonnetz_var'] = float(np.var(librosa.feature.tonnetz(y=y, sr=sr)))
            
            features['zcr_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            features['zcr_std'] = float(np.std(librosa.feature.zero_crossing_rate(y)))
            features['zcr_var'] = float(np.var(librosa.feature.zero_crossing_rate(y)))
            
            harmonic = librosa.effects.harmonic(y)
            features['harm_mean'] = float(np.mean(harmonic))
            features['harm_std'] = float(np.std(harmonic))
            features['harm_var'] = float(np.var(harmonic))
            
            percussive = librosa.effects.percussive(y)
            features['perc_mean'] = float(np.mean(percussive))
            features['perc_std'] = float(np.std(percussive))
            features['perc_var'] = float(np.var(percussive))
            
            melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            features['mel_mean'] = float(np.mean(melspectrogram))
            features['mel_std'] = float(np.std(melspectrogram))
            features['mel_var'] = float(np.var(melspectrogram))

            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    features[key] = value[0]
            
            # Save the features dictionary to a JSON file
            with open(output_path, 'w') as f:
                json.dump(features, f)
        
if __name__ == '__main__': 
    extract_feature('../../dataset/jamendo/mp3')





    
    
    
    # # Traversing over each file in path
    # file_data = [f for f in listdir(path) if isfile (join(path, f))]
    # for line in file_data:
    #     if ( line[-1:] == '\n' ):
    #         line = line[:-1]

    #     # Reading Song
    #     songname = path + line
        
        
    #     y, sr = librosa.load(songname, duration=30)
    #     S = np.abs(librosa.stft(y))
        
    #     # Extracting Features
    #     tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
    #     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    #     chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    #     chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        
    #     cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    #     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    #     contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    #     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    #     poly_features = librosa.feature.poly_features(S=S, sr=sr)
    #     tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    #     zcr = librosa.feature.zero_crossing_rate(y)
    #     harmonic = librosa.effects.harmonic(y)
    #     percussive = librosa.effects.percussive(y)

    #     rms = librosa.feature.rms(y=y)
        
    #     mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #     mfcc_delta = librosa.feature.delta(mfcc)
    
    #     onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    #     frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
    #     melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    #     mel_mean._set_value(id, np.mean(melspectrogram))  # melspectrogram
    #     mel_std._set_value(id, np.std(melspectrogram))
    #     mel_var._set_value(id, np.var(melspectrogram))
        
    #     # Transforming Features
    #     songname_vector._set_value(id, line)  # song name



    #     tempo_vector._set_value(id, tempo)  # tempo
    #     total_beats._set_value(id, sum(beats))  # beats
    #     average_beats._set_value(id, np.average(beats))
    #     chroma_stft_mean._set_value(id, np.mean(chroma_stft))  # chroma stft
    #     chroma_stft_std._set_value(id, np.std(chroma_stft))
    #     chroma_stft_var._set_value(id, np.var(chroma_stft))
    #     chroma_cq_mean._set_value(id, np.mean(chroma_cq))  # chroma cq
    #     chroma_cq_std._set_value(id, np.std(chroma_cq))
    #     chroma_cq_var._set_value(id, np.var(chroma_cq))
    #     chroma_cens_mean._set_value(id, np.mean(chroma_cens))  # chroma cens
    #     chroma_cens_std._set_value(id, np.std(chroma_cens))
    #     chroma_cens_var._set_value(id, np.var(chroma_cens))
    #     mfcc_mean._set_value(id, np.mean(mfcc))  # mfcc
    #     mfcc_std._set_value(id, np.std(mfcc))
    #     mfcc_var._set_value(id, np.var(mfcc))
    #     mfcc_delta_mean._set_value(id, np.mean(mfcc_delta))  # mfcc delta
    #     mfcc_delta_std._set_value(id, np.std(mfcc_delta))
    #     mfcc_delta_var._set_value(id, np.var(mfcc_delta))
    #     rms_mean._set_value(id, np.mean(rms))  # rmse
    #     rms_std._set_value(id, np.std(rms))
    #     rms_var._set_value(id, np.var(rms))
    #     cent_mean._set_value(id, np.mean(cent))  # cent
    #     cent_std._set_value(id, np.std(cent))
    #     cent_var._set_value(id, np.var(cent))
    #     spec_bw_mean._set_value(id, np.mean(spec_bw))  # spectral bandwidth
    #     spec_bw_std._set_value(id, np.std(spec_bw))
    #     spec_bw_var._set_value(id, np.var(spec_bw))
    #     contrast_mean._set_value(id, np.mean(contrast))  # contrast
    #     contrast_std._set_value(id, np.std(contrast))
    #     contrast_var._set_value(id, np.var(contrast))
    #     rolloff_mean._set_value(id, np.mean(rolloff))  # rolloff
    #     rolloff_std._set_value(id, np.std(rolloff))
    #     rolloff_var._set_value(id, np.var(rolloff))
    #     poly_mean._set_value(id, np.mean(poly_features))  # poly features
    #     poly_std._set_value(id, np.std(poly_features))
    #     poly_var._set_value(id, np.var(poly_features))
    #     tonnetz_mean._set_value(id, np.mean(tonnetz))  # tonnetz
    #     tonnetz_std._set_value(id, np.std(tonnetz))
    #     tonnetz_var._set_value(id, np.var(tonnetz))
    #     zcr_mean._set_value(id, np.mean(zcr))  # zero crossing rate
    #     zcr_std._set_value(id, np.std(zcr))
    #     zcr_var._set_value(id, np.var(zcr))
    #     harm_mean._set_value(id, np.mean(harmonic))  # harmonic
    #     harm_std._set_value(id, np.std(harmonic))
    #     harm_var._set_value(id, np.var(harmonic))
    #     perc_mean._set_value(id, np.mean(percussive))  # percussive
    #     perc_std._set_value(id, np.std(percussive))
    #     perc_var._set_value(id, np.var(percussive))
    #     frame_mean._set_value(id, np.mean(frames_to_time))  # frames
    #     frame_std._set_value(id, np.std(frames_to_time))
    #     frame_var._set_value(id, np.var(frames_to_time))
        
    #     print(songname)
    #     id = id+1
    
    # # Concatenating Features into one csv and json format
    # feature_set['song_name'] = songname_vector  # song name
    # feature_set['tempo'] = tempo_vector  # tempo 
    # feature_set['total_beats'] = total_beats  # beats
    # feature_set['average_beats'] = average_beats

    # feature_set['chroma_stft_mean'] = chroma_stft_mean  # chroma stft
    # feature_set['chroma_stft_std'] = chroma_stft_std
    # feature_set['chroma_stft_var'] = chroma_stft_var
    # feature_set['chroma_cq_mean'] = chroma_cq_mean  # chroma cq
    # feature_set['chroma_cq_std'] = chroma_cq_std
    # feature_set['chroma_cq_var'] = chroma_cq_var
    # feature_set['chroma_cens_mean'] = chroma_cens_mean  # chroma cens
    # feature_set['chroma_cens_std'] = chroma_cens_std
    # feature_set['chroma_cens_var'] = chroma_cens_var
    
    # feature_set['mfcc_mean'] = mfcc_mean  # mfcc
    # feature_set['mfcc_std'] = mfcc_std
    # feature_set['mfcc_var'] = mfcc_var
    # feature_set['mfcc_delta_mean'] = mfcc_delta_mean  # mfcc delta
    # feature_set['mfcc_delta_std'] = mfcc_delta_std
    # feature_set['mfcc_delta_var'] = mfcc_delta_var
    
    
    # feature_set['cent_mean'] = cent_mean  # cent
    # feature_set['cent_std'] = cent_std
    # feature_set['cent_var'] = cent_var

    # feature_set['rms_mean'] = rms_mean  # rms
    # feature_set['rms_std'] = rms_std
    # feature_set['rms_var'] = rms_var

    # feature_set['spec_bw_mean'] = spec_bw_mean  # spectral bandwidth
    # feature_set['spec_bw_std'] = spec_bw_std
    # feature_set['spec_bw_var'] = spec_bw_var
    # feature_set['contrast_mean'] = contrast_mean  # contrast
    # feature_set['contrast_std'] = contrast_std
    # feature_set['contrast_var'] = contrast_var
    # feature_set['rolloff_mean'] = rolloff_mean  # rolloff
    # feature_set['rolloff_std'] = rolloff_std
    # feature_set['rolloff_var'] = rolloff_var
    # feature_set['poly_mean'] = poly_mean  # poly features
    # feature_set['poly_std'] = poly_std
    # feature_set['poly_var'] = poly_var
    # feature_set['tonnetz_mean'] = tonnetz_mean  # tonnetz
    # feature_set['tonnetz_std'] = tonnetz_std
    # feature_set['tonnetz_var'] = tonnetz_var
    # feature_set['zcr_mean'] = zcr_mean  # zero crossing rate
    # feature_set['zcr_std'] = zcr_std
    # feature_set['zcr_var'] = zcr_var
    # feature_set['harm_mean'] = harm_mean  # harmonic
    # feature_set['harm_std'] = harm_std
    # feature_set['harm_var'] = harm_var
    # feature_set['perc_mean'] = perc_mean  # percussive
    # feature_set['perc_std'] = perc_std
    # feature_set['perc_var'] = perc_var
    # feature_set['frame_mean'] = frame_mean  # frames
    # feature_set['frame_std'] = frame_std
    # feature_set['frame_var'] = frame_var

    # feature_set['melspectrogram_mean'] = mel_mean  # melspectrogram
    # feature_set['melspectrogram_std'] = mel_std
    # feature_set['melspectrogram_var'] = mel_var

    # # Converting Dataframe into CSV Excel and JSON file
    # feature_set.to_csv('Emotion_features.csv')
    # feature_set.to_json('Emotion_features.json')
    

if __name__ == '__main__': 
    # Extracting Feature Function Call
    extract_feature('../../dataset/jamendo/mp3')