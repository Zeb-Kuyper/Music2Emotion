import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import time
import numpy as np
from torch import nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
# processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
def extract_features_from_segment(segment, sample_rate, save_path):
    input_audio = segment.float()
    model_inputs = processor(input_audio, sampling_rate=sample_rate, return_tensors="pt")
    model_inputs.to(device)

    with torch.no_grad():
        model_outputs = model(**model_inputs, output_hidden_states=True)

    # take a look at the output shape, there are 13 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(model_outputs.hidden_states).squeeze()[1:,:,:].unsqueeze(0)
    # print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]
    all_layer_hidden_states = all_layer_hidden_states.mean(dim=2)    
    # print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]
    features = all_layer_hidden_states.cpu().detach().numpy()
    np.save(save_path, features)

def split_audio(waveform, sample_rate, segment_duration=30):
    segment_samples = segment_duration * sample_rate
    total_samples = waveform.size(0)
    
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        # Only include segments that are exactly 30 seconds long
        if end <= total_samples:
            segment = waveform[start:end]
            segments.append(segment)
    
    return segments

# Function to process and extract features from a single audio file
def process_audio_file(file_path, output_dir):
    print(f"Processing {file_path}")
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    waveform = waveform.squeeze()

    resample_rate = processor.sampling_rate
    if sample_rate != resample_rate:
        # print(f"Resampling from {sample_rate} to {resample_rate}")
        resampler = T.Resample(sample_rate, resample_rate)
        waveform = resampler(waveform)
        sample_rate = resample_rate

    segments = split_audio(waveform, sample_rate)

    for i, segment in enumerate(segments):
        segment_save_path = os.path.join(output_dir, f"segment_{i}.npy")
        # Check if the output file already exists
        if os.path.exists(segment_save_path):
            #print(f"Skipping segment {i} as {segment_save_path} already exists.")
            continue

        extract_features_from_segment(segment, sample_rate, segment_save_path)
        # print(f"Saved features for segment {i} -> {segment_save_path}")

# Function to recursively traverse directory and process all MP3 files
def process_directory(input_dir, output_dir):
    gpu = 3
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                if gpu == 0:
                    if int(file_path.split("/")[-2]) >=0 and int(file_path.split("/")[-2]) < 25:
                        relative_path = os.path.relpath(file_path, input_dir)
                        output_file_dir = os.path.join(output_dir, os.path.splitext(relative_path)[0])
                        os.makedirs(output_file_dir, exist_ok=True)
                        process_audio_file(file_path, output_file_dir)
                elif gpu ==1 :
                    if int(file_path.split("/")[-2]) >=25 and int(file_path.split("/")[-2]) < 50:
                        relative_path = os.path.relpath(file_path, input_dir)
                        output_file_dir = os.path.join(output_dir, os.path.splitext(relative_path)[0])
                        os.makedirs(output_file_dir, exist_ok=True)
                        process_audio_file(file_path, output_file_dir)
                elif gpu ==2 :
                    if int(file_path.split("/")[-2]) >=50 and int(file_path.split("/")[-2]) < 75:
                        relative_path = os.path.relpath(file_path, input_dir)
                        output_file_dir = os.path.join(output_dir, os.path.splitext(relative_path)[0])
                        os.makedirs(output_file_dir, exist_ok=True)
                        process_audio_file(file_path, output_file_dir)
                elif gpu ==3 :
                    if int(file_path.split("/")[-2]) >=75 and int(file_path.split("/")[-2]) < 100:
                        relative_path = os.path.relpath(file_path, input_dir)
                        output_file_dir = os.path.join(output_dir, os.path.splitext(relative_path)[0])
                        os.makedirs(output_file_dir, exist_ok=True)
                        process_audio_file(file_path, output_file_dir)
                

# Main function to execute the process
def main():
    input_directory = "../dataset/jamendo/mp3"  # Replace with your input directory path
    output_directory = "../dataset/jamendo/mert_30s_all"  # Replace with your output directory path
    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()



    # Move inputs to the device
    # inputs = {key: value.to(device) for key, value in inputs.items()}
    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)
    # all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    # features = all_layer_hidden_states.cpu().detach().numpy()
    # print(features.shape)
    # # Move the output back to the CPU before saving
    # time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    # aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
    # aggregator.to(device)
    # weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
    # features = weighted_avg_hidden_states.cpu().detach().numpy()
    # shape : 768
    # Save features as a .npy file