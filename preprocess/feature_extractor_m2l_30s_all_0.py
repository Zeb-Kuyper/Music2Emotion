import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from music2latent import EncoderDecoder  # Import your custom model

# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the Music2latent model
encoder_decoder = EncoderDecoder(device=device)

# Function to process and extract features from a single audio segment
def extract_features_from_segment(segment, sample_rate, save_path):
    input_audio = segment.unsqueeze(0).to(device)  # Add batch dimension and move to the device

    with torch.no_grad():
        features_music2latent = encoder_decoder.encode(input_audio, extract_features=True)
        # encoded_features, _ = encoder_decoder.gen(input_audio)

    # Flatten and average the features along the time axis
    features = features_music2latent.mean(dim=-1).cpu().numpy()

    # Save features as a .npy file
    np.save(save_path, features)

def split_audio(waveform, sample_rate, segment_duration=30):
    segment_samples = segment_duration * sample_rate
    total_samples = waveform.size(0)
    
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        # Only include segments that are exactly 10 seconds long
        if end <= total_samples:
            segment = waveform[start:end]
            segments.append(segment)

    if len(segments) == 0:
        segment = waveform
        segments.append(segment)

    return segments

# Function to process and extract features from a single audio file
def process_audio_file(file_path, output_dir):
    print(f"Processing {file_path}")

    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    waveform = waveform.squeeze()

    resample_rate = 44100  # Assuming the Music2latent model expects 44100 Hz audio
    if sample_rate != resample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        waveform = resampler(waveform)
        sample_rate = resample_rate

    segments = split_audio(waveform, sample_rate)

    for i, segment in enumerate(segments):
        segment_save_path = os.path.join(output_dir, f"segment_{i}.npy")
        
        # Check if the output file already exists
        if os.path.exists(segment_save_path):
            continue

        extract_features_from_segment(segment, sample_rate, segment_save_path)
        # print(f"Saved features for segment {i} -> {segment_save_path}")

# Function to recursively traverse directory and process all MP3 files
def process_directory(input_dir, output_dir):
    gpu = 0
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
    input_directory = "dataset/jamendo/mp3"  # Replace with your input directory path
    output_directory = "dataset/jamendo/music2latent_30s_all"  # Replace with your output directory path

    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()
