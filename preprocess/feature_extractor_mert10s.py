import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import time

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and processor
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)



# Function to process and extract features from a single audio segment
def extract_features_from_segment(segment, sample_rate, save_path):
    input_audio = segment.float()
    inputs = processor(input_audio, sampling_rate=sample_rate, return_tensors="pt")

    # Move inputs to the device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    
    # Move the output back to the CPU before saving
    all_layer_hidden_states = all_layer_hidden_states.cpu()

    torch.save(all_layer_hidden_states, save_path)

# def split_audio(waveform, sample_rate, segment_duration=10):
#     segment_samples = segment_duration * sample_rate
#     total_samples = waveform.size(0)  # Using size(0) for 1-dimensional tensor
    
#     segments = []
#     for start in range(0, total_samples, segment_samples):
#         end = min(start + segment_samples, total_samples)
#         segment = waveform[start:end]  # No need for the extra dimension index
#         segments.append(segment)
    
#     return segments

def split_audio(waveform, sample_rate, segment_duration=10):
    segment_samples = segment_duration * sample_rate
    total_samples = waveform.size(0)
    
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        
        # Only include segments that are exactly 10 seconds long
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
        segment_save_path = os.path.join(output_dir, f"segment_{i}.pt")
        
        # Check if the output file already exists
        if os.path.exists(segment_save_path):
            #print(f"Skipping segment {i} as {segment_save_path} already exists.")
            continue

        extract_features_from_segment(segment, sample_rate, segment_save_path)
        # print(f"Saved features for segment {i} -> {segment_save_path}")

# Function to recursively traverse directory and process all MP3 files
def process_directory(input_dir, output_dir):

    start_mp3="1302083.mp3"
    isStart = False

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_dir)
                output_file_dir = os.path.join(output_dir, os.path.splitext(relative_path)[0])
                os.makedirs(output_file_dir, exist_ok=True)

                if isStart == False:
                    if "1302083.mp3" not in file_path:
                        continue
                    else:
                        isStart = True

                process_audio_file(file_path, output_file_dir)

# Main function to execute the process
def main():
    input_directory = "dataset/jamendo/mp3"  # Replace with your input directory path
    output_directory = "dataset/jamendo/mert"  # Replace with your output directory path

    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()
