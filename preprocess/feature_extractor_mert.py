import os
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm  # For progress bar
import time

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load model and processor
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

# # Load model and processor
# model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
# processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)


# Function to process and extract features from a single audio file
def extract_features(file_path, save_path):

    print("file path:", file_path)
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0).unsqueeze(0)
    waveform = waveform.squeeze()

    resample_rate = processor.sampling_rate
    if sample_rate != resample_rate:
        print(f"Resampling from {sample_rate} to {resample_rate}")
        resampler = T.Resample(sample_rate, resample_rate)
        waveform = resampler(waveform)

    input_audio = waveform.float()
    inputs = processor(input_audio, sampling_rate=processor.sampling_rate, return_tensors="pt")


    # Move inputs to the device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    
    # Move the output back to the CPU before saving
    all_layer_hidden_states = all_layer_hidden_states.cpu()


    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)

    # all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()


    # time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    # aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
    # weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
    
    torch.save(all_layer_hidden_states, save_path)

# Function to recursively traverse directory and process all MP3 files
def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, input_dir)
                save_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_path = save_path.replace('.mp3', '.pt')  # Change file extension for saved features

                # Check if the output file already exists
                if os.path.exists(save_path):
                    print(f"Skipping {file_path} as {save_path} already exists.")
                    continue

                # print(f"Processing {file_path} -> {save_path}")
                # start_time = time.time()

                # file_path = "dataset/jamendo/mp3/00/7400.mp3"
                # save_path = "dataset/jamendo/mert/00/7400.pt"
                extract_features(file_path, save_path)
                # print("--- %s seconds ---" % (time.time() - start_time))
                # assert(False)
                

# Main function to execute the process
def main():
    input_directory = "dataset/jamendo/mp3"  # Replace with your input directory path
    output_directory = "dataset/jamendo/mert"  # Replace with your output directory path

    process_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()
