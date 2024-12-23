import os
import json
import numpy as np
import math

def compute_statistics(json_dir):
    """Compute mean and std for each feature across all JSON files."""
    all_features = {}

    # Iterate through all JSON files
    for subdir, _, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(subdir, filename), 'r') as file:
                    data = json.load(file)

                # Accumulate feature values
                for key, value in data.items():
                    if key not in all_features:
                        all_features[key] = []

                    # Check for None or NaN and skip them
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        continue

                    all_features[key].append(value)

    # Compute mean and std (or min and max) for each feature
    statistics = {}
    for key, values in all_features.items():
        values = np.array(values)
        statistics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    return statistics

def normalize_dataset(json_dir, output_dir, statistics, method='standardize'):
    """Normalize the dataset using the provided statistics."""
    for subdir, _, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(subdir, filename), 'r') as file:
                    data = json.load(file)

                # Normalize each feature in the JSON file
                for key in data.keys():
                    value = data[key]

                    # Replace None or NaN with 0 before normalizing
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        value = 0
                    
                    if method == 'standardize':
                        data[key] = (value - statistics[key]['mean']) / statistics[key]['std']
                    elif method == 'min-max':
                        data[key] = (value - statistics[key]['min']) / (statistics[key]['max'] - statistics[key]['min'])

                # Determine the new output path
                relative_path = os.path.relpath(subdir, json_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Save the normalized data back to the output JSON file
                output_file_path = os.path.join(output_subdir, filename)
                with open(output_file_path, 'w') as file:
                    json.dump(data, file)

if __name__ == '__main__':
    json_dir = '../../dataset/jamendo/librosa'  # Replace with the path to your JSON files directory
    output_dir = '../../dataset/jamendo/librosa_norm'  # Replace with the path to your output directory
    method = 'standardize'  # Choose between 'standardize' or 'min-max'

    # Compute dataset-wide statistics
    statistics = compute_statistics(json_dir)

    # Normalize the dataset using the computed statistics
    normalize_dataset(json_dir, output_dir, statistics, method=method)