import requests
import pandas as pd
import os

# Define Deezer API URL template
DEEZER_API_URL = "https://api.deezer.com/track/{}"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('../data/dmdd/test.csv')

# Create a directory to store the audio previews
if not os.path.exists('dmdd_test'):
    os.makedirs('dmdd_test')

# Function to download the audio preview using the Deezer song ID
def download_preview(dzr_sng_id):
    try:
        # Fetch the track information from Deezer API
        url = DEEZER_API_URL.format(dzr_sng_id)
        response = requests.get(url)
        track_info = response.json()
        
        # Check if 'preview' is available and if it's a valid URL
        if 'preview' in track_info and track_info['preview']:
            preview_url = track_info['preview']
            
            # Download the preview
            preview_response = requests.get(preview_url)
            if preview_response.status_code == 200:
                # Save the audio preview as dzr_sng_id.mp3
                filename = f"{dzr_sng_id}.mp3"
                file_path = os.path.join('dmdd_test', filename)
                
                # Save the file
                with open(file_path, 'wb') as f:
                    f.write(preview_response.content)
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download preview for song ID {dzr_sng_id}")
        else:
            print(f"No preview available for song ID {dzr_sng_id}")
    except Exception as e:
        print(f"Error downloading song ID {dzr_sng_id}: {e}")

# Iterate over the rows in the DataFrame and download each track preview
for index, row in df.iterrows():
    dzr_sng_id = row['dzr_sng_id']
    download_preview(dzr_sng_id)

print("All downloads complete!")






## [ Lyrics ]

# import requests
# import pandas as pd
# import os

# # Define Deezer API URL template
# DEEZER_API_URL = "https://api.deezer.com/track/{}"

# # Load the CSV file into a pandas DataFrame
# df = pd.read_csv('../data/dmdd/train.csv')

# # Create a directory to store the audio previews
# if not os.path.exists('audio_lyrics'):
#     os.makedirs('audio_lyrics')

# # Function to download the audio preview using the Deezer song ID
# def download_lyrics(dzr_sng_id):
#     try:
#         # Fetch the track information from Deezer API
#         url = DEEZER_API_URL.format(dzr_sng_id)
#         response = requests.get(url)
#         track_info = response.json()

#         if track_info['explicit_lyrics']:
#             print(dzr_sng_id)

#     except Exception as e:
#         print(f"Error downloading song ID {dzr_sng_id}: {e}")

# # Iterate over the rows in the DataFrame and download each track preview
# for index, row in df.iterrows():
#     dzr_sng_id = row['dzr_sng_id']
#     download_lyrics(dzr_sng_id)

# print("All downloads complete!")