import csv
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

client_id = 'dd85d22904434f088c3368bd80f9e737'  # Replace with your Client ID
client_secret = '645999f592554ae1b7d64e88643e340c'  # Replace with your Client Secret
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def download_spotify_preview(spotify_id, file_name):
    try:
        track = sp.track(spotify_id)
        if track.get('preview_url'):
            preview_url = track['preview_url']
            print(f"Downloading preview for Spotify ID: {spotify_id} -> {file_name}")
            response = requests.get(preview_url)
            with open(file_name, 'wb') as file:
                file.write(response.content)        
            print(f"Preview for Spotify ID '{spotify_id}' downloaded successfully as {file_name}!")
        else:
            print(f"No preview available for Spotify ID: {spotify_id}")
    except Exception as e:
        print(f"Error for Spotify ID {spotify_id}: {e}")

def main():
    csv_file = '../data/muse_v3.csv'
    output_folder = 'muse'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the CSV file and process each row
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader, start=1):
            if i < 88134:
                continue
            spotify_id = row['spotify_id']
            if not spotify_id:
                print(f"Skipping row {i} due to missing Spotify ID")
                continue
            file_name = os.path.join(output_folder, f"{i}.mp3")
            download_spotify_preview(spotify_id, file_name)

if __name__ == "__main__":
    main()