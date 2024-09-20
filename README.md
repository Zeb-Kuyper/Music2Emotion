### Music2Emotion 

## TO DO LIST (order by priority)

- Extract MERT 30s (15-45) and test.
- Genre/Instrument as Input condition
- lyric transcription
- combine datasets
    - DMDD (VA, 30s)
    - Multi-modal mirex (Category, lyrics, audio)
- Audio augmentation
- Test with Jukebox audio encoder
- Test with AGC codec: https://github.com/AudiogenAI/agc

...

## Note
- Using all is better than first 30s (MERT)


## Version History
v1 first commit


## command
[cuda-install]
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# preprocessing
[single]
python preprocess/create_h5_dataset.py -audio-dir dataset/jamendo/mp3_missing --target-dir dataset/jamendo/dac_missing --no-encodec True --no-clap True

[multi]

## dac 
python preprocess/create_h5_dataset_multifolder.py -root-dir dataset/jamendo/mp3 --target-dir dataset/jamendo/dac --no-encodec True --no-clap True

## encodec
python preprocess/create_h5_dataset_multifolder.py -root-dir dataset/jamendo/mp3 --target-dir dataset/jamendo/encodec_10s --no-dac True --no-clap True --chunk-dur-sec 10 --min-sec 11

latest: ongoing dataset/jamendo/mp3/09/1029909.mp3

## clap
python preprocess/create_h5_dataset_multifolder.py -root-dir dataset/jamendo/mp3 --target-dir dataset/jamendo/clap --no-dac True--no-encodec True


# 10s chunk

python preprocess/create_h5_dataset_multifolder.py -root-dir dataset/jamendo/mp3 --target-dir dataset/jamendo/dac_10s --no-encodec True --no-clap True --chunk-dur-sec 10 --min-sec 11 --skip-existing

# download jamendo dataset
python3 scripts/download/download.py --dataset autotagging_moodtheme --type audio data_new --unpack
