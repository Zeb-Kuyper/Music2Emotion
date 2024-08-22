
import h5py
import os

# Get the list of all files and directories in the current directory
# contents = '../dataset/test_dac'

path = "../dataset/test_dac"

# for filename in os.listdir(path):
#     filepath = os.path.join(path,filename)

# Print the contents
for filename in os.listdir(path):
    filepath = os.path.join(path,filename)

    if h5py.is_hdf5(filepath):
        with h5py.File(filepath, 'r') as f:
        # Read a dataset from the file
            # keys = list(f.keys())
            # print(keys)
            data = f['0']
            print(data.keys())

            print( type( data[ 'dac_frame_len' ] ) )

            print( data[ 'dac_latents' ].shape )
            print( data[ 'dac_rvq' ].shape )
            

            # print(data['encodec_frame_len'][()])
            # print(data['encodec_latents'][:])
            # print(data['encodec_rvq'][:])
            # print(data['spectrogram'][:])

            # Print the data
            # print(data[:])
# Open the HDF5 file