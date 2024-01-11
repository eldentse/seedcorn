import mat73
import os
import numpy as np


def mat2npy(folder_path, split='01'):
    counter = 0
    features = []

    name = 'ConnMatPCA' + split

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith(name):
            file_path = os.path.join(folder_path, filename)
            counter += 1
            data = mat73.loadmat(file_path)
            temp = data['ConnMatPCA']['Mat'][:,:, 0, 0, :]
            reshaped_array = np.transpose(temp, (2, 0, 1))
            features.append(reshaped_array)
    
    print('There are', counter, 'of', split, 'files')
    filename = name + '.npy'
    np.save(filename, 
            np.array(features, dtype=object), 
            allow_pickle=True)
    print('Saved as npy!')
