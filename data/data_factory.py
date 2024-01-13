import mat73
import os
import numpy as np
import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


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


def get_dataset(dataset_folder, split, split_ratio, split_type, path_to_txt):

    if split_type == 'base':
        # Iterate over files in the folder
        name = 'ConnMatPCA' 
        for filename in os.listdir(dataset_folder):
            if filename.startswith(name):
                file_path = os.path.join(dataset_folder, filename)
                data = np.load(file_path, allow_pickle=True)
                key = filename.split('_')[-1].split('.')[0]
                if key == '01':
                    label = 0
                    temporal_data_01 = npy2temporal(data, label)
                elif key == '02':
                    label = 1
                    temporal_data_02 = npy2temporal(data, label)
                else:
                    assert False, 'Class should be binary!'

    else:
        assert False, "Not implemented yet!"

    # Split train/test
    train_dataset, test_dataset = split_dataset(temporal_data_01, 
                                                temporal_data_02,
                                                split_ratio)


    if split == "train":
        return train_dataset, test_dataset
    if split == "test":
        assert False, "We return testing split together with training split"
    if split == "val":
        assert False, "We do not consider validation split in this project"


def npy2temporal(input_data, label):
    num_data, time_steps, num_nodes, _ = input_data.shape
    num_edges = num_nodes**2
    num_classes = 2

    edge_index_seq = np.zeros((num_data, 2, num_edges)) 
    edge_attr_seq = np.zeros((num_data, time_steps, num_edges)) 
    node_features_seq = np.zeros((num_data, time_steps, num_nodes, num_nodes))
    target_seq = np.zeros((num_data, time_steps, num_classes)) 

    for i in range(num_data):
        seq = input_data[i]
        data_list = []
        for j in range(time_steps):
            temp = seq[j]
            temp = temp.astype(np.float64) + 1e-8
            edge_index, edge_attr, node_features = adj2coo(temp)

            if j == 0:
                edge_index_seq[i] = edge_index
            edge_attr_seq[i,j] = edge_attr
            node_features_seq[i,j] = node_features
            target_seq[i,j] = one_hot_encode(label, num_classes)
    
    dataset = DynamicGraphTemporalSignal(edge_index_seq, edge_attr_seq, node_features_seq, target_seq)
           
    return dataset 


def adj2coo(adj_matrix):
    adj_tensor = torch.tensor(adj_matrix)

    edge_index = adj_tensor.nonzero(as_tuple=False).t()
    edge_attr = adj_tensor[edge_index[0], edge_index[1]]

    num_nodes = adj_tensor.shape[0]
    node_features = torch.eye(num_nodes)

    return edge_index, edge_attr, node_features


def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x]


def split_dataset(dataset_1, dataset_2, split_ratio):
    idx_01 = int(dataset_1.snapshot_count*split_ratio)
    train_01 = dataset_1[idx_01:]
    test_01 = dataset_1[:idx_01]

    idx_02 = int(dataset_2.snapshot_count*split_ratio)
    train_02 = dataset_2[idx_02:]
    test_02 = dataset_2[:idx_02]

    train_dataset = concatenate_datasets(train_01, train_02)
    test_dataset = concatenate_datasets(test_01, test_02)

    return train_dataset, test_dataset


def concatenate_datasets(dataset_1, dataset_2):
    concat_edge_index_seq = np.concatenate((dataset_1.edge_indices, 
                                            dataset_2.edge_indices), axis=0)
    concat_edge_attr_seq = np.concatenate((dataset_1.edge_weights, 
                                           dataset_2.edge_weights), axis=0)
    concat_node_features_seq = np.concatenate((dataset_1.features, 
                                               dataset_2.features), axis=0)
    concat_target_seq = np.concatenate((dataset_1.targets, 
                                        dataset_2.targets), axis=0)

    concat_dataset = DynamicGraphTemporalSignal(concat_edge_index_seq, 
                                                concat_edge_attr_seq, 
                                                concat_node_features_seq, 
                                                concat_target_seq)
    return concat_dataset