import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.data_utils import to_lookup
from utils.common_utils import torch2numpy
from data.dataset_factory import split_data
from data.queries import BaseQueries


class DataLoaderX(DataLoader):
    """This function transforms arbitrary generator into a 
    background-thead generator that prefetches several 
    batches of data in a parallel background thead
    
    This is useful if you have a computationally heavy process 
    (CPU or GPU) that iteratively processes minibatches from 
    the generator while the generator consumes some other resource 
    (disk IO / loading from database / more CPU 
    if you have unused cores).

    Reference: https://github.com/justheuristic/prefetch_generator
    """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
    
def get_dataset(
        dataset_folder,
        split,
        split_ratio,
        apply_PCA,
        PCA_variance,
        split_type="base",
        path_to_txt=None,
        remove_outliers=False,
        outliers_threshold=3
        ):

    # Distribute dataset
    x, y, lookup_values, queries, x_column_names, lookup_df, classification_task, multi_task = split_data(dataset_folder, 
                                                                                                          split_type, 
                                                                                                          path_to_txt,
                                                                                                          remove_outliers, 
                                                                                                          outliers_threshold)        
                    
    if multi_task:
        label_info, y_train, y_test = {}, {}, {}
        for key, _ in y.items():
            label_info[key], y[key] = to_lookup(lookup_values[key], y[key])
            x_train, x_test, y_train[key], y_test[key] = train_test_split(x,
                                                                          y[key],
                                                                          test_size = split_ratio, 
                                                                          random_state = 0)

    else:
        label_info, y = to_lookup(lookup_values, y)
    
        try:
            x_train, x_test, y_train, y_test = train_test_split(x,
                                                                y,
                                                                stratify = y, 
                                                                test_size = split_ratio, 
                                                                random_state = 0)
        except ValueError:
            print('Cannot stratify. Split without it now!')
            x_train, x_test, y_train, y_test = train_test_split(x,
                                                                y,
                                                                test_size = split_ratio, 
                                                                random_state = 0)


    # Preprocessing - feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)

    if not classification_task:
        y_sc = StandardScaler()
        y_train = y_sc.fit_transform(y_train)
        y_test = y_sc.transform(y_test)
    else:
        y_sc = None

    # PCA
    if apply_PCA:
        pca = PCA(PCA_variance)
 
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        explained_variance = pca.explained_variance_ratio_
        print("Number of components required to capture 90% of variance:", pca.n_components_)
    
    
    if multi_task:
        dims = [X_train.shape[-1]] + [len(label_info[key]) for key, _ in label_info.items()]
    else:
        dims = [X_train.shape[-1], len(label_info.keys())]

    data_helper = Data_Helper(dims, label_info, queries, x_test, y_test, x_column_names, lookup_df, classification_task, sc, multi_task, pca, y_sc) if apply_PCA \
             else Data_Helper(dims, label_info, queries, x_test, y_test, x_column_names, lookup_df, classification_task, sc, multi_task)

    if classification_task and not multi_task:
        assert len(label_info.keys())==(max(y_train)+1), "Number of label class does not match!"

    if split == "train":
        if multi_task:
            train_dataset = multi_task_dataset(X_train, y_train, queries)
            test_dataset = multi_task_dataset(X_test, y_test, queries)
            return train_dataset, test_dataset, data_helper
        else:
            return torch.utils.data.TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)), torch.utils.data.TensorDataset(
                torch.from_numpy(X_test), torch.from_numpy(y_test)), data_helper
    if split == "test":
        assert False, "We return testing split together with training split"
    if split == "val":
        assert False, "We do not consider validation split in this project"


class Data_Helper:
    "Handles data precessing (inverse-)transformation and holds information about data"
    def __init__(self, dims, label_info, queries, x, y, x_column_names, lookup_df, classification_task=True, standardscaler=None, multi_task=False, pca=None, y_standardscaler=None):
        # Data info
        self.dims = dims
        self.label_info = label_info
        self.queries = queries
        self.x_column_names = x_column_names
        self.y_column_names = ['Prediction', 'GT']
        self.column_names = np.concatenate((self.x_column_names, self.y_column_names))
        self.lookup_df = lookup_df
        self.classification_task = classification_task
        self.multi_task = multi_task
        
        # Test split
        self.x = x
        self.y = y

        # Data prepocessing transformer
        self.sc = standardscaler
        self.pca = pca
        self.y_sc = y_standardscaler

    
    def forward_transform(self, input):
        if input.shape == 1:
            input = np.expand_dims(input, axis=0)

        if self.pca is not None:
            return self.pca.transform(self.sc.transform(input)) 
        else:
            return self.sc.transform(input)
    

    def inverse_transform(self, input):
        if input.shape == 1:
            input = np.expand_dims(input, axis=0)

        if self.pca is not None:
            return self.sc.inverse_transform(self.pca.inverse_transform(input))
        else:
            return self.sc.inverse_transform(input)
        
    
    def sampled_forward_pass(self, model, use_GPU=False, n_samples=5):
        idx = np.random.choice(self.x.shape[0], n_samples, replace=False)
        raw_inputs = self.x[idx]
        inputs = self.forward_transform(raw_inputs)
        y_gt = self.y[idx]

        inputs = torch.from_numpy(inputs).cuda() if use_GPU else inputs
        with torch.no_grad():
            model.eval()
            y_pred = torch2numpy(model(inputs))

        if self.classification_task:
            y_pred = np.expand_dims(y_pred.argmax(axis=1), axis=-1)
            y_gt = np.expand_dims(y_gt, axis=-1)
        else:
            return np.concatenate((raw_inputs, y_pred, y_gt), axis=-1)
        
        return np.concatenate((raw_inputs, y_pred, y_gt), axis=-1)


    def sampled_forward_pass_MT(self, model, use_GPU=False, n_samples=5):
        idx = np.random.choice(self.x.shape[0], n_samples, replace=False)
        raw_inputs = self.x[idx]
        inputs = self.forward_transform(raw_inputs)
        y_gt_SUMINS = self.y['SUMINS'][idx]
        y_gt_FILE_NAME = self.y['FILE_NAME'][idx]

        inputs = torch.from_numpy(inputs).cuda() if use_GPU else inputs
        with torch.no_grad():
            model.eval()
            y_pred = torch2numpy(model(inputs))
        
            y_pred_out, y_gt, batch = {}, {}, {}
            y_pred_out['SUMINS'] = np.expand_dims(y_pred['SUMINS'].argmax(axis=1), axis=-1)
            y_pred_out['FILE_NAME'] = np.expand_dims(y_pred['FILE_NAME'].argmax(axis=1), axis=-1)
            y_gt['SUMINS'] = np.expand_dims(y_gt_SUMINS, axis=-1)
            y_gt['FILE_NAME'] = np.expand_dims(y_gt_FILE_NAME, axis=-1)

            batch['raw_inputs'] = raw_inputs
            batch['y_pred'] = y_pred_out
            batch['y_gt'] = y_gt

        return batch
    

    def array_to_markdown(self, array):
        df = pd.DataFrame(array, columns=self.column_names)

        if self.classification_task:
            # Map y using label_info
            reverse_label_info = dict((v,k) for k,v in self.label_info.items())
            df['Prediction'].replace(reverse_label_info, inplace=True)
            df['GT'].replace(reverse_label_info, inplace=True)

            # Map x using lookup tables
            for name in self.x_column_names:
                key = name+'_lookup'
                if key in self.lookup_df:
                    col = self.lookup_df[key]
                    catagorical_col = df[name]
                    mapping = dict(zip(catagorical_col, col))
                    df[name].replace(mapping, inplace=True)
        
        return df.to_markdown()
    

class multi_task_dataset(Dataset): 
    def __init__(self, x, y, queries):

        self.dataset = {}
        self.dataset['inputs'] = x

        if BaseQueries.SUMINS in queries:
            self.dataset['SUMINS'] = y['SUMINS']
        if BaseQueries.FILE_NAME in queries:
            self.dataset['FILE_NAME'] = y['FILE_NAME']

    def __len__(self):
        return len(self.dataset['inputs'])

    def __getitem__(self, idx):
        sample = {}

        for key, _ in self.dataset.items():
            sample[key] = self.dataset[key][idx]
        
        return sample