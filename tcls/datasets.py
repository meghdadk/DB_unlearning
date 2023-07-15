import pandas as pd
import numpy as np
import json
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def read_data(path, cols):
    # Load the dataset

    data = pd.read_csv(path, usecols=cols, sep=',')
    data = data.dropna()

    return data


def delete_data(data, filters_path, frac=1.0):

    assert frac <= 1 and frac >= 0

    filters = None
    with open(filters_path, 'r') as f:
        filters = json.load(f)

    filters = filters['filters']

    original_data = data.copy()
    print ("data size before deletion: {}".format(len(data)))

    
    for i, _filter in enumerate(filters):
        if _filter['type'] == 'equality':
            sub_data = data[data[_filter['att']] == _filter['val']]
            sub_data = sub_data.sample(frac=1-frac)

            data = data[data[_filter['att']] != _filter['val']]

            data = pd.concat([data, sub_data])
        elif _filter['type'] == 'range_full':
            sub_data = data[
                        ((data[_filter['att']] >= _filter['min_val'])
                        & (data[_filter['att']] <= _filter['max_val']))
                    ]
            sub_data = sub_data.sample(frac=1-frac)

            data = data[
                        ~((data[_filter['att']] >= _filter['min_val'])
                        & (data[_filter['att']] <= _filter['max_val']))
                    ]
            data = pd.concat([data, sub_data])
        elif _filter['type'] == 'range_selective':
            sub_data = data[
                        ((data[_filter['att']] >= _filter['min_val'])
                        & (data[_filter['att']] <= _filter['max_val']))
                    ]
            
            sub_data = sub_data[np.arange(len(sub_data)) % 2 == 0]

            data = data[
                        ~((data[_filter['att']] >= _filter['min_val'])
                        & (data[_filter['att']] <= _filter['max_val']))
                    ]
            data = pd.concat([data, sub_data])

        else:
            raise ValueError('Filter unkown!')

    print ("data size after deletion: {}".format(len(data)))
    if len(data) == 0:
        raise ValueError('The filters deleted the whole data!')

    removed_rows = original_data[~original_data.isin(data)].dropna()

    #data.to_csv(self.data_path.replace('.csv','_reduced.csv'), index=None, header=True, sep=self.delimiter)
    #removed_rows.to_csv(self.data_path.replace('.csv','_deleted.csv'), index=None, header=True, sep=self.delimiter)

    return original_data, data, removed_rows

def _get_numericals_categoricals(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

    return numerical_cols, categorical_cols

def _encode(data, categorical_cols, type='label-encoding'):
    if type == 'label-encoding':
        for col in categorical_cols:
            data[col] = LabelEncoder().fit_transform(data[col])
    elif type == 'one-hot':
        data = pd.get_dummies(data, categorical_cols, dtype=float)

    return data

def _normalize(data, numerical_cols):
    data[numerical_cols] = StandardScaler().fit_transform(data[numerical_cols])

    return data

def _encode_labels(targets):
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(targets)
    
    return targets, label_encoder.classes_, label_encoder

def split_data(data, label, cat_cols, num_cols, bs, _return='loaders'):
    # Separate features and labels
    X = data.drop(label, axis=1)
    y = data[label]

    # Convert categorical attributes to binary encodings
    X = _encode(X, cat_cols, type='one-hot')

    # Normalize numerical attributes
    X = _normalize(X, num_cols)

    # Convert the label to numerical values
    y, classes, label_encoder = _encode_labels(y)


    # Split the dataset into train/val/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDatasets and DataLoaders for minibatch training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    if _return == 'tensors':
        return X_train, y_train, X_val, y_val, X_test, y_test
    elif _return == 'loaders':
        return train_loader, val_loader, test_loader
    