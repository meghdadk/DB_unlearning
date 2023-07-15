import pandas as pd
import numpy as np
import json
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

class DataProcessor:
    def __init__(self, path, cols, label=None, cat_cols=None, num_cols=None, filters_path=None):
        self.path = path
        self.cols = cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.label = label
        self.filters_path = filters_path

    def _read_data(self):
        data = pd.read_csv(self.path, usecols=self.cols, sep=',')
        data = data.dropna()
        return data

    def _delete_data(self, data, frac=1.0):
        assert frac <= 1 and frac >= 0

        filters = None
        with open(self.filters_path, 'r') as f:
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

        return original_data.reset_index(drop=True), data.reset_index(drop=True), removed_rows.reset_index(drop=True)

    def _get_numericals_categoricals(self, data):
        categorical_cols = data.select_dtypes(include=['object']).columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

        return numerical_cols, categorical_cols

    def _cat_encoders(self, data, update_params=True, type='label-encoding'):
        if update_params:
            self.cat_encoders = {}
        if type == 'label-encoding':
            for col in self.cat_cols:
                if update_params:
                    encoder = LabelEncoder()
                    encoder.fit(data[col])
                    self.cat_encoders[col] = encoder

                data[col] = self.cat_encoders[col].transform(data[col])
        elif type == 'one-hot':
            for col in self.cat_cols:
                if update_params:
                    encoder = OneHotEncoder(sparse=False)
                    encoder.fit(np.asarray(data[col]).reshape(-1,1))
                    self.cat_encoders[col] = encoder
                one_hot_encoded = self.cat_encoders[col].transform(np.asarray(data[col]).reshape(-1, 1))
                one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.cat_encoders[col].get_feature_names_out([col]))
                data = pd.concat([data.drop(col, axis=1), one_hot_df], axis=1)

        return data

    def _normalizers(self, data, update_params=True):
        if update_params:
            self.normalizer = {}
        for col in self.num_cols:
            if update_params:
                normalizer = StandardScaler()
                normalizer.fit(np.asarray(data[col]).reshape(-1,1))
                self.normalizer[col] = normalizer
            data[col] = self.normalizer[col].transform(np.asarray(data[col]).reshape(-1,1))

        return data

    def _label_encoder(self, targets, update_params=True):
        if update_params:
            label_encoder = LabelEncoder()
            label_encoder.fit(targets)
            self.classes = label_encoder.classes_
            self.label_encoder = label_encoder

        targets = self.label_encoder.transform(targets)
        
        return targets

    def _prepare_data(self, data, update_params, bs=128, test_frac=0.2, val_frac=0.2):
        # Separate features and labels
        X = data.drop(self.label, axis=1)
        y = data[self.label]

        # Convert categorical attributes to binary encodings
        X = self._cat_encoders(X, update_params, type='one-hot')

        # Normalize numerical attributes
        X = self._normalizers(X, update_params)

        # Convert the label to numerical values
        y = self._label_encoder(y, update_params)


        # Split the dataset into train/val/test sets
        X_train, y_train = X, y
        X_, y_val = None, None
        X_test, y_test = None, None
        if test_frac != 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42)
        if val_frac != 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_frac, random_state=42)


        # Get the loaders
        train_loader = self._get_loaders(X_train, y_train, bs)
        val_loader = None
        test_loader = None
        if test_frac != 0:
            test_loader = self._get_loaders(X_test, y_test, bs)
            
        if val_frac != 0:
            val_loader = self._get_loaders(X_val, y_val, bs)
            

        # Compute class weights
        class_counts = torch.bincount(torch.tensor(y_train))
        class_weights = 1.0 / class_counts


        if update_params:
            self.class_weights = class_weights
            self.train_size = X_train.shape
            if X_val is not None:
                self.val_size = X_val.shape
            if X_test is not None:
                self.test_size = X_test.shape

        return train_loader, val_loader, test_loader

    def _get_loaders(self, X, y, bs):
        # Convert data to PyTorch tensors
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # Create TensorDatasets and DataLoaders for minibatch training
        _dataset = TensorDataset(X, y)
        _loader = DataLoader(_dataset, batch_size=bs, shuffle=True)

        return _loader

        