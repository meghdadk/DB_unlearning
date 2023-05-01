import pandas as pd
import numpy as np
import random
import json
from itertools import groupby
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

random.seed(1234)
np.random.seed(1234)

class Normalizer:
    def __init__(self):
        self.mean = None
        self.width = None
        self.min = None
        self.max = None
        
    def get_mean(self):
        return self.mean
    
    def get_width(self):
        return self.width

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max

class Data:
    """Data loading and preparing for Mixture Density Network, including encoding and normalizing

    Args:
        data_path (string): Path to the dataset's file. It must be in csv format \
                            by default it assumes the csv file contains header and \
                            is comma delimited
        header (list-like): A list of the data headers if the original csv file \
                            does not contain headers. If this list is not empty \
                            the first line of the csv file is treate as data
        x_attributes (list-like): a list of x attributes
        y_attributes (list-like): a list of target attributes
        sep (char): csv file delimiter. by default is ','


    """


    def __init__(
            self,
            data_path,
            header=[],
            x_attributes=[],
            y_attributes=[],
            sep=','):

        self.data_path = data_path
        self.delimiter = sep
        self.header = [h.tolower() for h in header]
        self.x_attributes = [x.lower() for x in x_attributes]
        self.y_attributes = [y.lower() for y in y_attributes]
        self.encoders = {}
        self.normalizers = {}
        self.FTs = {}


    def read_data(self, _return="dataframe"):
        """Reads the data file and puts in x and y variables


        Args:
            _return (string): one the two options: dictionaries, dataframe\
                              dictionaries: reutrns x, y dictionaries
                              dataframe: returns a dataframe of data table 
        output:
            x_values (map-like): A dictionary of the {'att': list} format \
                                   that is a list of values for each attribute
            y_values (map-like): A dictionary of the {'att': list} format \
                                   that is a list of values for each attribute
            table (dataframe): A dataframe of the data table
        """
        if len(self.header) == 0:
            data = pd.read_csv(self.data_path, sep=self.delimiter)
        else:
            data = pd.read_csv(self.data_path, header=None, sep=self.delimiter, names=self.header)

        data.columns = [x.lower() for x in data.columns]
        data.columns = [x.replace(' ','_') for x in data.columns]

        for att in self.x_attributes:
            data[att] = data[att].astype(str)

        #data = data[self.x_attributes+self.y_attributes]
        data = data.dropna(subset=self.x_attributes + self.y_attributes)
        if _return == "dataframe":
            return data
        else:
            return self.get_x_y_dict(data)


    def delete_data(self, filters_path, frac=1.0):
        """Delete a part of data defined by filters

        Args:
            filters_path (string): address to a json file containing filters to remove
            frac (float): The fraction of tuples to be deleted. 
                          e.g. if frac=.2, 20% of rows that satisfy filters
                          will be deleted

        Output:
            A reduced data table saved in the self.data_path directory
        """

        assert frac <= 1 and frac >= 0

        filters = None
        with open(filters_path, 'r') as f:
            filters = json.load(f)

        filters = filters['filters']

        data = self.read_data(_return="dataframe")
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
                data = pd.concat([data,sub_data])
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
                data = pd.concat([data,sub_data])

            else:
                raise ValueError('Filter unkown!')

        print ("data size after deletion: {}".format(len(data)))
        if len(data) == 0:
            raise ValueError('The filters deleted the whole data!')

        
        x_values, y_values = self.get_x_y_dict(data)
        self.create_frequency_tables(x_values)  #update the frequency tables

        data.to_csv(self.data_path.replace('.csv','_reduced.csv'), index=None, header=True, sep=self.delimiter)

        removed_rows = original_data[~original_data.isin(data)].dropna()
        removed_rows.to_csv(self.data_path.replace('.csv','_deleted.csv'), index=None, header=True, sep=self.delimiter)

        return data, removed_rows


    def get_x_y_dict(self, data):
        """Convert a dataframe to x and y dictionaries


        Args:
            data (dataframe): data


        Output:
            x_values (map-like): A dictionary of the {'att': list} format \
                                   that is a list of values for each attribute
            y_values (map-like): A dictionary of the {'att': list} format \
                                   that is a list of values for each attribute        
        """

        x_values = {}
        y_values = {}
        for att in self.x_attributes:
            if data[att].dtype!=np.str:
                data[att] = data[att].astype(np.str)
            x_values[att] = data[att].str.lower().tolist()
        for att in self.y_attributes:
            y_values[att] = data[att].tolist()


        return x_values, y_values        

    def create_encoders(self, data):
        """Applies one-hot encoding to the categorical variables

        Args:
            data (map-like): A dictionary of the data to be encoded \
                             It must have a {'att': list} format

        Output:
            self.encoders (map-like): A dictionary of the trained encoders \
                                      {'att': encoder}

        """
        
        for key in data.keys():
            encoder = OneHotEncoder(categories='auto')        
            encoder.fit(np.asarray(data[key]).reshape(-1,1))
            self.encoders[key] = encoder

    def create_frequency_tables(self, data):
        """Creates frequency tables for the categorical variables

        Args:
            data (map-like): A dictionary of {'att': list} format

        Output:
            self.FTs (map-like): A dictionary of dictionaries {'att': {'val': freq}}

        """
        for key in data.keys():
            ft = Counter(data[key])
            self.FTs[key] = ft

    def create_normalizers(self, data=None, min_max=None):
        """Get normalizing stats for min_max standardization

        Args:
            data (map-like): A dictionary of {'att': list} format
            min_max (map-like): A dictionary of tuples. each tuple contains \
                                the (minimum, maximum) of the attribute

        Output:
            self.normalizers (map-like): A dictionary of Normalization class {'att': Normalizer}

        """
        if min_max is not None:
            for key in min_max:
                _mean = min_max[key][0]
                _width = min_max[key][1]
                norm = Normalizer()
                norm.mean = _mean
                norm.width = _width
                norm.min = min_max[key][0]
                norm.max = min_max[key][1]
                self.normalizers[key] = norm
                
        else:
            for key in data.keys():
                _mean = (np.min(data[key]) + np.max(data[key]))/2
                _width = (np.max(data[key]) - np.min(data[key]))
                norm = Normalizer()
                norm.mean = _mean
                norm.width = _width
                norm.min = np.min(data[key])
                norm.max = np.max(data[key])
                self.normalizers[key] = norm
        
    def normalize(self, data):

        """Normalizing numerical attributes

        Args:
            data (map-like): A dictionary of {'att': list} format

        Output:
            normalized_data (map-like): A dictionary of normalized values {'att': list}

        """

        normalized_data = {}
        for key in data.keys():
            s = self.normalizers[key]
            normed = [(x-s.mean)/s.width * 2 for x in data[key]]
            normalized_data[key] = normed
            
        return normalized_data

    def denormalize(self, data):
        """Denormalizing numerical attributes

        Args:
            data (map-like): A dictionary of {'att': list} format

        Output:
            normalized_data (map-like): A dictionary of denormalized values {'att': list}

        """
        denormalized_data = {}
        for key in data.keys():
            s = self.normalizing_stats[key]
            denormed = [0.5 * s.width * x + s.mean for x in data[key]]
            denormalized_data[key] = denormed
            
        return normalized_data



