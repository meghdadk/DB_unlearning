"""Dataset registrations."""
import os

import numpy as np
import json

import common
import pandas as pd


def load_census(filename="census.csv", batch_num=None, finetune=False):
    csv_file = '../tabular_data/census/{}'.format(filename)
    cols = [
        'age','workclass','fnlwgt','education',
		'marital_status','occupation','relationship',
		'race','sex','capital_gain','capital_loss',
		'hours_per_week','native_country'
    ]

    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    df = df.dropna(axis=1, how='all')
	
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    if batch_num!=None:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        df = df.iloc[:landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        return common.CsvTable('census', df, cols), landmarks


    #landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
    #df = df.iloc[:landmarks[5]] 

    print (df.shape)
    return common.CsvTable('census', df, cols)

def load_reduced_census(filters_path, filename="census.csv", frac=1.0):
    np.random.seed(1)
    csv_file = '../tabular_data/census/{}'.format(filename)
    cols = [
        'age','workclass','fnlwgt','education',
        'marital_status','occupation','relationship',
        'race','sex','capital_gain','capital_loss',
        'hours_per_week','native_country'
    ]

    data = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    data = data.dropna(axis=1, how='all')
    
    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    data.replace('', np.nan, inplace=True)
    data.dropna(inplace=True)


    assert frac <= 1 and frac >= 0

    original_data = data.copy()
    print ("data size before deltion: {}".format(len(data)))

    filters = None
    with open(filters_path, 'r') as f:
        filters = json.load(f)

    filters = filters['filters']

    original_data = data.copy()
    print ("data size before deltion: {}".format(len(data)))

    
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


    data.to_csv(csv_file.replace('.csv','_reduced.csv'), index=None, header=True)

    removed_rows = original_data[~original_data.isin(data)].dropna()
    removed_rows.to_csv(csv_file.replace('.csv', '_deleted.csv'), index=None, header=True)


    original = common.CsvTable('census', original_data, cols)
    retained = common.CsvTable('census', data, cols)
    removed = common.CsvTable('census', removed_rows, cols)


    return original, retained, removed

def load_forest(filename="forest.csv",batch_num=None,finetune=False):
    csv_file = '../tabular_data/forest/{}'.format(filename)
    cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
    'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points']

    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    df = df.dropna(axis=1, how='all')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    if batch_num!=None:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        df = df.iloc[:landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        return common.CsvTable('Forest', df, cols), landmarks


    print (df.shape)

    return common.CsvTable('Forest', df, cols)

def load_reduced_forest(filters_path, filename="forest.csv", frac=1.0):
    np.random.seed(1)
    csv_file = '../tabular_data/forest/{}'.format(filename)
    cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
    'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points']

    data = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    data = data.dropna(axis=1, how='all')
    
    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    data.replace('', np.nan, inplace=True)
    data.dropna(inplace=True)


    assert frac <= 1 and frac >= 0

    filters = None
    with open(filters_path, 'r') as f:
        filters = json.load(f)

    filters = filters['filters']

    original_data = data.copy()
    print ("data size before deltion: {}".format(len(data)))

    
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


    data.to_csv(csv_file.replace('.csv','_reduced.csv'), index=None, header=True)

    removed_rows = original_data[~original_data.isin(data)].dropna()
    removed_rows.to_csv(csv_file.replace('.csv', '_deleted.csv'), index=None, header=True)


    original = common.CsvTable('forest', original_data, cols)
    retained = common.CsvTable('forest', data, cols)
    removed = common.CsvTable('forest', removed_rows, cols)


    return original, retained, removed

def load_DMV(filename="DMV.csv",batch_num=None,finetune=False):
    csv_file = '../tabular_data/DMV/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Maximum Gross Weight', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]

    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    df = df.dropna(axis=1, how='all')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    if batch_num!=None:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        df = df.iloc[:landmarks[batch_num]]

    if finetune:
        landmarks = int(len(df)*10/12) + np.linspace(0, int((len(df)*10/12)*0.2), 6, dtype=np.int)
        return common.CsvTable('Forest', df, cols), landmarks


    print (df.shape)

    type_casts = {'Reg Valid Date': np.datetime64, 'Maximum Gross Weight': np.float64}
    return common.CsvTable('Forest', df, cols, type_casts)

def load_reduced_DMV(filters_path, filename="DMV.csv", frac=1.0):
    np.random.seed(1)
    csv_file = '../tabular_data/DMV/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Maximum Gross Weight', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]

    data = pd.read_csv(csv_file,usecols=cols, sep = ',')
    #df = df[cols]
    data = data.dropna(axis=1, how='all')
    
    df_obj = data.select_dtypes(['object'])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    data.replace('', np.nan, inplace=True)
    data.dropna(inplace=True)


    assert frac <= 1 and frac >= 0

    filters = None
    with open(filters_path, 'r') as f:
        filters = json.load(f)

    filters = filters['filters']

    original_data = data.copy()
    print ("data size before deltion: {}".format(len(data)))


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


    data.to_csv(csv_file.replace('.csv','_reduced.csv'), index=None, header=True)

    removed_rows = original_data[~original_data.isin(data)].dropna()
    removed_rows.to_csv(csv_file.replace('.csv', '_deleted.csv'), index=None, header=True)

    type_casts = {'Reg Valid Date': np.datetime64, 'Maximum Gross Weight': np.float64}
    original = common.CsvTable('forest', original_data, cols, type_casts)
    retained = common.CsvTable('forest', data, cols, type_casts)
    removed = common.CsvTable('forest', removed_rows, cols, type_casts)


    return original, retained, removed

def load_permuted_forest(filename="forest.csv", permute=True, size=1000):
    csv_file = '../tabular_data/forest/{}'.format(filename)
    cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology', 
	'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
	'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
	'Horizontal_Distance_To_Fire_Points']


    df = pd.read_csv(csv_file,usecols=cols, sep = ',')
    print (df.shape)
    df = df.dropna(axis=1, how='all')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

   
    if permute:
        columns_to_sort = df.columns

        sorted_columns = pd.concat([df[col].sort_values(ignore_index=True).reset_index(drop=True) for col in columns_to_sort], axis=1, ignore_index=True)
        sorted_columns.columns = df.columns
        update_sample = sorted_columns.sample(n=size)


    return common.CsvTable('forest', update_sample, cols=update_sample.columns)

if __name__=="__main__":
    load_permuted_census(permute=True)
    #load_partly_permuted_census()
    #load_reduced_census(filters={"equality":('marital_status', 'married')})
