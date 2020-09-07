"""A module, which contains utilities for manipulating the dataset."""

import json
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def resample_data(input_data, number_of_seconds=1, list_cols=['time', 'rss', 'dist']):

    tmp_data = copy.deepcopy(input_data)
    for data_point in tqdm(tmp_data):
        tmp_df = pd.DataFrame({col:data_point[col] for col in list_cols})
        tmp_df['datetime'] = pd.to_datetime(tmp_df.time, unit='ms')
        tmp_df = tmp_df.set_index('datetime')
        tmp_df = tmp_df.resample(str(number_of_seconds)+'S').mean().bfill()
        tmp_df['time'] = tmp_df.index.astype(int)//1000000
        tmp_df = tmp_df.reset_index()[list_cols]
        for col in list_cols:
            data_point[col] = tmp_df[col].values
    return tmp_data

def load_data(path_to_file, list_cols=['time', 'rss', 'dist']):
    # load data
    with open(path_to_file, 'r') as f:
        data = np.array(json.load(f))
    for d in data:
        for col in list_cols:
            d[col] = np.array(d[col])
    return np.array(data)

def data_to_meta(input_data):
    meta_data = {
        'scenario':[],
        'receiver_model':[],
        'receiver_id':[],
        'sender_model':[],
        'sender_id':[],
        'combination':[],
        'room':[],
        'exp':[],
        'number_of_contacts':[]
    }
    for data_point in tqdm(input_data):
        meta_data['scenario'].append(data_point['scenario'])
        meta_data['receiver_model'].append(data_point['receiver']['phone_model'])
        meta_data['receiver_id'].append(data_point['receiver']['id'])
        meta_data['sender_model'].append(data_point['sender']['phone_model'])
        meta_data['sender_id'].append(data_point['sender']['id'])
        meta_data['combination'].append((data_point['receiver']['phone_model'], data_point['sender']['phone_model']))
        meta_data['room'].append(int(data_point['scenario'][11]))
        meta_data['exp'].append(int(data_point['scenario'][11]))
        meta_data['number_of_contacts'].append( data_point['additional_info']['number_of_contacts'])
    return pd.DataFrame(meta_data)

def split_dataset(dataset, split_method, split_parameters):
    """
    Splits the dataset into training and validation splits.

    Parameters
    ----------
        dataset: list
            A list of data points.
        split_method: str
            The name of the method that is used to split the dataset. Currently the only supported method is "room", where the
            dataset is split by rooms.
        split_parameters: tuple
            A tuple containing paramters for the split method. In case of the split method "room", the tuple must contain two
            lists: a list of the room numbers for the training split and a list of the room numbers of the validation split.

    Returns
    -------
        tuple
            Returns a tuple containing to dataset splits in the same format as the dataset that was specified in the paramters.
    """

    if split_method == 'room':
        data_train, data_test = [],[]
        for data_point in dataset:
            if int(data_point['scenario'][11]) in split_parameters[0]:
                data_train.append(data_point)
            elif int(data_point['scenario'][11]) in split_parameters[1]:
                data_test.append(data_point)
            else:
                pass
    else:
        pass

    return data_train, data_test

def calibrate_dataset(dataset, method='pairwise', reference_device='samsung_SM-A405FN'):
    df_meta = data_to_meta(dataset)
    
    devices = np.unique(df_meta.receiver_model)

    mean_calibration_matrix = np.zeros((len(devices), len(devices)))
    var_calibration_matrix = np.zeros((len(devices), len(devices)))
    n_calibration_matrix = np.zeros((len(devices), len(devices)))

    if method == 'naive':
        for (sender, receiver) in np.unique(df_meta.combination):
            [idx, idy] = [np.where(devices == sender)[0][0], np.where(devices == receiver)[0][0]]

            sender_selection = dataset[((df_meta.receiver_model == receiver)&(df_meta.sender_model == sender))]
            receiver_selection = dataset[((df_meta.receiver_model == sender)&(df_meta.sender_model == receiver))]

            receiver_rss = np.concatenate([s['rss'] for s in receiver_selection])
            receiver_dist = np.concatenate([s['dist'] for s in receiver_selection])
            
            sender_rss = np.concatenate([s['rss'] for s in sender_selection])
            sender_dist = np.concatenate([s['dist'] for s in sender_selection])
            
            means = []
            for distance in np.unique(receiver_dist):
                sd_receiver = receiver_rss[receiver_dist == distance]
                sd_sender = sender_rss[sender_dist == distance]
                
                receiver_mean = np.mean(sd_receiver)
                sender_mean = np.mean(sd_sender)
                means.append(receiver_mean-sender_mean)
 
            mean_calibration_matrix[idx, idy] = np.mean(means)
            var_calibration_matrix[idx, idy] = np.std(means)
            n_calibration_matrix[idx, idy] += len(receiver_rss)
            
            tmp_data = copy.deepcopy(dataset)
            for d in tqdm(tmp_data):
                idx = np.where(devices == d['sender']['phone_model'])[0][0]
                idy = np.where(devices == reference_device)[0][0] 
                d['rss'] -= mean_calibration_matrix[idy,idx].astype(int)

    elif method == 'pairwise':
        
        standardizers = {}

        for (receiver, transmitter) in np.unique(df_meta.combination):
            selection = dataset[((df_meta.receiver_model == receiver)&(df_meta.sender_model == transmitter))]
            rss = np.concatenate([s['rss'] for s in selection])[:,np.newaxis]
            ss = StandardScaler()
            ss.fit(rss)
            standardizers[(receiver, transmitter)] = ss
            idx = np.where(devices == receiver)[0][0]
            idy = np.where(devices == transmitter)[0][0]
            mean_calibration_matrix[idx, idy] = ss.mean_
            var_calibration_matrix[idx, idy] = ss.var_
            n_calibration_matrix[idx, idy] += np.log(len(rss))

        tmp_data = copy.deepcopy(dataset)
        for d in tqdm(tmp_data):
            tmp = standardizers[(d['receiver']['phone_model'],d['sender']['phone_model'])].transform(d['rss'][:,np.newaxis])
            d['rss'] = standardizers[(reference_device, d['sender']['phone_model'])].inverse_transform(tmp)[:,0]
    
    return tmp_data

    
