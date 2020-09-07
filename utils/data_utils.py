"""A module, which contains utilities for loading the data."""

import json
import numpy as np


def load_data(path_to_file, list_cols=['time', 'rss', 'dist']):
    # load data
    with open(path_to_file, 'r') as f:
        data = np.array(json.load(f))
    for d in data:
        for col in list_cols:
            d[col] = np.array(d[col])
    return np.array(data)



    
