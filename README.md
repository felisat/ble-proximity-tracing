# Estimation of SARS-CoV-2 Transmission from Bluetooth Low Energy Measurements


This repository contains the Bluetooth Low Energy received signal strength measurements with corresponding ground-truth distances which were analyzed in the paper

[Estimation of SARS-CoV-2 Transmission from Bluetooth Low Energy Measurements](https://arxiv.org/abs/2004.11841)
by Felix Sattler, Jackie Ma, Patrick Wagner, David Neumann, Markus Wenzel, Ralf Schäfer, Wojciech Samek, Klaus-Robert Müller, Thomas Wiegand

A detailed description of the experiment setup can be found in the paper and the accompanying supplementary materials.

# How to use the data
The data is stored in .json files in the `data/` folder. To load the data you can use the python scripts available in `utils/`. For example, running

```python
from utils import data_utils
data_train = data_utils.load_data('data/bundeswehr-04-01/data_train.json')
```

will load a list of data items from the training split, which was collected on the 1st of April 2020. Every data item is a dictionary containing the following keys:

- `'rss'`: A sequence of RSSI measurements collected by the device.
- `'time'`: A sequence of corresponding time-stamps.
- `'dist'`: A sequence of corresponding groud-truth distance measurements.
- `'receiver_id'`: The unique ID of the receiving device.
- `'transmitter_id'`: The unique ID of the transmitting device.
- `'room'`: The location in which the data was colected. Rooms 1, 2, and 3 are indoor locations. Rooms 4 and 5 are outdoor locations.
- `'model'`: The phone model of the receiving device.
- `'scenario'`: Contains additional information about the experiment setup.
- `'holding_position'`: The holding position of the receiving device.


# Licensing

All code in this repository is available under the MIT License. Files in the data/ directory are available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.


# Citation
Cite this work as 

```
Sattler, Felix, et al. "Risk Estimation of SARS-CoV-2 Transmission from Bluetooth Low Energy Measurements." arXiv preprint arXiv:2004.11841 (2020).
```


# Contact

Contact wojciech.samek@hhi.fraunhofer.de with any questions about this repository.

