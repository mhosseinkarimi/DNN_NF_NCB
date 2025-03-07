import contextlib

import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from src.data.preprocessing import preprocessing_location_polar_output
from src.train.hptuning import HPModelSelection
from src.utils.misc import load_config

# Load the data
print("Loading the data...")
bf_points = scipy.io.loadmat("data/GNF 3 null HP tuning/LCMV_BF_points_3nulls_100000data_points_50range_4angle_sep.mat")["BF_point"]
null_points = scipy.io.loadmat("data/GNF 3 null HP tuning/LCMV_null_points_3nulls_100000data_points_50range_4angle_sep.mat")["null_point"]
weights = scipy.io.loadmat("data/GNF 3 null HP tuning/LCMV_weights_3nulls_100000data_points_50range_4angle_sep.mat")["W"]
print("Data loaded successfully!")

# adds one dimension for the case of 1 null comment in other cases
# null_points = np.expand_dims(null_points, axis=-1)

# Preprocessing
data, w_mag, w_phase = preprocessing_location_polar_output(bf_points, null_points, weights)
train_data, val_data, train_w_mag, val_w_mag, train_w_phase, val_w_phase = train_test_split(data, w_mag, w_phase, test_size=0.5, random_state=42)
del data, w_mag, w_phase


# Load config file
config = load_config()

# --------------- 4 Hidden Layers ----------------------
# Initialize the model selection class
ms = HPModelSelection(params=config['hptuning_4_layers'], input_size=train_data.shape[1], output_size=train_w_phase.shape[1])

# Run the model selection search
with open(config['hptuning_4_layers']['directory'] + '/' + config['hptuning_4_layers']['name'] + '/' + 'logs.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print("Starting the Model Selection Search...")
        ms.run(train_data, train_w_phase, train_w_mag, val_data, val_w_phase, val_w_mag)
        print("Search terminated successfully!")


# --------------- 5 Hidden Layers ----------------------
# Initialize the model selection class
ms = HPModelSelection(params=config['hptuning_5_layers'], input_size=train_data.shape[1], output_size=train_w_phase.shape[1])

# Run the model selection search
with open(config['hptuning_5_layers']['directory'] + '/' + config['hptuning_5_layers']['name'] + '/' + 'logs.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print("Starting the Model Selection Search...")
        ms.run(train_data, train_w_phase, train_w_mag, val_data, val_w_phase, val_w_mag)
        print("Search terminated successfully!")

# --------------- 6 Hidden Layers ----------------------
# Initialize the model selection class
ms = HPModelSelection(params=config['hptuning_6_layers'], input_size=train_data.shape[1], output_size=train_w_phase.shape[1])

# Run the model selection search
with open(config['hptuning_6_layers']['directory'] + '/' + config['hptuning_6_layers']['name'] + '/' + 'logs.txt', 'w') as f:
    with contextlib.redirect_stdout(f):
        print("Starting the Model Selection Search...")
        ms.run(train_data, train_w_phase, train_w_mag, val_data, val_w_phase, val_w_mag)
        print("Search terminated successfully!")