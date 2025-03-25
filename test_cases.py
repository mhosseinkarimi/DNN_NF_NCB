import numpy as np
import scipy
from src.data.preprocessing import preprocessing_location_polar_output
from src.model.dnn import FCDNN
from src.utils.losses import rmse, circular_mae
from src.utils.misc import load_config, deg2rad, db2mag


# Load config file
config = load_config()

sample_idx = np.random.randint(0, 1000000, size=10000)
bf_points = scipy.io.loadmat('data/General NF 2 nulls Region 8 deg separation/LCMV_BF_points_2nulls_1000000data_points.mat')['BF_point'][sample_idx]
null_points = scipy.io.loadmat('data/General NF 2 nulls Region 8 deg separation/LCMV_null_points_2nulls_1000000data_points.mat')['null_point'][sample_idx]
weights = scipy.io.loadmat('data/General NF 2 nulls Region 8 deg separation/LCMV_weights_2nulls_1000000data_points.mat')['W'][sample_idx]
# random weights to feed to preprocessing
# TODO: Refactor the preprocess method to remove the dependency to weights
# Preprocessing
data, w_mag, w_phase = preprocessing_location_polar_output(bf_points, null_points, weights)

# Loading Phase model
phase_model = FCDNN(
    num_layers=len(config['test']['phase']['structure']),
    units=config['test']['phase']['structure'],
    input_shape=data.shape[1],
    output_dim=24,
    dropout=None,
    loss=circular_mae,
    )                  

phase_model.model.load_weights('artifacts/models/train/phase/DNN_[1024, 1024, 1024, 1024, 1024, 1024]_time_2025-03-18-14-57')
phase_model.summary()

# Load Magnitude model
mag_model = FCDNN(
    num_layers=len(config['test']['mag']['structure']),
    units=config['test']['mag']['structure'],
    input_shape=data.shape[1],
    output_dim=24,
    dropout=None,
    loss=rmse,
    )
mag_model.model.load_weights('artifacts/models/train/magnitude/DNN_[1024, 1024, 1024, 1024, 1024, 1024]_time_2025-03-18-20-11')
mag_model.summary()

# Prediction
w_phase_pred = phase_model(data).numpy()
w_mag_pred = mag_model(data).numpy()
w_pred = db2mag(w_mag_pred) * np.exp(1j*w_phase_pred)

print(circular_mae(w_phase_pred, w_phase))
print(rmse(w_mag_pred, w_mag))
# Save test cases
scipy.io.savemat(f'{config["test"]["directory"]}/bf_points.mat', {'bf_points': bf_points})
scipy.io.savemat(f'{config["test"]["directory"]}/null_points.mat', {'null_points': null_points})
scipy.io.savemat(f'{config["test"]["directory"]}/w_pred.mat', {'w_pred': w_pred})
scipy.io.savemat(f'{config["test"]["directory"]}/w_lcmv.mat', {'w': weights})
