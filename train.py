import contextlib

import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from src.data.preprocessing import preprocessing_location_polar_output, preprocessing_location_sine_features
from src.model.dnn import FCDNN
from src.utils.misc import load_config
from src.utils.losses import circular_mae, rmse
from src.utils.lr_scheduler import LRScheduler
from pathlib import Path
from datetime import datetime


# Load the data
print('Loading the data...')
bf_points = scipy.io.loadmat('data/General NF 2 nulls Region 8 deg separation/LCMV_BF_points_2nulls_1000000data_points.mat')['BF_point']
null_points = scipy.io.loadmat('data/General NF 2 nulls Region 8 deg separation/LCMV_null_points_2nulls_1000000data_points.mat')['null_point']
weights = scipy.io.loadmat('data/General NF 2 nulls Region 8 deg separation/LCMV_weights_2nulls_1000000data_points.mat')['W']
print('Data loaded successfully!')

# adds one dimension for the case of 1 null comment in other cases
# null_points = np.expand_dims(null_points, axis=-1)

# Preprocessing
data, w_mag, w_phase = preprocessing_location_polar_output(bf_points, null_points, weights)
train_data, val_data, train_w_mag, val_w_mag, train_w_phase, val_w_phase = train_test_split(data, w_mag, w_phase, test_size=0.5, random_state=42)
del data, w_mag, w_phase


# Load config file
config = load_config()

# Make save and log directories
log_dir = Path(config['train']['directory']['logs'])
fig_dir = Path(config['train']['directory']['figures'])
model_dir = Path(config['train']['directory']['models'])

if not log_dir.exists():
    (log_dir/'phase'/'loss').mkdir(parents=True)
    (log_dir/'phase'/'train_log').mkdir(parents=True)

    (log_dir/'magnitude'/'loss').mkdir(parents=True)
    (log_dir/'magnitude'/'train_log').mkdir(parents=True)

if not fig_dir.exists():
    (fig_dir/'phase').mkdir(parents=True)
    (fig_dir/'magnitude').mkdir(parents=True)

if not model_dir.exists():
    (model_dir/'phase').mkdir(parents=True)
    (model_dir/'magnitude').mkdir(parents=True)

#---------------------- Phase Estimation -------------------------------
# Build model
phase_model = FCDNN(
    num_layers=len(config['train']['phase']['structure']),
    units=config['train']['phase']['structure'],
    input_shape=train_data.shape[1],
    output_dim=train_w_phase.shape[1],
    dropout=config['train']['phase']['dropout'],
    loss=circular_mae,
    model_save_dir=str(model_dir/'phase'/f'DNN_{config["train"]["phase"]["structure"]}_time_{datetime.now().strftime("%Y-%m-%d-%H-%M")}')
    )

# LR scheduling
lr_scheduler = LRScheduler(
    steps=list(range(1, config['train']['phase']['num_epochs'])),
    lr=config['train']['phase']['learning_rate'], 
    mode=config['train']['phase']['lr_scheduling'], 
    lr_step=config['train']['phase']['lr_step']
    )

# Training
print('Training the phase prediction model...')
with open(str(log_dir/'phase'/'train_log'/f'train_logs_DNN_{config["train"]["phase"]["structure"]}_time_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.txt'), 'w') as f:
    with contextlib.redirect_stdout(f):
        phase_model.summary()
        loss_dict = phase_model.train(
            train_data, train_w_phase, 
            val_data, val_w_phase,
            epochs=config['train']['phase']['num_epochs'],
            batch_size=config['train']['phase']['batch_size'],
            lr=config['train']['phase']['learning_rate'],
            lr_scheduler=lr_scheduler,
            device=config['train']['phase']['device']
            )

# Save the loss values
loss_df = pd.DataFrame(loss_dict)
loss_df.to_csv(str(log_dir/'phase'/'loss'/f'loss_DNN_{config["train"]["phase"]["structure"]}_time_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv'), index=False)

# Plot the loss values
# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.plot(loss_df['train_infer'], label='Training Loss', color='blue', linestyle='-', linewidth=2)
plt.plot(loss_df['val'], label='Validation Loss', color='red', linestyle='--', linewidth=2)

# Set Times New Roman font style
font_properties = {'fontname': 'Times New Roman', 'fontsize': 14, 'fontweight': 'bold'}

# Add title and labels with Times New Roman font
plt.title('Training vs Validation Loss', **font_properties)
plt.xlabel('Epochs', fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontname='Times New Roman', fontsize=12, fontweight='bold')

# Customize legend with Times New Roman font
plt.legend(fontsize=12, prop={'family': 'Times New Roman'})

# Add grid, ticks, and formatting
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10, fontname='Times New Roman')
plt.yticks(fontsize=10, fontname='Times New Roman')

# Save the figure
plot_path = fig_dir / 'phase' / f'loss_plot_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f'Loss plot saved to {plot_path}')

# --------------------------- Magnitude Estimator ------------------------------
# Build model
mag_model = FCDNN(
    num_layers=len(config['train']['mag']['structure']),
    units=config['train']['mag']['structure'],
    input_shape=train_data.shape[1],
    output_dim=train_w_mag.shape[1],
    dropout=config['train']['mag']['dropout'],
    loss=rmse,
    model_save_dir=str(model_dir/'magnitude'/f'DNN_{config["train"]["mag"]["structure"]}_time_{datetime.now().strftime("%Y-%m-%d-%H-%M")}')
    )

# LR scheduling
lr_scheduler = LRScheduler(
    steps=list(range(1, config['train']['mag']['num_epochs'])),
    lr=config['train']['mag']['learning_rate'], 
    mode=config['train']['mag']['lr_scheduling'], 
    lr_step=config['train']['mag']['lr_step']
    )

print('Training the magnitude prediction model...')
# Training
with open(str(log_dir/'magnitude'/'train_log'/f'train_logs_DNN_{config["train"]["mag"]["structure"]}_time_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.txt'), 'w') as f:
    with contextlib.redirect_stdout(f):
        mag_model.summary()
        loss_dict = mag_model.train(
            train_data, train_w_mag, 
            val_data, val_w_mag,
            epochs=config['train']['mag']['num_epochs'],
            batch_size=config['train']['mag']['batch_size'],
            lr=config['train']['mag']['learning_rate'],
            lr_scheduler=lr_scheduler,
            device=config['train']['mag']['device']
            )

# Save the loss values
loss_df = pd.DataFrame(loss_dict)
loss_df.to_csv(str(log_dir/'magnitude'/'loss'/f'loss_DNN_{config["train"]["mag"]["structure"]}_time_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv'), index=False)

# Plot the loss values
# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.plot(loss_df['train_infer'], label='Training Loss', color='blue', linestyle='-', linewidth=2)
plt.plot(loss_df['val'], label='Validation Loss', color='red', linestyle='--', linewidth=2)

# Set Times New Roman font style
font_properties = {'fontname': 'Times New Roman', 'fontsize': 14, 'fontweight': 'bold'}

# Add title and labels with Times New Roman font
plt.title('Training vs Validation Loss', **font_properties)
plt.xlabel('Epochs', fontname='Times New Roman', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontname='Times New Roman', fontsize=12, fontweight='bold')

# Customize legend with Times New Roman font
plt.legend(fontsize=12, prop={'family': 'Times New Roman'})

# Add grid, ticks, and formatting
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10, fontname='Times New Roman')
plt.yticks(fontsize=10, fontname='Times New Roman')

# Save the figure
plot_path = fig_dir / 'magnitude' / f'loss_plot_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f'Loss plot saved to {plot_path}')