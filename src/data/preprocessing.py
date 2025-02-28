import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.misc import mag2db


def preprocessing_location_polar_output(bf_points, null_points, w):
    # Reformatting and reshaping data
    bf_dist = bf_points[:, 1].reshape(-1, 1)
    bf_angle = bf_points[:, 0].reshape(-1, 1)
    null_dist = null_points[:, 1]
    null_angle = null_points[:, 0]
    data = np.hstack(
        [np.hstack([bf_dist, null_dist])/6.0,
         np.hstack([bf_angle, null_angle])/(np.pi/2),
         np.cos(np.hstack([bf_angle, null_angle])),
         np.sin(np.hstack([bf_angle, null_angle]))])
    w = w / np.sqrt(np.sum(np.abs(w)**2, axis=1)).reshape(-1, 1)
    w_mag = 20*np.log10(np.abs(w))
    w_phase = np.angle(w)
    w_phase = np.mod(w_phase - np.expand_dims(w_phase[:, 0], axis=1), 2*np.pi)
    return data, w_mag, w_phase

def preprocessing_steering_vector_covariance(steering_vector, w, calculate_cov=True):
    # Reformatting and reshaping data
    scaler = StandardScaler()
    if calculate_cov:
        data = np.zeros((steering_vector.shape[1], steering_vector.shape[0], steering_vector.shape[0] + steering_vector.shape[2], 2))
        for i in range(steering_vector.shape[1]):
            # calculate covaraince 
            sv_reshape = steering_vector[:, i, :]
            cov = sv_reshape.dot(np.matrix(sv_reshape).H)
            data[i, :, :, 0] = np.hstack((sv_reshape.real, cov.real))
            data[i, :, :, 1] = np.hstack((sv_reshape.imag, cov.imag))
            data[i, :, :, 0] = scaler.fit_transform(data[i, :, :, 0])
            data[i, :, :, 1] = scaler.fit_transform(data[i, :, :, 1])
    else:
        data = np.zeros((steering_vector.shape[1], steering_vector.shape[2], 2*steering_vector.shape[0]))
        for i in range(data.shape[0]):
            steering_vector_real = np.transpose(steering_vector[:, i, :]).real
            steering_vector_imag = np.transpose(steering_vector[:, i, :]).imag
            data[i] = scaler.fit_transform(np.hstack([steering_vector_real, steering_vector_imag]))
        data = np.expand_dims(data, axis=-1)
    
    w_mag = mag2db(np.abs(w))
    w_phase = np.angle(w)
    w_phase = np.mod(w_phase - np.expand_dims(w_phase[:, 0], axis=1), 2*np.pi)
    return data, w_mag, w_phase

def preprocessing_steering_vector(steering_vector, w):
    # Reformatting and reshaping data
    data = np.zeros((steering_vector.shape[1], steering_vector.shape[2]*2*steering_vector.shape[0]))
    for i in range(data.shape[0]):
        steering_vector_mag = mag2db(np.abs(np.transpose(steering_vector[:, i, :])))
        steering_vector_phase = np.mod(np.angle(np.transpose(steering_vector[:, i, :])), 2*np.pi)
        data[i] = np.vstack([steering_vector_mag, steering_vector_phase]).flatten()    
    w = w / np.sqrt(np.sum(np.abs(w)**2, axis=1)).reshape(-1, 1)
    w_mag = 20*np.log10(np.abs(w))
    w_phase = np.angle(w)
    w_phase = np.mod(w_phase - np.expand_dims(w_phase[:, 0], axis=1), 2*np.pi)
    return data, w_mag, w_phase

def preprocessing_multi_point(bf_points, null_points, w, elem_pos):
    data = np.zeros((len(bf_points), len(elem_pos)*(2 + 2*null_points.shape[2])))
    # finding BF points relative coordinates to the antenna elements
    for i in range(len(bf_points)):
        bf_points_cart = np.hstack([bf_points[i, 1]*np.cos(bf_points[i, 0]), bf_points[i, 1]*np.sin(bf_points[i, 0])])
        null_points_cart = np.vstack([np.hstack([null_points[i, 1, k] * np.cos(null_points[i, 0, k]), null_points[i, 1, k] * np.sin(null_points[i, 0, k])]) for k in range(null_points.shape[2])])
        sample_input = []
        for j in range(len(elem_pos)):
            bf_coord = np.stack([np.linalg.norm(bf_points_cart - elem_pos[j]), np.arctan2(bf_points_cart[1] - elem_pos[j][1], bf_points_cart[0] - elem_pos[j][0])])
            null_coord = np.stack([np.stack([np.linalg.norm(null_points_cart[k] - elem_pos[j]), np.arctan2(null_points_cart[k][1] - elem_pos[j][1], null_points_cart[k][0] - elem_pos[j][0])]) for k in range(null_points.shape[2])])
            sample_input.append(np.hstack([bf_coord, null_coord.flatten()]))
        data[i] = np.hstack(sample_input)
    w = w / np.sqrt(np.sum(np.abs(w)**2, axis=1)).reshape(-1, 1)
    w_mag = 20*np.log10(np.abs(w))
    w_phase = np.angle(w)
    w_phase = np.mod(w_phase - np.expand_dims(w_phase[:, 0], axis=1), 2*np.pi)
    return data, w_mag, w_phase
