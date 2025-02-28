import numpy as np
import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))

def rmse_modified(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=0))

def circular_mae(y_true, y_pred):
    error = tf.math.abs(y_pred - y_true)
    return tf.reduce_mean(tf.minimum(error, tf.math.abs(2*np.pi*tf.ones_like(error) - error)), axis=-1)

def circular_mae_modified(y_true, y_pred):
    error = tf.math.abs(y_pred - y_true)
    return tf.reduce_mean(tf.minimum(error, tf.math.abs(2*np.pi*tf.ones_like(error) - error)), axis=0)

def sin_mae(y_true, y_pred):
    error = tf.math.abs(y_pred - y_true)
    return tf.reduce_mean(tf.math.sin(error/2), axis=-1)

def sin_mae_modified(y_true, y_pred):
    error = tf.math.abs(y_pred - y_true)
    return tf.reduce_mean(tf.math.sin(error/2), axis=0)

def mag_l1_l2_loss(y_true, y_pred):
    y_true_lin = 10 ** (y_true/20)
    y_pred_lin = 10 ** (y_pred/20)
    
    y_true_pow = 10 * np.log10(np.sum(y_true_lin ** 2, axis=-1)).reshape(-1, 1)
    y_pred_pow = 10 * np.log10(np.sum(y_pred_lin ** 2, axis=-1)).reshape(-1, 1)
    
    return tf.reshape(rmse(y_true, y_pred), (-1, 1)) + tf.abs(y_true_pow - y_pred_pow)

if __name__ == "__main__":
    print(rmse)