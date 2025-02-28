import numpy as np
import scipy
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(data, label):
    """Creates a tf.Example message ready to be written to a file."""
    serialized_data = data.tobytes()
    serialize_label = label.tobytes()
    feature = {
        'data': _bytes_feature(serialized_data),
        'label': _bytes_feature(serialize_label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(file_path, dataset):
    with tf.io.TFRecordWriter(file_path) as writer:
        for data, label in dataset:
            example = serialize_example(data, label)
            writer.write(example)

def write_tfrecord_iteratively(
    file_path, data_file_list, label_file_list, 
    data_name, label_name,
    preprocess_fcn, label_type, **kwargs):
    
    with tf.io.TFRecordWriter(file_path) as writer:
        for data_path, label_path in zip(data_file_list, label_file_list):
            data = scipy.io.loadmat(data_path)[data_name]
            label = scipy.io.loadmat(label_path)[label_name]
            data, w_mag, w_phase = preprocess_fcn(data, label, **kwargs)
            if label_type == "mag":
                w = w_mag
            elif label_type == "phase":
                w = w_phase
            elif label_type == "combined":
                w = np.stack((w_mag, w_phase), axis=1)
                w = w.flatten()
            else:
                raise ValueError('The value for label_type should be selected from "mag" or "phase"')
            
            for data_ex, w_ex in zip(data, w):
                example = serialize_example(data_ex, w_ex)
                writer.write(example)

def write_tfrecord_location_iteratively(
    file_path, 
    bf_file_list,
    null_file_list,
    weight_file_list, 
    preprocess_fcn,
    label_type,
    **kwargs):
    
    with tf.io.TFRecordWriter(file_path) as writer:
        for bf_path, null_path, weight_path in zip(bf_file_list, null_file_list, weight_file_list):
            bf_points = scipy.io.loadmat(bf_path)["BF_point"]
            null_points = scipy.io.loadmat(null_path)["null_point"]
            w = scipy.io.loadmat(weight_path)["W"]
            data, w_mag, w_phase = preprocess_fcn(bf_points, null_points, w)
            if label_type == "mag":
                w = w_mag
            elif label_type == "phase":
                w = w_phase
            elif label_type == "combined":
                w = np.stack((w_mag, w_phase), axis=1)
                w = w.flatten()
            else:
                raise ValueError('The value for label_type should be selected from "mag" or "phase"')
            
            for data_ex, w_ex in zip(data, w):
                example = serialize_example(data_ex, w_ex)
                writer.write(example)
            
def parse_tfrecord_sv_cov(example_proto):
    # Define your feature description here
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.decode_raw(example['data'], tf.float64)
    data = tf.reshape(data, (24, 28, 2))  # Replace with the original shape of your data
    label = tf.io.decode_raw(example['label'], tf.float64)
    label = tf.reshape(label, (24,))/np.pi
    return data, label

def parse_tfrecord_sv(example_proto):
    # Define your feature description here
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.decode_raw(example['data'], tf.float64)
    data = tf.reshape(data, (4, 48))  # Replace with the original shape of your data
    data = tf.expand_dims(data, axis=-1)
    label = tf.io.decode_raw(example['label'], tf.float64)
    label = tf.reshape(label, (24,))/np.pi
    return data, label

def parse_tf_record_location(example_proto):
    # Define your feature description here
    feature_description = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    data = tf.io.decode_raw(example['data'], tf.float64)
    label = tf.io.decode_raw(example['label'], tf.float64)
    return data, label

def create_dataset(file_path, batch_size, parse_tfrecord_fn):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset
