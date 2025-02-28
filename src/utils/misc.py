import numpy as np
import yaml

def mag2db(mag):
    return 20*np.log10(np.abs(mag))

def db2mag(db):
    return 10 ** (db/20)

def rad2deg(rad):
    return 180 * rad/np.pi

def deg2rad(deg):
    return np.pi * deg/180

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config