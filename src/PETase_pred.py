#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
import math
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LeakyReLU, Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, MaxPooling1D
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow import keras
import os
import random
from imblearn.over_sampling import SMOTE

# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# model
model = load_model('saved_models/cnn_pb_abh_pet.h5', custom_objects = {'sensitivity': sensitivity})

petase_pb = np.load('../processed_data/PB_PETase.npz')['arr_0']
petase_pb = np.expand_dims(petase_pb, axis=1)

y_pred = model.predict(petase_pb)

print(y_pred)

'''
Ground Truth: [0,0,0,0,0,1]
Ground Truth: [49, 55, 56, 49, 58, 79]

Predictions CNN_PB_SMOTE_ALL_CLAS:
[[1.0000000e+00 2.3519826e-08] - 0
 [1.0000000e+00 2.8369568e-08] - 0 
 [9.9999976e-01 2.1806292e-07] - 0
 [1.0000000e+00 4.0027592e-11] - 0
 [9.9999905e-01 9.4503071e-07] - 0
 [9.9142951e-01 8.5705305e-03]] - 0

Predictions CNN_PB_ALL_CLAS:
[[9.9961984e-01 3.8015953e-04] - 0
 [9.9972147e-01 2.7857331e-04] - 0
 [9.9971801e-01 2.8191364e-04] - 0
 [9.9975854e-01 2.4149123e-04] - 0
 [9.9953949e-01 4.6047330e-04] - 0
 [9.9649519e-01 3.5047936e-03]] - 0

Predictions CNN_PB_ABH_CLAS:
[[0.9630317  0.03696835] - 0
 [0.9674205  0.03257949] - 0
 [0.9740097  0.02599025] - 0
 [0.959992   0.04000795] - 0
 [0.96564263 0.03435737] - 0
 [0.61113536 0.38886467]] - 0

 Predictions CNN_PB_ABH_PET_CLAS: BEST
 [[0.9797538  0.02024623] - 0
 [0.97991234 0.02008769] - 0
 [0.98267937 0.01732068] - 0
 [0.97580373 0.02419624] - 0
 [0.9801838  0.01981621] - 0
 [0.33680946 0.66319054]] - 1
'''