# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import math
import pickle
import re
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras import optimizers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import regularizers
from keras import backend as K
import keras

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

# dataset import 
ds = pd.read_csv('../../processed_data/seq_temp_abh.csv')
X = np.load('../../processed_data/PB_abh.npz')['arr_0']
X = np.array(X, dtype = 'f')
X = np.expand_dims(X, axis = 1)
y = np.asarray(list(ds["temperatures"]))
y = np.array(y, dtype='f')

# y process
print("Loaded X and y")

X, y = shuffle(X, y, random_state=42)
print("Shuffled")

# indices = np.arange(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Conducted Train-Test Split")
print(X_train.shape)

bs = 8

with tf.device('/cpu:0'): # use gpu

    # keras nn model
    input_ = Input(shape = (1,1024,))
    x = Flatten()(input_)
    x = Dense(1024, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = "relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    out = Dense(1, activation = 'linear')(x)
    model = Model(input_, out)

    print(model.summary())

    # adam optimizer
    opt = keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(optimizer = "adam", loss = "mean_squared_error", metrics=['mse'])

    # callbacks
    mcp_save = keras.callbacks.ModelCheckpoint('../saved_models/ann_pb.h5', save_best_only=True, monitor='val_mse', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks_list = [reduce_lr, mcp_save]

    print("Training")
    # training
    num_epochs = 500
    history = model.fit(X_train, y_train, batch_size = bs, epochs = num_epochs, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list)

'''
Results:
MSE: 394.21994 (BS 1)
MSE: 283.81451 (BS 2)
MSE: 73.92136 (BS 4) - Best
MSE: 113.67749 (BS 8) 
'''


