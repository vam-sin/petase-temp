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

# Apparently you may use different seed values at each stage
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

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
y = list(ds["temperatures"])
y = np.array(y, dtype='f')

# threshold 70
thresh = 70
y_70 = []

for i in y:
    if i >= thresh:
        y_70.append(1)
    else:
        y_70.append(0)

y_70 = np.asarray(y_70)
y_70 = to_categorical(y_70)
  
# y process
print("Loaded X and y")

X, y_70 = shuffle(X, y_70, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y_70, test_size=0.2, random_state=42)
print("Conducted Train-Test Split")

# keras nn model
input_ = Input(shape = (1, 1024,))
x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same")(input_)
x = LeakyReLU(alpha = 0.05)(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same")(x)
x = LeakyReLU(alpha = 0.05)(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform')(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform')(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform')(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x) 
out = Dense(2, activation = 'softmax')(x)
model = Model(input_, out)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy', sensitivity])

# callbacks
mcp_save = keras.callbacks.ModelCheckpoint('../saved_models/cnn_pb_abh.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
# parameters
bs = 4
num_epochs = 500
weights = {0:1, 1:1}

with tf.device('/gpu:0'): # use gpu
    #history = model.fit(X_train, y_train, batch_size = bs, epochs = num_epochs, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list, class_weight = weights)
    model = load_model('../saved_models/cnn_pb.h5', custom_objects = {'sensitivity': sensitivity})
    y_pred = model.predict(X_test)
    # print(y_test, y_pred)
    # Metrics
    print("Classification Report")
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))
'''
Classification Method:

Classes: 
0: Less than 70 degrees
1: >= 70

ProtBert Embeddings + Convolutional Neural Network

Classification Report
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        18
           1       1.00      1.00      1.00         4

    accuracy                           1.00        22
   macro avg       1.00      1.00      1.00        22
weighted avg       1.00      1.00      1.00        22

Confusion Matrix
[[18  0]
 [ 0  4]]
F1 Score
1.0
'''