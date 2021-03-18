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
ds = pd.read_csv('../../processed_data/seq_temp_all.csv')
ds = ds[ds['sequences'].notna()] # 20 sequences not available

X = np.load('../../processed_data/PB_all.npz')['arr_0']
X = np.array(X, dtype = 'f')
y = list(ds["temperatures"])
y = np.array(y, dtype='f')

# threshold 70
thresh = 70.0
y_70 = []

for i in y:
    if i >= thresh:
        y_70.append(1)
    else:
        y_70.append(0)

y_70 = np.asarray(y_70)

smote = SMOTE(random_state = 101)
X_res, y_res = smote.fit_resample(X, y_70)
X_res = np.expand_dims(X_res, axis = 1)
print(len(y), len(y_res))

y_res = to_categorical(y_res)
  
# y process
print("Loaded X and y")

X_res, y_res = shuffle(X_res, y_res, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print("Conducted Train-Test Split")

# keras nn model
input_ = Input(shape = (1, 1024,))
x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same")(input_)
x = LeakyReLU(alpha = 0.05)(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same")(x)
x = LeakyReLU(alpha = 0.05)(x)
x = Dropout(0.6)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform')(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.8)(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform')(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.8)(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform')(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.8)(x) 
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
mcp_save = keras.callbacks.ModelCheckpoint('../saved_models/cnn_pb_smote.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training
# parameters
bs = 32
num_epochs = 500
weights = {0:1, 1:30}

with tf.device('/gpu:0'): # use gpu
    history = model.fit(X_train, y_train, batch_size = bs, epochs = num_epochs, validation_data = (X_test, y_test), shuffle = False, callbacks = callbacks_list)
    model = load_model('../saved_models/cnn_pb_smote.h5', custom_objects = {'sensitivity': sensitivity})
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

Classification Report: BS 32
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      1117
           1       0.66      0.62      0.64       126

    accuracy                           0.93      1243
   macro avg       0.81      0.79      0.80      1243
weighted avg       0.93      0.93      0.93      1243


Confusion Matrix:

[[1076   41]
 [  48   78]]

F1 Score: 0.9274879902452139

Classification Report: BS 64
              precision    recall  f1-score   support

           0       0.96      0.94      0.95      1117
           1       0.57      0.69      0.63       126

    accuracy                           0.92      1243
   macro avg       0.77      0.82      0.79      1243
weighted avg       0.92      0.92      0.92      1243

Confusion Matrix
[[1052   65]
 [  39   87]]
F1 Score
0.9197514002509561

Classification Report: BS 128
              precision    recall  f1-score   support

           0       0.97      0.91      0.94      1117
           1       0.49      0.75      0.60       126

    accuracy                           0.90      1243
   macro avg       0.73      0.83      0.77      1243
weighted avg       0.92      0.90      0.91      1243

Confusion Matrix
[[1020   97]
 [  31   95]]
F1 Score
0.9061421394887608
'''
'''SMOTE Sampling:
Threshold 70
Accuracy: 0.97558
Classification Report: BS 32
              precision    recall  f1-score   support

           0       1.00      0.95      0.97      1121
           1       0.96      1.00      0.98      1131

    accuracy                           0.98      2252
   macro avg       0.98      0.98      0.98      2252
weighted avg       0.98      0.98      0.98      2252

Confusion Matrix
[[1069   52]
 [   3 1128]]
F1 Score
0.9755633329631406
'''