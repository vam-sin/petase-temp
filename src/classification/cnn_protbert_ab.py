#libraries
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, normalize
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
from tensorflow.keras.layers import LeakyReLU, Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, MaxPooling1D, Add
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
ds = pd.read_csv('../../processed_data/seq_temp_ab.csv')
# ds = ds[ds['sequences'].notna()] # 20 sequences not available

X = np.load('../../processed_data/PB_ab.npz')['arr_0']
X = np.array(X, dtype = 'f')
X = normalize(X)
y = list(ds["temperature"])
y = np.array(y, dtype='f')

# threshold 70
thresh = 55.0
y_55 = []

for i in y:
    if i >= thresh:
        y_55.append(1)
    else:
        y_55.append(0)

y_55 = np.asarray(y_55)

unique, counts = np.unique(y_55, return_counts=True)
print(unique, counts)

# Non-SMOTE
# y_55 = to_categorical(y_55)
# X = np.expand_dims(X, axis = 1)

# y process
print("Loaded X and y")

X, y = shuffle(X, y_55, random_state=42)
print("Shuffled")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# X_train = np.expand_dims(X_train, axis = 1)
# X_test = np.expand_dims(X_test, axis = 1)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# X_train = list(X_train)
# y_train = list(y_train)

# y_val = []
# X_val = []

# length = len(X_train)

# for i in range(length):
#     if y_train[i] == 1:
#         y_val.append(y_train[i])
#         X_val.append(X_train[i])
#         y_train.pop(i)
#         X_train.pop(i)
#     if len(y_val) == 100:
#         break

# for i in range(length):
#     if y_train[i] == 0:
#         y_val.append(y_train[i])
#         X_val.append(X_train[i])
#         y_train.pop(i)
#         X_train.pop(i)
#     if len(y_val) == 200:
#         break

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.125, random_state = 42) 

# SMOTE
smote = SMOTE(random_state = 101)
X_train, y_train = smote.fit_resample(X_train, y_train)
X_val, y_val = smote.fit_resample(X_val, y_val)

X_train = np.expand_dims(X_train, axis = 1)
X_val = np.expand_dims(X_val, axis = 1)
X_test = np.expand_dims(X_test, axis = 1)
# print("Conducted Train-Test Split")
# X_train = np.asarray(X_train)
# X_val = np.asarray(X_val)
# y_train = np.asarray(y_train)
# y_val = np.asarray(y_val)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

print(X_test.shape, X_train.shape, X_val.shape, y_test.shape, y_train.shape, y_val.shape)

# parameters
bs = 64
num_epochs = 400
weights = {0:1, 1:10}
dropout_val = 0.15
model_file_name = '../saved_models/cnn_pb_ab_clas_1_1_smote.h5'

def ResBlock(inp):
    x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(inp)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(x)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    # x = Add()([x, inp])

    return x

# keras nn model
input_ = Input(shape = (1, 1024,))
x = Conv1D(512, (3), kernel_initializer = 'glorot_uniform', padding="same", kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(input_)
x = LeakyReLU(alpha = 0.05)(x)
x = Dropout(0.7)(x)
x = BatchNormalization()(x)

# for blocks in range(1):
#     x = ResBlock(x)

x = Flatten()(x)
# x = Dense(1024, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(x)
# x = LeakyReLU(alpha = 0.05)(x)
# x = BatchNormalization()(x)
# x = Dropout(dropout_val)(x)
# x = Dense(1024, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(x)
# x = LeakyReLU(alpha = 0.05)(x)
# x = BatchNormalization()(x)
# x = Dropout(dropout_val)(x)
x = Dense(1024, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(x)
x = LeakyReLU(alpha = 0.05)(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x) 
out = Dense(2, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-4))(x)
model = Model(input_, out)

print(model.summary())

# adam optimizer
opt = keras.optimizers.Adam(learning_rate = 1e-4)

# sensitivity metric
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

# callbacks
mcp_save = keras.callbacks.ModelCheckpoint(model_file_name, save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', factor=0.1, patience=40, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

# training

with tf.device('/gpu:0'): # use gpu
    history = model.fit(X_train, y_train, batch_size = bs, epochs = num_epochs, validation_data = (X_val, y_val), shuffle = False, callbacks = callbacks_list, class_weight = weights)
    
    # Plot History
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model = load_model(model_file_name, custom_objects = {'sensitivity': sensitivity})
    
    y_pred = model.predict(X_val)
    # print(y_pred_prob)
    print("Classification Report Validation")
    print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Confusion Matrix")
    matrix = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))


    y_pred = model.predict(X_test)
    # print(y_pred_prob)
    print("Classification Report Test")
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average = 'weighted'))


'''55
bs = 128
num_epochs = 500
weights = {0:1, 1:1}
dropout_val = 0.7

Classification Report Validation
              precision    recall  f1-score   support

           0       0.65      0.93      0.77       100
           1       0.88      0.51      0.65       100

    accuracy                           0.72       200
   macro avg       0.77      0.72      0.71       200
weighted avg       0.77      0.72      0.71       200

Confusion Matrix
[[93  7]
 [49 51]]
F1 Score
0.7070823307877394
Classification Report Test
              precision    recall  f1-score   support

           0       0.87      0.93      0.90       521
           1       0.69      0.53      0.60       152

    accuracy                           0.84       673
   macro avg       0.78      0.73      0.75       673
weighted avg       0.83      0.84      0.83       673

Confusion Matrix
[[484  37]
 [ 71  81]]
F1 Score
0.8319558985179825

#########################
bs = 128
num_epochs = 500
weights = {0:1, 1:10}
dropout_val = 0.4

Classification Report Validation
              precision    recall  f1-score   support

           0       0.91      0.90      0.91       273
           1       0.61      0.63      0.62        63

    accuracy                           0.85       336
   macro avg       0.76      0.77      0.76       336
weighted avg       0.86      0.85      0.86       336

Confusion Matrix
[[247  26]
 [ 23  40]]
F1 Score
0.8554595485888047
Classification Report Test
              precision    recall  f1-score   support

           0       0.87      0.90      0.88       521
           1       0.61      0.55      0.58       152

    accuracy                           0.82       673
   macro avg       0.74      0.72      0.73       673
weighted avg       0.81      0.82      0.82       673

Confusion Matrix
[[467  54]
 [ 68  84]]
F1 Score
0.8155481196656839


#########
bs = 128
num_epochs = 500
weights = {0:1, 1:20}
dropout_val = 0.2
Classification Report Validation
              precision    recall  f1-score   support

           0       0.85      0.97      0.90       273
           1       0.64      0.25      0.36        63

    accuracy                           0.83       336
   macro avg       0.74      0.61      0.63       336
weighted avg       0.81      0.83      0.80       336

Confusion Matrix
[[264   9]
 [ 47  16]]
F1 Score
0.8027708592777086
Classification Report Test
              precision    recall  f1-score   support

           0       0.82      0.98      0.89       521
           1       0.77      0.28      0.41       152

    accuracy                           0.82       673
   macro avg       0.80      0.63      0.65       673
weighted avg       0.81      0.82      0.78       673

Confusion Matrix
[[508  13]
 [109  43]]
F1 Score
0.7845349536306535

'''